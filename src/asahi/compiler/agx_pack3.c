/*
 * Copyright 2026 hitori1403
 * SPDX-License-Identifier: MIT
 */

/*
 * The G15 (AGX3) instruction packer: AGX IR after register allocation to
 * AGX3 machine code, through the encoder tables gen-pack.py generates from
 * AGX3.xml (agx3_pack.h).
 *
 * Each instruction starts from a template: an encoding Apple's compiler
 * emitted (the applegpu-checked corpus, or the lab's running shader). Its
 * unpacked operands carry every field the IR does not model (the q flags,
 * the "expect" bits, the t/v consistency bits) at the values Apple uses.
 * The packer then overwrites the operands the IR does model. An operand
 * value is built from its encoding's first case (the disassembler's order)
 * whose register class matches and which offers the flags to set.
 */

#include <setjmp.h>
#include "util/bitscan.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "agx3_pack.h"
#include "agx_builder.h"
#include "agx_compiler.h"

/* agx3_kind_names' order */
enum a3_kind {
   A3_STACK,
   A3_CR,
   A3_UNIFORM,
   A3_REG,
   A3_IMM,
   A3_FIMM,
   A3_TS,
   A3_SS,
   A3_CF,
};

enum a3_mod {
   A3_BFLOAT,
   A3_CACHE,
   A3_SX,
   A3_ABS,
   A3_NEG,
   A3_SAT,
   A3_LU,
};

#define A3_MAX_OPS 32

struct a3 {
   const agx_instr *I;
   const struct agx3_pack_spec *spec;
   unsigned length;
   uint64_t ops[A3_MAX_OPS];
   uint64_t tmpl[A3_MAX_OPS];

   /* The values set, checked against the packed instruction */
   uint64_t want[A3_MAX_OPS];
   uint32_t set;

   /* The shader's scratch in bytes: the thread-private stack below it */
   unsigned scratch_B;
};

static unsigned
a3_hex(char c)
{
   return c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10;
}

static void
a3_begin(struct a3 *a, const agx_instr *I, enum agx3_instr instr,
         const char *tmpl)
{
   uint8_t in[AGX3_MAX_BYTES] = {0};
   unsigned n = strlen(tmpl) / 2;

   assert(n <= AGX3_MAX_BYTES);
   for (unsigned i = 0; i < n; i++)
      in[i] = (a3_hex(tmpl[2 * i]) << 4) | a3_hex(tmpl[2 * i + 1]);

   a->I = I;
   a->spec = agx3_spec(instr);
   a->length = n;
   a->set = 0;
   assert(a->spec->nops <= A3_MAX_OPS);
   agx3_unpack(in, a->spec, n, a->ops);
   memcpy(a->tmpl, a->ops, sizeof(a->ops));
}

/* Set an operand, clearing the bits other operands (display modifiers that
 * alias it) hold in the same instruction bits, so a template's value
 * cannot leak through the OR in agx3_pack. */
static void
a3_set(struct a3 *a, unsigned op, uint64_t v)
{
   const struct agx3_pack_spec *s = a->spec;
   uint64_t own[3] = {0};

   assert(op < s->nops);
   for (unsigned f = s->op_start[op]; f < s->op_start[op + 1]; f++) {
      for (unsigned b = 0; b < s->fields[f].size; b++) {
         unsigned bit = s->fields[f].start + b;
         own[bit / 64] |= BITFIELD64_BIT(bit % 64);
      }
   }

   for (unsigned j = 0; j < s->nops; j++) {
      if (j == op)
         continue;

      for (unsigned f = s->op_start[j]; f < s->op_start[j + 1]; f++) {
         for (unsigned b = 0; b < s->fields[f].size; b++) {
            unsigned bit = s->fields[f].start + b;
            if (own[bit / 64] & BITFIELD64_BIT(bit % 64))
               a->ops[j] &= ~BITFIELD64_BIT(s->fields[f].vpos + b);
         }
      }
   }

   a->ops[op] = v;
   a->want[op] = v;
   a->set |= BITFIELD_BIT(op);
}

static uint32_t
a3_offered(const struct agx3_case *c)
{
   uint32_t m = 0;
   for (unsigned f = c->nkind; f < c->nfields; f++)
      m |= BITFIELD_BIT(c->mod[f]);
   return m;
}

/* The first case of an operand with the kind and size that offers mods;
 * among those, the first that also offers prefer (the integer cases offer
 * sx where the float cases offer neg). */
static const struct agx3_case *
a3_case(const struct a3 *a, unsigned op, enum a3_kind kind, unsigned size,
        uint32_t mods, uint32_t prefer)
{
   const struct agx3_pack_spec *s = a->spec;
   const struct agx3_case *cases = s->cases + s->case_start[op];
   const struct agx3_case *fallback = NULL;

   for (unsigned i = 0; i < s->ncases[op]; i++) {
      const struct agx3_case *c = &cases[i];

      if (!c->nkind || c->kind != kind || (c->skey & AGX3_SKEY_BFLOAT))
         continue;
      if ((c->skey & AGX3_SKEY_SIZE_MASK) != size)
         continue;

      uint32_t offered = a3_offered(c);
      if (mods & ~offered)
         continue;
      if ((offered & prefer) == prefer)
         return c;
      if (!fallback)
         fallback = c;
   }

   return fallback;
}

/* The value bits of operand op that share instruction bits with an operand
 * already set, and those operands' values for them */
static uint64_t
a3_shared(const struct a3 *a, unsigned op, uint64_t *val)
{
   const struct agx3_pack_spec *s = a->spec;
   uint64_t mask = 0;

   *val = 0;
   for (unsigned f = s->op_start[op]; f < s->op_start[op + 1]; f++) {
      const struct agx3_field *fo = &s->fields[f];

      u_foreach_bit(j, a->set) {
         if (j == op)
            continue;

         for (unsigned g = s->op_start[j]; g < s->op_start[j + 1]; g++) {
            const struct agx3_field *fj = &s->fields[g];

            for (unsigned b = 0; b < fo->size; b++) {
               unsigned bit = fo->start + b;
               if (bit < fj->start || bit >= fj->start + fj->size)
                  continue;

               uint64_t vb = BITFIELD64_BIT(fo->vpos + b);
               mask |= vb;
               if ((a->ops[j] >> (fj->vpos + bit - fj->start)) & 1)
                  *val |= vb;
            }
         }
      }
   }

   return mask;
}

/* The carry bits are outside the disassembler's model, but some of them
 * keep an earlier case from matching (a register destination whose carry
 * bit is clear reads as a cr register), and some are bits another operand
 * owns (cmpsel's selected values carry the destination's size bit). Choose
 * them so that the disassembler's first match is c: the bits operands set
 * earlier hold, then the template's bits if they match, else the smallest
 * subset of the free carry bits that does. */
static uint64_t
a3_fix_carry(const struct a3 *a, unsigned op, const struct agx3_case *c,
             uint64_t v)
{
   const struct agx3_pack_spec *s = a->spec;
   const struct agx3_case *cases = s->cases + s->case_start[op];
   unsigned n = s->ncases[op];
   uint64_t pad = s->op_pad ? s->op_pad[op] : 0;
   uint64_t shared_val, shared = a3_shared(a, op, &shared_val);
   uint64_t fixed = c->carry_mask & shared;
   uint64_t free = c->carry_mask & ~shared;
   uint64_t base = (v & ~c->carry_mask) | (shared_val & fixed);

   /* The disassembler reads the constant bits of the operand too */
   uint64_t t = base | (a->tmpl[op] & free);
   if (agx3_case_match(cases, n, t | pad) == c)
      return t;

   unsigned bits[64], nbits = 0;
   u_foreach_bit64(b, free)
      bits[nbits++] = b;
   assert(nbits <= 12);

   for (unsigned pop = 0; pop <= nbits; pop++) {
      for (unsigned m = 0; m < (1u << nbits); m++) {
         if (util_bitcount(m) != pop)
            continue;

         uint64_t carry = 0;
         u_foreach_bit(i, m)
            carry |= BITFIELD64_BIT(bits[i]);

         if (agx3_case_match(cases, n, base | carry | pad) == c)
            return base | carry;
      }
   }

   return UINT64_MAX;
}

static unsigned
a3_size(enum agx_size size)
{
   return size == AGX_SIZE_16 ? 0 : size == AGX_SIZE_32 ? 1 : 2;
}

/* AGX3_SURVEY=1: count what the packer cannot pack, by message and opcode,
 * print the table at exit and keep going (the instruction is dropped) */
static jmp_buf a3_survey_jmp;
static bool a3_survey_on, a3_debug;
static unsigned a3_survey_count[AGX_NUM_OPCODES][8];
static const char *a3_survey_what[8];

static void
a3_survey_print(void)
{
   for (unsigned op = 0; op < AGX_NUM_OPCODES; op++) {
      for (unsigned w = 0; w < ARRAY_SIZE(a3_survey_what); w++) {
         if (a3_survey_count[op][w])
            fprintf(stderr, "AGX3 survey: %6u %-24s %s\n",
                    a3_survey_count[op][w], agx_opcodes_info[op].name,
                    a3_survey_what[w]);
      }
   }
}

static void NORETURN
a3_fail(const struct a3 *a, const char *what)
{
   if (a3_survey_on) {
      unsigned w = 0;
      while (w < ARRAY_SIZE(a3_survey_what) - 1 && a3_survey_what[w] &&
             strcmp(a3_survey_what[w], what))
         w++;
      a3_survey_what[w] = what;
      a3_survey_count[a->I->op][w]++;
      longjmp(a3_survey_jmp, 1);
   }

   fprintf(stderr, "AGX3: %s: ", what);
   agx_print_instr(a->I, stderr);
   UNREACHABLE("AGX3 packing failed");
}

/* A register or uniform operand */
static uint64_t
a3_reg(struct a3 *a, unsigned op, agx_index idx, uint32_t mods, uint32_t prefer,
       unsigned count_field)
{
   enum a3_kind kind;

   if (idx.type == AGX_INDEX_REGISTER)
      kind = A3_REG;
   else if (idx.type == AGX_INDEX_UNIFORM)
      kind = A3_UNIFORM;
   else
      a3_fail(a, "operand is not a register or uniform");

   const struct agx3_case *c =
      a3_case(a, op, kind, a3_size(idx.size), mods, prefer);
   if (!c)
      a3_fail(a, "no case for an operand");

   struct agx3_reg r = {
      .index = idx.value >> 1,
      .half = idx.value & 1,
      .count_field = count_field,
      .mods = mods,
   };

   uint64_t v = a3_fix_carry(a, op, c, agx3_reg_value(c, &r));
   if (v == UINT64_MAX)
      a3_fail(a, "no carry bits select the operand's case");

   return v;
}

/* An immediate operand of a register encoding: an integer, or for a float
 * operand the 8-bit float immediate (the case's first of the kind, falling
 * back to the other kind for operands that have only one) */
static uint64_t
a3_imm(struct a3 *a, unsigned op, uint32_t value, bool fl)
{
   const struct agx3_case *c = a3_case(a, op, fl ? A3_FIMM : A3_IMM, 0, 0, 0);
   uint64_t modvals[AGX3_MAX_CASE_FIELDS] = {0};

   if (!c)
      c = a3_case(a, op, fl ? A3_IMM : A3_FIMM, 0, 0, 0);
   if (!c)
      a3_fail(a, "no immediate case");

   unsigned bits = 0;
   for (unsigned f = 0; f < c->nkind; f++)
      bits += c->f[f].size;
   if (value >> bits)
      a3_fail(a, "immediate out of range");

   uint64_t v = a3_fix_carry(a, op, c, agx3_case_value(c, value, modvals, 0));
   if (v == UINT64_MAX)
      a3_fail(a, "no carry bits select the immediate's case");

   return v;
}

/* An operand of a kind without registers (a coordinate register, a
 * coefficient register): value is the number the disassembler shows,
 * mods the flags to set */
static uint64_t
a3_kind_value(struct a3 *a, unsigned op, enum a3_kind kind, uint32_t value,
              uint32_t mods)
{
   const struct agx3_case *c = a3_case(a, op, kind, 0, mods, 0);
   uint64_t modvals[AGX3_MAX_CASE_FIELDS] = {0};

   if (!c)
      a3_fail(a, "no case for an operand");

   unsigned bits = 0;
   for (unsigned f = 0; f < c->nkind; f++)
      bits += c->f[f].size;
   if (value >> bits)
      a3_fail(a, "operand out of range");

   for (unsigned f = c->nkind; f < c->nfields; f++)
      modvals[f - c->nkind] = (mods >> c->mod[f]) & 1;

   uint64_t v = a3_fix_carry(a, op, c, agx3_case_value(c, value, modvals, 0));
   if (v == UINT64_MAX)
      a3_fail(a, "no carry bits select the operand's case");

   return v;
}

/* A 32-bit stack register, stN (N in 32-bit units; Apple numbers them in
 * 16-bit units) */
static uint64_t
a3_stack(struct a3 *a, unsigned op, unsigned index, uint32_t mods)
{
   const struct agx3_case *c =
      a3_case(a, op, A3_STACK, a3_size(AGX_SIZE_32), mods, 0);
   if (!c)
      a3_fail(a, "no case for a stack register");

   struct agx3_reg r = {.index = index, .mods = mods};
   uint64_t v = a3_fix_carry(a, op, c, agx3_reg_value(c, &r));
   if (v == UINT64_MAX)
      a3_fail(a, "no carry bits select the operand's case");

   return v;
}

/* A source of an ALU instruction: a register, a uniform or an 8-bit
 * immediate. Float sources carry abs and neg; integer sources pick the
 * integer cases (which offer sx). */
static uint64_t
a3_src(struct a3 *a, unsigned op, agx_index idx, bool integer)
{
   if (idx.type == AGX_INDEX_IMMEDIATE)
      return a3_imm(a, op, idx.value, !integer);

   uint32_t mods = 0;
   if (idx.abs && !integer)
      mods |= BITFIELD_BIT(A3_ABS);
   if (idx.neg)
      mods |= BITFIELD_BIT(A3_NEG);

   return a3_reg(a, op, idx, mods, BITFIELD_BIT(integer ? A3_SX : A3_NEG), 0);
}

/* An extended integer source (iadd, imad, a memory index): the IR's abs is
 * G13's zero extension, so a source narrower than 64 bits without it is
 * sign-extended (sx) */
static uint64_t
a3_ext_src(struct a3 *a, unsigned op, agx_index idx)
{
   if (idx.type == AGX_INDEX_IMMEDIATE)
      return a3_imm(a, op, idx.value, false);

   bool sx = !idx.abs && idx.size != AGX_SIZE_64;
   return a3_reg(a, op, idx, sx ? BITFIELD_BIT(A3_SX) : 0, BITFIELD_BIT(A3_SX),
                 0);
}

/* A compared value: G15 extends a 16-bit register before it compares, and
 * without sx it zero-extends, so a signed compare of 16-bit values needs sx
 * (the 16-bit clamp of a signed integer render target turned every value into
 * -128; cmpsel verified, cmp and the ballots by analogy). Immediates are small
 * and positive. */
static uint64_t
a3_cmp_src(struct a3 *a, unsigned op, agx_index idx, bool integer,
           enum agx_icond icond)
{
   bool sgn = icond == AGX_ICOND_SLT || icond == AGX_ICOND_SGT;

   if (!integer || !sgn || idx.type == AGX_INDEX_IMMEDIATE ||
       idx.size != AGX_SIZE_16)
      return a3_src(a, op, idx, integer);

   return a3_reg(a, op, idx, BITFIELD_BIT(A3_SX), BITFIELD_BIT(A3_SX), 0);
}

/* A t/v consistency modifier (dst_vt): its first bit is the complement of
 * its second; the destination holds both as carry bits (and the group the
 * second). Set the first from the second as packed so far, in every
 * operand that holds it. */
static void
a3_vt(struct a3 *a, unsigned op)
{
   const struct agx3_pack_spec *s = a->spec;
   uint8_t out[AGX3_MAX_BYTES] = {0};
   const struct agx3_field *t = &s->fields[s->op_start[op]];
   const struct agx3_field *v = t + 1;

   assert(s->op_start[op + 1] - s->op_start[op] == 2);
   assert(t->size == 1 && v->size == 1);
   a->ops[op] = 0;
   agx3_pack(out, s, a->length, a->ops);

   bool vbit = (out[v->start / 8] >> (v->start % 8)) & 1;
   a->ops[op] = vbit ? 2 : 1;

   for (unsigned j = 0; j < s->nops; j++) {
      for (unsigned f = s->op_start[j]; f < s->op_start[j + 1]; f++) {
         const struct agx3_field *g = &s->fields[f];
         if (j == op || t->start < g->start || t->start >= g->start + g->size)
            continue;

         uint64_t bit = BITFIELD64_BIT(g->vpos + t->start - g->start);
         a->ops[j] = vbit ? a->ops[j] & ~bit : a->ops[j] | bit;
         a->want[j] = vbit ? a->want[j] & ~bit : a->want[j] | bit;
      }
   }
}

/* Pack the instruction into out and check it. Operands share instruction
 * bits: an operand set later can change one set earlier, and an operand
 * value can have bits the encoding cannot hold (a register too high for a
 * short form, a modifier it lacks). The instruction must read back as set. */
static bool
a3_check(const struct a3 *a, uint8_t *out)
{
   uint64_t back[A3_MAX_OPS];

   memset(out, 0, AGX3_MAX_BYTES);
   agx3_pack(out, a->spec, a->length, a->ops);
   agx3_unpack(out, a->spec, a->length, back);

   u_foreach_bit(op, a->set) {
      uint64_t pad = a->spec->op_pad ? a->spec->op_pad[op] : 0;

      if (back[op] != (a->want[op] | pad)) {
         if (a3_debug)
            fprintf(stderr,
                    "AGX3: %s operand %u: set 0x%" PRIx64 ", reads 0x%" PRIx64
                    "\n",
                    a->spec->name, op, a->want[op], back[op]);
         return false;
      }
   }

   return agx3_instr_length(out, AGX3_MAX_BYTES) == a->length;
}

static void
a3_emit(struct a3 *a, struct util_dynarray *emission)
{
   uint8_t out[AGX3_MAX_BYTES];
   bool ok = a3_check(a, out);

   /* A multi-length encoding's longer forms hold operand bits the template's
    * length cannot (ffma10's abs on its second source is bit 81 of the 12-byte
    * form), so fall back to them. */
   for (unsigned i = 0; !ok && a->spec->has_length && i < a->spec->nlengths;
        i++) {
      if (a->spec->lengths[i] > a->length) {
         a->length = a->spec->lengths[i];
         ok = a3_check(a, out);
      }
   }

   if (!ok)
      a3_fail(a, "the operands do not fit the encoding");

   memcpy(util_dynarray_grow_bytes(emission, 1, a->length), out, a->length);
}

/* G13 special registers to G15 (AGX3.xml's sr names) */
static unsigned
a3_sr(const struct a3 *a, enum agx_sr sr)
{
   switch (sr) {
   case AGX_SR_THREADGROUP_POSITION_IN_GRID_X:
   case AGX_SR_THREADGROUP_POSITION_IN_GRID_Y:
   case AGX_SR_THREADGROUP_POSITION_IN_GRID_Z:
      return 28 + (sr - AGX_SR_THREADGROUP_POSITION_IN_GRID_X);
   case AGX_SR_THREADS_PER_THREADGROUP_X:
   case AGX_SR_THREADS_PER_THREADGROUP_Y:
   case AGX_SR_THREADS_PER_THREADGROUP_Z:
      return 24 + (sr - AGX_SR_THREADS_PER_THREADGROUP_X);
   case AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_X:
   case AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_Y:
   case AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_Z:
      return 40 + (sr - AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_X);
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_X:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_Y:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_Z:
      return 36 + (sr - AGX_SR_THREAD_POSITION_IN_THREADGROUP_X);
   case AGX_SR_THREAD_INDEX_IN_THREADGROUP:
      return 39;
   case AGX_SR_THREAD_INDEX_IN_SUBGROUP:
      return 2;
   case AGX_SR_SUBGROUP_INDEX_IN_THREADGROUP:
      return 5;
   case AGX_SR_ACTIVE_THREAD_INDEX_IN_QUAD:
      return 20;
   case AGX_SR_ACTIVE_THREAD_INDEX_IN_SUBGROUP:
      return 21;
   case AGX_SR_IS_ACTIVE_THREAD:
      return 4;
   case AGX_SR_THREAD_POSITION_IN_GRID_X:
   case AGX_SR_THREAD_POSITION_IN_GRID_Y:
   case AGX_SR_THREAD_POSITION_IN_GRID_Z:
      return 32 + (sr - AGX_SR_THREAD_POSITION_IN_GRID_X);
   /* Apple's names (libLLVM's register table) PVQUAD and PVSIMD are the
    * active thread indices above; TVQUAD and TVSIMD read as the totals */
   case AGX_SR_TOTAL_ACTIVE_THREADS_IN_QUAD:
      return 22;
   case AGX_SR_TOTAL_ACTIVE_THREADS_IN_SUBGROUP:
      return 23;
   case AGX_SR_BACKFACING:
      return 69;
   /* The tile's layer: Apple's end-of-tile program reads sr67 into its
    * layered block writes' pack_imgwir (smp.md) */
   case AGX_SR_LAYER_ID:
      return 67;
   /* SR_CVGINFO, the pixel's sample coverage: Apple's per-sample shading
    * masks its tile stores with its bits (draws.md, Multisampling) */
   case AGX_SR_COVERAGE_MASK:
      return 66;
   /* demo15's vertex shader reads sr93 (VERTEXID in libLLVM's register
    * table) for its vertex; the instance ID is sr88 (INSTID, not verified
    * on the hardware). Whether sr93 includes the base vertex is not known. */
   case AGX_SR_VERTEX_ID:
      return 93;
   case AGX_SR_INSTANCE_ID:
      return 88;
   default:
      a3_fail(a, "special register not mapped");
   }
}

/* The device_load/store component mask: the IR's xyzw bitmask to the
 * hardware's enumeration (AGX3.xml device_load_mask) */
static const uint8_t a3_mem_mask[16] = {
   0, 1, 8, 9, 10, 11, 12, 13, 14, 15, 2, 3, 4, 5, 6, 7,
};

static unsigned
a3_mem_type(const struct a3 *a, enum agx_format format)
{
   switch (format) {
   case AGX_FORMAT_I8:
      return 2;
   case AGX_FORMAT_I16:
      return 0;
   case AGX_FORMAT_I32:
      return 1;
   default:
      a3_fail(a, "memory format not supported");
   }
}

static unsigned
a3_format_shift(enum agx_format format)
{
   return format == AGX_FORMAT_I8 ? 0 : format == AGX_FORMAT_I16 ? 1 : 2;
}

/* The memreg operand: the register, the type and the mask in one value.
 * The case's count field is the mask; its exact bits pick the type. */
static uint64_t
a3_memreg(struct a3 *a, unsigned op, agx_index reg, unsigned type,
          unsigned mask)
{
   const struct agx3_pack_spec *s = a->spec;
   const struct agx3_case *cases = s->cases + s->case_start[op];
   unsigned size = type == 1 ? 1 : 0;

   if (reg.type != AGX_INDEX_REGISTER)
      a3_fail(a, "memory register is not a register");

   for (unsigned i = 0; i < s->ncases[op]; i++) {
      const struct agx3_case *c = &cases[i];
      uint64_t tbits = (uint64_t)type << 8;

      if (!c->nkind || c->kind != A3_REG ||
          (c->skey & AGX3_SKEY_SIZE_MASK) != size)
         continue;
      if ((tbits & c->exact_mask) != (c->exact & BITFIELD64_MASK(10)))
         continue;

      struct agx3_reg r = {
         .index = reg.value >> 1,
         .half = reg.value & 1,
         .count_field = mask,
         .carry = tbits,
      };

      uint64_t v = agx3_reg_value(c, &r);
      if (agx3_case_match(cases, s->ncases[op], v) != c)
         a3_fail(a, "memreg value does not select its case");

      return v;
   }

   a3_fail(a, "no memreg case");
}

static uint64_t
a3_membase(struct a3 *a, unsigned op, agx_index base)
{
   if (base.size != AGX_SIZE_64)
      a3_fail(a, "memory base is not 64-bit");

   return a3_reg(a, op, base, 0, 0, 0);
}

static uint64_t
a3_memidx(struct a3 *a, unsigned op, agx_index idx)
{
   return a3_ext_src(a, op, idx);
}

/* The memory index shift in bytes (G15's shift and offset are in bytes,
 * G13's in elements) to the field: 1 = none, 2-4 = lsl 1-3, and 0 = lsl 4
 * (Apple's encoding; boot #143: the 5 applegpu names lsl 4 read the
 * unscaled index). 5-7 are not used. */
static unsigned
a3_memshift(const struct a3 *a, unsigned bytes)
{
   if (bytes > 4)
      a3_fail(a, "memory index shift out of range");

   return bytes == 4 ? 0 : bytes + 1;
}

/* Templates (little-endian bytes) */
#define T_MOV_IMM32 "2c80020000000000"             /* mov_imm r2, 0 */
#define T_GET_SR    "fca05006"                     /* get_sr 0, r31, sr32 */
#define T_IADD      "9f015400020410a81105"         /* iadd r0, r1, r2 */
#define T_BFI       "270054000200001090100800"     /* bfi r0, r1, 0, 0 */
#define T_MOV       "0b020100"                     /* mov r0, r1 */
#define T_LOAD      "67005400061f2000570100404000" /* device_load 5, ... */
#define T_STORE                                                                \
   "e7005400020000001100009008"                                                \
   "00"
#define T_WAIT     "0604" /* wait 0 */
#define T_STOP     "0e000000"
#define T_BITOP    "0b030605000800000000" /* andn1 r0, r1, r2 */
#define T_CMPSEL   "02030305020605000008" /* cmpsel ult, r0..r4 */
#define T_CMPSEL14 T_CMPSEL "00000000"
#define T_IMADD    "9f00540002041030d0212a00" /* imadd r0, r1, r2, r3 */
#define T_IADD64   "9f015400020820501105"     /* iadd dr0, dr2, dr4 */
#define T_IADD_SH  "9f0154000200f8c81004"     /* iadd r0, 0, r0, lsl 1 */
#define T_BFEIL    "a70054000200001090100800" /* bfeil r0, r1, 0, 0 */
#define T_ASR      "a7015400020408182200"     /* asr r0, r1, r2 */
#define T_IUNOP    "2705540002044c04"         /* popcount r0, r1 */
#define T_F2I      "27075400020494480300"     /* convert f_to_s32 */
#define T_I2F      "a707540002048c20"         /* convert u32_to_f */
#define T_FUNOP    "2f005400020490400200"     /* floor r0, r1 */
#define T_FADD     "01000400020000000000"     /* fadd (10 B) */
#define T_FFMA     "01000600020000000000"     /* ffma (10 B) */
#define T_ZERO10   "00000000000000000000"
#define T_ICMP_BAL "1707540002041038a202"     /* icmp_ballot ieq */
#define T_FCMP_BAL "97075400020410a888840000" /* fcmp_ballot flt */
#define T_SHUFFLE  "c70454000204022c0000"     /* simd_shuffle_xor */
#define T_IF       "0f055401"                 /* if cc0, 1 */
#define T_ELSE     "0f042401"                 /* else cc0, 1 */
#define T_WHILE    "8f045405"                 /* while cc1, 1 */
#define T_BREAK    "8f0554010000"             /* break cc0, 1 */
#define T_POP_EXEC "0f0604010000"             /* pop_exec 1 */
#define T_JMP      "0f015402a0c900000100"     /* jmp_exec_none */
#define T_ZERO12   "000000000000000000000000"
#define T_ZERO14   "0000000000000000000000000000"

/* libLLVM 32023's encoder (GPUCompiler.framework, macOS 26A434) with the
 * operands below: wait none, the hint bits at their defaults */
#define T_LDTG      "6702540200000000440000400000" /* threadgroup_load r1, x */
#define T_STTG      "e702540200000000440000100000" /* threadgroup_store */
#define T_ATG       "670354010002000002000004"     /* threadgroup_atomic add */
#define T_ADEV      "6701540000048182000000008003" /* device_atomic add */
#define T_ADEV_RET  "6701540000048182000600008003" /* ... 3, st0, add */
#define T_GET_STACK "0c80094300800000"         /* get_stack 2, 2, r0, ^st0 */
#define T_LDTPS     "6f0054000000112000000000" /* ldtps 0, i32, x, r0, 0 */
#define T_STTPS     "ef005400000011080000"     /* sttps i32, x, r0, 0 */
#define T_BARRIER   "070454610300"             /* threadgroup_barrier */
#define T_ST_VAR    "5706540000404800"         /* st_var ^cr0, 0, 0 */
#define T_ST_TILE   "e70654000000012e00000000" /* tile_store i32 */
#define T_LD_TILE   "670654000000012e01000000" /* tile_load 2, i32 */
#define T_WAIT_PIX  "8702540c0800"             /* wait_pix 12, f1 */
#define T_SIG_PIX   "0702540c0000"             /* signal_pix 12 */
#define T_LDCF      "1f0354000002"             /* ldcf 2, r0, 0 */
#define T_ITER      "2f055400020000021000"     /* iter r0, cf0, center */
#define T_ITER_COORDS "2f055404030202022000"   /* iter $r2, cf1, ^r2 */
#define T_ITRCOORDS   "af04540403080a03"       /* itrcoords r2, $r2, sample */
#define T_ZS_EMIT     "d70454000003"           /* zs_emit ^cr0, 0, z */
#define T_MOVMSK      "570454000000"           /* movmsk ^cr0, 0 */
#define T_PACK_IMGWIR                                                          \
   "9705540400080000d0401000" /* pack_imgwir cr2, 0, u2l, 0, 0, 4 */
/* Apple's end-of-tile layer pack (pack_imgwir cr2, 2, ^r4l, 0x204, 0, 0) */
#define T_PACK_IMGWIR_LAYER "9705540400101000f0d00000"
#define T_PACK_SMPIR                                                           \
   "1705540600000004f0c00800" /* pack_smpir cr3, 0, ^r0l, 0, 10 */
#define T_IMGWBLK                                                              \
   "5710540000000080a8750000" /* block_image_store ^cr0, 0, du0, 0, 7530 */
/* Apple's rw.write() of the imgw capture (macOS 27.0.1): image_write ^cr8,
 * group 0, du2, index 1, level 0, xyzw, wait 0 */
#define T_IMGW "d71054102040000100040000883a0900"

/* Apple's texture instructions (the 2026-10-16 capture's kernels and the
 * QuartzCore corpus, texture.md) */
#define T_TEX_L0 "05820cb830250900000010000100" /* texture_sample 2D, level 0 */
#define T_TEX_AUTO                                                             \
   "05000cb8b0000000010110000100" /* texture_sample 2D, auto LOD */
#define T_TEX_LOAD "25800c38b0001701000100000100" /* texture_load 2D, level 0 */
#define T_TEX_LEVELW                                                           \
   "15000c38b0000001000110000100" /* texture_sample 2D, level in cr3 */
#define T_TEX_BIASW                                                            \
   "05000cb8b0240400000010000100" /* texture_sample 2D, bias in cr3 */

/* Operand positions (AGX3.xml order) */
enum {
   MOV_IMM_DST = 0,
   MOV_IMM_IMM = 1,
   GET_SR_G = 0,
   GET_SR_DST = 1,
   GET_SR_SR = 2,
   IADD_DST = 0,
   IADD_A = 3,
   IADD_B = 4,
   IADD_SAT = 7,
   BFI_DST = 0,
   BFI_A = 2,
   BFI_B = 3,
   BFI_C = 4,
   BFI_MASK = 5,
   MOV_DST = 0,
   MOV_SRC = 1,
   MEM_G = 0,
   MEM_REG = 3,
   MEM_BASE = 4,
   MEM_INDEX = 5,
   MEM_SHIFT = 6,
   MEM_OFFSET = 7,
   WAIT_MASK = 0,
   BITOP_TT = 0,
   CMPSEL_CC = 0,
   VT = 1,
   IADD_WT = 2,
   IADD_SHIFT = 5,
   CONVERT_MODE = 0,
   CONVERT_DST = 1,
   CONVERT_VT = 2,
   CONVERT_SRC = 3,
   CONVERT_RND = 4,
   STTG_REG = 2,
   STTG_BASE = 3,
   STTG_INDEX = 4,
   STTG_SHIFT = 5,
   STTG_OFFSET = 6,
   ATG_G = 0,
   ATG_DST = 1,
   ATG_DATA = 2,
   ATG_BASE = 3,
   ATG_INDEX = 4,
   ATG_OP = 5,
   ATG_SHIFT = 6,
   ATG_OFFSET = 7,
   ADEV_DATA = 0,
   ADEV_BASE = 1,
   ADEV_INDEX = 2,
   ADEV_OP = 3,
   ADEV_SHIFT = 4,
   ADEV_OFFSET = 5,
   ADEV_RET_G = 0,
   ADEV_RET_DST = 1,
   GET_STACK_G = 0,
   GET_STACK_Q = 1,
   GET_STACK_DST = 2,
   GET_STACK_SRC = 3,
   GET_STACK_WAIT = 4,
   TPS_G = 0,
   TPS_REG = 3,
   TPS_INDEX = 4,
   TPS_OFFSET = 5,
   BARRIER_FLAGS = 0,
   BARRIER_WAIT = 1,
   ST_VAR_CR = 0,
   ST_VAR_SLOT = 1,
   ST_VAR_G = 2,
   ST_VAR_WAIT = 3,
   ST_TILE_REG = 2,
   LD_TILE_G = 0,
   LD_TILE_REG = 3,
   PIX_MASK = 0,
   WAIT_PIX_F0 = 1,
   WAIT_PIX_F1 = 2,
   LDCF_G = 0,
   LDCF_DST = 1,
   LDCF_CF = 2,
   ITER_DST = 0,
   ITER_CF = 1,
   ITER_MODE = 2,
   ITER_PERSP = 3,
   ITER_COORDS_DST = 0,
   ITER_COORDS_CF = 1,
   ITER_COORDS_CRD = 2,
   ITRCOORDS_DST = 0,
   ITRCOORDS_SRC = 1,
   ITRCOORDS_SAMPLE = 2,
   ITRCOORDS_DCACHE = 3,
   ZS_CR = 0,
   ZS_G = 1,
   ZS_MODE = 2,
   ZS_WAIT = 3,
   MOVMSK_CR = 0,
   MOVMSK_G = 1,
   MOVMSK_WAIT = 2,
   PACK_IMGWIR_DST = 0,
   PACK_IMGWIR_G = 1,
   PACK_IMGWIR_A = 2,
   PACK_IMGWIR_B = 3,
   PACK_IMGWIR_C = 4,
   PACK_IMGWIR_LAYOUT = 5,
   PACK_IMGWIR_WAIT = 6,
   PACK_SMPIR_DST = 0,
   PACK_SMPIR_G = 1,
   PACK_SMPIR_A = 2,
   PACK_SMPIR_C = 3,
   PACK_SMPIR_MODE = 4,
   PACK_SMPIR_WAIT = 5,
   IMGWBLK_CR = 0,
   IMGWBLK_G = 1,
   IMGWBLK_PTR = 2,
   IMGWBLK_OFFSET = 3,
   IMGWBLK_FLAGS = 4,
   IMGWBLK_WAIT = 7,
   IMGW_CR = 0,
   IMGW_G = 1,
   IMGW_BASE = 2,
   IMGW_INDEX = 3,
   IMGW_LOD = 4,
   IMGW_DATA = 5,
   IMGW_WAIT = 6,
   TEX_G0 = 0,
   TEX_G1 = 1,
   TEX_MASK = 2,
   TEX_DST = 3,
   TEX_TS = 4,
   TEX_CR = 5,
   TEX_Q1 = 6,
   TEX_Q2 = 7,
   TEX_WAIT = 8,
   TEX_SS = 10,
   TEX_USMP = 11,
   TEX_UTEX = 12,
   TEX_LOD = 13,
   TEX_Q3 = 13,
   TEX_OFFSET = 14,
   TEX_MODE = 15,
   /* a gather's component sits where the others' mask does */
   TEX_COMPONENT = TEX_MASK,
   SPECLM_SAMPLES = 0,
   SPECLM_STRIDE = 1,
   SPECLM_SHAPE = 2,
   SPECLM_IMAGEBLOCK = 3,
   SPECLM_THREADGROUP = 4,
   SPECLM_SIZE = 5,
   LDSHDR_VA = 0,
   LDSHDR_SIZE = 1,
   SPECTPR_CR = 0,
   SPECTPR_STACK = 1,
   SPECTPR_STACK2 = 2,
};

/* An ALU instruction: the destination, the sources in IR order at the
 * operand positions srcs, the t/v modifiers and the saturate flag (-1 for
 * none) */
struct a3_alu {
   enum agx3_instr instr;
   const char *tmpl;
   uint8_t srcs[4];
   int8_t vt, wt, sat;
   bool integer, ext;
};

static void
a3_alu(struct a3 *a, const agx_instr *I, const struct a3_alu *d,
       const agx_index *srcs, struct util_dynarray *emission)
{
   unsigned nsrcs = agx_opcodes_info[I->op].nr_srcs;

   a3_begin(a, I, d->instr, d->tmpl);
   a3_set(a, 0, a3_reg(a, 0, I->dest[0], 0, 0, 0));

   for (unsigned i = 0; i < nsrcs; i++) {
      a3_set(a, d->srcs[i],
             d->ext ? a3_ext_src(a, d->srcs[i], srcs[i])
                    : a3_src(a, d->srcs[i], srcs[i], d->integer));
   }

   if (d->sat >= 0)
      a3_set(a, d->sat, I->saturate);
   else if (I->saturate)
      a3_fail(a, "saturate not supported");

   if (d->vt >= 0)
      a3_vt(a, d->vt);
   if (d->wt >= 0)
      a3_vt(a, d->wt);

   a3_emit(a, emission);
}

static const struct a3_alu a3_imadd = {AGX3_IMADD, T_IMADD, {2, 3, 4}, 1,
                                       -1,         7,       true,      true};
static const struct a3_alu a3_imsub = {AGX3_IMSUB, T_IMADD, {2, 3, 4}, 1,
                                       -1,         7,       true,      true};
static const struct a3_alu a3_iadd = {AGX3_IADD, T_IADD, {3, 4}, 1,
                                      2,         7,      true,   true};
static const struct a3_alu a3_isub = {AGX3_ISUB, T_IADD, {3, 4}, 1,
                                      2,         7,      true,   true};
static const struct a3_alu a3_iadd64 = {AGX3_IADD64, T_IADD64, {3, 4}, 1,
                                        2,           -1,       true,   true};
static const struct a3_alu a3_isub64 = {AGX3_ISUB64, T_IADD64, {3, 4}, 1,
                                        2,           -1,       true,   true};
static const struct a3_alu a3_iadd_sh = {
   AGX3_IADD_SHIFT, T_IADD_SH, {3, 4}, 1, 2, -1, true, true};
static const struct a3_alu a3_isub_sh = {
   AGX3_ISUB_SHIFT, T_IADD_SH, {3, 4}, 1, 2, -1, true, true};
static const struct a3_alu a3_bfi = {AGX3_BFI, T_BFI, {2, 3, 4}, 1,
                                     -1,       -1,    true};
static const struct a3_alu a3_bfeil = {AGX3_BFEIL, T_BFEIL, {2, 3, 4}, 1,
                                       -1,         -1,      true};
static const struct a3_alu a3_extr = {AGX3_EXTR, T_BFEIL, {2, 3, 4}, 1,
                                      -1,        -1,      true};
static const struct a3_alu a3_asr = {AGX3_ASR, T_ASR, {2, 3}, 1, -1, -1, true};
static const struct a3_alu a3_bitrev = {AGX3_BITREV, T_IUNOP, {2}, 1,
                                        -1,          -1,      true};
static const struct a3_alu a3_popcount = {
   AGX3_POPCOUNT, T_IUNOP, {2}, 1, -1, -1, true};
static const struct a3_alu a3_ffs = {AGX3_FFS, T_IUNOP, {2}, 1, -1, -1, true};
static const struct a3_alu a3_fadd = {AGX3_FADD10, T_FADD, {1, 2}, -1,
                                      -1,          5,      false};
static const struct a3_alu a3_fmul = {AGX3_FMUL12, T_ZERO12, {1, 2}, -1,
                                      -1,          5,        false};
static const struct a3_alu a3_ffma = {AGX3_FFMA10, T_FFMA, {1, 2, 3}, -1,
                                      -1,          6,      false};
static const struct a3_alu a3_hadd = {AGX3_HADD10, T_ZERO10, {1, 2}, -1,
                                      -1,          4,        false};
static const struct a3_alu a3_hmul = {AGX3_HMUL10, T_ZERO10, {1, 2}, -1,
                                      -1,          4,        false};
static const struct a3_alu a3_hfma = {AGX3_HFMA10, T_FFMA, {1, 2, 3}, -1,
                                      -1,          5,      false};

/* The float unary operations: dest, t/v, source */
static enum agx3_instr
a3_funop(enum agx_opcode op)
{
   switch (op) {
   case AGX_OPCODE_FLOOR:
      return AGX3_FLOOR;
   case AGX_OPCODE_CEIL:
      return AGX3_CEIL;
   case AGX_OPCODE_TRUNC:
      return AGX3_TRUNC;
   case AGX_OPCODE_ROUNDEVEN:
      return AGX3_RINT;
   case AGX_OPCODE_RCP:
      return AGX3_RCP;
   case AGX_OPCODE_RSQRT:
      return AGX3_RSQRT;
   case AGX_OPCODE_SRSQRT:
      return AGX3_RSQRT_SPECIAL;
   case AGX_OPCODE_LOG2:
      return AGX3_LOG2;
   case AGX_OPCODE_EXP2:
      return AGX3_EXP2;
   case AGX_OPCODE_SIN_PT_2:
      return AGX3_SINC; /* sin(A pi / 2) / A */
   default:
      return AGX3_NUM_INSTRS;
   }
}

/* bitop: the truth table bit (a | b << 1) is the IR's. Tables that read
 * only b misbehave (applegpu: 0011 acts as and, 1100 hangs): those read
 * a with the sources swapped. */
static void
a3_bitop(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   unsigned tt = I->truth_table & 0xf;
   agx_index x = I->src[0], y = I->src[1];

   if (tt == 0xc || tt == 0x3) {
      tt = (tt & 0x9) | ((tt & 2) << 1) | ((tt & 4) >> 1);
      x = I->src[1];
      y = I->src[0];
   }

   a3_begin(a, I, AGX3_BITOP10, T_BITOP);
   a3_set(a, BITOP_TT, tt);
   a3_set(a, 1, a3_reg(a, 1, I->dest[0], 0, 0, 0));
   a3_set(a, 2, a3_src(a, 2, x, true));
   a3_set(a, 3, a3_src(a, 3, y, true));
   a3_emit(a, emission);
}

/* icmpsel/fcmpsel: D = A cc B ? X : Y. The selected values are moved as
 * integers unless they carry float modifiers. */
static void
a3_cmpsel(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   static const uint8_t icc[] = {
      [AGX_ICOND_UEQ] = 15, [AGX_ICOND_ULT] = 5, [AGX_ICOND_UGT] = 4,
      [AGX_ICOND_SEQ] = 15, [AGX_ICOND_SLT] = 7, [AGX_ICOND_SGT] = 6,
   };
   static const uint8_t fcc[] = {
      [AGX_FCOND_EQ] = 8,  [AGX_FCOND_LT] = 3,  [AGX_FCOND_GT] = 2,
      [AGX_FCOND_LTN] = 1, [AGX_FCOND_GE] = 10, [AGX_FCOND_LE] = 11,
      [AGX_FCOND_GTN] = 0,
   };
   bool integer = I->op == AGX_OPCODE_ICMPSEL;
   unsigned cc = integer ? icc[I->icond] : fcc[I->fcond];
   bool sel_int = true;

   for (unsigned i = 2; i < 4; i++)
      sel_int &= !I->src[i].abs && !I->src[i].neg;

   /* The 10-byte form lacks abs on the compared values and the high
    * destination bits; the 14-byte form has them */
   for (unsigned len = 10;; len = 14) {
      uint8_t out[AGX3_MAX_BYTES];

      a3_begin(a, I, len == 10 ? AGX3_CMPSEL10 : AGX3_CMPSEL14,
               len == 10 ? T_CMPSEL : T_CMPSEL14);
      a3_set(a, CMPSEL_CC, cc);
      a3_set(a, 1, a3_reg(a, 1, I->dest[0], 0, 0, 0));
      a3_set(a, 2, a3_cmp_src(a, 2, I->src[0], integer, I->icond));
      a3_set(a, 3, a3_cmp_src(a, 3, I->src[1], integer, I->icond));
      a3_set(a, 4, a3_src(a, 4, I->src[2], sel_int));
      a3_set(a, 5, a3_src(a, 5, I->src[3], sel_int));

      if (len == 14 || a3_check(a, out))
         break;
   }

   a3_emit(a, emission);
}

/* iadd: D = A + (B << shift), isub for a negated B; 64-bit adds have no
 * shift */
static void
a3_iadd_op(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   agx_index srcs[2] = {I->src[0], I->src[1]};
   bool sub = srcs[1].neg;
   srcs[1].neg = false;

   if (srcs[0].neg)
      a3_fail(a, "iadd with a negated first source");

   if (I->dest[0].size == AGX_SIZE_64) {
      if (I->shift)
         a3_fail(a, "64-bit iadd with a shift");

      a3_alu(a, I, sub ? &a3_isub64 : &a3_iadd64, srcs, emission);
   } else if (I->shift) {
      if (I->shift > 4 || I->saturate)
         a3_fail(a, "iadd shift form not supported");

      const struct a3_alu *d = sub ? &a3_isub_sh : &a3_iadd_sh;
      a3_begin(a, I, d->instr, d->tmpl);
      a3_set(a, 0, a3_reg(a, 0, I->dest[0], 0, 0, 0));
      a3_set(a, 3, a3_ext_src(a, 3, srcs[0]));
      a3_set(a, 4, a3_ext_src(a, 4, srcs[1]));
      a3_set(a, IADD_SHIFT, I->shift & 3);
      a3_vt(a, 1);
      a3_vt(a, 2);
      a3_emit(a, emission);
   } else {
      a3_alu(a, I, sub ? &a3_isub : &a3_iadd, srcs, emission);
   }
}

/* convert: src[0] is the G13 mode, src[1] the value */
static void
a3_convert(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   static const int8_t f2i[16] = {
      [0 ... 15] = -1,
      [AGX_CONVERT_F_TO_U16] = 4,
      [AGX_CONVERT_F_TO_S16] = 3,
      [AGX_CONVERT_F_TO_U32] = 0,
      [AGX_CONVERT_F_TO_S32] = 5,
   };
   static const int8_t i2f[16] = {
      [0 ... 15] = -1,
      [AGX_CONVERT_U8_TO_F] = 1,
      [AGX_CONVERT_S8_TO_F] = 3,
      [AGX_CONVERT_U16_TO_F] = 0,
      [AGX_CONVERT_S16_TO_F] = 2,
      [AGX_CONVERT_U32_TO_F] = 4,
      [AGX_CONVERT_S32_TO_F] = 6,
   };

   if (I->src[0].type != AGX_INDEX_IMMEDIATE || I->src[0].value > 15)
      a3_fail(a, "convert mode is not an immediate");

   unsigned mode = I->src[0].value;
   bool to_int = f2i[mode] >= 0;

   if (!to_int && i2f[mode] < 0)
      a3_fail(a, "convert mode not supported");

   a3_begin(a, I, to_int ? AGX3_CONVERT_F2I : AGX3_CONVERT_I2F,
            to_int ? T_F2I : T_I2F);
   a3_set(a, CONVERT_MODE, to_int ? f2i[mode] : i2f[mode]);
   a3_set(a, CONVERT_DST, a3_reg(a, CONVERT_DST, I->dest[0], 0, 0, 0));
   a3_set(a, CONVERT_SRC, a3_src(a, CONVERT_SRC, I->src[1], !to_int));
   if (!to_int)
      a3_set(a, CONVERT_RND, I->round == AGX_ROUND_RTZ ? 1 : 0);
   a3_vt(a, CONVERT_VT);
   a3_emit(a, emission);
}

static void
a3_mov_imm(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   a3_begin(a, I, AGX3_MOV_IMM32, T_MOV_IMM32);
   a3_set(a, MOV_IMM_DST, a3_reg(a, MOV_IMM_DST, I->dest[0], 0, 0, 0));
   a3_set(a, MOV_IMM_IMM, (uint32_t)I->imm);
   a3_emit(a, emission);
}

static void
a3_mem(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   bool store = I->op == AGX_OPCODE_DEVICE_STORE;
   agx_index reg = store ? I->src[0] : I->dest[0];
   agx_index base = I->src[store ? 1 : 0];
   agx_index index = I->src[store ? 2 : 1];
   unsigned type = a3_mem_type(a, I->format);

   a3_begin(a, I, store ? AGX3_DEVICE_STORE : AGX3_DEVICE_LOAD,
            store ? T_STORE : T_LOAD);
   /* A store with g bit 0 set frees its source register: it reads 0 after
    * (boot #264). Apple's stores use g = 0. */
   a3_set(a, MEM_G, store ? 0 : I->scoreboard);
   a3_set(a, MEM_REG, a3_memreg(a, MEM_REG, reg, type, a3_mem_mask[I->mask]));
   a3_set(a, MEM_BASE, a3_membase(a, MEM_BASE, base));
   a3_set(a, MEM_INDEX, a3_memidx(a, MEM_INDEX, index));
   a3_set(a, MEM_SHIFT, a3_memshift(a, I->shift + a3_format_shift(I->format)));
   a3_set(a, MEM_OFFSET, 0);
   a3_emit(a, emission);
}

/* The IR's local_load and local_atomic are synchronous; G15's ldtg and
 * atomictg deliver in a load group. They use group 2 (the scoreboard
 * assigns 0 and 1) and wait for it. */
#define A3_SYNC_G 2

static void
a3_wait(struct a3 *a, const agx_instr *I, unsigned g,
        struct util_dynarray *emission)
{
   a3_begin(a, I, AGX3_WAIT, T_WAIT);
   a3_set(a, WAIT_MASK, BITFIELD_BIT(g));
   a3_emit(a, emission);
}

/* A threadgroup memory base or index: a 16-bit register or uniform, or an
 * immediate (the IR's zero) */
static uint64_t
a3_tgsrc(struct a3 *a, unsigned op, agx_index idx)
{
   if (idx.type == AGX_INDEX_IMMEDIATE)
      return a3_imm(a, op, idx.value, false);

   if (idx.size != AGX_SIZE_16)
      a3_fail(a, "threadgroup address is not 16-bit");

   return a3_reg(a, op, idx, 0, 0, 0);
}

/* local_load/local_store (ldtg/sttg): the IR's base is a byte address in a
 * 16-bit register (or zero) and its index zero; the index is scaled by the
 * format as on device memory. */
static void
a3_local_mem(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   bool store = I->op == AGX_OPCODE_LOCAL_STORE;
   agx_index reg = store ? I->src[0] : I->dest[0];
   agx_index base = I->src[store ? 1 : 0];
   agx_index index = I->src[store ? 2 : 1];
   unsigned type = a3_mem_type(a, I->format);
   unsigned r = store ? STTG_REG : MEM_REG;

   a3_begin(a, I, store ? AGX3_THREADGROUP_STORE : AGX3_THREADGROUP_LOAD,
            store ? T_STTG : T_LDTG);
   if (!store)
      a3_set(a, MEM_G, A3_SYNC_G);
   a3_set(a, r, a3_memreg(a, r, reg, type, a3_mem_mask[I->mask]));
   a3_set(a, r + 1, a3_tgsrc(a, r + 1, base));
   a3_set(a, r + 2, a3_tgsrc(a, r + 2, index));
   a3_set(a, r + 3, a3_memshift(a, a3_format_shift(I->format)));
   a3_set(a, r + 4, 0);
   a3_emit(a, emission);

   if (!store)
      a3_wait(a, I, A3_SYNC_G, emission);
}

/* stack_load/stack_store (ldtps/sttps): the thread-private stack, which
 * the entry program's specTPR sizes (agx3_stack_size). The IR's offset in
 * bytes is the index (a register or a 16-bit immediate), unscaled: the
 * index shift stays 0, as in the corpus's ldtps through a byte pointer
 * (Apple's stack kernel scales an element index by 4 instead). 32-bit data
 * moves as 16-bit halves, as in every ldtps/sttps of Apple's, so one
 * instruction moves at most two 32-bit components: NIR splits wider
 * scratch accesses (mem_access_size_align_cb) and a wider spill, whose
 * offset is an immediate, is split here. A load delivers in its
 * scoreboard's group; a store keeps its data register (lu clear).
 * vkcompute's stk set passes on the hardware. */
static void
a3_stack_mem(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   bool store = I->op == AGX_OPCODE_STACK_STORE;
   agx_index reg = store ? I->src[0] : I->dest[0];
   agx_index offset = I->src[store ? 1 : 0];
   bool halves = I->format == AGX_FORMAT_I32;
   unsigned type = a3_mem_type(a, halves ? AGX_FORMAT_I16 : I->format);
   unsigned comps = util_last_bit(I->mask);
   unsigned step = halves ? 2 : 4;

   if (I->mask != BITFIELD_MASK(comps))
      a3_fail(a, "stack access mask is not contiguous");

   for (unsigned c = 0; c < comps; c += step) {
      unsigned n = MIN2(comps - c, step);
      agx_index r = reg;
      r.value += halves ? 2 * c : c;

      a3_begin(a, I, store ? AGX3_STTPS : AGX3_LDTPS,
               store ? T_STTPS : T_LDTPS);
      if (!store)
         a3_set(a, TPS_G, I->scoreboard);
      a3_set(a, TPS_REG,
             a3_memreg(a, TPS_REG, r, type,
                       a3_mem_mask[BITFIELD_MASK(halves ? 2 * n : n)]));

      if (offset.type == AGX_INDEX_IMMEDIATE) {
         unsigned bytes = c * (halves ? 4 : agx_size_align_16(r.size) * 2);
         a3_set(a, TPS_INDEX,
                a3_imm(a, TPS_INDEX, offset.value + bytes, false));
      } else if (c == 0) {
         a3_set(a, TPS_INDEX,
                a3_reg(a, TPS_INDEX, offset, 0, BITFIELD_BIT(A3_SX), 0));
      } else {
         a3_fail(a, "wide stack access at a register offset");
      }

      a3_set(a, TPS_OFFSET, 0);
      a3_emit(a, emission);
   }
}

/* The IR's atomic operations (G13's numbering, which Apple's compiler also
 * uses: the low 4 bits of its memory flags operand) to the hardware's code,
 * Apple's table */
static unsigned
a3_atomic_op(const struct a3 *a, enum agx_atomic_opc op)
{
   static const uint8_t hw[] = {
      [AGX_ATOMIC_OPC_ADD] = 0,   [AGX_ATOMIC_OPC_SUB] = 11,
      [AGX_ATOMIC_OPC_XCHG] = 14, [AGX_ATOMIC_OPC_CMPXCHG] = 2,
      [AGX_ATOMIC_OPC_UMIN] = 13, [AGX_ATOMIC_OPC_IMIN] = 5,
      [AGX_ATOMIC_OPC_UMAX] = 12, [AGX_ATOMIC_OPC_IMAX] = 4,
      [AGX_ATOMIC_OPC_AND] = 1,   [AGX_ATOMIC_OPC_OR] = 6,
      [AGX_ATOMIC_OPC_XOR] = 15,
   };

   if (op >= ARRAY_SIZE(hw))
      a3_fail(a, "atomic operation not mapped");

   return hw[op];
}

/* An atomic's data: a 32-bit register; cmpxchg's is a vector of two (the
 * new value, then the compared one, as on G13) in consecutive registers,
 * encoded by the first */
static uint64_t
a3_atomic_data(struct a3 *a, unsigned op, agx_index value)
{
   if (value.size != AGX_SIZE_32)
      a3_fail(a, "64-bit atomics not supported");

   return a3_reg(a, op, value, 0, 0, 0);
}

/* local_atomic (atomictg): the result in a load group (its field is the
 * group + 1, 0 for none), the base a byte address */
static void
a3_local_atomic(struct a3 *a, const agx_instr *I,
                struct util_dynarray *emission)
{
   bool result = !agx_is_null(I->dest[0]);

   a3_begin(a, I, AGX3_THREADGROUP_ATOMIC, T_ATG);
   a3_set(a, ATG_G, result ? A3_SYNC_G + 1 : 0);
   a3_set(a, ATG_DST, result ? a3_reg(a, ATG_DST, I->dest[0], 0, 0, 0) : 0);
   a3_set(a, ATG_DATA, a3_atomic_data(a, ATG_DATA, I->src[0]));
   a3_set(a, ATG_BASE, a3_tgsrc(a, ATG_BASE, I->src[1]));
   a3_set(a, ATG_INDEX, a3_tgsrc(a, ATG_INDEX, I->src[2]));
   a3_set(a, ATG_OP, a3_atomic_op(a, I->atomic_opc));
   a3_set(a, ATG_SHIFT, a3_memshift(a, 2));
   a3_set(a, ATG_OFFSET, 0);
   a3_emit(a, emission);

   if (result)
      a3_wait(a, I, A3_SYNC_G, emission);
}

/* atomic (atomicdev). The index counts 32-bit elements. G15 returns a
 * device atomic's result only to a stack register: the result goes to the
 * stack word after the shader's scratch in group 2 (the result form's
 * operands are the other form's after g and the destination), get_stack
 * waits for it and copies it to the destination in group 2 (its source
 * tagged with the atomic's group, as Apple's compiler tags a pending
 * source: q), then a wait. The entry program allocates the stack
 * (agx3_stack_size). */
static unsigned
a3_atomic_stack(unsigned scratch_B)
{
   return DIV_ROUND_UP(scratch_B, 4);
}

static void
a3_atomic(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   bool result = !agx_is_null(I->dest[0]);
   unsigned o = result ? 2 : 0;

   if (result) {
      a3_begin(a, I, AGX3_DEVICE_ATOMIC_RET, T_ADEV_RET);
      a3_set(a, ADEV_RET_G, A3_SYNC_G + 1);
      a3_set(a, ADEV_RET_DST,
             a3_stack(a, ADEV_RET_DST, a3_atomic_stack(a->scratch_B), 0));
   } else {
      a3_begin(a, I, AGX3_DEVICE_ATOMIC, T_ADEV);
   }

   a3_set(a, o + ADEV_DATA, a3_atomic_data(a, o + ADEV_DATA, I->src[0]));
   a3_set(a, o + ADEV_BASE, a3_membase(a, o + ADEV_BASE, I->src[1]));
   a3_set(a, o + ADEV_INDEX, a3_memidx(a, o + ADEV_INDEX, I->src[2]));
   a3_set(a, o + ADEV_OP, a3_atomic_op(a, I->atomic_opc));
   a3_set(a, o + ADEV_SHIFT, a3_memshift(a, 2));
   a3_set(a, o + ADEV_OFFSET, 0);
   a3_emit(a, emission);

   if (!result)
      return;

   a3_begin(a, I, AGX3_GET_STACK, T_GET_STACK);
   a3_set(a, GET_STACK_G, A3_SYNC_G);
   a3_set(a, GET_STACK_Q, A3_SYNC_G);
   a3_set(a, GET_STACK_DST, a3_reg(a, GET_STACK_DST, I->dest[0], 0, 0, 0));
   a3_set(a, GET_STACK_SRC,
          a3_stack(a, GET_STACK_SRC, a3_atomic_stack(a->scratch_B),
                   BITFIELD_BIT(A3_LU)));
   a3_set(a, GET_STACK_WAIT, BITFIELD_BIT(A3_SYNC_G));
   a3_emit(a, emission);

   a3_wait(a, I, A3_SYNC_G, emission);
}

static void a3_mov_cr(struct a3 *a, const agx_instr *I, unsigned cr,
                      agx_index value, struct util_dynarray *emission);
static uint64_t a3_src9(struct a3 *a, unsigned op, agx_index idx);

/* The texture instructions' result masks (texture.md): Apple's 4-bit code
 * for each IR mask (x = 1, y = 2, z = 4, w = 8), 0xff for none */
static const uint8_t a3_tex_mask[16] = {
   0xff, 0, 1, 5, 2, 8, 9, 6, 3, 4, 10, 13, 11, 12, 14, 7,
};

/* A texture table's pair, as the instruction names it: n for u(2n):u(2n+1),
 * or for r(2n):r(2n+1) with bit 83 (the texture's) or 82 (the sampler's) set
 * (texture.md), so the table base must be a 64-bit uniform below u32 or a
 * 64-bit register below r32 */
static unsigned
a3_tex_pair(struct a3 *a, agx_index base)
{
   if ((base.type != AGX_INDEX_UNIFORM && base.type != AGX_INDEX_REGISTER) ||
       base.size != AGX_SIZE_64 || (base.value & 3) || base.value >= 64)
      a3_fail(a, "texture table base is not a pair below u32 or r32");

   return base.value >> 2;
}

/* The operand of the current instruction that is the single bit b, or -1 */
static int
a3_op_at_bit(const struct a3 *a, unsigned b)
{
   const struct agx3_pack_spec *s = a->spec;

   for (unsigned op = 0; op < s->nops; op++) {
      if (s->op_start[op + 1] == s->op_start[op] + 1 &&
          s->fields[s->op_start[op]].start == b &&
          s->fields[s->op_start[op]].size == 1)
         return op;
   }

   return -1;
}

/* A texture or sampler index: an immediate, or a 16-bit register of the
 * low 64 (the field counts halves, boot #323); a 32-bit register is read as
 * its low half */
static uint64_t
a3_tex_index(struct a3 *a, unsigned op, enum a3_kind kind, agx_index idx)
{
   if (idx.type == AGX_INDEX_IMMEDIATE) {
      if (idx.value >= 256)
         a3_fail(a, "texture or sampler index immediate out of range");
      return a3_kind_value(a, op, kind, idx.value, 0);
   }

   if (idx.type != AGX_INDEX_REGISTER || idx.size > AGX_SIZE_32 ||
       idx.value >= 128)
      a3_fail(a, "texture or sampler index is not a register below r64");

   idx.size = AGX_SIZE_16;
   return a3_reg(a, op, idx, 0, 0, 0);
}

/* The texture instruction of each form for each IR dimension (AGX3.xml: bits
 * 48:63 code the LOD mode, the dimension and the array kind, from libLLVM
 * 32023's smp tree). AGX3_NUM_INSTRS where the tree has none: 1D and
 * multisampled textures have no level, so they sample at the automatic LOD
 * and fetch without a level only. */
#define A3_NONE AGX3_NUM_INSTRS
static const enum agx3_instr a3_tex_l0[] = {
   [AGX_DIM_1D] = A3_NONE,
   [AGX_DIM_1D_ARRAY] = A3_NONE,
   [AGX_DIM_2D] = AGX3_TEXTURE_SAMPLE,
   [AGX_DIM_2D_ARRAY] = AGX3_TEXTURE_SAMPLE_2D_ARRAY,
   [AGX_DIM_2D_MS] = A3_NONE,
   [AGX_DIM_3D] = AGX3_TEXTURE_SAMPLE_3D,
   [AGX_DIM_CUBE] = AGX3_TEXTURE_SAMPLE_CUBE,
   [AGX_DIM_CUBE_ARRAY] = AGX3_TEXTURE_SAMPLE_CUBE_ARRAY,
   [AGX_DIM_2D_MS_ARRAY] = A3_NONE,
};

static const enum agx3_instr a3_tex_auto[] = {
   [AGX_DIM_1D] = AGX3_TEXTURE_SAMPLE_AUTO_1D,
   [AGX_DIM_1D_ARRAY] = AGX3_TEXTURE_SAMPLE_AUTO_1D_ARRAY,
   [AGX_DIM_2D] = AGX3_TEXTURE_SAMPLE_AUTO,
   [AGX_DIM_2D_ARRAY] = AGX3_TEXTURE_SAMPLE_AUTO_2D_ARRAY,
   [AGX_DIM_2D_MS] = AGX3_TEXTURE_SAMPLE_AUTO_2D_MS,
   [AGX_DIM_3D] = AGX3_TEXTURE_SAMPLE_AUTO_3D,
   [AGX_DIM_CUBE] = AGX3_TEXTURE_SAMPLE_AUTO_CUBE,
   [AGX_DIM_CUBE_ARRAY] = AGX3_TEXTURE_SAMPLE_AUTO_CUBE_ARRAY,
   [AGX_DIM_2D_MS_ARRAY] = AGX3_TEXTURE_SAMPLE_AUTO_2D_MS_ARRAY,
};

static const enum agx3_instr a3_tex_load[] = {
   [AGX_DIM_1D] = A3_NONE,
   [AGX_DIM_1D_ARRAY] = A3_NONE,
   [AGX_DIM_2D] = AGX3_TEXTURE_LOAD,
   [AGX_DIM_2D_ARRAY] = AGX3_TEXTURE_LOAD_2D_ARRAY,
   [AGX_DIM_2D_MS] = A3_NONE,
   [AGX_DIM_3D] = AGX3_TEXTURE_LOAD_3D,
   [AGX_DIM_CUBE] = AGX3_TEXTURE_LOAD_CUBE,
   [AGX_DIM_CUBE_ARRAY] = AGX3_TEXTURE_LOAD_CUBE_ARRAY,
   [AGX_DIM_2D_MS_ARRAY] = A3_NONE,
};

static const enum agx3_instr a3_tex_load_nolod[] = {
   [AGX_DIM_1D] = AGX3_TEXTURE_LOAD_NOLOD_1D,
   [AGX_DIM_1D_ARRAY] = AGX3_TEXTURE_LOAD_NOLOD_1D_ARRAY,
   [AGX_DIM_2D] = AGX3_TEXTURE_LOAD_NOLOD,
   [AGX_DIM_2D_ARRAY] = AGX3_TEXTURE_LOAD_NOLOD_2D_ARRAY,
   [AGX_DIM_2D_MS] = AGX3_TEXTURE_LOAD_NOLOD_2D_MS,
   [AGX_DIM_3D] = AGX3_TEXTURE_LOAD_NOLOD_3D,
   [AGX_DIM_CUBE] = AGX3_TEXTURE_LOAD_NOLOD_CUBE,
   [AGX_DIM_CUBE_ARRAY] = AGX3_TEXTURE_LOAD_NOLOD_CUBE_ARRAY,
   [AGX_DIM_2D_MS_ARRAY] = AGX3_TEXTURE_LOAD_NOLOD_2D_MS_ARRAY,
};

/* With F42's constant texel offset (auto LOD, level 0, a fetch's level) */
static const enum agx3_instr a3_tex_offset[] = {
   [AGX_DIM_2D] = AGX3_TEXTURE_SAMPLE_OFFSET,
   [AGX_DIM_2D_ARRAY] = AGX3_TEXTURE_SAMPLE_OFFSET_2D_ARRAY,
   [AGX_DIM_3D] = AGX3_TEXTURE_SAMPLE_OFFSET_3D,
};

static const enum agx3_instr a3_tex_l0_offset[] = {
   [AGX_DIM_2D] = AGX3_TEXTURE_SAMPLE_L0_OFFSET,
   [AGX_DIM_2D_ARRAY] = AGX3_TEXTURE_SAMPLE_L0_OFFSET_2D_ARRAY,
   [AGX_DIM_3D] = AGX3_TEXTURE_SAMPLE_L0_OFFSET_3D,
};

static const enum agx3_instr a3_tex_load_offset[] = {
   [AGX_DIM_2D] = AGX3_TEXTURE_LOAD_OFFSET,
   [AGX_DIM_2D_ARRAY] = AGX3_TEXTURE_LOAD_OFFSET_2D_ARRAY,
   [AGX_DIM_3D] = AGX3_TEXTURE_LOAD_OFFSET_3D,
};

/* Gathers, at the automatic LOD (Apple's gather has no LOD operand) */
static const enum agx3_instr a3_tex_gather[] = {
   [AGX_DIM_2D] = AGX3_TEXTURE_GATHER,
   [AGX_DIM_2D_ARRAY] = AGX3_TEXTURE_GATHER_2D_ARRAY,
};

static const enum agx3_instr a3_tex_gather_offset[] = {
   [AGX_DIM_2D] = AGX3_TEXTURE_GATHER_OFFSET,
   [AGX_DIM_2D_ARRAY] = AGX3_TEXTURE_GATHER_OFFSET_2D_ARRAY,
};

/* The mode codes (bits 48:63; 48:58 with F28, whose immediate is then 0) of
 * texture_sample_mode, texture_gather_mode and texture_load_mode, generated
 * from libLLVM 32023's smp encoder tree (instruction 180: the LOD mode f139,
 * the offset kind f141, the compare f143, the dimension f76, the array kind
 * f77 = 2, a register; a gather takes its sample form's code). Indexed by the
 * dimension, sample or fetch, the LOD form (enum a3_tex_lod), the offset kind
 * (none, F42's immediate, the LOD word) and the compare; A3_NO_CODE where the
 * tree has none. */
enum a3_tex_lod {
   A3_LOD_AUTO,
   A3_LOD_BIAS_F28,
   A3_LOD_LEVEL_F28,
   A3_LOD_BIAS_WORD,
   A3_LOD_LEVEL_WORD,
   A3_LOD_GRAD,
};

#define A3_NO_CODE 0xffff
#define N          A3_NO_CODE
static const uint16_t a3_tex_code[AGX_DIM_CUBE_ARRAY + 1][2][6][3][2] =
   {
      [AGX_DIM_2D] =
         {
            {{{0, 32}, {513, 545}, {64, 96}},
             {{7, 39}, {779, 811}, {71, 103}},
             {{9, 41}, {539, 571}, {73, 105}},
             {{4, 36}, {641, 673}, {68, 100}},
             {{256, 288}, {529, 561}, {320, 352}},
             {{260, 292}, {521, 553}, {324, 356}}},
            {{{1152, N}, {649, N}, {1216, N}},
             {{N, N}, {N, N}, {N, N}},
             {{279, N}, {515, N}, {343, N}},
             {{N, N}, {N, N}, {N, N}},
             {{9344, N}, {713, N}, {9408, N}},
             {{N, N}, {N, N}, {N, N}}},
         },
      [AGX_DIM_2D_ARRAY] =
         {
            {{{1024, 1056}, {769, 801}, {1088, 1120}},
             {{263, 295}, {843, 875}, {327, 359}},
             {{265, 297}, {603, 635}, {329, 361}},
             {{8, 40}, {897, 929}, {72, 104}},
             {{264, 296}, {593, 625}, {328, 360}},
             {{268, 300}, {777, 809}, {332, 364}}},
            {{{1184, N}, {905, N}, {1248, N}},
             {{N, N}, {N, N}, {N, N}},
             {{407, N}, {579, N}, {471, N}},
             {{N, N}, {N, N}, {N, N}},
             {{9376, N}, {969, N}, {9440, N}},
             {{N, N}, {N, N}, {N, N}}},
         },
      [AGX_DIM_3D] =
         {
            {{{384, N}, {657, N}, {416, N}},
             {{59, N}, {661, N}, {187, N}},
             {{57, N}, {917, N}, {185, N}},
             {{400, N}, {665, N}, {436, N}},
             {{8584, N}, {537, N}, {8616, N}},
             {{412, N}, {793, N}, {444, N}}},
            {{{388, N}, {785, N}, {420, N}},
             {{N, N}, {N, N}, {N, N}},
             {{121, N}, {899, N}, {249, N}},
             {{N, N}, {N, N}, {N, N}},
             {{408, N}, {601, N}, {440, N}},
             {{N, N}, {N, N}, {N, N}}},
         },
      [AGX_DIM_CUBE] =
         {
            {{{16, 48}, {N, N}, {N, N}},
             {{273, 305}, {N, N}, {N, N}},
             {{19, 51}, {N, N}, {N, N}},
             {{20, 52}, {N, N}, {N, N}},
             {{768, 800}, {N, N}, {832, 864}},
             {{772, 804}, {N, N}, {836, 868}}},
            {{{4224, N}, {N, N}, {N, N}},
             {{N, N}, {N, N}, {N, N}},
             {{311, N}, {N, N}, {N, N}},
             {{N, N}, {N, N}, {N, N}},
             {{12416, N}, {N, N}, {N, N}},
             {{N, N}, {N, N}, {N, N}}},
         },
      [AGX_DIM_CUBE_ARRAY] =
         {
            {{{1040, 1072}, {N, N}, {N, N}},
             {{401, 433}, {N, N}, {N, N}},
             {{83, 115}, {N, N}, {N, N}},
             {{24, 56}, {N, N}, {N, N}},
             {{776, 808}, {N, N}, {840, 872}},
             {{780, 812}, {N, N}, {844, 876}}},
            {{{4256, N}, {N, N}, {N, N}},
             {{N, N}, {N, N}, {N, N}},
             {{375, N}, {N, N}, {N, N}},
             {{N, N}, {N, N}, {N, N}},
             {{12448, N}, {N, N}, {N, N}},
             {{N, N}, {N, N}, {N, N}}},
         },
};
#undef N

/* Whether I takes its LOD or bias through the LOD word: a bias, or a level
 * other than a sample's level 0 and a fetch's immediate (F28's) */
static bool
a3_tex_lod_word(const agx_instr *I)
{
   agx_index lod = I->src[1];
   bool load =
      I->op == AGX_OPCODE_TEXTURE_LOAD || I->op == AGX_OPCODE_IMAGE_LOAD;

   if (I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS)
      return true;
   if (I->lod_mode != AGX_LOD_MODE_LOD_MIN)
      return false;
   if (lod.type != AGX_INDEX_IMMEDIATE)
      return true;
   return load ? lod.value >= 32 : lod.value != 0;
}

/* An instruction of one of the tables above, A3_NONE for a dimension it
 * leaves out (a designated initialiser leaves those 0, a real instruction) */
static enum agx3_instr
a3_tex_pick(const enum agx3_instr *table, enum agx_dim dim)
{
   switch (dim) {
   case AGX_DIM_2D:
   case AGX_DIM_2D_ARRAY:
   case AGX_DIM_3D:
      if (table == a3_tex_gather || table == a3_tex_gather_offset)
         return dim == AGX_DIM_3D ? A3_NONE : table[dim];
      return table[dim];
   default:
      return A3_NONE;
   }
}

/* The IR's coordinate vector, G13's layout: the spatial coordinates (a
 * cube's direction), then for an array or a multisampled texture one 32-bit
 * word holding the layer, or the sample index in its low half and the layer
 * in its high half */
static unsigned
a3_tex_coords(const agx_instr *I)
{
   return I->src[0].channels_m1 + 1;
}

static bool
a3_tex_has_layer_word(const agx_instr *I)
{
   switch (I->dim) {
   case AGX_DIM_1D_ARRAY:
   case AGX_DIM_2D_ARRAY:
   case AGX_DIM_2D_MS:
   case AGX_DIM_2D_MS_ARRAY:
   case AGX_DIM_CUBE_ARRAY:
      return true;
   default:
      return false;
   }
}

/* The cr slot of IR coordinate word c (smp.md, Dimensions and arrays): the
 * spatial coordinates in slots 0-2, the layer word in slot 3 (a 2D array's
 * layer in slot 2 is ignored), but a multisampled read's word (the sample
 * index at bit 12, an array's layer below it, lower_regular_texture) right
 * after y, in slot 2 (in slot 3 both are ignored) */
static unsigned
a3_tex_slot(const agx_instr *I, unsigned c)
{
   if (I->dim == AGX_DIM_2D_MS || I->dim == AGX_DIM_2D_MS_ARRAY)
      return c;

   return (a3_tex_has_layer_word(I) && c == a3_tex_coords(I) - 1) ? 3 : c;
}

/* The offset of I that is not F42's immediate: the IR's offset word (with a
 * compare, the second word of the reference's pair), in the LOD word's
 * layout (lower_regular_texture) */
static agx_index
a3_tex_offset_word(const agx_instr *I)
{
   agx_index offset = I->src[5];

   if (I->shadow) {
      offset.value += 2;
      offset.channels_m1 = 0;
   }

   return offset;
}

/* Whether I's texel offset goes through the LOD word (cr3), not F42 */
static bool
a3_tex_offset_in_word(const agx_instr *I)
{
   return I->offset && (I->shadow || I->src[5].type != AGX_INDEX_IMMEDIATE);
}

/* The coordinates of a gradient sample: the IR's (G13's) order, dP/dx and
 * dP/dy of each coordinate in turn */
static unsigned
a3_tex_grad_words(const agx_instr *I)
{
   return I->lod_mode == AGX_LOD_MODE_LOD_GRAD ? I->src[1].channels_m1 + 1 : 0;
}

/* texture_sample / texture_load (texture.md; boots #252 and #256 run
 * Apple's 2D forms): G15 has no texture state registers. The texture is
 * entry ts of the 32 B state array at a uniform pair (the IR's bindless base,
 * its index counting 32 B descriptors), the sampler entry ss of the 32 B
 * sampler table at another pair (src[6]). The coordinates go to cr slots
 * (smp.md), in group 0, which the instruction waits for: x, y and z in
 * cr0-cr2, a depth reference in cr2, the LOD word in cr3 (an array's layer at
 * bits 0:10, a texel offset's x at 12:15, y at 28:31 and z at 0:3 when it is
 * not F42's immediate, the LOD or bias at 16:27 through pack_smpir), the
 * gradients from cr4; the result arrives in group A3_SYNC_G and
 * is waited for at once. Done: every dimension, automatic LOD (fragment
 * shaders) and level 0, a fetch at a constant level, a LOD, bias or fetch
 * level in a register (2D and 2D array samples and 2D fetches run on the
 * hardware), gradients, immediate or register indices, constant offsets and
 * offsets in a register, gathers, 2D and 2D array depth compares (the
 * reference in cr2). Gallium samples every cube in cube array mode, the
 * face-local (s, t) and the cube and face in the LOD word
 * (agx_nir_lower_cube_g15). */
static void
a3_texture(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   bool load =
      I->op == AGX_OPCODE_TEXTURE_LOAD || I->op == AGX_OPCODE_IMAGE_LOAD;
   bool auto_lod = I->lod_mode == AGX_LOD_MODE_AUTO_LOD;
   bool grad = I->lod_mode == AGX_LOD_MODE_LOD_GRAD;
   agx_index coords = I->src[0], lod = I->src[1];
   agx_index texture = I->src[3], sampler = I->src[4];

   /* An image load is a fetch with the txf sampler (agx_emit_image_load) */
   if (I->op == AGX_OPCODE_IMAGE_LOAD && agx_is_null(I->src[6]))
      a3_fail(a, "image load without the txf sampler");
   if (I->dim >= ARRAY_SIZE(a3_tex_auto))
      a3_fail(a, "texture dimension");
   if (I->query_lod)
      a3_fail(a, "texture LOD query");
   /* A cube is read in cube array mode (agx_nir_lower_cube_g15), whose cr2
    * (z) is free for the reference as a 2D array's is */
   if (I->shadow && I->dim != AGX_DIM_2D && I->dim != AGX_DIM_2D_ARRAY &&
       I->dim != AGX_DIM_CUBE_ARRAY)
      a3_fail(a, "texture compare other than 2D, 2D array or cube array");

   bool gather = I->gather != AGX_GATHER_NONE;
   /* A sparse read's mask carries its residency code in bit 4, which the
    * hardware writes after the components (bit 81, texture.md) */
   unsigned mask = I->mask & 0xf;
   bool off_word = a3_tex_offset_in_word(I);
   agx_index offset = off_word ? a3_tex_offset_word(I) : I->src[5];
   /* A fetch's level is F28's 12-bit immediate (Apple's loads have 0), in
    * q5.6 as the LOD word's (smp.md) */
   bool imm_level = load && I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
                    lod.type == AGX_INDEX_IMMEDIATE && lod.value < 32;
   bool level0 = I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
                 lod.type == AGX_INDEX_IMMEDIATE && lod.value == 0;
   bool lodw = a3_tex_lod_word(I);
   bool bias = I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS;

   if (!auto_lod && !imm_level && !level0 && !lodw && !grad)
      a3_fail(a, "texture LOD other than automatic, a level, a bias or "
                 "gradients");
   if (gather && (load || lodw || grad))
      a3_fail(a, "gather of a fetch or at a LOD");
   if (load && auto_lod)
      a3_fail(a, "texture_load with an automatic LOD");
   if (coords.size != AGX_SIZE_32 || a3_tex_coords(I) > 4)
      a3_fail(a, "texture coordinates are not 32-bit or more than four");
   if (mask == 0 || (I->mask > 15 && !I->sparse))
      a3_fail(a, "texture result mask");
   if (off_word &&
       (offset.type != AGX_INDEX_REGISTER || offset.size != AGX_SIZE_32))
      a3_fail(a, "texture offset is not a 32-bit register");

   /* The forms with a mode code: compares, a LOD word, gradients and an
    * offset in the LOD word (the code table); the others by name */
   bool coded =
      I->shadow || lodw || grad || off_word ||
      (gather && (I->dim == AGX_DIM_CUBE || I->dim == AGX_DIM_CUBE_ARRAY));
   enum agx3_instr instr = A3_NONE;
   const char *tmpl = load ? T_TEX_LOAD : auto_lod ? T_TEX_AUTO : T_TEX_L0;
   unsigned code = A3_NO_CODE;

   if (coded) {
      enum a3_tex_lod form = grad ? A3_LOD_GRAD
                             : lodw
                                ? (bias ? A3_LOD_BIAS_WORD : A3_LOD_LEVEL_WORD)
                             : (auto_lod || gather) ? A3_LOD_AUTO
                                                    : A3_LOD_LEVEL_F28;

      /* F28's immediate stays 0 in these forms */
      if (form == A3_LOD_LEVEL_F28 && !level0)
         a3_fail(a, "texture fetch at a level with a compare or an offset in "
                    "a register");

      switch (I->dim) {
      case AGX_DIM_2D:
      case AGX_DIM_2D_ARRAY:
      case AGX_DIM_3D:
      case AGX_DIM_CUBE:
      case AGX_DIM_CUBE_ARRAY:
         code = a3_tex_code[I->dim][load][form][off_word    ? 2
                                                : I->offset ? 1
                                                            : 0][I->shadow];
         break;
      default:
         break;
      }

      instr = gather ? AGX3_TEXTURE_GATHER_MODE
              : load ? AGX3_TEXTURE_LOAD_MODE
                     : AGX3_TEXTURE_SAMPLE_MODE;
      if (gather || (auto_lod && !load))
         tmpl = T_TEX_AUTO;
      else if (grad || (lodw && !bias && !load))
         tmpl = T_TEX_LEVELW;
      else if (lodw && bias)
         tmpl = T_TEX_BIASW;
   } else if (gather) {
      /* whatever the stage: a gather reads one level */
      instr =
         a3_tex_pick(I->offset ? a3_tex_gather_offset : a3_tex_gather, I->dim);
      tmpl = T_TEX_AUTO;
   } else if (I->offset) {
      instr = a3_tex_pick(load       ? a3_tex_load_offset
                          : auto_lod ? a3_tex_offset
                                     : a3_tex_l0_offset,
                          I->dim);
   } else if (load) {
      /* A dimension without levels fetches without F28 */
      instr = a3_tex_load[I->dim];
      if (instr == A3_NONE) {
         if (!level0)
            a3_fail(a, "texture_load at a level of a texture without levels");
         instr = a3_tex_load_nolod[I->dim];
      }
   } else {
      instr = auto_lod ? a3_tex_auto[I->dim] : a3_tex_l0[I->dim];
   }

   if (instr == A3_NONE || (coded && code == A3_NO_CODE))
      a3_fail(a, "texture form not in the encoder tree for this dimension");

   bool has_lod = load && !lodw && a3_tex_load[I->dim] != A3_NONE;

   /* The layer word goes into the LOD word when there is one */
   bool word = lodw || off_word;
   agx_index layer = agx_null();
   for (unsigned c = 0; c < a3_tex_coords(I); ++c) {
      agx_index v = coords;
      v.value += 2 * c;
      v.channels_m1 = 0;
      if (word && a3_tex_slot(I, c) == 3)
         layer = v;
      else
         a3_mov_cr(a, I, a3_tex_slot(I, c), v, emission);
   }

   /* The LOD word: pack_smpir's A the f16 LOD or bias as q5.6 (mode 0xa,
    * Apple's for both) or a fetch's 16-bit level (mode 0xe, s16 to q5.6), at
    * bits 16:27; with none, the layer clamped to 0x7ff (mode 4, Apple's code
    * 4 at position 0); its C the offset word or the layer, ORed in. An
    * array's offset word already holds the layer (lower_regular_texture),
    * which a LOD and an offset need: C is one register */
   if (word) {
      agx_index pa = lodw ? lod : layer;
      agx_index pc = off_word ? offset : lodw ? layer : agx_null();

      if (agx_is_null(pa)) {
         a3_mov_cr(a, I, 3, offset, emission);
      } else {
         a3_begin(a, I, AGX3_PACK_SMPIR, T_PACK_SMPIR);
         a3_set(a, PACK_SMPIR_DST,
                a3_kind_value(a, PACK_SMPIR_DST, A3_CR, 3, 0));
         a3_set(a, PACK_SMPIR_G, 0);
         a3_set(a, PACK_SMPIR_A, a3_src9(a, PACK_SMPIR_A, pa));
         if (!agx_is_null(pc))
            a3_set(a, PACK_SMPIR_C, a3_reg(a, PACK_SMPIR_C, pc, 0, 0, 0));
         a3_set(a, PACK_SMPIR_MODE, !lodw ? 0x4 : load ? 0xe : 0xa);
         a3_set(a, PACK_SMPIR_WAIT, 0);
         a3_emit(a, emission);
      }
   }

   /* The depth reference goes to cr2, the slot a 2D or 2D array read leaves
    * free (texture.md: a 2D sweep on the hardware, 2026-10-18) */
   if (I->shadow) {
      agx_index ref = I->src[5];
      ref.channels_m1 = 0;
      a3_mov_cr(a, I, 2, ref, emission);
   }

   /* The gradients from cr4 in the IR's order, dP/dx and dP/dy of each
    * coordinate in turn (libLLVM 32023's pack sequence for them,
    * sub_25092FFB0: four 32-bit words for 2D and cube, six for 3D; the order
    * from the CTS's texturegrad cases on the hardware) */
   for (unsigned i = 0; i < a3_tex_grad_words(I); ++i) {
      agx_index v = lod;
      v.value += 2 * i;
      v.channels_m1 = 0;
      a3_mov_cr(a, I, 4 + i, v, emission);
   }

   a3_begin(a, I, instr, tmpl);
   a3_set(a, TEX_G0, A3_SYNC_G);
   a3_set(a, TEX_G1, A3_SYNC_G);
   /* Bit 64 of a sample (q3) ends the quad's helper lanes: after it,
    * derivatives read 0 across an uncovered pixel, and a later sample's
    * implicit LOD comes out wrong (the last column of an odd-width
    * viewport). Apple sets it only on a fragment shader's last sample when
    * no sample or derivative follows (the QuartzCore corpus: 222 of 222
    * flagged samples, none on an earlier one); T_TEX_AUTO is such a sample.
    * Keep the helpers to the end. */
   if (!load) {
      int q3 = a3_op_at_bit(a, 64);
      if (q3 >= 0)
         a3_set(a, q3, 0);
   }
   if (gather) {
      /* four values of one component (AGX_GATHER_R, _G, _B, _A) */
      a3_set(a, TEX_COMPONENT, I->gather >> 1);
      a3_set(a, TEX_DST,
             a3_reg(a, TEX_DST, I->dest[0], 0, 0, a3_tex_mask[0xf]));
   } else {
      a3_set(a, TEX_MASK, a3_tex_mask[mask]);
      a3_set(a, TEX_DST,
             a3_reg(a, TEX_DST, I->dest[0], 0, 0, a3_tex_mask[mask]));
   }
   if (I->sparse)
      a3_set(a, a3_op_at_bit(a, 81), 1);
   a3_set(a, TEX_TS, a3_tex_index(a, TEX_TS, A3_TS, texture));
   a3_set(a, TEX_CR, a3_kind_value(a, TEX_CR, A3_CR, 0, 0));
   a3_set(a, TEX_Q1, 1);
   /* 0 for a fetch and Apple's sample at a level in cr3 (gradients: a
    * guess) */
   a3_set(a, TEX_Q2, (load || grad || (lodw && !bias)) ? 0 : 1);
   a3_set(a, TEX_WAIT, 1);
   a3_set(a, TEX_SS, a3_tex_index(a, TEX_SS, A3_SS, sampler));
   a3_set(a, TEX_USMP, a3_tex_pair(a, I->src[6]));
   a3_set(a, TEX_UTEX, a3_tex_pair(a, I->src[2]));

   /* a fetch's level is F28's 12-bit immediate */
   if (has_lod && !coded)
      a3_set(a, TEX_LOD, imm_level ? lod.value << 6 : 0);
   if (I->offset && !off_word)
      a3_set(a, TEX_OFFSET, offset.value);
   if (coded)
      a3_set(a, TEX_MODE, code);

   a3_emit(a, emission);

   /* The table pairs in registers: bits 83 and 82, which no operand holds */
   if (I->src[2].type == AGX_INDEX_REGISTER ||
       I->src[6].type == AGX_INDEX_REGISTER) {
      uint8_t *out = (uint8_t *)emission->data + emission->size - a->length;

      assert(a->length >= 14);
      if (I->src[2].type == AGX_INDEX_REGISTER)
         out[10] |= BITFIELD_BIT(83 - 80);
      if (I->src[6].type == AGX_INDEX_REGISTER)
         out[10] |= BITFIELD_BIT(82 - 80);
   }

   a3_wait(a, I, A3_SYNC_G, emission);
}

/* The coordinate registers the packing of I writes, the highest plus one,
 * for the entry program to back (the specTPR's 41:31: Apple's kernels and
 * render programs set it to their highest cr written plus one, pack_imgwir's
 * destination aside, 2026-10-16) */
unsigned
agx3_cr_count(const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_TEXTURE_SAMPLE:
   case AGX_OPCODE_TEXTURE_LOAD:
   case AGX_OPCODE_IMAGE_LOAD: {
      unsigned n = MAX2(a3_tex_slot(I, a3_tex_coords(I) - 1) + 1,
                        (a3_tex_lod_word(I) || a3_tex_offset_in_word(I)) ? 4
                        : I->shadow                                      ? 3
                                                                         : 2);
      return MAX2(n, a3_tex_grad_words(I) ? 4 + a3_tex_grad_words(I) : 0);
   }
   case AGX_OPCODE_ZS_EMIT:
      /* the coverage word (with the stencil), then the depth (a3_zs_emit) */
      return (I->zs & 1) ? 2 : 1;
   case AGX_OPCODE_BLOCK_IMAGE_STORE:
      /* x, y and pack_imgwir's cr2; a layered target's layer pack in cr2
       * and the offset's in cr3 */
      return (I->dim == AGX_DIM_2D_ARRAY || I->dim == AGX_DIM_2D_MS_ARRAY) ? 4
                                                                           : 2;
   case AGX_OPCODE_IMAGE_WRITE:
      return I->vary_cr + 8;
   case AGX_OPCODE_ST_VARY:
      return I->vary_cr + 1;
   case AGX_OPCODE_SAMPLE_MASK:
      /* the mov to cr0 before movmsk: Apple's discard shader has 1 here;
       * with 0, movmsk does not kill (boots #333-#360) */
      return 1;
   default:
      return 0;
   }
}

/* The thread-private stack (bytes per thread) the packing of I uses, for
 * the entry program to allocate (specTPR): the scratch, then a device
 * atomic's result word */
unsigned
agx3_stack_size(const agx_instr *I, unsigned scratch_B)
{
   switch (I->op) {
   case AGX_OPCODE_ATOMIC:
      return agx_is_null(I->dest[0]) ? 0 : 4 * (a3_atomic_stack(scratch_B) + 1);
   case AGX_OPCODE_STACK_LOAD:
   case AGX_OPCODE_STACK_STORE:
      return scratch_B;
   default:
      return 0;
   }
}

/* Barriers: G15 has one barrier instruction whose flags (AGX3.xml) say
 * what it orders, as Apple's compiler sets them (sub_251C56ED8: Metal's
 * memory flags, ordering and scope). The IR's G13 sequences map to:
 * threadgroup_barrier = threadgroup_barrier(mem_threadgroup); the memory
 * barrier = a device memory acq_rel fence at threadgroup scope, the first
 * of the device-scope sequence (unknown_barrier_1) the same at device
 * scope, the rest of it nothing; image barriers 1 and 3 a texture release
 * and acquire, 2 and 4 nothing; flush_memory_to_texture (device memory
 * written by atomics, then read as a texture) an acq_rel fence of device and
 * texture memory at device scope. The texture fences carry bit 10, as
 * Apple's do with texture writes not L2-persistent (imgwci bit 107, which
 * a3_image_write sets), and the release waits for the image writes' group
 * (every imgwci delivers in group 0), as Apple's release waits for its
 * imgwci's: the corpus's 0x650 and 0x6D0. Not verified on the hardware. */
static void
a3_barrier(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   unsigned flags, wait = 0;

   switch (I->op) {
   case AGX_OPCODE_THREADGROUP_BARRIER:
      flags = 0x361;
      break;
   case AGX_OPCODE_MEMORY_BARRIER:
      flags = 0x284;
      break;
   case AGX_OPCODE_UNKNOWN_BARRIER_1:
      flags = 0x684;
      break;
   case AGX_OPCODE_IMAGE_BARRIER_1:
      flags = 0x650;
      wait = BITFIELD_BIT(0);
      break;
   case AGX_OPCODE_IMAGE_BARRIER_3:
      flags = 0x6d0;
      break;
   case AGX_OPCODE_FLUSH_MEMORY_TO_TEXTURE:
      flags = 0x694;
      break;
   case AGX_OPCODE_MEMORY_BARRIER_2:
   case AGX_OPCODE_MEMORY_BARRIER_3:
   case AGX_OPCODE_UNKNOWN_BARRIER_2:
   case AGX_OPCODE_DEVICE_BARRIER_1:
   case AGX_OPCODE_DEVICE_BARRIER_2:
   case AGX_OPCODE_IMAGE_BARRIER_2:
   case AGX_OPCODE_IMAGE_BARRIER_4:
      return;
   default:
      a3_fail(a, "barrier not mapped");
   }

   a3_begin(a, I, AGX3_BARRIER, T_BARRIER);
   a3_set(a, BARRIER_FLAGS, flags);
   a3_set(a, BARRIER_WAIT, wait);
   a3_emit(a, emission);
}

/* uniform_store: a mov to the uniform per element. The value is a vector of
 * the mask's length (16 or 32-bit elements) or one 64-bit register, which is
 * two 32-bit movs. */
static void
a3_uniform_store(struct a3 *a, const agx_instr *I,
                 struct util_dynarray *emission)
{
   agx_index value = I->src[0];
   agx_index offset = I->src[1];

   if (offset.type != AGX_INDEX_IMMEDIATE)
      a3_fail(a, "uniform_store with a register offset");
   if (value.type != AGX_INDEX_REGISTER)
      a3_fail(a, "uniform_store of a non-register");

   unsigned count = MAX2(util_last_bit(I->mask), 1);
   enum agx_size size = value.size == AGX_SIZE_16 ? AGX_SIZE_16 : AGX_SIZE_32;
   unsigned stride = agx_size_align_16(size);

   if (value.size == AGX_SIZE_64) {
      assert(count == 1);
      count = 2;
   }

   for (unsigned w = 0; w < count; w++) {
      agx_index src = value;
      src.size = size;
      src.value += stride * w;
      src.abs = src.neg = false;

      agx_index dst = agx_uniform(offset.value + stride * w, size);

      /* The 4-byte mov reaches registers and uniforms below 64 */
      for (unsigned len = 4;; len = 10) {
         uint8_t out[AGX3_MAX_BYTES];

         a3_begin(a, I, len == 4 ? AGX3_MOV : AGX3_MOV10,
                  len == 4 ? T_MOV : T_ZERO10);
         a3_set(a, MOV_DST, a3_reg(a, MOV_DST, dst, 0, 0, 0));
         a3_set(a, MOV_SRC, a3_reg(a, MOV_SRC, src, 0, BITFIELD_BIT(A3_SX), 0));

         if (len == 10 || a3_check(a, out))
            break;
      }

      a3_emit(a, emission);
   }
}

/*
 * Fragment and vertex I/O, from libLLVM 32023's encoder. demo15 runs
 * st_var from a cr, tile_store, wait_pix and signal_pix on the hardware;
 * the rest is not verified there.
 */

/* A 16-bit source in the 11-bit form (src9): a 16-bit register, a uniform
 * or an 8-bit immediate */
static uint64_t
a3_src9(struct a3 *a, unsigned op, agx_index idx)
{
   if (idx.type == AGX_INDEX_IMMEDIATE)
      return a3_imm(a, op, idx.value, false);

   idx.size = AGX_SIZE_16;
   return a3_reg(a, op, idx, 0, 0, 0);
}

/* A register to coordinate register cr (as the disassembler numbers it),
 * in group 0: the 4-byte mov reaches a cr below 64 in group 0 only and has
 * one size bit for both operands; the 10-byte one has its own for the
 * destination (vdst bit 15, a carry of the cr case), set for a 32-bit value
 * as Apple's 10-byte movs to cr4096 have it */
static void
a3_mov_cr(struct a3 *a, const agx_instr *I, unsigned cr, agx_index value,
          struct util_dynarray *emission)
{
   if (value.type != AGX_INDEX_REGISTER)
      a3_fail(a, "coordinate register value is not a register");

   value.abs = value.neg = false;

   for (unsigned len = 4;; len = 10) {
      uint8_t out[AGX3_MAX_BYTES];

      a3_begin(a, I, len == 4 ? AGX3_MOV : AGX3_MOV10,
               len == 4 ? T_MOV : T_ZERO10);
      uint64_t dst = a3_kind_value(a, MOV_DST, A3_CR, cr, 0);
      if (value.size == AGX_SIZE_32)
         dst |= BITFIELD64_BIT(15);

      a3_set(a, MOV_DST, dst);
      a3_set(a, MOV_SRC, a3_reg(a, MOV_SRC, value, 0, BITFIELD_BIT(A3_SX), 0));

      if (len == 10 || a3_check(a, out))
         break;
   }

   a3_emit(a, emission);
}

#define A3_VARY_CRS    8
#define A3_IMGW_TUPLES 2

/* G15's st_var and image write read their coordinate registers after they
 * issue, until their group (0) completes, and a later write to those crs does
 * not wait for it: a mov to the cr of the st_var just before stored the new
 * value in both slots (the point size and the first varying of a vertex
 * shader with two varyings, the GS copy shader's st_vars through cr0 with
 * dynamic slots). An image write likewise: of two writes from cr0-cr7, the
 * first stored the second's data, and the second, its tuple freed by the
 * first, wrote zero to texel (0, 0); a texture fetch whose coordinates went
 * to cr0 right after a write read texel (0, 0) (dEQP-VK
 * texel_view_compatible's decompress pass). The st_vars take their crs in
 * turn, as Apple's vertex shaders take a cr per output, and the image writes
 * take tuples from cr8 in turn, clear of the crs other instructions write
 * from cr0, as Apple's shaders give each write its own tuple, or the next
 * one free when that one is still being read. A cr still being read is
 * reused only after a wait for group 0, as is one another instruction
 * writes. A block starts with the crs still being read at the end of each
 * predecessor and of the block before it (a SIMD group runs both sides of a
 * branch, one after the other): a wait on one side does not cover the other
 * (dEQP-VK descriptorset_random: a write in one branch, a wait in another,
 * and the write after the join stored zero to texel (0, 0)). A loop's back
 * edge may bring the last iteration's.
 */
void
agx3_assign_crs(agx_context *ctx)
{
   unsigned next_vary = 0, next_imgw = 0;
   uint64_t *pending_out = calloc(ctx->num_blocks, sizeof(uint64_t));
   unsigned *order = calloc(ctx->num_blocks, sizeof(unsigned));
   agx_block *prev = NULL;
   unsigned n = 0;

   agx_foreach_block(ctx, block)
      order[block->index] = n++;

   agx_foreach_block(ctx, block) {
      uint64_t pending = prev ? pending_out[prev->index] : 0;
      agx_foreach_predecessor(block, pred) {
         if (order[(*pred)->index] >= order[block->index])
            pending = ~0ull;
         else
            pending |= pending_out[(*pred)->index];
      }

      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_WAIT && I->scoreboard == 0)
            pending = 0;

         uint64_t crs;
         if (I->op == AGX_OPCODE_ST_VARY) {
            I->vary_cr = (next_vary++) % A3_VARY_CRS;
            crs = BITFIELD64_BIT(I->vary_cr);
         } else if (I->op == AGX_OPCODE_IMAGE_WRITE) {
            unsigned t = next_imgw++;
            for (unsigned i = 0; i < A3_IMGW_TUPLES; ++i) {
               unsigned cr = 8 * (1 + (t + i) % A3_IMGW_TUPLES);
               if (!(pending & BITFIELD64_RANGE(cr, 8))) {
                  t += i;
                  break;
               }
            }
            I->vary_cr = 8 * (1 + t % A3_IMGW_TUPLES);
            crs = BITFIELD64_RANGE(I->vary_cr, 8);
         } else {
            /* Another instruction's crs (a vertex shader's texture fetch after
             * the point size's st_var: the fetch read the wrong texel) */
            crs = BITFIELD64_MASK(MIN2(agx3_cr_count(I), 64));
         }

         if (pending & crs) {
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            agx_wait(&b, 0);
            pending = 0;
         }

         if (I->op == AGX_OPCODE_ST_VARY || I->op == AGX_OPCODE_IMAGE_WRITE)
            pending |= crs;
      }

      pending_out[block->index] = pending;
      prev = block;
   }

   free(pending_out);
   free(order);
}

/* st_vary: G15 stores a varying from a coordinate register that an ALU
 * instruction wrote in a group (uvsw; demo15's transform writes cr0-cr3
 * in groups 0-3 and each st_var waits for its group). The value goes
 * through cr (agx3_assign_crs) in group 0; st_var waits for group 0
 * and frees the cr. The slot counts 32-bit words, as G13's. G15 has no last
 * flag: the IR's is dropped. */
static void
a3_st_vary(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   agx_index slot = I->src[0];
   agx_index value = I->src[1];
   unsigned cr = I->vary_cr;

   if (value.size != AGX_SIZE_32)
      a3_fail(a, "st_vary of a value that is not 32-bit");

   a3_mov_cr(a, I, cr, value, emission);

   a3_begin(a, I, AGX3_ST_VAR, T_ST_VAR);
   a3_set(a, ST_VAR_CR,
          a3_kind_value(a, ST_VAR_CR, A3_CR, cr, BITFIELD_BIT(A3_LU)));
   a3_set(a, ST_VAR_SLOT, a3_src9(a, ST_VAR_SLOT, slot));
   a3_set(a, ST_VAR_G, 0);
   a3_set(a, ST_VAR_WAIT, BITFIELD_BIT(0));
   a3_emit(a, emission);
}

/* The tile formats G15 can move: its tile instructions have no format
 * (demo15 converts with fltnorm before tile_store) and pack 8-bit data
 * two per 16-bit register, where G13 has one per register */
static unsigned
a3_tile_type(const struct a3 *a, const agx_instr *I)
{
   switch (I->format) {
   case AGX_FORMAT_I8:
      if (util_bitcount(I->mask) > 1)
         a3_fail(a, "8-bit tile vector (G15 packs the bytes)");
      return 2;
   case AGX_FORMAT_I16:
   case AGX_FORMAT_F16:
      return 0;
   case AGX_FORMAT_I32:
      return 1;
   default:
      a3_fail(a, "tile format conversion (G15's tile data is raw)");
   }
}

/* ld_tile/st_tile (lldcol/lstcol): the sample mask and the pixel offset go
 * to the two 16-bit sources demo15 sets to 1 and 0 (the mapping and the
 * offset's unit are guesses); the load is synchronous in the IR. */
static void
a3_tile(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   bool load = I->op == AGX_OPCODE_LD_TILE;
   agx_index reg = load ? I->dest[0] : I->src[0];
   agx_index samples = I->src[load ? 0 : 1];
   agx_index coords = I->src[load ? 1 : 2];
   unsigned r = load ? LD_TILE_REG : ST_TILE_REG;
   unsigned type = a3_tile_type(a, I);

   /* Explicit coordinates are x and y as two 16-bit halves (the compute
    * blit's local invocation id): the encoding names them as one 32-bit
    * register */
   if (I->explicit_coords && coords.type == AGX_INDEX_REGISTER &&
       coords.size == AGX_SIZE_16 && agx_channels(coords) == 2 &&
       !(coords.value & 1))
      coords = agx_register(coords.value, AGX_SIZE_32);

   a3_begin(a, I, load ? AGX3_TILE_LOAD : AGX3_TILE_STORE,
            load ? T_LD_TILE : T_ST_TILE);
   if (load)
      a3_set(a, LD_TILE_G, A3_SYNC_G);
   a3_set(a, r, a3_memreg(a, r, reg, type, a3_mem_mask[I->mask]));
   a3_set(a, r + 1, I->explicit_coords ? a3_reg(a, r + 1, coords, 0, 0, 0) : 0);
   a3_set(a, r + 2, a3_src9(a, r + 2, samples));
   a3_set(a, r + 3, a3_imm(a, r + 3, I->pixel_offset, false));
   a3_emit(a, emission);

   if (load)
      a3_wait(a, I, A3_SYNC_G, emission);
}

/* wait_pix/signal_pix (pixwait/pixrel), G13's masks. G13 releases its fences
 * at the end of the shader; demo15's fragment shader signals what it waited
 * for (0x20C) and its entry program the rest (0x2F3): G15's linked program
 * ends with a signal_pix of what it waited for (agx_fast_link), and its entry
 * program releases the others. demo15 waits for 0x200 and 0xC with two
 * instructions, each with its own flag (bit 34 for 0x200, bit 35 for 0xC);
 * what they mean is not known. Other fences are waited for without a flag
 * (a guess). */
static void
a3_pixwait(struct a3 *a, const agx_instr *I, unsigned mask, bool f0, bool f1,
           struct util_dynarray *emission)
{
   a3_begin(a, I, AGX3_WAIT_PIX, T_WAIT_PIX);
   a3_set(a, PIX_MASK, mask);
   a3_set(a, WAIT_PIX_F0, f0);
   a3_set(a, WAIT_PIX_F1, f1);
   a3_emit(a, emission);
}

static void
a3_pix(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   unsigned mask = I->writeout;

   if (mask >> 10)
      a3_fail(a, "pixel mask out of range");

   if (I->op == AGX_OPCODE_WAIT_PIX) {
      if (mask & 0x200)
         a3_pixwait(a, I, 0x200, true, false, emission);
      if (mask & 0xC)
         a3_pixwait(a, I, mask & 0xC, false, true, emission);
      if (mask & ~0x20C)
         a3_pixwait(a, I, mask & ~0x20C, false, false, emission);
      return;
   }

   a3_begin(a, I, AGX3_SIGNAL_PIX, T_SIG_PIX);
   a3_set(a, PIX_MASK, mask);
   a3_emit(a, emission);
}

/* ldcf (ldcfr): a coefficient register's A, B and C, as G13's ldcf of one
 * channel, in a load group the IR's synchronous ldcf waits for */
static void
a3_ldcf(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   if (I->channels != 1)
      a3_fail(a, "ldcf of more than one channel");

   a3_begin(a, I, AGX3_LDCF, T_LDCF);
   a3_set(a, LDCF_G, A3_SYNC_G);
   a3_set(a, LDCF_DST, a3_reg(a, LDCF_DST, I->dest[0], 0, 0, 0));
   a3_set(a, LDCF_CF, a3_src9(a, LDCF_CF, I->src[0]));
   a3_emit(a, emission);

   a3_wait(a, I, A3_SYNC_G, emission);
}

/* iter (f32interp): one instruction per channel, coefficient registers cf,
 * cf + 1, ...; synchronous. persp is Apple's flag on the iteration of W's
 * coefficients, whose result only the coefficient-register forms of rcp and
 * fmul that follow it in Apple's code consume (a plain read gives 0; the
 * compiler does not set it). G15 has no projected
 * iteration: agx_emit_load_vary lowers iterproj. Centroid and a dynamic
 * sample need itrcoords into a register first. */
static void
a3_iter(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   agx_index cf = I->src[0];
   agx_index sample = I->src[1];
   agx_index dest = I->dest[0];
   unsigned mode = 0;

   if (I->op == AGX_OPCODE_ITERPROJ)
      a3_fail(a, "iterproj (G15 iterates W, takes its reciprocal and "
                 "multiplies: agx_emit_load_vary)");

   if (cf.type != AGX_INDEX_IMMEDIATE)
      a3_fail(a, "iter with a coefficient register in a GPR");

   switch (I->interpolation) {
   case AGX_INTERPOLATION_CENTER:
      break;
   case AGX_INTERPOLATION_SAMPLE:
      if (sample.type != AGX_INDEX_IMMEDIATE || sample.value > 3)
         a3_fail(a, "iter at a dynamic sample (G15: itrcoords)");
      mode = 4 | sample.value;
      break;
   case AGX_INTERPOLATION_COORDS:
      /* At the coordinates itrcoords wrote to the register sample */
      if (I->perspective)
         a3_fail(a, "projected iter at coordinates");

      for (unsigned i = 0; i < I->channels; i++) {
         agx_index d = dest;
         d.value += i * (dest.size == AGX_SIZE_32 ? 2 : 1);

         a3_begin(a, I, AGX3_ITER_COORDS, T_ITER_COORDS);
         a3_set(a, ITER_COORDS_DST, a3_reg(a, ITER_COORDS_DST, d, 0, 0, 0));
         a3_set(a, ITER_COORDS_CF,
                a3_kind_value(a, ITER_COORDS_CF, A3_CF, cf.value + i, 0));
         a3_set(a, ITER_COORDS_CRD,
                a3_reg(a, ITER_COORDS_CRD, sample, 0, 0, 0));
         a3_emit(a, emission);
      }
      return;
   default:
      a3_fail(a, "iter at the centroid (G15: itrcoords)");
   }

   for (unsigned i = 0; i < I->channels; i++) {
      agx_index d = dest;
      d.value += i * (dest.size == AGX_SIZE_32 ? 2 : 1);

      a3_begin(a, I, AGX3_ITER, T_ITER);
      a3_set(a, ITER_DST, a3_reg(a, ITER_DST, d, 0, 0, 0));
      a3_set(a, ITER_CF, a3_kind_value(a, ITER_CF, A3_CF, cf.value + i, 0));
      a3_set(a, ITER_MODE, mode);
      a3_set(a, ITER_PERSP, I->perspective);
      a3_emit(a, emission);
   }
}

/* itrcoords: the coordinates of a sample by its index (bit 57 set), or of the
 * centroid of the coverage and the sample mask input (bit 57 clear), from a
 * 32-bit register: the index in its low half, or the two masks (AGX3.xml).
 * The compiler builds the 32-bit source as a vector of two 16-bit halves,
 * which register allocation aligns to a 32-bit register. Synchronous like
 * iter. */
static void
a3_itrcoords(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   agx_index src = I->src[0];

   if (I->interpolation != AGX_INTERPOLATION_SAMPLE &&
       I->interpolation != AGX_INTERPOLATION_CENTROID)
      a3_fail(a, "itrcoords of a location other than a sample or centroid");

   if (src.type == AGX_INDEX_REGISTER && src.size == AGX_SIZE_16 &&
       agx_channels(src) == 2 && !(src.value & 1)) {
      src.size = AGX_SIZE_32;
      src.channels_m1 = 0;
   }

   if (src.type != AGX_INDEX_REGISTER || src.size != AGX_SIZE_32)
      a3_fail(a, "itrcoords of a source other than a 32-bit register");

   a3_begin(a, I, AGX3_ITRCOORDS, T_ITRCOORDS);
   a3_set(a, ITRCOORDS_DST, a3_reg(a, ITRCOORDS_DST, I->dest[0], 0, 0, 0));
   a3_set(a, ITRCOORDS_SRC, a3_reg(a, ITRCOORDS_SRC, src, 0, 0, 0));
   a3_set(a, ITRCOORDS_SAMPLE, I->interpolation == AGX_INTERPOLATION_SAMPLE);
   a3_set(a, ITRCOORDS_DCACHE, 0);
   a3_emit(a, emission);
}

/* zs_emit (depthf) reads movmsk's coverage word and the depth from the
 * coordinate registers Apple's depth-output shader writes before it: its
 * 10-byte mov and cmpsel name cr4096 and cr4097 (the destination's y field,
 * vdst bit 13, set), 32-bit, and its zs_emit names cr0 (draws.md, depth
 * output). With the same values in cr0 and cr1 the 3D job hangs in a
 * punch-through pass (kernel #281). The stencil's place, after the depth,
 * or the depth's when alone, is a guess (the IR widens it to 32 bits with
 * depth). */
#define A3_ZS_CR 4096

static void
a3_zs_emit(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   static const uint8_t mode[] = {[1] = 1, [2] = 0, [3] = 3};

   if (I->zs < 1 || I->zs > 3)
      a3_fail(a, "zs_emit of nothing");

   /* cr4096: the coverage word with the stencil in bits 16:23 (the
    * compiler builds it); cr4097: the depth, the first of zs_emit's
    * second source. The stencil alone (mode 0) writes nothing on the
    * hardware, so the drivers add the depth (agx_nir_lower_stencil_only_g15) */
   a3_mov_cr(a, I, A3_ZS_CR + 0, I->src[0], emission);
   if (I->zs & 1) {
      agx_index z = I->src[1];
      z.size = AGX_SIZE_32;
      a3_mov_cr(a, I, A3_ZS_CR + 1, z, emission);
   }

   a3_begin(a, I, AGX3_ZS_EMIT, T_ZS_EMIT);
   a3_set(a, ZS_CR, a3_kind_value(a, ZS_CR, A3_CR, 0, BITFIELD_BIT(A3_LU)));
   a3_set(a, ZS_G, 0);
   a3_set(a, ZS_MODE, mode[I->zs]);
   a3_set(a, ZS_WAIT, BITFIELD_BIT(0));
   a3_emit(a, emission);
}

/* sample_mask: G15's version has one source, the 32-bit value for cr0 that
 * the compiler built (LIVE | active << 4 | TARGET << 8, see movmsk in
 * AGX3.xml); a mov writes cr0 and movmsk waits for it and applies it. The
 * second source is unused. */
static void
a3_sample_mask(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   a3_mov_cr(a, I, 0, I->src[0], emission);

   a3_begin(a, I, AGX3_MOVMSK, T_MOVMSK);
   a3_set(a, MOVMSK_CR,
          a3_kind_value(a, MOVMSK_CR, A3_CR, 0, BITFIELD_BIT(A3_LU)));
   a3_set(a, MOVMSK_G, 0);
   a3_set(a, MOVMSK_WAIT, BITFIELD_BIT(0));
   a3_emit(a, emission);
}

/* imgwblk bits 77:79, by the tilebuffer format (smp.md): Metal's driver gives
 * each render target a tile format index (TextureFormatTable entry +0x20:
 * 2/3 8-bit snorm/unorm, 4/5 8-bit sint/uint, 6/7 16-bit snorm/unorm, 8/9
 * 16-bit sint/uint, 10/11 32-bit sint/uint, 12 16-bit float, 14 the packed
 * 16-bit unorms, 17 32-bit float), from which its end-of-tile program picks
 * 3, 2, 4, 0, 1, 0, 0 and 1. RGB10A2 and RG11B10F take 3 (probed on the
 * hardware); RGB9E5 is not known. */
static unsigned
a3_imgwblk_format(enum agx_format format)
{
   switch (format) {
   case AGX_FORMAT_I8:
      return 2;
   case AGX_FORMAT_I16:
   case AGX_FORMAT_F16:
      return 0;
   case AGX_FORMAT_I32:
      return 1;
   case AGX_FORMAT_U16NORM:
   case AGX_FORMAT_S16NORM:
      return 4;
   default:
      return 3;
   }
}

/* block_image_store (imgwblk): the tile to the image a PBE descriptor
 * describes. G15 takes the descriptor's address in a 64-bit uniform (the IR's
 * base, which the entry program loads) and coordinate registers: x and y of
 * the tile's first pixel (the IR's coords, a 32-bit vec2) in cr0 and cr1, and
 * pack_imgwir's register in cr2 (cr3 for a multisampled or layered target),
 * all written in group 0 as demo15's
 * end-of-tile program does (it ran on the hardware: imadd to cr0 and cr1,
 * pack_imgwir to cr4 in 16-bit units, imgwblk waiting for group 0).
 * pack_imgwir's first source is the render target's offset in the tilebuffer
 * in bytes (boot #406: a second target at 4 comes out right with it, and
 * every offset in imgwblk's own operand reads another descriptor); its
 * other two are not known, the supported forms, 2D with no layer, take
 * demo15's: zero sources and the layout it uses (Apple's multisampled pass
 * too), with the flags of the image type. imgwblk's own operand stays 0,
 * each render target has its own
 * descriptor pointer. A following block write overwrites cr2 before the
 * earlier one has read it, so each is followed by a wait (boots #404, #406:
 * without it the two swap their data). The coordinates move through mov to
 * cr, which demo15 does not use (its imadd writes the cr), the way st_vary
 * does. */
static void
a3_block_image_store(struct a3 *a, const agx_instr *I,
                     struct util_dynarray *emission)
{
   agx_index base = I->src[0];
   agx_index offset = I->src[2];
   agx_index coords = I->src[3];

   bool array = I->dim == AGX_DIM_2D_ARRAY || I->dim == AGX_DIM_2D_MS_ARRAY;

   /* The tilebuffer offset's pack is the fourth coordinate register of every
    * form but the 1-sample 2D write's: a multisampled 2D write leaves cr2
    * unwritten (Apple's end-of-tile program, its 0x08a writes) */
   unsigned offset_cr = (array || I->dim == AGX_DIM_2D_MS) ? 3 : 2;

   if (base.type != AGX_INDEX_UNIFORM || base.size != AGX_SIZE_64 ||
       coords.type != AGX_INDEX_REGISTER || coords.size != AGX_SIZE_32 ||
       agx_channels(coords) != (array ? 3 : 2) || !I->explicit_coords ||
       (I->dim != AGX_DIM_2D && I->dim != AGX_DIM_2D_MS && !array))
      a3_fail(a, "block_image_store of a bindless image or of this dimension");

   agx_index x = agx_register(coords.value, AGX_SIZE_32);
   agx_index y = agx_register(coords.value + 2, AGX_SIZE_32);

   a3_mov_cr(a, I, 0, x, emission);
   a3_mov_cr(a, I, 1, y, emission);

   /* A layered target (Apple's end-of-tile program, its 2D array writes):
    * the layer packed into cr2 by pack_imgwir B code 0x204, the tilebuffer
    * offset's pack moved to cr3 */
   if (array) {
      agx_index layer = agx_register(coords.value + 4, AGX_SIZE_16);

      a3_begin(a, I, AGX3_PACK_IMGWIR, T_PACK_IMGWIR_LAYER);
      a3_set(a, PACK_IMGWIR_DST,
             a3_kind_value(a, PACK_IMGWIR_DST, A3_CR, 2, 0));
      a3_set(a, PACK_IMGWIR_G, 0);
      a3_set(a, PACK_IMGWIR_A, a3_src9(a, PACK_IMGWIR_A, layer));
      a3_emit(a, emission);
   }

   a3_begin(a, I, AGX3_PACK_IMGWIR, T_PACK_IMGWIR);
   a3_set(a, PACK_IMGWIR_DST,
          a3_kind_value(a, PACK_IMGWIR_DST, A3_CR, offset_cr, 0));
   a3_set(a, PACK_IMGWIR_G, 0);
   a3_set(a, PACK_IMGWIR_A, a3_src9(a, PACK_IMGWIR_A, offset));
   a3_set(a, PACK_IMGWIR_B, a3_imm(a, PACK_IMGWIR_B, 0, false));
   a3_set(a, PACK_IMGWIR_C, a3_imm(a, PACK_IMGWIR_C, 0, false));
   a3_emit(a, emission);

   a3_begin(a, I, AGX3_BLOCK_IMAGE_STORE, T_IMGWBLK);
   a3_set(a, IMGWBLK_CR,
          a3_kind_value(a, IMGWBLK_CR, A3_CR, 0, BITFIELD_BIT(A3_LU)));
   a3_set(a, IMGWBLK_G, 0);
   a3_set(a, IMGWBLK_PTR, a3_membase(a, IMGWBLK_PTR, base));
   a3_set(a, IMGWBLK_OFFSET, a3_imm(a, IMGWBLK_OFFSET, 0, false));
   /* The image type in the flags (smp.md): 2D 0x1d6a, 2D multisampled
    * 0x1c8a, the dimension's fields only; bits 77:79 (flags 11:13) the tile
    * format's class */
   unsigned flags = I->dim == AGX_DIM_2D_MS         ? 0x1c8a
                    : I->dim == AGX_DIM_2D_MS_ARRAY ? 0x1c0a
                    : I->dim == AGX_DIM_2D_ARRAY    ? 0x1d4a
                                                    : 0x1d6a;
   flags = (flags & ~(7u << 11)) | (a3_imgwblk_format(I->format) << 11);
   a3_set(a, IMGWBLK_FLAGS, flags);
   a3_set(a, IMGWBLK_WAIT, BITFIELD_BIT(0));
   a3_emit(a, emission);

   a3_wait(a, I, 0, emission);
}

/* image_write (imgwci): the PBE descriptor is entry index of the 32 B array
 * at the IR's bindless base (a 64-bit uniform or register pair), the index an
 * immediate or a register. The coordinates and the data go to a tuple of
 * eight 32-bit coordinate registers from vary_cr (agx3_assign_crs), written
 * in group 0, which the instruction waits for and frees. With two
 * coordinates (2D, a 1D array's layer as the second): x, y, data x, data y in
 * slots 0-3, data z and w in 6 and 7 (Apple's code). A 2D array's layer is
 * slot 3 and moves data y to slot 5; a 3D image's z is slot 2 and moves the
 * data to slots 4-7 (both on the hardware, kernels #274 and #276; the 1D
 * array is not verified). Apple's builder clamps a layer to 0x7ff (pack_args'
 * min7ff).
 * Cube images are 2D arrays here. The level is the instruction's immediate;
 * all four data components are given. */

static void
a3_image_write(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   static const enum agx3_instr instr[] = {
      [AGX_DIM_1D] = AGX3_IMAGE_WRITE_1D,
      [AGX_DIM_1D_ARRAY] = AGX3_IMAGE_WRITE_1D_ARRAY,
      [AGX_DIM_2D] = AGX3_IMAGE_WRITE,
      [AGX_DIM_2D_ARRAY] = AGX3_IMAGE_WRITE_2D_ARRAY,
      [AGX_DIM_2D_MS] = A3_NONE,
      [AGX_DIM_3D] = AGX3_IMAGE_WRITE_3D,
      [AGX_DIM_CUBE] = AGX3_IMAGE_WRITE_CUBE,
      [AGX_DIM_CUBE_ARRAY] = AGX3_IMAGE_WRITE_CUBE_ARRAY,
      [AGX_DIM_2D_MS_ARRAY] = A3_NONE,
   };
   agx_index data = I->src[0], coords = I->src[1], lod = I->src[2];
   agx_index base = I->src[3], index = I->src[4];

   if (I->dim >= ARRAY_SIZE(instr) || instr[I->dim] == A3_NONE)
      a3_fail(a, "image write to a multisampled image (lowered on G13)");
   if (lod.type != AGX_INDEX_IMMEDIATE || lod.value >= 4096)
      a3_fail(a, "image write at a level that is not an immediate");
   if (data.size != AGX_SIZE_32 || coords.size != AGX_SIZE_32)
      a3_fail(a, "image write of 16-bit data or coordinates");

   unsigned ncoords = coords.channels_m1 + 1;

   /* The third coordinate: a 3D image's z or an array's layer */
   bool is_3d = I->dim == AGX_DIM_3D;
   const uint8_t coord_slot[3] = {0, 1, is_3d ? 2 : 3};
   static const uint8_t data_slots[3][4] = {
      {2, 3, 6, 7}, /* two coordinates */
      {2, 5, 6, 7}, /* an array's layer */
      {4, 5, 6, 7}, /* a 3D image's z */
   };
   const uint8_t *data_slot = data_slots[ncoords > 2 ? (is_3d ? 2 : 1) : 0];

   for (unsigned c = 0; c < ncoords; ++c) {
      agx_index v = coords;
      v.value += 2 * c;
      v.channels_m1 = 0;
      a3_mov_cr(a, I, I->vary_cr + coord_slot[c], v, emission);
   }

   for (unsigned c = 0; c < 4; ++c) {
      agx_index v = data;
      v.value += 2 * c;
      v.channels_m1 = 0;
      a3_mov_cr(a, I, I->vary_cr + data_slot[c], v, emission);
   }

   a3_begin(a, I, instr[I->dim], T_IMGW);
   a3_set(a, IMGW_CR,
          a3_kind_value(a, IMGW_CR, A3_CR, I->vary_cr, BITFIELD_BIT(A3_LU)));
   a3_set(a, IMGW_G, 0);
   a3_set(a, IMGW_BASE, a3_membase(a, IMGW_BASE, base));

   if (index.type == AGX_INDEX_IMMEDIATE) {
      a3_set(a, IMGW_INDEX, a3_imm(a, IMGW_INDEX, index.value, false));
   } else {
      if (index.size == AGX_SIZE_64)
         a3_fail(a, "image index is 64-bit");
      index.abs = index.neg = false;
      a3_set(a, IMGW_INDEX, a3_reg(a, IMGW_INDEX, index, 0, 0, 0));
   }

   a3_set(a, IMGW_LOD, lod.value);
   a3_set(a, IMGW_DATA, 4);
   a3_set(a, IMGW_WAIT, BITFIELD_BIT(0));
   a3_emit(a, emission);
}

/* icmp_ballot/fcmp_ballot (and the quad forms): cc, dest, t/v, A, B. The
 * inverted conditions are the condition + 8. G15 has no NaN-aware ballot
 * compare (G13's LTN and GTN). */
static void
a3_ballot(struct a3 *a, const agx_instr *I, struct util_dynarray *emission)
{
   static const int8_t icc[] = {
      [AGX_ICOND_UEQ] = 1, [AGX_ICOND_ULT] = 6, [AGX_ICOND_UGT] = 5,
      [AGX_ICOND_SEQ] = 1, [AGX_ICOND_SLT] = 3, [AGX_ICOND_SGT] = 2,
   };
   static const int8_t fcc[] = {
      [AGX_FCOND_EQ] = 0,   [AGX_FCOND_LT] = 4, [AGX_FCOND_GT] = 2,
      [AGX_FCOND_LTN] = -1, [AGX_FCOND_GE] = 1, [AGX_FCOND_LE] = 3,
      [AGX_FCOND_GTN] = -1,
   };
   bool fl =
      I->op == AGX_OPCODE_FCMP_BALLOT || I->op == AGX_OPCODE_FCMP_QUAD_BALLOT;
   bool quad = I->op == AGX_OPCODE_ICMP_QUAD_BALLOT ||
               I->op == AGX_OPCODE_FCMP_QUAD_BALLOT;
   int cc = fl ? fcc[I->fcond] : icc[I->icond];

   if (cc < 0)
      a3_fail(a, "ballot condition not supported");

   a3_begin(a, I,
            fl ? (quad ? AGX3_FCMP_QUAD_BALLOT : AGX3_FCMP_BALLOT)
               : (quad ? AGX3_ICMP_QUAD_BALLOT : AGX3_ICMP_BALLOT),
            fl ? T_FCMP_BAL : T_ICMP_BAL);
   a3_set(a, 0, cc + (I->invert_cond ? 8 : 0));
   a3_set(a, 1, a3_reg(a, 1, I->dest[0], 0, 0, 0));
   a3_set(a, 3, a3_cmp_src(a, 3, I->src[0], !fl, I->icond));
   a3_set(a, 4, a3_cmp_src(a, 4, I->src[1], !fl, I->icond));
   a3_vt(a, 2);
   a3_emit(a, emission);
}

/* The shuffles: dest, t/v, the value, the lane (index, mask or delta) */
static enum agx3_instr
a3_shuffle(enum agx_opcode op)
{
   switch (op) {
   case AGX_OPCODE_SHUFFLE:
      return AGX3_SIMD_SHUFFLE;
   case AGX_OPCODE_SHUFFLE_XOR:
      return AGX3_SIMD_SHUFFLE_XOR;
   case AGX_OPCODE_SHUFFLE_UP:
      return AGX3_SIMD_SHUFFLE_UP;
   case AGX_OPCODE_SHUFFLE_DOWN:
      return AGX3_SIMD_SHUFFLE_DOWN;
   case AGX_OPCODE_QUAD_SHUFFLE:
      return AGX3_QUAD_SHUFFLE;
   case AGX_OPCODE_QUAD_SHUFFLE_XOR:
      return AGX3_QUAD_SHUFFLE_XOR;
   case AGX_OPCODE_QUAD_SHUFFLE_UP:
      return AGX3_QUAD_SHUFFLE_UP;
   case AGX_OPCODE_QUAD_SHUFFLE_DOWN:
      return AGX3_QUAD_SHUFFLE_DOWN;
   default:
      return AGX3_NUM_INSTRS;
   }
}

/* Control flow. G15 keeps the exec mask stack in hardware, not in r0l
 * (boot #145: r0l = 1 + pop_exec 0 masks nothing). The stack instructions
 * (if, else, while, break; push_exec = if cc_always, pop_exec) take a
 * condition code cc0-cc5 that a cmp sets, and otherwise behave as G13's:
 * if, else cc_always, nested ifs, pop_exec 2, while and break (also from
 * inside an if) all match G13's model on the hardware (#145). else cc
 * re-enables only the threads whose cc is true. Jumps count from the jump
 * instruction's start. The IR's compare-and-branch instructions are a cmp
 * to CF_CC and the instruction on it. */
#define CF_CC      0
#define CC_ALWAYS  6
#define CMP_CC     0
#define CMP_D      1
#define CMP_A      2
#define CMP_B      3
#define EXEC_A     0
#define EXEC_N     1
#define JMP_TARGET 0

struct agx3_fixup {
   agx_block *block;
   uint32_t offset;
   bool skip_to_end;
};

static unsigned
a3_icc(enum agx_icond c)
{
   static const uint8_t icc[] = {
      [AGX_ICOND_UEQ] = 15, [AGX_ICOND_ULT] = 5, [AGX_ICOND_UGT] = 4,
      [AGX_ICOND_SEQ] = 15, [AGX_ICOND_SLT] = 7, [AGX_ICOND_SGT] = 6,
   };
   return icc[c];
}

static unsigned
a3_fcc(enum agx_fcond c)
{
   static const uint8_t fcc[] = {
      [AGX_FCOND_EQ] = 8,  [AGX_FCOND_LT] = 3,  [AGX_FCOND_GT] = 2,
      [AGX_FCOND_LTN] = 1, [AGX_FCOND_GE] = 10, [AGX_FCOND_LE] = 11,
      [AGX_FCOND_GTN] = 0,
   };
   return fcc[c];
}

/* cmp: cc = A cond B, inverted by invert (cmpselcc4: 32 + cond, 48 + the
 * inverted cond). The 10-byte form lacks bits of the compared values (an
 * abs on A); the 14-byte form has them, as cmpsel's. */
static void
a3_cmp(struct a3 *a, const agx_instr *I, bool fl, unsigned cond, bool invert,
       unsigned cc, struct util_dynarray *emission)
{
   for (unsigned len = 10;; len = 14) {
      uint8_t out[AGX3_MAX_BYTES];

      a3_begin(a, I, len == 10 ? AGX3_CMPCC10 : AGX3_CMPCC14,
               len == 10 ? T_ZERO10 : T_ZERO14);
      a3_set(a, CMP_CC, (invert ? 48 : 32) + cond);
      a3_set(a, CMP_D, cc);
      a3_set(a, CMP_A, a3_cmp_src(a, CMP_A, I->src[0], !fl, I->icond));
      a3_set(a, CMP_B, a3_cmp_src(a, CMP_B, I->src[1], !fl, I->icond));

      if (len == 14 || a3_check(a, out))
         break;
   }

   a3_emit(a, emission);
}

/* An exec stack instruction on a condition code */
static void
a3_exec(struct a3 *a, const agx_instr *I, enum agx3_instr instr,
        const char *tmpl, unsigned cc, unsigned n,
        struct util_dynarray *emission)
{
   a3_begin(a, I, instr, tmpl);
   a3_set(a, EXEC_A, cc);
   a3_set(a, EXEC_N, n);
   a3_emit(a, emission);
}

/* if/else/while: the IR's compare, then the instruction on its result.
 * A compare of 0.0 == 0.0 (emit_if's unconditional else, agx_push_exec's
 * if) is cc_always: the if is push_exec. */
static void
a3_cf_cmp(struct a3 *a, const agx_instr *I, enum agx3_instr instr,
          const char *tmpl, struct util_dynarray *emission)
{
   bool fl = I->op == AGX_OPCODE_IF_FCMP || I->op == AGX_OPCODE_ELSE_FCMP ||
             I->op == AGX_OPCODE_WHILE_FCMP ||
             I->op == AGX_OPCODE_BREAK_IF_FCMP;
   unsigned cond = fl ? a3_fcc(I->fcond) : a3_icc(I->icond);
   unsigned cc = CF_CC;

   bool always = fl && I->fcond == AGX_FCOND_EQ && !I->invert_cond &&
                 agx_is_equiv(I->src[0], agx_zero()) &&
                 agx_is_equiv(I->src[1], agx_zero());

   if (always)
      cc = CC_ALWAYS;
   else
      a3_cmp(a, I, fl, cond, I->invert_cond, cc, emission);

   a3_exec(a, I, instr, tmpl, cc, I->nest, emission);
}

/* A jump: the target is patched by agx3_fixup_branches */
static void
a3_jmp(struct a3 *a, const agx_instr *I, enum agx3_instr instr,
       struct util_dynarray *emission, struct util_dynarray *fixups)
{
   if (!I->target || !fixups)
      a3_fail(a, "jump without a target");

   struct agx3_fixup fixup = {
      .block = I->target,
      .offset = emission->size,
      .skip_to_end = I->op == AGX_OPCODE_JMP_EXEC_NONE_AFTER,
   };
   util_dynarray_append(fixups, fixup);

   a3_begin(a, I, instr, T_JMP);
   a3_set(a, JMP_TARGET, 0);
   a3_emit(a, emission);
}

/* The branch operand: bit 0 clear (relative), the offset in halfwords from
 * the jump instruction's start in bits 1:47; at instruction bits 24:71 */
void
agx3_fixup_branches(struct util_dynarray *emission,
                    struct util_dynarray *fixups)
{
   util_dynarray_foreach(fixups, struct agx3_fixup, fix) {
      uint8_t *p = (uint8_t *)emission->data + fix->offset;
      uint32_t target =
         fix->skip_to_end ? fix->block->last_offset : fix->block->offset;
      int64_t bytes = (int64_t)target - (int64_t)fix->offset;
      uint64_t v = ((uint64_t)(bytes >> 1) & BITFIELD64_MASK(47)) << 1;

      assert((bytes & 1) == 0);
      for (unsigned i = 0; i < 6; i++)
         p[3 + i] = v >> (8 * i);
   }
}

void
agx3_pack_instr(struct util_dynarray *emission, struct util_dynarray *fixups,
                const agx_instr *I, unsigned scratch_B)
{
   struct a3 a = {.I = I, .scratch_B = scratch_B};

   static bool once;
   if (!once) {
      once = true;
      assert(!strcmp(agx3_kind_names[A3_CR], "cr"));
      assert(!strcmp(agx3_kind_names[A3_CF], "cf"));
      assert(!strcmp(agx3_kind_names[A3_SS], "ss"));
      a3_survey_on = debug_get_bool_option("AGX3_SURVEY", false);
      a3_debug = debug_get_bool_option("AGX3_DEBUG", false);
      if (a3_survey_on)
         atexit(a3_survey_print);
   }

   if (a3_survey_on && setjmp(a3_survey_jmp))
      return;

   switch (I->op) {
   case AGX_OPCODE_MOV_IMM:
      a3_mov_imm(&a, I, emission);
      return;

   case AGX_OPCODE_GET_SR:
   case AGX_OPCODE_GET_SR_COVERAGE:
   case AGX_OPCODE_GET_SR_BARRIER:
      /* G15 delivers the value in wait group g (Apple's code waits on it
       * at the first use); the IR expects G13's synchronous get_sr. */
      a3_begin(&a, I, AGX3_GET_SR, T_GET_SR);
      a3_set(&a, GET_SR_G, 0);
      a3_set(&a, GET_SR_DST, a3_reg(&a, GET_SR_DST, I->dest[0], 0, 0, 0));
      a3_set(&a, GET_SR_SR, a3_sr(&a, I->sr));
      a3_emit(&a, emission);

      a3_begin(&a, I, AGX3_WAIT, T_WAIT);
      a3_set(&a, WAIT_MASK, BITFIELD_BIT(0));
      a3_emit(&a, emission);
      return;

   case AGX_OPCODE_IADD:
      a3_iadd_op(&a, I, emission);
      return;

   case AGX_OPCODE_IMAD: {
      agx_index srcs[3] = {I->src[0], I->src[1], I->src[2]};
      bool sub = srcs[2].neg;
      srcs[2].neg = false;

      if (I->shift)
         a3_fail(&a, "imad with a shift");

      a3_alu(&a, I, sub ? &a3_imsub : &a3_imadd, srcs, emission);
      return;
   }

   case AGX_OPCODE_BFI:
   case AGX_OPCODE_BFEIL:
   case AGX_OPCODE_EXTR: {
      const struct a3_alu *d = I->op == AGX_OPCODE_BFI     ? &a3_bfi
                               : I->op == AGX_OPCODE_BFEIL ? &a3_bfeil
                                                           : &a3_extr;
      a3_begin(&a, I, d->instr, d->tmpl);
      a3_set(&a, 0, a3_reg(&a, 0, I->dest[0], 0, 0, 0));
      for (unsigned i = 0; i < 3; i++)
         a3_set(&a, d->srcs[i], a3_src(&a, d->srcs[i], I->src[i], true));
      a3_set(&a, BFI_MASK, I->bfi_mask);
      a3_vt(&a, VT);
      a3_emit(&a, emission);
      return;
   }

   case AGX_OPCODE_ASR:
      /* asr widens a 16-bit value before it shifts, as the compares do, so
       * the value is an extended source: without sx, 0xea00 >> 8 is 0x00ea
       * and every 8- and 16-bit sign extension (i2i16 is bfi then asr) came
       * out unsigned. The shift amount has no sx bit.
       */
      a3_begin(&a, I, a3_asr.instr, a3_asr.tmpl);
      a3_set(&a, 0, a3_reg(&a, 0, I->dest[0], 0, 0, 0));
      a3_set(&a, 2, a3_ext_src(&a, 2, I->src[0]));
      a3_set(&a, 3, a3_src(&a, 3, I->src[1], true));
      if (I->saturate)
         a3_fail(&a, "saturate not supported");
      a3_vt(&a, a3_asr.vt);
      a3_emit(&a, emission);
      return;

   case AGX_OPCODE_BITREV:
      a3_alu(&a, I, &a3_bitrev, I->src, emission);
      return;

   case AGX_OPCODE_POPCOUNT:
      a3_alu(&a, I, &a3_popcount, I->src, emission);
      return;

   case AGX_OPCODE_FFS:
      a3_alu(&a, I, &a3_ffs, I->src, emission);
      return;

   case AGX_OPCODE_BITOP:
      a3_bitop(&a, I, emission);
      return;

   case AGX_OPCODE_ICMPSEL:
   case AGX_OPCODE_FCMPSEL:
      a3_cmpsel(&a, I, emission);
      return;

   case AGX_OPCODE_CONVERT:
      a3_convert(&a, I, emission);
      return;

   case AGX_OPCODE_FADD:
      a3_alu(&a, I, &a3_fadd, I->src, emission);
      return;

   case AGX_OPCODE_FMUL:
      a3_alu(&a, I, &a3_fmul, I->src, emission);
      return;

   case AGX_OPCODE_FFMA:
      a3_alu(&a, I, &a3_ffma, I->src, emission);
      return;

   case AGX_OPCODE_HADD:
      a3_alu(&a, I, &a3_hadd, I->src, emission);
      return;

   case AGX_OPCODE_HMUL:
      a3_alu(&a, I, &a3_hmul, I->src, emission);
      return;

   case AGX_OPCODE_HFMA:
      a3_alu(&a, I, &a3_hfma, I->src, emission);
      return;

   case AGX_OPCODE_FLOOR:
   case AGX_OPCODE_CEIL:
   case AGX_OPCODE_TRUNC:
   case AGX_OPCODE_ROUNDEVEN:
   case AGX_OPCODE_RCP:
   case AGX_OPCODE_RSQRT:
   case AGX_OPCODE_SRSQRT:
   case AGX_OPCODE_LOG2:
   case AGX_OPCODE_EXP2:
   case AGX_OPCODE_SIN_PT_2: {
      struct a3_alu d = {a3_funop(I->op), T_FUNOP, {2}, VT, -1, -1, false};
      a3_alu(&a, I, &d, I->src, emission);
      return;
   }

   case AGX_OPCODE_DEVICE_LOAD:
   case AGX_OPCODE_DEVICE_STORE:
      a3_mem(&a, I, emission);
      return;

   case AGX_OPCODE_UNIFORM_STORE:
      a3_uniform_store(&a, I, emission);
      return;

   case AGX_OPCODE_LOCAL_LOAD:
   case AGX_OPCODE_LOCAL_STORE:
      a3_local_mem(&a, I, emission);
      return;

   case AGX_OPCODE_LOCAL_ATOMIC:
      a3_local_atomic(&a, I, emission);
      return;

   case AGX_OPCODE_STACK_LOAD:
   case AGX_OPCODE_STACK_STORE:
      a3_stack_mem(&a, I, emission);
      return;

   case AGX_OPCODE_TEXTURE_SAMPLE:
   case AGX_OPCODE_TEXTURE_LOAD:
   case AGX_OPCODE_IMAGE_LOAD:
      a3_texture(&a, I, emission);
      return;

   case AGX_OPCODE_ATOMIC:
      a3_atomic(&a, I, emission);
      return;

   case AGX_OPCODE_THREADGROUP_BARRIER:
   case AGX_OPCODE_MEMORY_BARRIER:
   case AGX_OPCODE_MEMORY_BARRIER_2:
   case AGX_OPCODE_MEMORY_BARRIER_3:
   case AGX_OPCODE_UNKNOWN_BARRIER_1:
   case AGX_OPCODE_UNKNOWN_BARRIER_2:
   case AGX_OPCODE_DEVICE_BARRIER_1:
   case AGX_OPCODE_DEVICE_BARRIER_2:
   case AGX_OPCODE_IMAGE_BARRIER_1:
   case AGX_OPCODE_IMAGE_BARRIER_2:
   case AGX_OPCODE_IMAGE_BARRIER_3:
   case AGX_OPCODE_IMAGE_BARRIER_4:
   case AGX_OPCODE_FLUSH_MEMORY_TO_TEXTURE:
      a3_barrier(&a, I, emission);
      return;

   case AGX_OPCODE_ICMP_BALLOT:
   case AGX_OPCODE_ICMP_QUAD_BALLOT:
   case AGX_OPCODE_FCMP_BALLOT:
   case AGX_OPCODE_FCMP_QUAD_BALLOT:
      a3_ballot(&a, I, emission);
      return;

   case AGX_OPCODE_SHUFFLE:
   case AGX_OPCODE_SHUFFLE_XOR:
   case AGX_OPCODE_SHUFFLE_UP:
   case AGX_OPCODE_SHUFFLE_DOWN:
   case AGX_OPCODE_QUAD_SHUFFLE:
   case AGX_OPCODE_QUAD_SHUFFLE_XOR:
   case AGX_OPCODE_QUAD_SHUFFLE_UP:
   case AGX_OPCODE_QUAD_SHUFFLE_DOWN: {
      /* The lane operand is a 16-bit register or an immediate: a 32-bit
       * lane is read through its low half */
      agx_index srcs[2] = {I->src[0], I->src[1]};
      if (srcs[1].type != AGX_INDEX_IMMEDIATE)
         srcs[1].size = AGX_SIZE_16;

      struct a3_alu d = {
         a3_shuffle(I->op), T_SHUFFLE, {2, 3}, VT, -1, -1, true};
      a3_alu(&a, I, &d, srcs, emission);
      return;
   }

   case AGX_OPCODE_IF_ICMP:
   case AGX_OPCODE_IF_FCMP:
      a3_cf_cmp(&a, I, AGX3_IF, T_IF, emission);
      return;

   case AGX_OPCODE_ELSE_ICMP:
   case AGX_OPCODE_ELSE_FCMP:
      a3_cf_cmp(&a, I, AGX3_ELSE, T_ELSE, emission);
      return;

   case AGX_OPCODE_WHILE_ICMP:
   case AGX_OPCODE_WHILE_FCMP:
      a3_cf_cmp(&a, I, AGX3_WHILE, T_WHILE, emission);
      return;

   case AGX_OPCODE_BREAK_IF_ICMP:
   case AGX_OPCODE_BREAK_IF_FCMP:
      a3_cf_cmp(&a, I, AGX3_BREAK, T_BREAK, emission);
      return;

   case AGX_OPCODE_BREAK:
      a3_exec(&a, I, AGX3_BREAK, T_BREAK, CC_ALWAYS, I->nest, emission);
      return;

   case AGX_OPCODE_POP_EXEC:
      a3_begin(&a, I, AGX3_POP_EXEC, T_POP_EXEC);
      a3_set(&a, 0, I->nest);
      a3_emit(&a, emission);
      return;

   case AGX_OPCODE_JMP_EXEC_ANY:
      a3_jmp(&a, I, AGX3_JMP_EXEC_ANY, emission, fixups);
      return;

   case AGX_OPCODE_JMP_EXEC_NONE:
   case AGX_OPCODE_JMP_EXEC_NONE_AFTER:
      a3_jmp(&a, I, AGX3_JMP_EXEC_NONE, emission, fixups);
      return;

   case AGX_OPCODE_WAIT:
      a3_begin(&a, I, AGX3_WAIT, T_WAIT);
      a3_set(&a, WAIT_MASK, BITFIELD_BIT(I->scoreboard));
      a3_emit(&a, emission);
      return;

   case AGX_OPCODE_ST_VARY:
      a3_st_vary(&a, I, emission);
      return;

   case AGX_OPCODE_NO_VARYINGS:
      /* G15's st_var has no last flag to replace (not verified) */
      return;

   case AGX_OPCODE_LD_TILE:
   case AGX_OPCODE_ST_TILE:
      a3_tile(&a, I, emission);
      return;

   case AGX_OPCODE_WAIT_PIX:
   case AGX_OPCODE_SIGNAL_PIX:
      a3_pix(&a, I, emission);
      return;

   case AGX_OPCODE_LDCF:
      a3_ldcf(&a, I, emission);
      return;

   case AGX_OPCODE_ITER:
   case AGX_OPCODE_ITERPROJ:
      a3_iter(&a, I, emission);
      return;

   case AGX_OPCODE_ITRCOORDS:
      a3_itrcoords(&a, I, emission);
      return;

   case AGX_OPCODE_ZS_EMIT:
      a3_zs_emit(&a, I, emission);
      return;

   case AGX_OPCODE_SAMPLE_MASK:
      a3_sample_mask(&a, I, emission);
      return;

   case AGX_OPCODE_BLOCK_IMAGE_STORE:
      a3_block_image_store(&a, I, emission);
      return;

   case AGX_OPCODE_IMAGE_WRITE:
      a3_image_write(&a, I, emission);
      return;

   case AGX_OPCODE_STOP:
      a3_begin(&a, I, AGX3_STOP, T_STOP);
      a3_emit(&a, emission);
      return;

   case AGX_OPCODE_EXPORT:
      /* A zero-byte pseudo-op, as on G13 */
      return;

   default:
      a3_fail(&a, "opcode not supported");
   }
}

/*
 * Building blocks for the G15 entry program (lib/agx_esl.c), through the
 * same packer: registers and uniforms are 32-bit, numbered as rN/uN.
 */

/* The drivers build an entry program for every pipeline they emit, and the
 * packer checks each instruction it packs, so the building blocks keep their
 * encodings by their operands: direct-mapped, key 0 empty */
#define A3_MEMO_BITS 10

enum a3_memo_tag {
   A3_MEMO_MOV_IMM = 1,
   A3_MEMO_LOAD,
   A3_MEMO_WAIT,
   A3_MEMO_UNIFORM_STORE,
   A3_MEMO_STOP,
   A3_MEMO_SIGNAL_PIX,
};

static struct {
   uint64_t key;
   uint8_t length;
   uint8_t code[AGX3_MAX_BYTES];
} a3_memo[1 << A3_MEMO_BITS];

static simple_mtx_t a3_memo_lock = SIMPLE_MTX_INITIALIZER;

static unsigned
a3_memo_slot(uint64_t key)
{
   return (key * 0x9e3779b97f4a7c15ull) >> (64 - A3_MEMO_BITS);
}

static uint64_t
a3_memo_key(enum a3_memo_tag tag, uint64_t operands)
{
   assert(operands < BITFIELD64_BIT(56));
   return ((uint64_t)tag << 56) | operands;
}

static void
a3_memo_pack(struct util_dynarray *out, uint64_t key, const agx_instr *I)
{
   unsigned slot = a3_memo_slot(key);

   simple_mtx_lock(&a3_memo_lock);
   if (a3_memo[slot].key == key) {
      memcpy(util_dynarray_grow_bytes(out, 1, a3_memo[slot].length),
             a3_memo[slot].code, a3_memo[slot].length);
      simple_mtx_unlock(&a3_memo_lock);
      return;
   }
   simple_mtx_unlock(&a3_memo_lock);

   size_t at = out->size;
   agx3_pack_instr(out, NULL, I, 0);
   size_t length = out->size - at;

   if (length > AGX3_MAX_BYTES)
      return;

   simple_mtx_lock(&a3_memo_lock);
   a3_memo[slot].key = key;
   a3_memo[slot].length = length;
   memcpy(a3_memo[slot].code, (uint8_t *)out->data + at, length);
   simple_mtx_unlock(&a3_memo_lock);
}

/* mov_imm's immediate is a plain operand: the memo keeps the encoding with 0,
 * and the immediate is deposited into its fields */
void
agx3_emit_mov_imm(struct util_dynarray *out, unsigned reg, uint32_t imm)
{
   agx_instr I = {.op = AGX_OPCODE_MOV_IMM, .imm = 0};
   agx_index dst = agx_register(2 * reg, AGX_SIZE_32);
   I.dest = &dst;

   size_t at = out->size;
   assert(reg < 128);
   a3_memo_pack(out, a3_memo_key(A3_MEMO_MOV_IMM, reg), &I);

   const struct agx3_pack_spec *spec = agx3_spec(AGX3_MOV_IMM32);
   uint8_t *code = (uint8_t *)out->data + at;
   unsigned length = out->size - at;
   uint64_t ops[A3_MAX_OPS];

   agx3_unpack(code, spec, length, ops);
   ops[MOV_IMM_IMM] = imm;
   agx3_pack(code, spec, length, ops);
}

void
agx3_emit_load(struct util_dynarray *out, unsigned slot, unsigned dst,
               unsigned base, unsigned index16, unsigned count)
{
   agx_index d = agx_register(2 * dst, AGX_SIZE_32);
   agx_index srcs[2] = {
      agx_register(2 * base, AGX_SIZE_64),
      agx_immediate(index16),
   };
   agx_instr I = {
      .op = AGX_OPCODE_DEVICE_LOAD,
      .format = AGX_FORMAT_I32,
      .mask = BITFIELD_MASK(count),
      .shift = 2, /* the index counts 16 B: lsl 4 for i32 */
      .scoreboard = slot,
      .dest = &d,
      .src = srcs,
   };

   assert(slot < 8 && dst < 128 && base < 128 && index16 < (1 << 16) &&
          count <= 4);
   a3_memo_pack(
      out,
      a3_memo_key(A3_MEMO_LOAD, slot | (dst << 3) | (base << 10) |
                                   (index16 << 17) | ((uint64_t)count << 33)),
      &I);
}

void
agx3_emit_wait(struct util_dynarray *out, unsigned slot)
{
   agx_instr I = {.op = AGX_OPCODE_WAIT, .scoreboard = slot};
   a3_memo_pack(out, a3_memo_key(A3_MEMO_WAIT, slot), &I);
}

void
agx3_emit_uniform_store(struct util_dynarray *out, unsigned uniform,
                        unsigned src)
{
   agx_index srcs[2] = {
      agx_register(2 * src, AGX_SIZE_32),
      agx_immediate(2 * uniform),
   };
   agx_instr I = {.op = AGX_OPCODE_UNIFORM_STORE, .src = srcs};

   assert(uniform < (1 << 16) && src < 128);
   a3_memo_pack(out, a3_memo_key(A3_MEMO_UNIFORM_STORE, src | (uniform << 7)),
                &I);
}

void
agx3_emit_stop(struct util_dynarray *out)
{
   agx_instr I = {.op = AGX_OPCODE_STOP};
   a3_memo_pack(out, a3_memo_key(A3_MEMO_STOP, 0), &I);
}

void
agx3_emit_signal_pix(struct util_dynarray *out, unsigned mask)
{
   agx_instr I = {.op = AGX_OPCODE_SIGNAL_PIX, .writeout = mask};
   a3_memo_pack(out, a3_memo_key(A3_MEMO_SIGNAL_PIX, mask), &I);
}

/* The entry program's setup instructions, which no IR instruction packs:
 * their operands straight into AGX3.xml's fields */
static void
a3_emit_ops(struct util_dynarray *out, enum agx3_instr instr,
            const uint64_t *ops)
{
   const struct agx3_pack_spec *spec = agx3_spec(instr);
   uint8_t *code = util_dynarray_grow(out, uint8_t, spec->lengths[0]);

   assert(spec->nlengths == 1);
   agx3_pack(code, spec, spec->lengths[0], ops);
}

void
agx3_emit_speclm(struct util_dynarray *out, unsigned samples, unsigned stride,
                 unsigned shape, bool imageblock, unsigned threadgroup_64B)
{
   uint64_t ops[A3_MAX_OPS] = {
      [SPECLM_SAMPLES] = samples,
      [SPECLM_STRIDE] = stride,
      [SPECLM_SHAPE] = shape,
      [SPECLM_IMAGEBLOCK] = imageblock,
      [SPECLM_THREADGROUP] = threadgroup_64B != 0,
      [SPECLM_SIZE] = threadgroup_64B,
   };

   assert(samples < 4 && stride < 32 && shape < 256 &&
          threadgroup_64B < (1 << 10));
   a3_emit_ops(out, AGX3_SPECLM, ops);
}

void
agx3_emit_ldshdr(struct util_dynarray *out, uint64_t va, unsigned size_class)
{
   uint64_t ops[A3_MAX_OPS] = {
      [LDSHDR_VA] = va >> 6,
      [LDSHDR_SIZE] = size_class,
   };

   assert((va & 63) == 0 && va < BITFIELD64_BIT(48) && size_class < 4);
   a3_emit_ops(out, AGX3_LDSHDR, ops);
}

void
agx3_emit_spectpr(struct util_dynarray *out, unsigned crs, unsigned stack_words)
{
   uint64_t ops[A3_MAX_OPS] = {
      [SPECTPR_CR] = crs,
      [SPECTPR_STACK] = stack_words,
      [SPECTPR_STACK2] = stack_words,
   };

   assert(crs < (1 << 11) && stack_words < (1 << 16));
   a3_emit_ops(out, AGX3_SPECTPR, ops);
}

/*
 * The sample loop of agx_fast_link, G13's sample_loop_header and
 * sample_loop_footer in G15 instructions: r1l (AGX_ABI_FIN_SAMPLE_MASK) holds
 * the sample bit the loop body shades. The loop is the IR's top-level loop
 * (emit_loop): no push before it, a break on a condition code (G15 packs the
 * IR's break_if as break, agx_opt_break_if's nest 1), jmp_exec_any back to the
 * body and pop_exec 1 after it.
 */
void
agx3_emit_sample_loop_header(struct util_dynarray *out)
{
   agx_index r1l = agx_register(2, AGX_SIZE_16);
   agx_instr I = {.op = AGX_OPCODE_MOV_IMM, .imm = 1};
   I.dest = &r1l;
   agx3_pack_instr(out, NULL, &I, 0);
}

/* The footer at program offset footer_offset; the loop body starts at
 * loop_begin. nr_samples > 1. */
void
agx3_emit_sample_loop_footer(struct util_dynarray *out, unsigned nr_samples,
                             unsigned footer_offset, unsigned loop_begin)
{
   agx_index r1l = agx_register(2, AGX_SIZE_16);

   /* r1l <<= 1 */
   {
      agx_index srcs[2] = {agx_zero(), r1l};
      agx_instr I = {.op = AGX_OPCODE_IADD, .shift = 1};
      I.dest = &r1l;
      I.src = srcs;
      agx3_pack_instr(out, NULL, &I, 0);
   }

   /* Break the threads past the last sample: !(r1l < 1 << nr_samples) */
   {
      agx_index srcs[2] = {r1l, agx_immediate(1 << nr_samples)};
      agx_instr I = {
         .op = AGX_OPCODE_BREAK_IF_ICMP,
         .icond = AGX_ICOND_ULT,
         .invert_cond = true,
         .nest = 1,
      };
      I.src = srcs;
      agx3_pack_instr(out, NULL, &I, 0);
   }

   /* jmp_exec_any to the loop body, patched as agx3_fixup_branches does */
   {
      struct a3 a = {0};
      agx_instr I = {.op = AGX_OPCODE_JMP_EXEC_ANY};
      unsigned at = out->size;

      a3_begin(&a, &I, AGX3_JMP_EXEC_ANY, T_JMP);
      a3_set(&a, JMP_TARGET, 0);
      a3_emit(&a, out);

      uint8_t *p = (uint8_t *)out->data + at;
      int64_t bytes = (int64_t)loop_begin - (int64_t)(footer_offset + at);
      uint64_t v = ((uint64_t)(bytes >> 1) & BITFIELD64_MASK(47)) << 1;
      for (unsigned i = 0; i < 6; i++)
         p[3 + i] = v >> (8 * i);
   }

   {
      agx_instr I = {.op = AGX_OPCODE_POP_EXEC, .nest = 1};
      agx3_pack_instr(out, NULL, &I, 0);
   }
}

/* The offset of the first stop in the size bytes of instructions at code, -1
 * if none comes before the end or an instruction that does not decode */
int
agx3_stop_offset(const uint8_t *code, size_t size)
{
   for (size_t at = 0; at + 4 <= size;) {
      const uint8_t *p = code + at;
      if (p[0] == 0x0e && p[1] == 0 && p[2] == 0 && p[3] == 0)
         return at;

      unsigned len = agx3_instr_length(p, size - at);
      if (!len)
         return -1;

      at += len;
   }

   return -1;
}

/* The length of the AGX3 instruction in the size bytes at code, 0 if none
 * matches within them */
unsigned
agx3_instr_length(const uint8_t *code, size_t size)
{
   uint64_t raw[3] = {0};
   for (unsigned i = 0; i < MIN2(size, AGX3_MAX_BYTES); i++)
      raw[i / 8] |= (uint64_t)code[i] << (8 * (i % 8));

   if (!size)
      return 0;

   for (unsigned k = agx3_byte0_start[code[0]];
        k < agx3_byte0_start[code[0] + 1]; k++) {
      const struct agx3_pack_spec *spec = agx3_spec(agx3_byte0_instrs[k]);
      unsigned length = spec->lengths[0];

      if (spec->has_length) {
         unsigned idx =
            (raw[spec->length_bit / 64] >> (spec->length_bit % 64)) &
            (spec->nlengths - 1);
         length = spec->lengths[idx];
      }

      uint64_t r[3] = {raw[0], raw[1], raw[2]};
      if (length * 8 < 64) {
         r[0] &= BITFIELD64_MASK(length * 8);
         r[1] = r[2] = 0;
      } else if (length * 8 < 128) {
         r[1] &= BITFIELD64_MASK(length * 8 - 64);
         r[2] = 0;
      }

      if (length <= size && (r[0] & spec->exact[0]) == spec->value[0] &&
          (r[1] & spec->exact[1]) == spec->value[1] &&
          (r[2] & spec->exact[2]) == spec->value[2])
         return length;
   }

   return 0;
}
