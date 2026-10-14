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

#include "util/bitscan.h"
#include "agx_compiler.h"
#include "agx3_pack.h"

enum a3_kind {
   A3_STACK,
   A3_CR,
   A3_UNIFORM,
   A3_REG,
   A3_IMM,
   A3_FIMM,
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

/* The carry bits are outside the disassembler's model, but some of them
 * keep an earlier case from matching (a register destination whose carry
 * bit is clear reads as a cr register). Choose them so that the
 * disassembler's first match is c: the template's bits if they do, else
 * the smallest subset of the case's carry mask that does. */
static uint64_t
a3_fix_carry(const struct a3 *a, unsigned op, const struct agx3_case *c,
             uint64_t v)
{
   const struct agx3_pack_spec *s = a->spec;
   const struct agx3_case *cases = s->cases + s->case_start[op];
   unsigned n = s->ncases[op];
   uint64_t base = v & ~c->carry_mask;

   uint64_t t = base | (a->tmpl[op] & c->carry_mask);
   if (agx3_case_match(cases, n, t) == c)
      return t;

   unsigned bits[64], nbits = 0;
   u_foreach_bit64(b, c->carry_mask)
      bits[nbits++] = b;
   assert(nbits <= 12);

   for (unsigned pop = 0; pop <= nbits; pop++) {
      for (unsigned m = 0; m < (1u << nbits); m++) {
         if (util_bitcount(m) != pop)
            continue;

         uint64_t carry = 0;
         u_foreach_bit(i, m)
            carry |= BITFIELD64_BIT(bits[i]);

         if (agx3_case_match(cases, n, base | carry) == c)
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

static void NORETURN
a3_fail(const struct a3 *a, const char *what)
{
   fprintf(stderr, "AGX3: %s: ", what);
   agx_print_instr(a->I, stderr);
   UNREACHABLE("AGX3 packing failed");
}

/* A register or uniform operand */
static uint64_t
a3_reg(struct a3 *a, unsigned op, agx_index idx, uint32_t mods,
       uint32_t prefer, unsigned count_field)
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

/* An immediate operand of a register encoding */
static uint64_t
a3_imm(struct a3 *a, unsigned op, uint32_t value)
{
   const struct agx3_case *c = a3_case(a, op, A3_IMM, 0, 0, 0);
   uint64_t modvals[AGX3_MAX_CASE_FIELDS] = {0};

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

/* A source of an ALU instruction: a register, a uniform or an 8-bit
 * immediate. For integer sources the IR's abs is sign extension. */
static uint64_t
a3_src(struct a3 *a, unsigned op, agx_index idx, bool integer)
{
   if (idx.type == AGX_INDEX_IMMEDIATE)
      return a3_imm(a, op, idx.value);

   uint32_t mods = 0;
   if (idx.abs)
      mods |= BITFIELD_BIT(integer ? A3_SX : A3_ABS);
   if (idx.neg)
      mods |= BITFIELD_BIT(A3_NEG);

   return a3_reg(a, op, idx, mods, BITFIELD_BIT(integer ? A3_SX : A3_NEG),
                 0);
}

static void
a3_emit(struct a3 *a, struct util_dynarray *emission)
{
   agx3_pack(util_dynarray_grow_bytes(emission, 1, a->length), a->spec,
             a->length, a->ops);
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
   if (idx.type == AGX_INDEX_IMMEDIATE)
      return a3_imm(a, op, idx.value);

   return a3_reg(a, op, idx, idx.abs ? BITFIELD_BIT(A3_SX) : 0, 0, 0);
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
#define T_STORE     "e7005400020000001100009008" "00"
#define T_WAIT      "0604"                         /* wait 0 */
#define T_STOP      "0e000000"

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
};

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
   a3_set(a, MEM_G, I->scoreboard);
   a3_set(a, MEM_REG, a3_memreg(a, MEM_REG, reg, type, a3_mem_mask[I->mask]));
   a3_set(a, MEM_BASE, a3_membase(a, MEM_BASE, base));
   a3_set(a, MEM_INDEX, a3_memidx(a, MEM_INDEX, index));
   a3_set(a, MEM_SHIFT,
          a3_memshift(a, I->shift + a3_format_shift(I->format)));
   a3_set(a, MEM_OFFSET, 0);
   a3_emit(a, emission);
}

/* uniform_store: a mov to the uniform. The value is a 16, 32 or 64-bit
 * register; a 64-bit value is two 32-bit movs. */
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

   unsigned words = value.size == AGX_SIZE_64 ? 2 : 1;
   enum agx_size size = value.size == AGX_SIZE_16 ? AGX_SIZE_16 : AGX_SIZE_32;

   for (unsigned w = 0; w < words; w++) {
      agx_index src = value;
      src.size = size;
      src.value += 2 * w;
      src.abs = src.neg = false;

      agx_index dst = agx_uniform(offset.value + 2 * w, size);

      a3_begin(a, I, AGX3_MOV, T_MOV);
      a3_set(a, MOV_DST, a3_reg(a, MOV_DST, dst, 0, 0, 0));
      a3_set(a, MOV_SRC, a3_reg(a, MOV_SRC, src, 0, BITFIELD_BIT(A3_SX), 0));
      a3_emit(a, emission);
   }
}

void
agx3_pack_instr(struct util_dynarray *emission, const agx_instr *I)
{
   struct a3 a = {.I = I};

   switch (I->op) {
   case AGX_OPCODE_MOV_IMM:
      a3_mov_imm(&a, I, emission);
      return;

   case AGX_OPCODE_GET_SR:
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
      if (I->shift != 0 || I->dest[0].size != AGX_SIZE_32)
         a3_fail(&a, "iadd form not supported yet");

      a3_begin(&a, I, AGX3_IADD, T_IADD);
      a3_set(&a, IADD_DST, a3_reg(&a, IADD_DST, I->dest[0], 0, 0, 0));
      a3_set(&a, IADD_A, a3_src(&a, IADD_A, I->src[0], true));
      a3_set(&a, IADD_B, a3_src(&a, IADD_B, I->src[1], true));
      a3_set(&a, IADD_SAT, I->saturate);
      a3_emit(&a, emission);
      return;

   case AGX_OPCODE_BFI:
      if (I->dest[0].size != AGX_SIZE_32)
         a3_fail(&a, "bfi form not supported yet");

      a3_begin(&a, I, AGX3_BFI, T_BFI);
      a3_set(&a, BFI_DST, a3_reg(&a, BFI_DST, I->dest[0], 0, 0, 0));
      a3_set(&a, BFI_A, a3_src(&a, BFI_A, I->src[0], true));
      a3_set(&a, BFI_B, a3_src(&a, BFI_B, I->src[1], true));
      a3_set(&a, BFI_C, a3_src(&a, BFI_C, I->src[2], true));
      a3_set(&a, BFI_MASK, I->bfi_mask);
      a3_emit(&a, emission);
      return;

   case AGX_OPCODE_DEVICE_LOAD:
   case AGX_OPCODE_DEVICE_STORE:
      a3_mem(&a, I, emission);
      return;

   case AGX_OPCODE_UNIFORM_STORE:
      a3_uniform_store(&a, I, emission);
      return;

   case AGX_OPCODE_WAIT:
      a3_begin(&a, I, AGX3_WAIT, T_WAIT);
      a3_set(&a, WAIT_MASK, BITFIELD_BIT(I->scoreboard));
      a3_emit(&a, emission);
      return;

   case AGX_OPCODE_STOP:
      a3_begin(&a, I, AGX3_STOP, T_STOP);
      a3_emit(&a, emission);
      return;

   default:
      a3_fail(&a, "opcode not supported");
   }
}

/*
 * Building blocks for the G15 entry program (lib/agx_esl.c), through the
 * same packer: registers and uniforms are 32-bit, numbered as rN/uN.
 */
void
agx3_emit_mov_imm(struct util_dynarray *out, unsigned reg, uint32_t imm)
{
   agx_instr I = {.op = AGX_OPCODE_MOV_IMM, .imm = imm};
   agx_index dst = agx_register(2 * reg, AGX_SIZE_32);
   I.dest = &dst;
   agx3_pack_instr(out, &I);
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
   agx3_pack_instr(out, &I);
}

void
agx3_emit_wait(struct util_dynarray *out, unsigned slot)
{
   agx_instr I = {.op = AGX_OPCODE_WAIT, .scoreboard = slot};
   agx3_pack_instr(out, &I);
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
   agx3_pack_instr(out, &I);
}

void
agx3_emit_stop(struct util_dynarray *out)
{
   agx_instr I = {.op = AGX_OPCODE_STOP};
   agx3_pack_instr(out, &I);
}

/* The length of the AGX3 instruction at code, 0 if none matches */
unsigned
agx3_instr_length(const uint8_t *code)
{
   uint64_t raw[3] = {0};
   for (unsigned i = 0; i < AGX3_MAX_BYTES; i++)
      raw[i / 8] |= (uint64_t)code[i] << (8 * (i % 8));

   for (unsigned i = 0; i < AGX3_NUM_INSTRS; i++) {
      const struct agx3_pack_spec *spec = agx3_spec(i);
      unsigned length = spec->lengths[0];

      if (spec->has_length) {
         unsigned idx = (raw[spec->length_bit / 64] >> (spec->length_bit % 64)) &
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

      if ((r[0] & spec->exact[0]) == spec->value[0] &&
          (r[1] & spec->exact[1]) == spec->value[1] &&
          (r[2] & spec->exact[2]) == spec->value[2])
         return length;
   }

   return 0;
}
