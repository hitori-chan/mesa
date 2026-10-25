#!/usr/bin/env python3
# Copyright 2026 hitori1403
# SPDX-License-Identifier: GPL-2.0-only

# gen-pack.py ISA.xml: generates the AGX3 (G15, M3) packer tables, the
# inverse of the disassembler. The same scatter model the disassembler
# loads from the XML: an instruction is a set of fixed bits (exact) plus
# operand fields (dests, srcs, immediates and modifiers, in XML order),
# each a list of bit ranges with constant padding in between. The values
# the packer deposits are in the same space the disassembler's extract
# produces, padding bits included.
#
# On top of that, for each src/dest operand the cases of its encoding
# (the disassembler's case matching, in the same order) are emitted with
# their value-space layout: the case's exact bits, the register/immediate
# (kind) bit positions, and the named flag modifiers. The compiler picks
# a case by the operand it has (register class, size, flags) and
# assembles the operand value with agx3_case_value; agx3_pack then
# deposits the value into the instruction.
#
# The logic is verified by the round-trip against cases-agx3.txt, a
# corpus checked against applegpu (the C test runs this header): every
# corpus encoding must round-trip, and every operand value must
# decompose to exactly one case.

import sys

from isa import load, MAX_BITS

isa = load(sys.argv[1])

# The value space is one 64-bit word at most (asserted below)
MAX_FIELDS = 8  # the most a case holds (kind ranges + modifiers)


def to_alphanum(name):
    for i, j in {
        ' ': '_', '/': '_', '[': '', ']': '', '(': '', ')': '',
        '-': '_', ':': '', '.': '', ',': '', '=': '', '>': '',
        '#': '', '&': '', '*': '', '"': '', '+': '', '\'': '',
    }.items():
        name = name.replace(i, j)
    return name


def safe_name(name):
    name = to_alphanum(name)
    if not name[0].isalpha():
        name = '_' + name
    return name.lower()


def chunk(x):
    chunks = [hex((x >> (64 * i)) & ((1 << 64) - 1))
              for i in range(MAX_BITS // 64)]
    while len(chunks) > 1 and chunks[-1] == '0x0':
        chunks.pop()
    return f'{{{", ".join(chunks)}}}'


W = MAX_BITS // 64
B = MAX_BITS // 8

# The operand value layouts: names, in the order they appear in the
# XML, for the case-table reference
kind_names = []
mod_names = []
enc_order = list(isa.encodings)


def kind_name(n):
    if n not in kind_names:
        kind_names.append(n)
    return kind_names.index(n)


def mod_name(n):
    if n not in mod_names:
        mod_names.append(n)
    return mod_names.index(n)


def enc_index(n):
    return enc_order.index(n)


# The case's selection key (skey): how the compiler picks the case for
# the operand it has. Packed: bits 0:1 the size (0 = 16, 1 = 32,
# 2 = 64), bit 2 the bfloat property, bits 3:8 the kind name index
# (0x3f = none), bits 9:24 a mask over the modifiers the case offers.
# The kind name, the size and the bfloat property are the operand's
# register class; the offered modifiers say which flags the case can
# carry (e.g. sx makes the integer cases, abs/neg the float ones).
# When two cases of an encoding share a key (the pack formats, the
# count/size exact bits, ...), the encoding is not struct-key-unique and
# the selector also matches the case's exact pattern.
SKEY_KIND_NONE = 0x3f
SKEY_KIND_SHIFT = 3
SKEY_BFLOAT = 1 << 2
SKEY_SIZE32 = 1
SKEY_SIZE64 = 2
SKEY_MOD_SHIFT = 9


def case_key(c):
    size = 0
    if 'size32' in c.properties:
        size = SKEY_SIZE32
    elif 'size64' in c.properties:
        size = SKEY_SIZE64
    kind = kind_name(c.kind.name) if c.kind is not None else SKEY_KIND_NONE
    modmask = 0
    for m in c.modifiers:
        modmask |= 1 << mod_name(m.name)
    skey = size | (('bfloat' in c.properties) and SKEY_BFLOAT or 0) \
        | (kind << SKEY_KIND_SHIFT) | (modmask << 9)
    return skey


# Per encoding: the structural key is enough, or the selector must also
# match the exact pattern. Asserted along the way: the (key, exact)
# pair is unique in every encoding, so the selector is never ambiguous
# and the disassembler's first-match order is total.
enc_struct_key = {}
for k, enc in isa.encodings.items():
    seen = set()
    keys = set()
    for c in enc.cases:
        skey = case_key(c)
        keys.add(skey)
        pair = (skey, c.exact.value, c.exact.mask)
        assert pair not in seen, (k, c)
        seen.add(pair)
    enc_struct_key[k] = len(keys) == len(enc.cases)
for k, ok in enc_struct_key.items():
    if not ok:
        print(f'# note: {k} needs the exact pattern to select (the '
              f'structural key is ambiguous)', file=sys.stderr)


def case_layout(c):
    """The case's value-space fields: (is_kind, name_index, start, size),
    kind ranges first, then the modifiers in XML order."""
    f = []
    if c.kind is not None:
        for r in c.kind.bits.ranges:
            f.append((1, kind_name(c.kind.name), r.start, r.size))
    for m in c.modifiers:
        # One field per modifier: modvals and table_mod index the
        # modifiers by f - nkind
        assert len(m.bits.ranges) == 1, (c, m.name)
        for r in m.bits.ranges:
            f.append((0, mod_name(m.name), r.start, r.size))
    assert len(f) <= MAX_FIELDS, f
    owned = c.exact.mask
    for is_kind, ni, start, size in f:
        assert start + size <= 64, (start, size)
        owned |= (1 << size) - 1 << start
    # The carry bits (the encoder may set them; the disassembler does not
    # model them) must be disjoint from the modelled bits and in the value
    assert c.carry & c.exact.mask == 0, (c, c.carry)
    assert c.carry & owned == 0, (c, c.carry)
    assert c.carry <= (1 << 64) - 1, c.carry
    return f


# Collect the layouts up front (asserts run for every case, and the
# name tables are complete before the header is printed)
for k, enc in isa.encodings.items():
    for c in enc.cases:
        case_layout(c)

# The modifier tables (the count modifiers), deduplicated and
# concatenated; a case with a table modifier references its slice
mod_table = []
mod_table_index = {}

def table_ref(c):
    """(off, len, mod_index) for the case's table modifier, or None"""
    refs = [(i, m.table) for i, m in enumerate(c.modifiers)
            if m.table is not None]
    assert len(refs) <= 1, (c, refs)
    if not refs:
        return None
    i, table = refs[0]
    key = tuple(table)
    if key not in mod_table_index:
        mod_table_index[key] = (len(mod_table), len(key))
        mod_table.extend(key)
    return mod_table_index[key][0], mod_table_index[key][1], i

header = f'''
/* Generated by gen-pack.py from {sys.argv[1]}. Do not edit.

AGX3, the G15 (Apple M3) shader ISA. An instruction is a set of fixed
bits (exact) plus operand fields (dests, srcs, immediates and modifiers,
in XML order), each a list of bit ranges with constant padding in
between. The values the packer deposits are in the same space the
disassembler's extract produces, padding bits included.

A src/dest operand value is built from one of its encoding's cases: the
case's exact bits, plus the register/immediate (the kind) at its bit
positions, plus the flag modifiers. agx3_case_value assembles it;
agx3_pack deposits it into the instruction.
*/

#ifndef AGX3_PACK_H
#define AGX3_PACK_H

#include <assert.h>
#include <stdint.h>

#define AGX3_MAX_BYTES {B}
#define AGX3_MAX_CASE_FIELDS {MAX_FIELDS}

/* One non-constant operand field: the instruction bits at
 * [start, start + size) hold the value-space bits at [vpos, vpos + size).
 * Constant padding ranges are not fields: the disassembler's extract
 * counts them in the value space (as their constant value, see
 * agx3_pack_spec.op_pad), and the packer ignores them (they have no
 * instruction bits). */
struct agx3_field {{
   uint16_t start;
   uint8_t size;
   uint8_t vpos;
}};

/* One value-space field of a case: bits [start, start + size) of the
 * operand value. */
struct agx3_case_field {{
   uint8_t start;
   uint8_t size;
}};

/* A case of an operand's encoding, the disassembler's matching in the
 * same order. f[0]..f[nkind - 1] is the kind (register/immediate),
 * followed by the modifiers in XML order; the kind and modifier names
 * index agx3_kind_names / agx3_mod_names.
 *
 * carry_mask is the rest of the case's ownership: bits the encoder may
 * set that the disassembler does not model (per-instruction desc-
 * dependent flags, or don't-cares where the operand is a null register).
 * They pass through agx3_case_value / agx3_pack untouched. Proven by the
 * applegpu-checked corpus (cases-agx3.txt).
 *
 * shift is the case's kind shift (the register model, AGX3.xml's
 * header): base = dense << shift. table_off/table_len slice
 * agx3_mod_table for the case's table modifier (the count: the table
 * maps the field to the component count, not one to one, so the
 * encoder sets the field itself; table_mod is the modifier's slot,
 * 0xff when the case has no table). */
struct agx3_case {{
   uint64_t exact;
   uint64_t exact_mask;
   uint64_t carry_mask;
   uint8_t nfields;
   uint8_t nkind; /* 0: the case has no kind (no register or immediate) */
   uint8_t kind;
   struct agx3_case_field f[AGX3_MAX_CASE_FIELDS];
   uint8_t mod[AGX3_MAX_CASE_FIELDS];

   /* The selection key: size (bits 0:1), bfloat (bit 2), the kind name
    * index (bits 3:8, {SKEY_KIND_NONE:#x} = none), and a mask over the
    * modifiers the case offers (bits 9:24, agx3_mod_names). See
    * agx3_case_select. */
   uint32_t skey; /* the selection key (agx3_case_select) */
   uint8_t shift; /* the kind's shift (base = dense << shift) */
   uint8_t table_off; /* the modifier table, in agx3_mod_table */
   uint8_t table_len; /* 0: the case has no table modifier */
   uint8_t table_mod; /* the table modifier's slot (0xff = none) */
}};

struct agx3_pack_spec {{
   const char *name;

   /* Multi-length instructions carry a length field that is not one of
    * the operands; the caller picks the length, and agx3_pack sets the
    * field to the index of that length in lengths. */
   uint8_t has_length;
   uint8_t length_bit;
   uint8_t nlengths;
   uint8_t lengths[3];

   uint16_t nops;
   const uint16_t *op_start;
   const struct agx3_field *fields;
   const uint64_t *op_pad;

   /* The operand cases: operand i's start in the flat case array and
    * its count (0 for an operand without an encoding, e.g. modifiers
    * and immediates). op_enc[i] is the encoding's index in
    * agx3_enc_names ({SKEY_KIND_NONE:#x} when the operand has no cases). */
   const struct agx3_case *cases;
   const uint16_t *case_start;
   const uint8_t *ncases;
   const uint8_t *op_enc;

   uint64_t exact[{W}];
   uint64_t value[{W}];
   uint64_t mask[{W}];
}};

static const char *const agx3_kind_names[] = {{
{", ".join(f'"{n}"' for n in kind_names)}
}};

static const char *const agx3_mod_names[] = {{
{", ".join(f'"{n}"' for n in mod_names)}
}};

static const char *const agx3_enc_names[] = {{
{", ".join(f'"{n}"' for n in enc_order)}
}};

/* 1: the case's structural key (skey) selects it on its own; 0: the
 * selector must also match the exact pattern (the key alone is
 * ambiguous in this encoding). */
static const uint8_t agx3_enc_struct_key[] = {{
{", ".join(str(int(enc_struct_key[n])) for n in enc_order)}
}};

/* The modifier tables (the count modifiers), concatenated */
static const uint8_t agx3_mod_table[] = {{
{", ".join(str(x) for x in mod_table) if mod_table else "0"}
}};

enum {{ AGX3_NUM_ENCODINGS = {len(enc_order)} }};

/* The case's selection key (agx3_case_select): the size (bits 0:1),
 * the bfloat property (bit 2), the kind name index (bits 3:8,
 * {SKEY_KIND_NONE:#x} = none) and a mask over the modifiers the case
 * offers (bits 9:24, agx3_mod_names). The modmask is the wanted case's
 * full offering, not the flags being set (agx3_reg.mods). */
#define AGX3_SKEY_SIZE_MASK 0x3u
#define AGX3_SKEY_BFLOAT (1u << 2)
#define AGX3_SKEY_KIND_SHIFT {SKEY_KIND_SHIFT}
#define AGX3_SKEY_KIND_NONE {SKEY_KIND_NONE}u
#define AGX3_SKEY_MOD_SHIFT {SKEY_MOD_SHIFT}

static inline uint32_t
agx3_make_skey(uint32_t kind, uint32_t size, uint32_t bfloat, uint32_t modmask)
{{
   return (size & AGX3_SKEY_SIZE_MASK)
      | (bfloat ? AGX3_SKEY_BFLOAT : 0)
      | (kind << AGX3_SKEY_KIND_SHIFT)
      | (modmask << AGX3_SKEY_MOD_SHIFT);
}}

/* Deposit one value-space field of a case */
static inline uint64_t
agx3_case_field_deposit(uint64_t value, const struct agx3_case_field *f)
{{
   uint64_t m = f->size == 64 ? ~0ull : ((1ull << f->size) - 1);
   return (value & m) << f->start;
}}

/* Assemble an operand value from a case: the case's exact bits, the
 * kind spread over its field(s), the modifiers (modvals in the case's
 * modifier order), and the carry bits (masked by c->carry_mask). */
static inline uint64_t
agx3_case_value(const struct agx3_case *c, uint64_t kind,
                const uint64_t *modvals, uint64_t carry)
{{
   uint64_t v = c->exact;

   for (unsigned f = 0; f < c->nkind; f++) {{
      v |= agx3_case_field_deposit(kind, &c->f[f]);
      kind >>= c->f[f].size;
   }}

   for (unsigned f = c->nkind; f < c->nfields; f++)
      v |= agx3_case_field_deposit(modvals[f - c->nkind], &c->f[f]);

   v |= carry & c->carry_mask;

   return v;
}}

/* A register operand the compiler wants to encode (the register model
 * is in AGX3.xml's header): index is the N of pN{{lh}} (p = r, u, st),
 * half the 16-bit l/h, mods a mask over the one-bit flag modifiers to
 * set (agx3_mod_names; the count is not one), count_field the field of
 * a table-count case (agx3_reg_count gives its component count; the
 * table is not one to one: e.g. memreg's 2 and 4 both count 2), and
 * carry the carry bits to set (masked by the case). */
struct agx3_reg {{
   uint32_t index;
   uint8_t half;
   uint8_t count_field;
   uint64_t mods;
   uint64_t carry;
}};

/* The component count of a case for a count field (1 for a case
 * without a table modifier) */
static inline unsigned
agx3_reg_count(const struct agx3_case *c, unsigned count_field)
{{
   if (c->table_mod == 0xff)
      return 1;
   assert(count_field < c->table_len);
   return agx3_mod_table[c->table_off + count_field];
}}

/* The dense register field for a case: 2*i+h for 16-bit, i for 32-bit,
 * i/2 for the 64-bit with shift 2 (the du of membase). */
static inline uint64_t
agx3_reg_dense(const struct agx3_case *c, const struct agx3_reg *r)
{{
   if ((c->skey & AGX3_SKEY_SIZE_MASK) == 0)
      return 2 * r->index + r->half;
   if (c->shift == 2)
      return r->index >> 1;
   return r->index;
}}

/* The operand value in the value space for a case and a register */
static inline uint64_t
agx3_reg_value(const struct agx3_case *c, const struct agx3_reg *r)
{{
   uint64_t modvals[AGX3_MAX_CASE_FIELDS] = {{ 0 }};

   for (unsigned f = c->nkind; f < c->nfields; f++) {{
      unsigned k = f - c->nkind;
      if (c->table_mod == k)
         modvals[k] = r->count_field;
      else
         modvals[k] = (r->mods >> c->mod[f]) & 1;
   }}

   return agx3_case_value(c, agx3_reg_dense(c, r), modvals, r->carry);
}}

/* The first case matching a value, the disassembler's order */
static inline const struct agx3_case *
agx3_case_match(const struct agx3_case *cases, unsigned n, uint64_t value)
{{
   for (unsigned i = 0; i < n; i++)
      if ((value & cases[i].exact_mask) == cases[i].exact)
         return &cases[i];

   return NULL;
}}

/* The case for an operand the compiler wants to encode: its register
 * class (the kind, the size, bfloat) and the flags it carries (a mask
 * over agx3_mod_names). For an encoding whose structural key is
 * ambiguous (agx3_enc_struct_key[enc] == 0), the exact pattern
 * disambiguates: pass the case's exact value and mask (0/0 is
 * accepted only when the key is unique). Exactly one case matches, or
 * NULL if the operand is not expressible. */
static inline const struct agx3_case *
agx3_case_select(unsigned enc, const struct agx3_case *cases, unsigned n,
                 uint32_t skey, uint64_t exact, uint64_t exact_mask)
{{
   for (unsigned i = 0; i < n; i++) {{
      if (cases[i].skey != skey)
         continue;
      if (agx3_enc_struct_key[enc])
         return &cases[i];
      if (cases[i].exact == exact && cases[i].exact_mask == exact_mask)
         return &cases[i];
   }}

   return NULL;
}}

/* Unpack the operand values, in the disassembler's value space (the
 * constant padding bits sit at their positions, as their constant
 * value). */
static inline void
agx3_unpack(const uint8_t *in, const struct agx3_pack_spec *spec,
            uint8_t length, uint64_t *ops)
{{
   uint64_t w[{W}] = {{ 0 }};

   for (unsigned i = 0; i < length; i++)
      w[i / 8] |= (uint64_t)in[i] << (8 * (i % 8));

   for (unsigned i = 0; i < spec->nops; i++) {{
      uint64_t v = spec->op_pad ? spec->op_pad[i] : 0;

      for (unsigned f = spec->op_start[i]; f < spec->op_start[i + 1]; f++) {{
         const struct agx3_field *fd = &spec->fields[f];
         uint64_t m = fd->size == 64 ? ~0ull : ((1ull << fd->size) - 1);

         v |= ((w[fd->start / 64] >> (fd->start % 64)) & m) << fd->vpos;
      }}

      ops[i] = v;
   }}
}}

static inline void
agx3_pack(uint8_t *out, const struct agx3_pack_spec *spec, uint8_t length,
          const uint64_t *ops)
{{
   uint64_t w[{W}] = {{
{", ".join(f"spec->value[{i}]" for i in range(W))}
   }};

   if (spec->has_length) {{
      for (unsigned i = 0; i < spec->nlengths; i++)
         if (spec->lengths[i] == length) {{
            w[spec->length_bit / 64] |= (uint64_t)i << (spec->length_bit % 64);
            break;
         }}
   }}

   /* The constant padding bits are already in the instruction's fixed
    * value. */
   for (unsigned i = 0; i < spec->nops; i++) {{
      uint64_t v = ops[i];

      for (unsigned f = spec->op_start[i]; f < spec->op_start[i + 1]; f++) {{
         const struct agx3_field *fd = &spec->fields[f];
         uint64_t m = fd->size == 64 ? ~0ull : ((1ull << fd->size) - 1);

         w[fd->start / 64] |= ((v >> fd->vpos) & m) << (fd->start % 64);
      }}
   }}

   for (unsigned i = 0; i < length; i++)
      out[i] = (w[i / 8] >> (8 * (i % 8))) & 0xff;
}}

enum agx3_instr {{
{",\n".join(f"   AGX3_{safe_name(k).upper()}" for k in isa.instrs)},
   AGX3_NUM_INSTRS
}};

'''

decls = []
entries = []
for k, instr in isa.instrs.items():
    name = safe_name(k)
    ops = instr.ordered
    fields = []
    op_start = [0]
    op_pad = []
    for cat, op in ops:
        vpos = 0
        pad = 0
        for r in op.bits.ranges:
            if r.const is None:
                # Split fields that cross a 64-bit word: the C packer works
                # on one word per field
                start, size = r.start, r.size
                while size > 0:
                    room = 64 - (start % 64)
                    fields.append((start, min(size, room), vpos))
                    start += room
                    vpos += min(size, room)
                    size -= room
            else:
                pad |= r.const << vpos
                vpos += r.size
        op_start.append(len(fields))
        op_pad.append(pad)

    # The operand cases
    cases = []
    case_start = [0]
    ncases = []
    op_enc = []
    for cat, op in ops:
        if cat not in ('src', 'dest') or op.kind not in isa.encodings:
            ncases.append(0)
            case_start.append(len(cases))
            op_enc.append(SKEY_KIND_NONE)
            continue
        for c in isa.encodings[op.kind].cases:
            layout = case_layout(c)
            has_kind = c.kind is not None
            cf = []
            mod = []
            for is_kind, ni, start, size in layout:
                cf.append(f'{{{start}, {size}}}')
                mod.append('0xff' if is_kind else str(ni))
            nkind = sum(1 for x in layout if x[0])
            ref = table_ref(c)
            shift = c.kind.shift if has_kind else 0
            if ref:
                toff, tlen, tmod = f'{ref[0]}', f'{ref[1]}', f'{ref[2]}'
            else:
                toff, tlen, tmod = '0', '0', '0xff'
            cases.append(
                f'   {{ {hex(c.exact.value)}, {hex(c.exact.mask)}, '
                f'{hex(c.carry)}, '
                f'{len(layout)}, {nkind}, '
                f'{kind_name(c.kind.name) if has_kind else 0}, '
                f'{{ {", ".join(cf)} }}, '
                f'{{ {", ".join(mod)} }}, '
                f'{case_key(c):#x}, {shift}, {toff}, {tlen}, {tmod} }},')
        case_start.append(len(cases))
        ncases.append(len(isa.encodings[op.kind].cases))
        op_enc.append(enc_index(op.kind))

    assert len(op_enc) == len(ops), (k, len(op_enc), len(ops))

    if fields:
        decls.append(
            f'static const struct agx3_field {name}_fields[] = {{\n'
            + ''.join(f'   {{{s}, {z}, {v}}},\n' for s, z, v in fields)
            + '};\n')
    decls.append(
        f'static const uint64_t {name}_op_pad[] = '
        f'{{{", ".join(hex(x) for x in op_pad)}}};\n')
    decls.append(
        f'static const uint16_t {name}_op_start[] = '
        f'{{{", ".join(str(x) for x in op_start)}}};\n')
    decls.append(
        f'static const uint8_t {name}_ncases[] = '
        f'{{{", ".join(str(x) for x in ncases)}}};\n')
    decls.append(
        f'static const uint8_t {name}_op_enc[] = '
        f'{{{", ".join(str(x) for x in op_enc)}}};\n')
    decls.append(
        f'static const uint16_t {name}_case_start[] = '
        f'{{{", ".join(str(x) for x in case_start)}}};\n')
    if cases:
        decls.append(
            f'static const struct agx3_case {name}_cases[] = {{\n'
            + '\n'.join(cases)
            + '\n};\n')

    length = instr.length
    entry = (
        f'   [AGX3_{name.upper()}] = {{\n'
        f'      .name = "{instr.display}",\n'
        f'      .has_length = {int(instr.length_spec is not None)},\n'
        f'      .length_bit = {instr.length_bit},\n'
        f'      .nlengths = {len(length)},\n'
        f'      .lengths = {{{", ".join(str(x) for x in length)}}},\n'
        f'      .nops = {len(ops)},\n'
        f'      .op_start = {name}_op_start,\n'
        + (f'      .fields = {name}_fields,\n' if fields
           else '      .fields = NULL,\n')
        + f'      .op_pad = {name}_op_pad,\n'
        + (f'      .cases = {name}_cases,\n' if cases
           else '      .cases = NULL,\n')
        + f'      .case_start = {name}_case_start,\n'
        + f'      .ncases = {name}_ncases,\n'
        + f'      .op_enc = {name}_op_enc,\n'
        + f'      .exact = {chunk(instr.exact.mask)},\n'
        + f'      .value = {chunk(instr.exact.value)},\n'
        + f'      .mask = {chunk(instr.mask())},\n'
        + '   },\n')
    entries.append(entry)

# The instructions whose fixed bits allow each first byte, in table order:
# agx3_instr_length tries only these
assert len(isa.instrs) <= 256
byte0 = [[i for i, instr in enumerate(isa.instrs.values())
          if b & instr.exact.mask & 0xff == instr.exact.value & 0xff]
         for b in range(256)]
byte0_start = [0]
for ins in byte0:
    byte0_start.append(byte0_start[-1] + len(ins))

footer = f'''
static const struct agx3_pack_spec agx3_specs[AGX3_NUM_INSTRS] = {{
{"".join(entries)}}};

static const uint16_t agx3_byte0_start[257] = {{
   {", ".join(str(x) for x in byte0_start)}
}};

static const uint8_t agx3_byte0_instrs[] = {{
   {", ".join(str(i) for ins in byte0 for i in ins)}
}};

static inline const struct agx3_pack_spec *
agx3_spec(unsigned instr)
{{
   return &agx3_specs[instr];
}}

#endif /* AGX3_PACK_H */
'''

sys.stdout.write(header)
sys.stdout.write('\n'.join(decls))
sys.stdout.write(footer)
