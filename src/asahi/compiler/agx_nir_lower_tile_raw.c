/*
 * SPDX-License-Identifier: MIT
 */

#include "asahi/layout/layout.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "agx_nir.h"

/*
 * G15's tile loads and stores move raw data: there is no format conversion in
 * the instruction (Apple's compiler converts with fltnorm and the like before
 * lstcol), only the size of the access: 8, 16 or 32 bits per component, the
 * bytes of an 8-bit access packed two per 16-bit register. G13's load_local_
 * pixel_agx and store_local_pixel_agx name a format and the hardware converts.
 *
 * This pass rewrites them for G15 into accesses of raw I8 (one byte in a
 * 16-bit register), I16 and I32 data, with the conversion in NIR. The tile
 * layout is unchanged: the same bytes land where G13's conversion put them.
 */

static nir_def *
to_f32(nir_builder *b, nir_def *x)
{
   return x->bit_size == 32 ? x : nir_f2f32(b, x);
}

/* A 32-bit float vector of exactly n channels, a missing alpha being 1.0 */
static nir_def *
resize_f32(nir_builder *b, nir_def *x, unsigned n)
{
   nir_def *f = to_f32(b, x);
   nir_def *c[4];

   for (unsigned i = 0; i < n; i++) {
      c[i] = i < f->num_components ? nir_channel(b, f, i)
                                   : nir_imm_float(b, i == 3 ? 1.0 : 0.0);
   }

   return nir_vec(b, c, n);
}

static nir_def *
channels(nir_builder *b, nir_def *x, unsigned first, unsigned count)
{
   return nir_channels(b, x, BITFIELD_RANGE(first, count));
}

static void
store_raw(nir_builder *b, nir_intrinsic_instr *intr, nir_def *value,
          unsigned base, unsigned mask, enum ail_isa_format format)
{
   nir_store_local_pixel_agx(b, value, intr->src[1].ssa, intr->src[2].ssa,
                             .base = base, .write_mask = mask,
                             .format = (enum pipe_format)format,
                             .explicit_coord = nir_intrinsic_explicit_coord(intr));
}

static nir_def *
load_raw(nir_builder *b, nir_intrinsic_instr *intr, unsigned comps,
         unsigned bit_size, unsigned base, enum ail_isa_format format)
{
   return nir_load_local_pixel_agx(b, comps, bit_size, intr->src[0].ssa,
                                   .base = base,
                                   .format = (enum pipe_format)format);
}

/* Stores the low `nbytes` bytes of a 32-bit word at `base` */
static void
store_bytes(nir_builder *b, nir_intrinsic_instr *intr, nir_def *word,
            unsigned nbytes, unsigned base)
{
   switch (nbytes) {
   case 1:
      store_raw(b, intr, nir_u2u16(b, nir_iand_imm(b, word, 0xff)), base, 0x1,
                AIL_ISA_FORMAT_I8);
      break;
   case 2:
      store_raw(b, intr, nir_u2u16(b, nir_iand_imm(b, word, 0xffff)), base,
                0x1, AIL_ISA_FORMAT_I16);
      break;
   case 3:
      store_raw(b, intr, nir_u2u16(b, nir_iand_imm(b, word, 0xffff)), base,
                0x1, AIL_ISA_FORMAT_I16);
      store_raw(b, intr, nir_u2u16(b, nir_ubitfield_extract_imm(b, word, 16, 8)),
                base + 2, 0x1, AIL_ISA_FORMAT_I8);
      break;
   case 4:
      store_raw(b, intr, word, base, 0x1, AIL_ISA_FORMAT_I32);
      break;
   default:
      UNREACHABLE("byte count");
   }
}

/* Loads `nbytes` bytes at `base` into the low bytes of a 32-bit word */
static nir_def *
load_bytes(nir_builder *b, nir_intrinsic_instr *intr, unsigned nbytes,
           unsigned base)
{
   switch (nbytes) {
   case 1:
      return nir_u2u32(b, load_raw(b, intr, 1, 16, base, AIL_ISA_FORMAT_I8));
   case 2:
      return nir_u2u32(b, load_raw(b, intr, 1, 16, base, AIL_ISA_FORMAT_I16));
   case 3: {
      nir_def *lo = nir_u2u32(b, load_raw(b, intr, 1, 16, base, AIL_ISA_FORMAT_I16));
      nir_def *hi =
         nir_u2u32(b, load_raw(b, intr, 1, 16, base + 2, AIL_ISA_FORMAT_I8));
      return nir_ior(b, lo, nir_ishl_imm(b, hi, 16));
   }
   case 4:
      return load_raw(b, intr, 1, 32, base, AIL_ISA_FORMAT_I32);
   default:
      UNREACHABLE("byte count");
   }
}

/* Bytes of an 8-bit-per-component store: n 32-bit values 0..255 (the byte of
 * each channel). Stores the channels of `mask` at their offsets: the leading
 * run as one packed access, the rest a byte at a time. */
static void
store_byte_channels(nir_builder *b, nir_intrinsic_instr *intr, nir_def *bytes,
                    unsigned mask, unsigned base)
{
   unsigned run = 0;
   while (run < 4 && (mask & BITFIELD_BIT(run)))
      run++;

   bytes = nir_iand_imm(b, bytes, 0xff);

   if (run > 0) {
      nir_def *word = nir_channel(b, bytes, 0);
      for (unsigned i = 1; i < run; i++) {
         word = nir_ior(b, word, nir_ishl_imm(b, nir_channel(b, bytes, i), 8 * i));
      }
      store_bytes(b, intr, word, run, base);
   }

   u_foreach_bit(i, mask & ~BITFIELD_MASK(run)) {
      store_bytes(b, intr, nir_channel(b, bytes, i), 1, base + i);
   }
}

static nir_def *
unpack_bytes(nir_builder *b, nir_def *word, unsigned n)
{
   nir_def *c[4];
   for (unsigned i = 0; i < n; i++)
      c[i] = nir_ubitfield_extract_imm(b, word, 8 * i, 8);

   return nir_vec(b, c, n);
}

static const unsigned bits8[4] = {8, 8, 8, 8};
static const unsigned bits16[4] = {16, 16, 16, 16};
static const unsigned bits1010102[4] = {10, 10, 10, 2};

static bool
lower_store(nir_builder *b, nir_intrinsic_instr *intr)
{
   enum ail_isa_format format = (enum ail_isa_format)nir_intrinsic_format(intr);
   unsigned base = nir_intrinsic_base(intr);
   unsigned mask = nir_intrinsic_write_mask(intr);
   nir_def *value = intr->src[0].ssa;
   unsigned n = value->num_components;

   switch (format) {
   case AIL_ISA_FORMAT_I32:
      return false;

   case AIL_ISA_FORMAT_I16:
      if (value->bit_size == 16)
         return false;
      store_raw(b, intr, nir_u2u16(b, value), base, mask, format);
      return true;

   case AIL_ISA_FORMAT_F16:
      if (value->bit_size == 16)
         return false;
      store_raw(b, intr, nir_f2f16(b, value), base, mask, format);
      return true;

   case AIL_ISA_FORMAT_I8:
      if (n == 1 && mask == 1 && value->bit_size == 16)
         return false;
      store_byte_channels(b, intr, nir_u2u32(b, value), mask, base);
      return true;

   case AIL_ISA_FORMAT_U8NORM:
      store_byte_channels(b, intr,
                          nir_format_float_to_unorm(b, to_f32(b, value), bits8),
                          mask, base);
      return true;

   case AIL_ISA_FORMAT_S8NORM: {
      nir_def *s = nir_format_float_to_snorm(b, to_f32(b, value), bits8);
      store_byte_channels(b, intr, nir_iand_imm(b, s, 0xff), mask, base);
      return true;
   }

   case AIL_ISA_FORMAT_U16NORM:
      store_raw(b, intr,
                nir_u2u16(b, nir_format_float_to_unorm(b, to_f32(b, value), bits16)),
                base, mask, AIL_ISA_FORMAT_I16);
      return true;

   case AIL_ISA_FORMAT_S16NORM:
      store_raw(b, intr,
                nir_i2i16(b, nir_format_float_to_snorm(b, to_f32(b, value), bits16)),
                base, mask, AIL_ISA_FORMAT_I16);
      return true;

   case AIL_ISA_FORMAT_SRGBA8: {
      nir_def *f = to_f32(b, value);
      if (n >= 3) {
         nir_def *rgb = nir_format_linear_to_srgb(b, channels(b, f, 0, 3));
         nir_def *c[4] = {nir_channel(b, rgb, 0), nir_channel(b, rgb, 1),
                          nir_channel(b, rgb, 2),
                          n == 4 ? nir_channel(b, f, 3) : nir_imm_float(b, 1.0)};
         f = nir_vec(b, c, 4);
      } else {
         f = nir_format_linear_to_srgb(b, f);
      }
      store_byte_channels(b, intr, nir_format_float_to_unorm(b, f, bits8),
                          mask, base);
      return true;
   }

   case AIL_ISA_FORMAT_RGB10A2: {
      nir_def *u =
         nir_format_float_to_unorm(b, resize_f32(b, value, 4), bits1010102);
      store_bytes(b, intr, nir_format_pack_uint(b, u, bits1010102, 4), 4, base);
      return true;
   }

   case AIL_ISA_FORMAT_RG11B10F:
      store_bytes(b, intr,
                  nir_format_pack_11f11f10f(b, resize_f32(b, value, 3)),
                  4, base);
      return true;

   case AIL_ISA_FORMAT_RGB9E5:
      store_bytes(b, intr,
                  nir_format_pack_r9g9b9e5(b, resize_f32(b, value, 3)),
                  4, base);
      return true;

   default:
      UNREACHABLE("tile format");
   }
}

static bool
lower_load(nir_builder *b, nir_intrinsic_instr *intr)
{
   enum ail_isa_format format = (enum ail_isa_format)nir_intrinsic_format(intr);
   unsigned base = nir_intrinsic_base(intr);
   unsigned k = intr->def.num_components;
   unsigned bs = intr->def.bit_size;
   nir_def *res;

   switch (format) {
   case AIL_ISA_FORMAT_I32:
      return false;

   case AIL_ISA_FORMAT_I16:
   case AIL_ISA_FORMAT_F16:
      if (bs == 16)
         return false;
      res = load_raw(b, intr, k, 16, base, format);
      res = format == AIL_ISA_FORMAT_F16 ? nir_f2f32(b, res) : nir_u2u32(b, res);
      break;

   case AIL_ISA_FORMAT_I8:
      if (k == 1 && bs == 16)
         return false;
      res = unpack_bytes(b, load_bytes(b, intr, k, base), k);
      if (bs == 16)
         res = nir_u2u16(b, res);
      break;

   case AIL_ISA_FORMAT_U8NORM:
      res = nir_format_unorm_to_float(b, unpack_bytes(b, load_bytes(b, intr, k, base), k),
                                      bits8);
      break;

   case AIL_ISA_FORMAT_S8NORM: {
      nir_def *s = nir_format_sign_extend_ivec(
         b, unpack_bytes(b, load_bytes(b, intr, k, base), k), bits8);
      res = nir_format_snorm_to_float(b, s, bits8);
      break;
   }

   case AIL_ISA_FORMAT_U16NORM:
      res = nir_format_unorm_to_float(b, nir_u2u32(b, load_raw(b, intr, k, 16, base, AIL_ISA_FORMAT_I16)),
                                      bits16);
      break;

   case AIL_ISA_FORMAT_S16NORM: {
      nir_def *s = nir_i2i32(b, load_raw(b, intr, k, 16, base, AIL_ISA_FORMAT_I16));
      res = nir_format_snorm_to_float(b, s, bits16);
      break;
   }

   case AIL_ISA_FORMAT_SRGBA8: {
      nir_def *f = nir_format_unorm_to_float(
         b, unpack_bytes(b, load_bytes(b, intr, 4, base), 4), bits8);
      nir_def *rgb = nir_format_srgb_to_linear(b, channels(b, f, 0, 3));
      nir_def *c[4] = {nir_channel(b, rgb, 0), nir_channel(b, rgb, 1),
                       nir_channel(b, rgb, 2), nir_channel(b, f, 3)};
      res = nir_trim_vector(b, nir_vec(b, c, 4), k);
      break;
   }

   case AIL_ISA_FORMAT_RGB10A2: {
      nir_def *u = nir_format_unpack_uint(b, load_bytes(b, intr, 4, base),
                                          bits1010102, 4);
      res = nir_trim_vector(b, nir_format_unorm_to_float(b, u, bits1010102), k);
      break;
   }

   case AIL_ISA_FORMAT_RG11B10F:
      res = nir_trim_vector(
         b, nir_format_unpack_11f11f10f(b, load_bytes(b, intr, 4, base)), k);
      break;

   case AIL_ISA_FORMAT_RGB9E5:
      res = nir_trim_vector(
         b, nir_format_unpack_r9g9b9e5(b, load_bytes(b, intr, 4, base)), k);
      break;

   default:
      UNREACHABLE("tile format");
   }

   /* The conversions above are in 32 bits */
   if (bs == 16 && res->bit_size == 32) {
      bool is_float = format != AIL_ISA_FORMAT_I8 && format != AIL_ISA_FORMAT_I16;
      res = is_float ? nir_f2f16(b, res) : nir_u2u16(b, res);
   }

   nir_def_rewrite_uses(&intr->def, res);
   return true;
}

static bool
lower_intr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_local_pixel_agx &&
       intr->intrinsic != nir_intrinsic_load_local_pixel_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   bool progress = intr->intrinsic == nir_intrinsic_store_local_pixel_agx
                      ? lower_store(b, intr)
                      : lower_load(b, intr);

   if (progress)
      nir_instr_remove(&intr->instr);

   return progress;
}

bool
agx_nir_lower_tile_raw(nir_shader *shader)
{
   return nir_shader_intrinsics_pass(shader, lower_intr,
                                     nir_metadata_control_flow, NULL);
}
