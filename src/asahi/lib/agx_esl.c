/*
 * Copyright 2026 hitori1403
 * SPDX-License-Identifier: MIT
 */

#include "agx_esl.h"
#include <stdio.h>
#include "compiler/agx_compile.h"
#include "util/bitset.h"
#include "util/u_debug.h"
#include "util/u_math.h"
#include "agx_pack.h"

/* The imageblock of a render stage: the tile's shape, the samples and their
 * stride, from the USC SHARED word */
struct esl_tile {
   unsigned width_log2, height_log2, samples_log2, stride_8B;
};

/* The entry program's registers: r0:r1 the buffer address, r2 up the loads */
#define ESL_BASE 0
#define ESL_DATA 2
#define ESL_SLOT 0

/* The highest register it writes: an entry program that wrote r95 hung the
 * GPU, one that stopped at r94 ran (compute-program.md, Registers) */
#define ESL_LAST_REG 93

#define ESL_MAX_RANGES 32

/* The entry program cannot read back the uniforms it writes: a preamble's
 * load from a pushed pointer read a stale one in about half the SIMD groups of
 * a fragment program, the same pointer loaded from memory into a register
 * worked (boots #318, #319). So each pushed word N within reach is loaded into
 * register N + AGX_G15_PUSH_REG, moved to uN from there, and left there for
 * the preamble, which reads it from the register (agx_compile.c). */
#define ESL_PUSH_REGS AGX_G15_PUSH_WORDS

struct esl_range {
   unsigned start_halfs, size_halfs;
   uint64_t buffer;
};

static void
esl_u64(struct util_dynarray *out, uint64_t v)
{
   util_dynarray_append(out, v);
}

static void
esl_u16(struct util_dynarray *out, uint16_t v)
{
   util_dynarray_append(out, v);
}

/* Load a range's 16-bit uniforms from its buffer into the uniforms from
 * start_halfs on, the words out of reach of the push registers: through r2
 * up, free until the push registers are loaded, a batch's loads out before
 * one wait. The uniform file is filled in 32-bit words; an odd count rounds
 * up (the extra half belongs to the buffer's padding). */
static void
esl_uniforms(struct util_dynarray *out, const struct esl_range *r)
{
   assert((r->start_halfs & 1) == 0 && "uniform ranges start 32-bit aligned");
   unsigned first = r->start_halfs / 2;
   unsigned end = first + DIV_ROUND_UP(r->size_halfs, 2);
   unsigned from = MAX2(first, ESL_PUSH_REGS);

   if (from >= end)
      return;

   uint64_t buffer = r->buffer + 4 * (from - first);
   agx3_emit_mov_imm(out, ESL_BASE, buffer);
   agx3_emit_mov_imm(out, ESL_BASE + 1, buffer >> 32);

   for (unsigned batch = from; batch < end; batch += ESL_PUSH_REGS) {
      unsigned to = MIN2(end, batch + ESL_PUSH_REGS);

      for (unsigned u = batch; u < to; u += 4) {
         agx3_emit_load(out, ESL_SLOT, ESL_DATA + (u - batch), ESL_BASE,
                        (u - from) / 4, MIN2(4, to - u));
      }

      agx3_emit_wait(out, ESL_SLOT);

      for (unsigned u = batch; u < to; u++)
         agx3_emit_uniform_store(out, u, ESL_DATA + (u - batch));
   }
}

/* The words within reach, from every range, each into its push register: all
 * the loads go out before one wait. Each range's buffer address takes a
 * register pair of its own, r0:r1 or a pair no pushed word lands in, so that
 * no load's address is overwritten while it is in flight; a range left
 * without one waits for the loads before it and takes r0:r1. */
static void
esl_uniforms_reach(struct util_dynarray *out, const struct esl_range *ranges,
                   unsigned nr_ranges)
{
   BITSET_DECLARE(used, ESL_LAST_REG + 1) = {0};
   unsigned from[ESL_MAX_RANGES], to[ESL_MAX_RANGES];
   bool any = false;

   for (unsigned i = 0; i < nr_ranges; i++) {
      assert((ranges[i].start_halfs & 1) == 0 &&
             "uniform ranges start 32-bit aligned");
      from[i] = ranges[i].start_halfs / 2;
      to[i] =
         MIN2(from[i] + DIV_ROUND_UP(ranges[i].size_halfs, 2), ESL_PUSH_REGS);

      for (unsigned u = from[i]; u < to[i]; u++)
         BITSET_SET(used, u + AGX_G15_PUSH_REG);
   }

   for (unsigned i = 0, next = ESL_BASE; i < nr_ranges; i++) {
      if (from[i] >= to[i])
         continue;

      unsigned base = ESL_BASE;
      while (next + 1 <= ESL_LAST_REG &&
             (BITSET_TEST(used, next) || BITSET_TEST(used, next + 1)))
         next += 2;

      if (next + 1 <= ESL_LAST_REG) {
         base = next;
         next += 2;
      } else if (any) {
         agx3_emit_wait(out, ESL_SLOT);
      }

      agx3_emit_mov_imm(out, base, ranges[i].buffer);
      agx3_emit_mov_imm(out, base + 1, ranges[i].buffer >> 32);

      for (unsigned u = from[i]; u < to[i]; u += 4) {
         agx3_emit_load(out, ESL_SLOT, u + AGX_G15_PUSH_REG, base,
                        (u - from[i]) / 4, MIN2(4, to[i] - u));
      }

      any = true;
   }

   if (!any)
      return;

   agx3_emit_wait(out, ESL_SLOT);

   for (unsigned i = 0; i < nr_ranges; i++) {
      for (unsigned u = from[i]; u < to[i]; u++)
         agx3_emit_uniform_store(out, u, u + AGX_G15_PUSH_REG);
   }
}

/* specLM: threadgroup memory, 64 B units */
static void
esl_speclm(struct util_dynarray *out, unsigned bytes)
{
   uint64_t v = 0x412A0077;

   if (bytes) {
      unsigned units = DIV_ROUND_UP(bytes, 64);
      assert(units < (1 << 10));
      v |= BITFIELD64_BIT(39) | ((uint64_t)(units & 0xff) << 40) |
           ((uint64_t)(units >> 8) << 48);
   }

   esl_u64(out, v);
}

/* ldshdr: the shader, 64 B aligned; the size class from its size */
static void
esl_ldshdr(struct util_dynarray *out, uint64_t va, unsigned size)
{
   static const uint8_t size_class[4] = {2, 0, 1, 3};
   assert((va & 63) == 0 && va < BITFIELD64_BIT(48));

   esl_u64(out, AGX_ESL_LDSHDR | ((va >> 6) << 23));
   esl_u16(out, ((va >> 47) & 1) | (size_class[MIN2(size >> 7, 3)] << 1));
}

/* specTPR: the per-thread memory (metal-compute.md, from Apple's kernels and
 * render programs): the coordinate registers the shader writes at 41:31 (the
 * highest plus one, 8 B each), and the private (st) stack in words at 57:42
 * (its low 12 bits) and 73:58, both the same count: the shader's stack plus 4
 * words, as Apple's kernels have it. Without these the texture instructions
 * read no coordinates and get_stack is never written. */
static void
esl_spectpr(struct util_dynarray *out, unsigned stack_bytes, unsigned crs)
{
   unsigned words = stack_bytes ? DIV_ROUND_UP(stack_bytes, 4) + 4 : 0;
   assert(crs < (1 << 11) && words < (1 << 12));

   esl_u64(out, 0x2A00F7 | ((uint64_t)crs << 31) | ((uint64_t)words << 42) |
                   ((uint64_t)(words & 0x3F) << 58));
   esl_u16(out, words >> 6);
}

/* specLM of the render stages (demo15's programs: 0x4B432A0077), from the
 * imageblock the USC SHARED word describes. Apple's encoder (libLLVM 32023)
 * takes, besides the imageblock flag (bit 38), the tile's shape as a pair
 * (f19, f20) of which exactly the pairs with f19 + f20 <= 10 are legal (a
 * tile of at most 1024 threads: the log2 of its width and height, f19 the
 * width is a guess), the sample count as its log2 (f21, bits 23:24, 0-2) and
 * the sample stride in bytes (f22, bits 25:29 in 8-byte units). The shape
 * goes to bits 30:37 through this table (f19 rows, f20 columns, from the
 * encoder). Only demo15's configuration has run on the hardware: 32x32, one
 * sample, a stride of 8. */
static const uint8_t speclm_shape[11][11] = {
   {0x01, 0x47, 0x67, 0xa6, 0x26, 0x66, 0x87, 0x07, 0x46, 0x86, 0x06},
   {0x2a, 0x4a, 0x6a, 0x8b, 0x0b, 0x4b, 0x8a, 0x0a, 0x2b, 0x6b, 0},
   {0x2e, 0x4e, 0x6e, 0x8e, 0x0f, 0x4f, 0x6f, 0x0e, 0x2f, 0, 0},
   {0x34, 0x35, 0x55, 0x75, 0x15, 0x54, 0x74, 0x14, 0, 0, 0},
   {0x05, 0x24, 0x44, 0x64, 0x04, 0x25, 0x45, 0, 0, 0, 0},
   {0x0d, 0x2c, 0x4c, 0x4d, 0x0c, 0x2d, 0, 0, 0, 0, 0},
   {0x13, 0x32, 0x33, 0x52, 0x12, 0, 0, 0, 0, 0, 0},
   {0x02, 0x03, 0x22, 0x23, 0, 0, 0, 0, 0, 0, 0},
   {0x08, 0x09, 0x28, 0, 0, 0, 0, 0, 0, 0, 0},
   {0x10, 0x11, 0, 0, 0, 0, 0, 0, 0, 0, 0},
   {0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
};

static void
esl_speclm_imageblock(struct util_dynarray *out, const struct esl_tile *tile)
{
   unsigned samples_log2 = tile->samples_log2;
   unsigned stride = MIN2(tile->stride_8B, 31);
   unsigned w = tile->width_log2, h = tile->height_log2;

   assert(w <= 10 && h <= 10 && w + h <= 10 && samples_log2 <= 2);

   /* demo15's with the fields replaced */
   uint64_t v = 0x0000004B432A0077ull;
   v &= ~(BITFIELD64_MASK(2) << 23);
   v &= ~(BITFIELD64_MASK(5) << 25);
   v &= ~(BITFIELD64_MASK(8) << 30);

   static const uint8_t samples_code[3] = {2, 0, 1};
   v |= (uint64_t)samples_code[samples_log2] << 23;
   v |= (uint64_t)stride << 25;
   v |= (uint64_t)speclm_shape[w][h] << 30;

   esl_u64(out, v);
}

/* G15's pixel fences: demo15's fragment shader waits on 0x200 and then 0xC
 * and signals both at its end, and its entry program releases 0x2F3. Every
 * fence is released once, the shader's by the shader, the others by the entry
 * program, except 0x200, which both release: the universe is 0x2FF (a guess
 * from that one pair; fence 0x100 is in neither). */
#define ESL_PIX_ALL  0x2FF
#define ESL_PIX_BOTH 0x200

/* pixrel: the fragment entry releases the fences its program does not wait on */
static void
esl_pixrel(struct util_dynarray *out, unsigned waited)
{
   unsigned mask = ESL_PIX_ALL & ~(waited & ~ESL_PIX_BOTH);

   if (mask)
      agx3_emit_signal_pix(out, mask);
}

/* Inline the preamble (Apple's constant-calculation program) up to its
 * stop. It runs after ldshdr, as Metal's branch to it does. */
static bool
esl_preamble(struct util_dynarray *out, const struct agx_esl_code *code,
             uint64_t va)
{
   if (!code || va < code->va || va >= code->va + code->size)
      return false;

   const uint8_t *p = code->cpu + (va - code->va);
   const uint8_t *end = code->cpu + code->size;

   while (p + 4 <= end) {
      if (p[0] == 0x0e && p[1] == 0 && p[2] == 0 && p[3] == 0)
         return true;

      unsigned len = agx3_instr_length(p, end - p);
      if (!len)
         return false;

      memcpy(util_dynarray_grow_bytes(out, 1, len), p, len);
      p += len;
   }

   return false;
}

bool
agx_esl_from_usc(struct util_dynarray *out, const uint8_t *usc,
                 uint64_t shader_base, const struct agx_esl_code *code,
                 enum agx_esl_stage stage)
{
   uint64_t main_va = 0, preamble_va = 0;
   struct esl_range ranges[ESL_MAX_RANGES];
   unsigned nr_ranges = 0;
   unsigned shared_bytes = 0;

   /* demo15's: 32x32, one sample of 8 bytes */
   struct esl_tile tile = {5, 5, 0, 1};

   for (bool done = false; !done;) {
      switch (usc[0]) {
      case AGX_USC_CONTROL_UNIFORM:
      case AGX_USC_CONTROL_UNIFORM_HIGH: {
         agx_unpack(NULL, usc, USC_UNIFORM, u);
         unsigned high = usc[0] == AGX_USC_CONTROL_UNIFORM_HIGH ? 256 : 0;
         if (nr_ranges == ARRAY_SIZE(ranges)) {
            fprintf(stderr, "G15: too many uniform ranges\n");
            return false;
         }
         ranges[nr_ranges++] = (struct esl_range){
            .start_halfs = u.start_halfs + high,
            .size_halfs = u.size_halfs,
            .buffer = u.buffer,
         };
         usc += AGX_USC_UNIFORM_LENGTH;
         break;
      }

      case AGX_USC_CONTROL_SHARED: {
         agx_unpack(NULL, usc, USC_SHARED, s);
         if (s.uses_shared_memory)
            shared_bytes = s.bytes_per_threadgroup;

         switch (s.layout) {
         case AGX_SHARED_LAYOUT_32X32:
            tile.width_log2 = tile.height_log2 = 5;
            break;
         case AGX_SHARED_LAYOUT_32X16:
            tile.width_log2 = 5;
            tile.height_log2 = 4;
            break;
         case AGX_SHARED_LAYOUT_16X16:
            tile.width_log2 = tile.height_log2 = 4;
            break;
         default:
            break;
         }

         if (s.layout != AGX_SHARED_LAYOUT_VERTEX_COMPUTE) {
            tile.samples_log2 = util_logbase2(MAX2(s.sample_count, 1));
            tile.stride_8B = s.sample_stride_in_8_bytes;
         }

         usc += AGX_USC_SHARED_LENGTH;
         break;
      }

      case AGX_USC_CONTROL_SHADER: {
         agx_unpack(NULL, usc, USC_SHADER, s);
         main_va = shader_base + s.code;
         usc += AGX_USC_SHADER_LENGTH;
         break;
      }

      case AGX_USC_CONTROL_REGISTERS: {
         agx_unpack(NULL, usc, USC_REGISTERS, r);
         if (r.spill_size) {
            fprintf(stderr, "G15: spilling shaders are not supported yet\n");
            return false;
         }
         usc += AGX_USC_REGISTERS_LENGTH;
         break;
      }

      case AGX_USC_CONTROL_FRAGMENT_PROPERTIES:
         usc += AGX_USC_FRAGMENT_PROPERTIES_LENGTH;
         break;

      case AGX_USC_CONTROL_PRESHADER: {
         agx_unpack(NULL, usc, USC_PRESHADER, p);
         preamble_va = shader_base + p.code;
         done = true;
         break;
      }

      case AGX_USC_CONTROL_NO_PRESHADER:
         done = true;
         break;

      default:
         fprintf(stderr, "G15: USC word 0x%02x is not supported yet\n", usc[0]);
         return false;
      }
   }

   if (!main_va)
      return false;

   /* The words out of reach go through r2-r5 first, before the push
    * registers hold anything */
   for (unsigned i = 0; i < nr_ranges; i++)
      esl_uniforms(out, &ranges[i]);

   esl_uniforms_reach(out, ranges, nr_ranges);

   /* The preamble runs before the tail, as G13's preshader runs before the
    * shader (AGX_G15_PREAMBLE_LAST=1: after it, where Metal branches to its
    * constant-calculation program). */
   bool last = debug_get_bool_option("AGX_G15_PREAMBLE_LAST", false);

   if (preamble_va && !last && !esl_preamble(out, code, preamble_va))
      goto no_preamble;

   switch (stage) {
   case AGX_ESL_COMPUTE:
      esl_speclm(out, shared_bytes);
      esl_ldshdr(out, main_va, code ? code->main_size : 0);
      esl_spectpr(out, code ? code->stack_size : 0, code ? code->cr_count : 0);
      break;

   case AGX_ESL_VERTEX:
      /* Apple's vertex programs (the 2026-10-16 triangle) have specLM,
       * ldshdr and specTPR. demo15's has a 0x6F7-family word with every
       * field set after ldshdr: after the specTPR it loses the stack, so a
       * spilling vertex program reads its spills back wrong. */
      esl_ldshdr(out, main_va, code ? code->main_size : 0);
      esl_spectpr(out, code ? code->stack_size : 0, code ? code->cr_count : 0);
      break;

   case AGX_ESL_FRAGMENT:
      /* Apple's fragment programs have a specTPR after ldshdr */
      esl_speclm_imageblock(out, &tile);
      esl_ldshdr(out, main_va, code ? code->main_size : 0);
      if (!getenv("AGX_G15_FS_NOTPR"))
         esl_spectpr(out, code ? code->stack_size : 0,
                     code ? code->cr_count : 0);
      esl_pixrel(out, code ? code->pix_wait_mask : 0);
      break;

   case AGX_ESL_BACKGROUND:
      esl_speclm_imageblock(out, &tile);
      esl_ldshdr(out, main_va, code ? code->main_size : 0);

      /* a program that loads render targets has its coordinates in cr (a
       * clearing one has none, demo15's) */
      if (code && (code->cr_count || code->stack_size))
         esl_spectpr(out, code->stack_size, code->cr_count);
      break;

   case AGX_ESL_END_OF_TILE:
      /* demo15's: the block write's coordinates in cr0 and cr1; a layered
       * write's packs in cr2 and cr3 too (agx3_cr_count) */
      esl_speclm_imageblock(out, &tile);
      esl_ldshdr(out, main_va, code ? code->main_size : 0);
      esl_spectpr(out, code ? code->stack_size : 0,
                  MAX2(2, code ? code->cr_count : 0));
      break;
   }

   if (preamble_va && last && !esl_preamble(out, code, preamble_va))
      goto no_preamble;

   agx3_emit_stop(out);
   return true;

no_preamble:
   fprintf(stderr, "G15: the preamble is not in the shader's code\n");
   return false;
}
