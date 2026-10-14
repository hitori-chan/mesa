/*
 * Copyright 2026 hitori1403
 * SPDX-License-Identifier: MIT
 */

#include "agx_esl.h"
#include <stdio.h>
#include "util/u_debug.h"
#include "compiler/agx_compile.h"
#include "agx_pack.h"

/* The entry program's registers: r0:r1 the buffer address, r2-r5 a load */
#define ESL_BASE 0
#define ESL_DATA 2
#define ESL_SLOT 0

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

/* Load size_halfs 16-bit uniforms from buffer into the uniforms from
 * start_halfs on. The uniform file is filled in 32-bit words; an odd
 * count rounds up (the extra half belongs to the buffer's padding). */
static void
esl_uniforms(struct util_dynarray *out, unsigned start_halfs,
             unsigned size_halfs, uint64_t buffer)
{
   assert((start_halfs & 1) == 0 && "uniform ranges start 32-bit aligned");
   unsigned words = DIV_ROUND_UP(size_halfs, 2);

   agx3_emit_mov_imm(out, ESL_BASE, buffer);
   agx3_emit_mov_imm(out, ESL_BASE + 1, buffer >> 32);

   for (unsigned w = 0; w < words; w += 4) {
      unsigned n = MIN2(4, words - w);

      agx3_emit_load(out, ESL_SLOT, ESL_DATA, ESL_BASE, w / 4, n);
      agx3_emit_wait(out, ESL_SLOT);

      for (unsigned i = 0; i < n; i++)
         agx3_emit_uniform_store(out, start_halfs / 2 + w + i, ESL_DATA + i);
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

   esl_u64(out, 0x2A0177 | ((va >> 6) << 23));
   esl_u16(out, ((va >> 47) & 1) | (size_class[MIN2(size >> 7, 3)] << 1));
}

/* specTPR: no spill (TODO); the stack (41:31) in 8 B units */
static void
esl_spectpr(struct util_dynarray *out, unsigned stack_bytes)
{
   unsigned units = DIV_ROUND_UP(stack_bytes, 8);
   assert(units < (1 << 11));

   esl_u64(out, 0x2A00F7 | ((uint64_t)units << 31));
   esl_u16(out, 0);
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

      unsigned len = agx3_instr_length(p);
      if (!len || p + len > end)
         return false;

      memcpy(util_dynarray_grow_bytes(out, 1, len), p, len);
      p += len;
   }

   return false;
}

bool
agx_esl_from_usc(struct util_dynarray *out, const uint8_t *usc,
                 uint64_t shader_base, const struct agx_esl_code *code)
{
   uint64_t main_va = 0, preamble_va = 0;
   unsigned shared_bytes = 0;

   for (bool done = false; !done;) {
      switch (usc[0]) {
      case AGX_USC_CONTROL_UNIFORM:
      case AGX_USC_CONTROL_UNIFORM_HIGH: {
         agx_unpack(NULL, usc, USC_UNIFORM, u);
         unsigned high = usc[0] == AGX_USC_CONTROL_UNIFORM_HIGH ? 256 : 0;
         esl_uniforms(out, u.start_halfs + high, u.size_halfs, u.buffer);
         usc += AGX_USC_UNIFORM_LENGTH;
         break;
      }

      case AGX_USC_CONTROL_SHARED: {
         agx_unpack(NULL, usc, USC_SHARED, s);
         if (s.uses_shared_memory)
            shared_bytes = s.bytes_per_threadgroup;
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
         fprintf(stderr, "G15: USC word 0x%02x is not supported yet\n",
                 usc[0]);
         return false;
      }
   }

   if (!main_va)
      return false;

   /* The preamble runs before the tail, as G13's preshader runs before the
    * shader (AGX_G15_PREAMBLE_LAST=1: after it, where Metal branches to its
    * constant-calculation program). */
   bool last = debug_get_bool_option("AGX_G15_PREAMBLE_LAST", false);

   if (preamble_va && !last && !esl_preamble(out, code, preamble_va))
      goto no_preamble;

   esl_speclm(out, shared_bytes);
   esl_ldshdr(out, main_va, code ? code->main_size : 0);
   esl_spectpr(out, code ? code->stack_size : 0);

   if (preamble_va && last && !esl_preamble(out, code, preamble_va))
      goto no_preamble;

   agx3_emit_stop(out);
   return true;

no_preamble:
   fprintf(stderr, "G15: the preamble is not in the shader's code\n");
   return false;
}
