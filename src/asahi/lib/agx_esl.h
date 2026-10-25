/*
 * Copyright 2026 hitori1403
 * SPDX-License-Identifier: GPL-2.0-only
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "util/macros.h"
#include "util/u_dynarray.h"

/*
 * G15 (M3) has no USC control words: a compute launch points at an entry
 * program (Apple's "entry state loader") that loads the uniforms, names
 * the shader with ldshdr and stops; the GPU then runs the shader. This
 * translates a G13 USC word list into that program, so the drivers keep
 * building USC words.
 */

/* The code a USC word list refers to: the shader binary at its GPU
 * address, the main shader's size (ldshdr's size class), its stack
 * registers in bytes per thread and its coordinate registers
 * (agx_shader_info::stack_size, cr_count) */
struct agx_esl_code {
   uint64_t va;
   const uint8_t *cpu;
   uint32_t size;

   /* The preamble's offset and its bytes before the stop
    * (agx_shader_info::preamble_size), 0 to find them
    */
   uint32_t preamble_offset, preamble_size;
   uint32_t main_size;
   uint32_t stack_size;
   uint32_t cr_count;

   /* A fragment program's pixel fences: it waits on these and signals them at
    * its end; its entry program releases the others */
   uint16_t pix_wait_mask;
};

/* The program the entry program runs: each stage ends its entry program in
 * its own way (noopwafel's demo15 for the render stages) */
enum agx_esl_stage {
   AGX_ESL_COMPUTE,
   AGX_ESL_VERTEX,
   AGX_ESL_FRAGMENT,
   AGX_ESL_BACKGROUND,
   AGX_ESL_END_OF_TILE,
};

/* ldshdr's low 23 bits: the shader's VA follows, bits 46:6 from bit 23 and bit
 * 47 at bit 64, then its size class */
#define AGX_ESL_LDSHDR 0x2A0177

/* The shader the ldshdr at p loads, 0 if p holds another instruction */
static inline uint64_t
agx_esl_ldshdr_va(const uint8_t *p)
{
   uint64_t w;
   memcpy(&w, p, sizeof(w));

   if ((w & BITFIELD64_MASK(23)) != AGX_ESL_LDSHDR)
      return 0;

   return ((w >> 23) << 6) | ((uint64_t)(p[8] & 1) << 47);
}

/* Append the entry program for the USC words at usc (code offsets are
 * relative to shader_base) to out. Returns false for a word G15 does not
 * support yet. */
bool agx_esl_from_usc(struct util_dynarray *out, const uint8_t *usc,
                      uint64_t shader_base, const struct agx_esl_code *code,
                      enum agx_esl_stage stage);
