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

struct agx_tilebuffer_layout;

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

   /* A render stage's tilebuffer, whose imageblock its specLM declares: G13's
    * USC SHARED word cannot hold G15's samples and tiles. NULL, or a layout
    * without samples yet, gives demo15's 32x32 tile of one 8 B sample. */
   const struct agx_tilebuffer_layout *tib;
};

/* The most thread-private stack a specTPR gives a thread, in bytes: its word
 * count is 12 bits and holds the stack plus 4 words (esl_spectpr) */
#define AGX_ESL_MAX_STACK_B ((((1 << 12) - 1) - 4) * 4)

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

/* The target of the absolute branch at p (agx3_emit_branch_abs: 0f 00 54,
 * then the target with bit 0 set in bits 24-71), 0 if p holds another
 * instruction */
static inline uint64_t
agx_esl_branch_abs_va(const uint8_t *p)
{
   if (p[0] != 0x0f || p[1] != 0x00 || p[2] != 0x54 || !(p[3] & 1))
      return 0;

   uint64_t va = 0;
   for (unsigned i = 0; i < 6; i++)
      va |= (uint64_t)p[3 + i] << (8 * i);

   return va & ~1ull;
}

/* Append the entry program for the USC words at usc (code offsets are
 * relative to shader_base) to out. Returns false for a word G15 does not
 * support yet. */
bool agx_esl_from_usc(struct util_dynarray *out, const uint8_t *usc,
                      uint64_t shader_base, const struct agx_esl_code *code,
                      enum agx_esl_stage stage);

/*
 * A stub and a shared entry program (experimental, Honeykrisp's
 * HK_G15_STUB): the per-draw uniform words of a vertex program (its first
 * vertex, base instance and draw ID) go into their push registers in a stub of
 * a few mov_imm, which branches to an entry program the draws share, as
 * Apple's indirect command buffers do. The shared program is the one
 * agx_esl_from_usc builds, but for the provided words' ranges: no mov_imm or
 * load for them, their registers kept from every other range's address pair,
 * their uniform_store kept.
 */
struct agx_esl_shared {
   /* The uniform words (32-bit) the stub provides: whole ranges, within the
    * push registers' reach */
   uint64_t provided[2];

   /* NULL: ldshdr and specTPR stay in the shared program, after the stub's
    * branch. Else they go here, for the stub to run before its branch. */
   struct util_dynarray *tail;
};

/* At most a stub's size: three mov_imm, ldshdr, specTPR and the branch */
#define AGX_ESL_STUB_MAX 64

/* The shared entry program for the USC words at usc, as agx_esl_from_usc.
 * Returns false where a stub cannot provide the words: a uniform word out of
 * the push registers' reach (the program loads those through r2 up), a
 * range only partly provided, or a stage other than the vertex one. */
bool agx_esl_shared_from_usc(struct util_dynarray *out, const uint8_t *usc,
                             uint64_t shader_base,
                             const struct agx_esl_code *code,
                             enum agx_esl_stage stage,
                             const struct agx_esl_shared *shared);

/* A draw's stub: each value into its uniform word's push register, the tail
 * (agx_esl_shared), then the branch to the shared program */
void agx_esl_stub(struct util_dynarray *out, const unsigned *words,
                  const uint32_t *values, unsigned count, const void *tail,
                  size_t tail_size, uint64_t shared_va);
