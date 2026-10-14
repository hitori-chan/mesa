/*
 * Copyright 2026 hitori1403
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "util/u_dynarray.h"

/*
 * G15 (M3) has no USC control words: a compute launch points at an entry
 * program (Apple's "entry state loader") that loads the uniforms, names
 * the shader with ldshdr and stops; the GPU then runs the shader. This
 * translates a G13 USC word list into that program, so the drivers keep
 * building USC words.
 */

/* The code a USC word list refers to: the shader binary at its GPU
 * address, the main shader's size (ldshdr's size class) and its stack
 * registers in bytes per thread (agx_shader_info::stack_size) */
struct agx_esl_code {
   uint64_t va;
   const uint8_t *cpu;
   uint32_t size;
   uint32_t main_size;
   uint32_t stack_size;

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

/* Append the entry program for the USC words at usc (code offsets are
 * relative to shader_base) to out. Returns false for a word G15 does not
 * support yet. */
bool agx_esl_from_usc(struct util_dynarray *out, const uint8_t *usc,
                      uint64_t shader_base, const struct agx_esl_code *code,
                      enum agx_esl_stage stage);
