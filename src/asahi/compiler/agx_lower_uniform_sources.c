/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */
#include "agx_builder.h"
#include "agx_compiler.h"

/*
 * Not all instructions can take uniforms. Memory instructions can take
 * uniforms, but only for their base (first) source and only in the
 * low-half of the uniform file.
 *
 * This pass lowers invalid uniforms.
 */
bool
agx_instr_accepts_uniform(enum agx_opcode op, unsigned src_index,
                          unsigned value, enum agx_size size)
{
   /* Some instructions only seem able to access uniforms in the low half */
   bool high = value >= 256;

   /* ALU cannot access 64-bit uniforms */
   bool is_64 = size == AGX_SIZE_64;

   switch (op) {
   case AGX_OPCODE_IMAGE_LOAD:
   case AGX_OPCODE_TEXTURE_LOAD:
   case AGX_OPCODE_TEXTURE_SAMPLE:
      /* Unknown if this works, but the driver will never hit this. */
      assert(!(src_index == 2 && high) && "texture heap always low");
      return !high && (src_index == 1 || src_index == 2 || src_index == 6);

   case AGX_OPCODE_TEX_STATE_STORE:
   case AGX_OPCODE_SAMPLER_STATE_STORE:
      assert(!high && "texture heap always low");
      return !high && src_index == 0;

   case AGX_OPCODE_DEVICE_LOAD:
      return src_index == 0 && !high;
   case AGX_OPCODE_DEVICE_STORE:
   case AGX_OPCODE_ATOMIC:
      return src_index == 1 && !high;
   case AGX_OPCODE_LOCAL_LOAD:
      return src_index == 0;
   case AGX_OPCODE_LOCAL_STORE:
      return src_index == 1;
   case AGX_OPCODE_IMAGE_WRITE:
      return src_index == 3;
   case AGX_OPCODE_BLOCK_IMAGE_STORE:
   case AGX_OPCODE_ZS_EMIT:
   case AGX_OPCODE_ST_TILE:
   case AGX_OPCODE_LD_TILE:
   case AGX_OPCODE_UNIFORM_STORE:
   case AGX_OPCODE_ST_VARY:
   case AGX_OPCODE_LOCAL_ATOMIC:
   case AGX_OPCODE_SAMPLE_MASK:
   case AGX_OPCODE_ITER:
   case AGX_OPCODE_ITERPROJ:
   case AGX_OPCODE_STACK_LOAD:
   case AGX_OPCODE_STACK_STORE:
   case AGX_OPCODE_BALLOT:
   case AGX_OPCODE_FCMP_BALLOT:
   case AGX_OPCODE_ICMP_BALLOT:
   case AGX_OPCODE_QUAD_BALLOT:
   case AGX_OPCODE_FCMP_QUAD_BALLOT:
   case AGX_OPCODE_ICMP_QUAD_BALLOT:
      return false;
   case AGX_OPCODE_EXPORT:
   case AGX_OPCODE_PHI:
      /* We would fail validation otherwise */
      return true;
   default:
      return !is_64;
   }
}

/*
 * G15: a memory instruction whose base is a uniform the preamble wrote reads a
 * stale address: the preamble runs inline in the entry program right before
 * the shader, and the entry program cannot read back the uniforms it writes
 * (agx_esl.c). The geometry shader's dispatches stored to wrong addresses with
 * such bases and right ones with the base moved to registers first. Pushed
 * uniforms (below reserved_preamble) are written early and work as bases.
 */
static bool
agx_g15_stale_uniform_base(agx_context *ctx, agx_instr *I, unsigned s)
{
   if (ctx->key->dev.gpu_generation < 15 || ctx->is_preamble)
      return false;

   unsigned base_src;
   switch (I->op) {
   case AGX_OPCODE_DEVICE_LOAD:
      base_src = 0;
      break;
   case AGX_OPCODE_DEVICE_STORE:
   case AGX_OPCODE_ATOMIC:
      base_src = 1;
      break;
   default:
      return false;
   }

   return s == base_src && I->src[s].value >= ctx->key->reserved_preamble;
}

/*
 * G15's shuffles read their value from a register only. Its rcp, rsqrt, log2,
 * exp2 and sin read a uniform the preamble wrote as it was before: a main
 * program's rcp of the preamble's uniforms got the values a previous program
 * left there, while the ALU operations reading them got the new ones (KHR-GL46
 * shader_subroutine.four_subroutines_with_two_uniforms). Uniforms the entry
 * program loaded are not known to be safe either, so they move too.
 */
static bool
agx_g15_register_only(agx_context *ctx, agx_instr *I, unsigned s)
{
   if (ctx->key->dev.gpu_generation < 15)
      return false;

   switch (I->op) {
   case AGX_OPCODE_SHUFFLE:
   case AGX_OPCODE_SHUFFLE_XOR:
   case AGX_OPCODE_SHUFFLE_UP:
   case AGX_OPCODE_SHUFFLE_DOWN:
   case AGX_OPCODE_QUAD_SHUFFLE:
   case AGX_OPCODE_QUAD_SHUFFLE_XOR:
   case AGX_OPCODE_QUAD_SHUFFLE_UP:
   case AGX_OPCODE_QUAD_SHUFFLE_DOWN:
      return s == 0;
   case AGX_OPCODE_RCP:
   case AGX_OPCODE_RSQRT:
   case AGX_OPCODE_SRSQRT:
   case AGX_OPCODE_LOG2:
   case AGX_OPCODE_EXP2:
   case AGX_OPCODE_SIN_PT_2:
      return true;
   default:
      return false;
   }
}

void
agx_lower_uniform_sources(agx_context *ctx)
{
   agx_foreach_instr_global_safe(ctx, I) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));

      agx_foreach_src(I, s) {
         /* G15's block write takes its descriptor's address in a uniform
          * pair (its 7-bit field reaches the low half only) */
         if (I->op == AGX_OPCODE_BLOCK_IMAGE_STORE && s == 0 &&
             ctx->key->dev.gpu_generation >= 15 &&
             I->src[s].type == AGX_INDEX_UNIFORM &&
             I->src[s].size == AGX_SIZE_64 && I->src[s].value < 256)
            continue;

         if (I->src[s].type == AGX_INDEX_UNIFORM &&
             (!agx_instr_accepts_uniform(I->op, s, I->src[s].value,
                                         I->src[s].size) ||
              agx_g15_stale_uniform_base(ctx, I, s) ||
              agx_g15_register_only(ctx, I, s))) {

            agx_index idx = I->src[s];
            idx.abs = idx.neg = false;
            agx_replace_src(I, s, agx_mov(&b, idx));
         }
      }
   }
}
