/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2021 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "util/macros.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "agx_nir_texture.h"
#include "nir.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "nir_opcodes.h"

static nir_preamble_class
preamble_class(nir_def *def)
{
   nir_intrinsic_instr *intr = nir_def_as_intrinsic_or_null(def);
   if (!intr || (nir_intrinsic_has_desc_set(intr) &&
                 nir_intrinsic_desc_set(intr) >= 32 /* encoding restriction */))
      return nir_preamble_class_general;

   if (intr->intrinsic == nir_intrinsic_bindless_image_agx)
      return nir_preamble_class_image;
   else if (intr->intrinsic == nir_intrinsic_bindless_sampler_agx)
      return nir_preamble_class_sampler;
   else
      return nir_preamble_class_general;
}

static void
def_size(nir_def *def, unsigned *size, unsigned *align,
         nir_preamble_class *class)
{
   unsigned bit_size = MAX2(def->bit_size, 16);

   *size = (bit_size * def->num_components) / 16;
   *align = bit_size / 16;
   *class = preamble_class(def);
}

static bool
all_uses_float(nir_def *def)
{
   nir_foreach_use_including_if(use, def) {
      if (nir_src_is_if(use))
         return false;

      nir_instr *use_instr = nir_src_use_instr(use);
      if (use_instr->type != nir_instr_type_alu)
         return false;

      nir_alu_instr *use_alu = nir_instr_as_alu(use_instr);
      unsigned src_index = ~0;
      for (unsigned i = 0; i < nir_op_infos[use_alu->op].num_inputs; i++) {
         if (&use_alu->src[i].src == use) {
            src_index = i;
            break;
         }
      }

      assert(src_index != ~0);
      nir_alu_type src_type = nir_alu_type_get_base_type(
         nir_op_infos[use_alu->op].input_types[src_index]);

      if (src_type != nir_type_float)
         return false;

      /* No float modifiers on G13 */
      if (use_alu->op == nir_op_fmax || use_alu->op == nir_op_fmin)
         return false;
   }

   return true;
}

static float
alu_cost(nir_alu_instr *alu)
{
   /* TODO: Model 64-bit better */
   if (alu->def.bit_size == 64)
      return 10.0f;

   switch (alu->op) {
   case nir_op_fsat:
   case nir_op_f2fmp:
   case nir_op_f2f16:
   case nir_op_f2f16_rtne:
   case nir_op_fadd:
   case nir_op_fmul:
   case nir_op_ffma:
   case nir_op_iadd:
   case nir_op_inot:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
   case nir_op_feq:
   case nir_op_flt:
   case nir_op_fge:
   case nir_op_fneu:
   case nir_op_ieq:
   case nir_op_ine:
   case nir_op_ilt:
   case nir_op_ige:
   case nir_op_ult:
   case nir_op_uge:
   case nir_op_fmin:
   case nir_op_fmax:
   case nir_op_imin:
   case nir_op_imax:
   case nir_op_umin:
   case nir_op_umax:
   case nir_op_isub:
   case nir_op_ineg:
   case nir_op_bcsel:
   case nir_op_b2b1:
   case nir_op_b2b32:
   case nir_op_b2i8:
   case nir_op_b2i16:
   case nir_op_b2i32:
   case nir_op_b2f16:
   case nir_op_b2f32:
   case nir_op_i2i32:
   case nir_op_i2i16:
   case nir_op_u2u32:
   case nir_op_u2u16:
   case nir_op_u2u8:
   case nir_op_i2i8:
   case nir_op_iadd_sat:
   case nir_op_isub_sat:
   case nir_op_uadd_sat:
   case nir_op_usub_sat:
   case nir_op_iabs:
      /* SCIB */
      return 1.0;

   case nir_op_ffloor:
   case nir_op_fceil:
   case nir_op_ftrunc:
   case nir_op_fround_even:
   case nir_op_bit_count:
   case nir_op_bitfield_reverse:
   case nir_op_ufind_msb:
   case nir_op_imul:
   case nir_op_imadshl_agx:
   case nir_op_imsubshl_agx:
   case nir_op_ishl:
   case nir_op_ishr:
   case nir_op_ushr:
   case nir_op_flog2:
   case nir_op_fexp2:
   case nir_op_extr_agx:
   case nir_op_ubitfield_extract:
   case nir_op_f2i8:
   case nir_op_f2i16:
   case nir_op_f2i32:
   case nir_op_f2u8:
   case nir_op_f2u16:
   case nir_op_f2u32:
   case nir_op_i2fmp:
   case nir_op_i2f16:
   case nir_op_i2f32:
   case nir_op_u2fmp:
   case nir_op_u2f16:
   case nir_op_u2f32:
   case nir_op_interleave:
      /* IC */
      return 4.0;

   case nir_op_frcp:
      /* IC */
      return 6.0;

   case nir_op_frsq:
      /* IC */
      return 8.0;

   case nir_op_fsqrt:
      /* IC + F32 */
      return 8.5;

   case nir_op_imul_high:
   case nir_op_umul_high:
   case nir_op_imul_2x32_64:
   case nir_op_umul_2x32_64:
      /* IC */
      return 8.0;

   case nir_op_fsin_agx:
      /* 2 IC + 1 F32 in parallel */
      return 8.5;

   case nir_op_fneg:
   case nir_op_fabs:
   case nir_op_f2f32:
      /* Float source modifiers will be propagated */
      return all_uses_float(&alu->def) ? 0.0 : 1.0;

   case nir_op_mov:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   case nir_op_pack_32_2x16_split:
   case nir_op_pack_64_2x32_split:
   case nir_op_unpack_64_2x32_split_x:
   case nir_op_unpack_64_2x32_split_y:
   case nir_op_unpack_32_2x16_split_x:
   case nir_op_unpack_32_2x16_split_y:
   case nir_op_extract_i8:
   case nir_op_extract_u8:
   case nir_op_extract_i16:
   case nir_op_extract_u16:
      /* We optimistically assume that moves get coalesced */
      return 0.0;

   default:
      /* Shrug */
      return 2.0;
   }
}

static float
instr_cost(nir_instr *instr, const void *data)
{
   switch (instr->type) {
   case nir_instr_type_intrinsic:
      switch (nir_instr_as_intrinsic(instr)->intrinsic) {
      case nir_intrinsic_load_global:
      case nir_intrinsic_load_agx:
      case nir_intrinsic_load_global_constant:
      case nir_intrinsic_load_constant_agx:
      case nir_intrinsic_load_ubo:
         return 10.0;
      case nir_intrinsic_ddx:
      case nir_intrinsic_ddx_fine:
      case nir_intrinsic_ddx_coarse:
      case nir_intrinsic_ddy:
      case nir_intrinsic_ddy_fine:
      case nir_intrinsic_ddy_coarse:
         return 1.0;
      case nir_intrinsic_bindless_image_agx:
      case nir_intrinsic_bindless_sampler_agx:
         /* It's worth promoting even with a constant source, but it doesn't
          * turn into instructions so should be less than any other normal
          * instruction... But just enough to get over the image rewrite_cost.
          */
         return 2.5;
      default:
         /* Assume it's a sysval or something */
         return 0.0;
      }

   case nir_instr_type_tex:
      /* Texturing involes lots of memory bandwidth */
      return 20.0;

   case nir_instr_type_alu:
      return alu_cost(nir_instr_as_alu(instr));

   default:
      return 1.0;
   }
}

static float
rewrite_cost(nir_def *def, const void *data)
{
   bool mov_needed = false, vectorizable = true;
   nir_foreach_use(use, def) {
      nir_instr *parent_instr = nir_src_use_instr(use);
      if (parent_instr->type == nir_instr_type_tex) {
         /* TODO: Maybe check the source index, but biases can be uniform */
         break;
      } else if (parent_instr->type == nir_instr_type_phi) {
         /* Assume we'd eat a move anyway */
      } else if (parent_instr->type != nir_instr_type_alu) {
         mov_needed = true;
         vectorizable = false;
         break;
      } else {
         nir_alu_instr *alu = nir_instr_as_alu(parent_instr);
         if (alu->op == nir_op_vec2 || alu->op == nir_op_vec3 ||
             alu->op == nir_op_vec4) {
            mov_needed = true;
            break;
         } else if (alu->op == nir_op_mov) {
            mov_needed = true;
            vectorizable = false;
         } else {
            /* Assume for non-moves that the const is folded into the src */
         }
      }
   }

   return mov_needed ? ((float)(def->num_components * def->bit_size) /
                        (vectorizable ? 32.0 : 16.0))
                     : 0;
}

static bool
avoid_instr(const nir_instr *instr, const void *data)
{
   return false;
}

/* G15 has no texture or sampler state registers: its texture instructions
 * name the tables' uniform pairs and index them, so the handles stay in the
 * shader for the backend */
static bool
avoid_instr_g15(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   return intr->intrinsic == nir_intrinsic_bindless_image_agx ||
          intr->intrinsic == nir_intrinsic_bindless_image_base_agx ||
          intr->intrinsic == nir_intrinsic_bindless_sampler_agx;
}

/* G15 inlines the preamble into the entry program. A gather there handed the
 * main program wrong values (0 in the first one or two of the four uniforms it
 * stored, the rest right; the same gather in the main program was right:
 * dEQP-VK's border_swizzle gathers at constant coordinates), so G15 keeps
 * texture instructions in the shader. Why is not known.
 */
static bool
can_move_instr_g15(const nir_instr *instr, const void *data)
{
   return instr->type != nir_instr_type_tex;
}

static const nir_opt_preamble_options preamble_options = {
   .drawid_uniform = true,
   .subgroup_size_uniform = true,
   /* not supported in hardware */
   .load_workgroup_size_allowed = false,
   .def_size = def_size,
   .instr_cost_cb = instr_cost,
   .rewrite_cost_cb = rewrite_cost,
   .avoid_instr_cb = avoid_instr,

   /* hardware size is 512, but it's polite to leave some wiggle room to push
    * hot constants so we don't end up rematerializing all over the place.
    * 480 seems to be a sweetspot, based on a few minutes of shader-db.
    */
   .preamble_storage_size[nir_preamble_class_general] = 480,

   /* We have at least 32 texture state registers. TODO: check for more? */
   .preamble_storage_size[nir_preamble_class_image] = 32,

   /* We have at least 16 sampler state registers. TODO: check for more? */
   .preamble_storage_size[nir_preamble_class_sampler] = 16,
};

/*
 * Bindless image handles can't be stored to uniforms, so we move them back to
 * the main shader. Effectively un-optimizing the preamble.
 */
static bool
lower_store_preamble(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   int16_t *heaps = data;
   if (intr->intrinsic != nir_intrinsic_store_preamble ||
       nir_intrinsic_preamble_class(intr) == nir_preamble_class_image)
      return false;

   nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);
   if (!handle || handle->intrinsic != nir_intrinsic_bindless_image_agx)
      return false;

   heaps[nir_intrinsic_base(intr)] = nir_intrinsic_desc_set(handle);
   nir_src_rewrite(&intr->src[0], handle->src[0].ssa);
   return true;
}

struct lower_preamble_state {
   int16_t *heaps;
   bool g15;
};

static bool
lower_preamble(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower_preamble_state *state = data;

   if (intr->intrinsic == nir_intrinsic_bindless_sampler_agx && !state->g15) {
      /* Rematerialize bindless_sampler_agx before store_preamble with only the
       * byte offset (first source), not the sampler index.
       */
      nir_foreach_use_safe(use, &intr->def) {
         nir_instr *parent = nir_src_use_instr(use);
         if (parent->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *pintr = nir_instr_as_intrinsic(parent);
         if (pintr->intrinsic != nir_intrinsic_store_preamble ||
             nir_intrinsic_preamble_class(pintr) != nir_preamble_class_sampler)
            continue;

         b->cursor = nir_before_src(use);
         nir_def *repl =
            nir_bindless_sampler_agx(b, intr->src[0].ssa, nir_undef(b, 1, 16),
                                     .desc_set = nir_intrinsic_desc_set(intr));
         nir_src_rewrite(use, repl);
      }

      /* Replace other uses with just the sampler index. */
      nir_def_replace(&intr->def, intr->src[1].ssa);
      return true;
   }

   if (intr->intrinsic != nir_intrinsic_load_preamble)
      return false;

   int16_t *heaps = state->heaps;
   b->cursor = nir_after_instr(&intr->instr);

   unsigned base = nir_intrinsic_base(intr);
   nir_def *new_ = NULL;
   bool ts = nir_intrinsic_preamble_class(intr) == nir_preamble_class_image;
   bool ss = nir_intrinsic_preamble_class(intr) == nir_preamble_class_sampler;
   if (!ts && heaps[base] >= 0) {
      new_ = nir_bindless_image_agx(b, &intr->def, .desc_set = heaps[base]);
   }

   nir_foreach_use_safe(use, &intr->def) {
      nir_instr *parent = nir_src_use_instr(use);

      if (parent->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *pintr = nir_instr_as_intrinsic(parent);

         if (ts) {
            nir_rewrite_image_intrinsic(pintr, nir_imm_intN_t(b, base / 2, 16),
                                        nir_image_intrinsic_type_default);
         } else if (new_ != NULL &&
                    pintr->intrinsic != nir_intrinsic_bindless_image_agx) {
            nir_src_rewrite(use, new_);
         }
      } else if (parent->type == nir_instr_type_tex) {
         nir_tex_instr *tex = nir_instr_as_tex(parent);
         nir_tex_src *src = (nir_tex_src *)use;

         if (src->src_type == nir_tex_src_sampler_handle && ss) {
            nir_steal_tex_src(tex, nir_tex_src_sampler_handle);
            tex->sampler_index = base;
         } else if (src->src_type == nir_tex_src_texture_handle && ts) {
            nir_steal_tex_src(tex, nir_tex_src_texture_handle);
            tex->texture_index = base / 2;
         } else if (src->src_type == nir_tex_src_texture_handle) {
            assert(new_ != NULL);
            nir_src_rewrite(use, new_);
         }
      }
   }

   return true;
}

/* G15: the store of the preamble's value at base, or NULL */
static nir_intrinsic_instr *
find_store_preamble(nir_function_impl *preamble, unsigned base)
{
   nir_foreach_block(block, preamble) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_preamble &&
             nir_intrinsic_preamble_class(intr) == nir_preamble_class_general &&
             nir_intrinsic_base(intr) == base)
            return intr;
      }
   }
   return NULL;
}

/* G15: whether the packer takes tex's bias in F28 (agx_pack3.c's
 * a3_tex_code, its LOD form A3_LOD_BIAS_F28): a 2D, 2D array, 3D, cube or
 * cube array sample with no minimum LOD and no offset in the LOD word */
static bool
g15_bias_in_f28(nir_tex_instr *tex)
{
   if (tex->op != nir_texop_txb ||
       nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_lod_bias_min_agx) >= 0)
      return false;

   switch (tex->sampler_dim) {
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_CUBE:
      break;
   case GLSL_SAMPLER_DIM_3D:
      if (tex->is_array)
         return false;
      break;
   default:
      return false;
   }

   /* An offset goes in F42 only as a constant without a compare (the packed
    * offset, backend2, once agx_nir_lower_texture has run) */
   int oi = nir_tex_instr_src_index(tex, nir_tex_src_backend2);
   if (oi < 0)
      oi = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   return oi < 0 || (nir_src_is_const(tex->src[oi].src) &&
                     nir_tex_instr_src_index(tex, nir_tex_src_comparator) < 0);
}

/* G15: a bias the preamble computes reaches the sample as F28's uniform,
 * q5.6 in the uniform's low half, as Apple's compiler gives it: its preamble
 * converts the bias once (pack_smpir mode 0xa, the q5.6 moved to the low
 * half, the high half cleared; the QuartzCore corpus's 0x1A3 fragment
 * programs, smp.md), where Mesa packed it into cr3 at every sample. The
 * preamble stores the converted bias beside the f16 one, stored or pushed
 * (rounded to the nearest 1/64, clamped to the 12-bit field), and the
 * sample's bias source becomes its load, flagged AGX_TEXTURE_FLAG_BIAS_Q56. Before the main
 * shader's other preamble loads are lowered. */
static bool
agx_nir_g15_bias_q56(nir_shader *nir, unsigned *size)
{
   nir_function_impl *preamble = nir_shader_get_preamble(nir);
   if (!preamble || (agx_compiler_debug & AGX_DBG_BIASCR))
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   int16_t q56[512];
   memset(q56, ~0, sizeof(q56));
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;

         nir_tex_instr *tex = nir_instr_as_tex(instr);
         int bi = nir_tex_instr_src_index(tex, nir_tex_src_bias);
         if (bi < 0 || !g15_bias_in_f28(tex))
            continue;

         nir_intrinsic_instr *load = nir_src_as_intrinsic(tex->src[bi].src);
         if (!load || load->intrinsic != nir_intrinsic_load_preamble ||
             load->def.bit_size != 16 || load->def.num_components != 1 ||
             nir_intrinsic_preamble_class(load) != nir_preamble_class_general)
            continue;

         unsigned base = nir_intrinsic_base(load);
         if (base >= ARRAY_SIZE(q56))
            continue;

         if (q56[base] < 0) {
            /* the F28 uniform is 9 bits in 16-bit units, but a texture takes
             * uniforms below 256 (agx_instr_accepts_uniform) */
            if (*size >= 256)
               continue;

            /* The bias the preamble stores, or a pushed one (the GL driver's
             * sampler bias), which a G15 preamble reads from its register
             * (agx_emit_load_preamble) */
            nir_intrinsic_instr *store = find_store_preamble(preamble, base);
            nir_builder b;
            nir_def *v;
            if (store) {
               b = nir_builder_at(nir_after_instr(&store->instr));
               v = store->src[0].ssa;
            } else if (base + 2 <= 2 * AGX_G15_PUSH_WORDS) {
               b = nir_builder_at(nir_before_impl(preamble));
               v = nir_load_preamble(
                  &b, 1, 16, .base = base,
                  .preamble_class = nir_preamble_class_general);
            } else {
               continue;
            }

            nir_def *f = nir_fmul_imm(&b, nir_f2f32(&b, v), 64.0);
            nir_def *q = nir_f2i32(&b, nir_fround_even(&b, f));
            q = nir_imin(&b, nir_imax(&b, q, nir_imm_int(&b, -2048)),
                         nir_imm_int(&b, 2047));
            nir_store_preamble(&b, nir_i2i16(&b, q), .base = *size,
                               .preamble_class = nir_preamble_class_general);
            q56[base] = (*size)++;
         }

         nir_builder b = nir_builder_at(nir_before_instr(&tex->instr));
         nir_def *q = nir_load_preamble(
            &b, 1, 16, .base = q56[base],
            .preamble_class = nir_preamble_class_general);
         nir_src_rewrite(&tex->src[bi].src, q);
         tex->backend_flags |= AGX_TEXTURE_FLAG_BIAS_Q56;
         progress = true;
      }
   }

   nir_progress(progress, impl, nir_metadata_control_flow);
   nir_progress(progress, preamble, nir_metadata_control_flow);
   return progress;
}

bool
agx_nir_opt_preamble(nir_shader *nir, unsigned *sizes, bool g15)
{
   bool progress = false;
   nir_opt_preamble_options options = preamble_options;
   if (g15) {
      options.avoid_instr_cb = avoid_instr_g15;
      options.can_move_instr_cb = can_move_instr_g15;
   }

   NIR_PASS(progress, nir, nir_opt_preamble, &options, sizes);

   if (g15 && progress)
      NIR_PASS(_, nir, agx_nir_g15_bias_q56, &sizes[0]);

   int16_t heap[512];
   memset(heap, ~0, sizeof(heap));

   if (progress) {
      nir_function_intrinsics_pass(nir_shader_get_preamble(nir),
                                   lower_store_preamble,
                                   nir_metadata_control_flow, heap);
   }

   struct lower_preamble_state state = {heap, g15};
   NIR_PASS(progress, nir, nir_shader_intrinsics_pass, lower_preamble,
            nir_metadata_control_flow, &state);

   return progress;
}
