/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/bitset.h"
#include "agx_state.h"
#include "nir.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"

/*
 * Lower binding table textures and images to texture state registers and (if
 * necessary) bindless access into an internal table mapped like additional
 * texture state registers. The following layout is used:
 *
 *    1. Textures
 *    2. Images (read/write interleaved)
 */

static bool
lower_sampler(nir_builder *b, nir_tex_instr *tex)
{
   if (!nir_tex_instr_need_sampler(tex))
      return false;

   nir_def *index = nir_steal_tex_src(tex, nir_tex_src_sampler_offset);
   if (!index)
      index = nir_imm_int(b, tex->sampler_index);

   nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle,
                         nir_load_sampler_handle_agx(b, index));
   return true;
}

/*
 * G15 has no sampler state registers: a texture instruction names its sampler
 * by index into the stage's sampler table (entry 0 the txf sampler, API
 * sampler i at i + 1), whose base the instruction takes from a uniform pair.
 */
static bool
lower_sampler_g15(nir_builder *b, nir_tex_instr *tex)
{
   bool txf = tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms;

   if (nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0 ||
       !(nir_tex_instr_need_sampler(tex) || txf))
      return false;

   nir_def *index = nir_steal_tex_src(tex, nir_tex_src_sampler_offset);
   if (!nir_tex_instr_need_sampler(tex))
      index = nir_imm_int(b, 0);
   else if (!index)
      index = nir_imm_int(b, tex->sampler_index + 1);
   else
      index = nir_iadd_imm(b, index, 1);

   nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle,
                         nir_load_sampler_handle_agx(b, index));
   return true;
}

struct lower_bindings_state {
   bool *uses_bindless_samplers;
   bool g15;
};

static bool
lower(nir_builder *b, nir_instr *instr, void *data)
{
   const struct lower_bindings_state *state = data;
   bool *uses_bindless_samplers = state->uses_bindless_samplers;
   bool progress = false;

   /* G15 has no texture state registers: everything is bindless */
   bool force_bindless = state->g15 || agx_nir_needs_texture_crawl(instr);
   b->cursor = nir_before_instr(instr);

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

      switch (intr->intrinsic) {
      case nir_intrinsic_image_load:
      case nir_intrinsic_image_store:
      case nir_intrinsic_image_size:
      case nir_intrinsic_image_samples:
      case nir_intrinsic_image_atomic:
      case nir_intrinsic_image_atomic_swap:
         break;
      default:
         return false;
      }

      nir_def *index = intr->src[0].ssa;
      nir_scalar index_scalar = nir_scalar_resolved(index, 0);

      /* Remap according to the driver layout */
      unsigned offset = BITSET_LAST_BIT(b->shader->info.textures_used);

      /* For reads and queries, we use the texture descriptor which is first.
       * Writes and atomics use the PBE descriptor.
       */
      if (intr->intrinsic != nir_intrinsic_image_load &&
          intr->intrinsic != nir_intrinsic_image_size &&
          intr->intrinsic != nir_intrinsic_image_samples)
         offset++;

      /* If we can determine statically that the image fits in texture state
       * registers, avoid lowering to bindless access.
       */
      if (nir_scalar_is_const(index_scalar) && !force_bindless) {
         unsigned idx = (nir_scalar_as_uint(index_scalar) * 2) + offset;

         if (idx < AGX_NUM_TEXTURE_STATE_REGS) {
            nir_src_rewrite(&intr->src[0], nir_imm_intN_t(b, idx, 16));
            return true;
         }
      }

      /* Otherwise, lower to bindless...
       *
       * The driver uploads enough null texture/PBE descriptors for robustness
       * given the shader limit, but we still need to clamp since we're lowering
       * to bindless so the hardware doesn't know the limit.
       *
       * The GL spec says out-of-bounds image indexing is undefined, but
       * faulting is not acceptable for robustness.
       */
      index = nir_umin(
         b, index,
         nir_imm_intN_t(b, b->shader->info.num_images - 1, index->bit_size));

      index = nir_iadd_imm(b, nir_imul_imm(b, index, 2), offset);

      nir_rewrite_image_intrinsic(intr, nir_load_texture_handle_agx(b, index),
                                  nir_image_intrinsic_type_bindless);
   } else if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      if (state->g15) {
         progress |= lower_sampler_g15(b, tex);
      } else if (((BITSET_COUNT(b->shader->info.samplers_used) > 16) &&
                  (nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset) >=
                      0 ||
                   tex->sampler_index >= 16)) &&
                 lower_sampler(b, tex)) {
         progress = true;
         *uses_bindless_samplers = true;
      }

      /* Nothing to do for "real" bindless */
      if (nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) >= 0)
         return progress;

      /* Textures are mapped 1:1, so if we can prove it fits in a texture state
       * register, use the texture state register.
       */
      if (tex->texture_index < AGX_NUM_TEXTURE_STATE_REGS &&
          nir_tex_instr_src_index(tex, nir_tex_src_texture_offset) == -1 &&
          !force_bindless)
         return progress;

      /* Otherwise, lower to bindless. Could be optimized. */
      nir_def *index = nir_steal_tex_src(tex, nir_tex_src_texture_offset);
      if (!index)
         index = nir_imm_int(b, tex->texture_index);

      /* As above */
      index = nir_umin(
         b, index,
         nir_imm_intN_t(b, b->shader->info.num_textures - 1, index->bit_size));

      nir_tex_instr_add_src(tex, nir_tex_src_texture_handle,
                            nir_load_texture_handle_agx(b, index));
   }

   return true;
}

/*
 * G15 has no image load (libLLVM 32023 knows none): an image is read as its
 * texture descriptor by a fetch with the txf sampler, the coordinates as
 * agx_nir_lower_texture leaves them (a 1D or buffer image as 2D, a cube as a
 * 2D array, the layer robust). Multisampled loads are left alone.
 */
static bool
lower_image_load_g15(nir_builder *b, nir_intrinsic_instr *intr)
{
   if (intr->intrinsic != nir_intrinsic_bindless_image_load)
      return false;

   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool array = nir_intrinsic_image_array(intr);

   if (dim == GLSL_SAMPLER_DIM_MS)
      return false;

   if (dim == GLSL_SAMPLER_DIM_CUBE) {
      dim = GLSL_SAMPLER_DIM_2D;
      array = true;
   }

   assert(dim == GLSL_SAMPLER_DIM_2D || dim == GLSL_SAMPLER_DIM_3D);
   unsigned n = glsl_get_sampler_dim_coordinate_components(dim) + array;

   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 3);
   tex->op = nir_texop_txf;
   tex->sampler_dim = dim;
   tex->is_array = array;
   tex->coord_components = n;
   tex->dest_type = nir_intrinsic_dest_type(intr);
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1,
                                     nir_trim_vector(b, intr->src[1].ssa, n));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, intr->src[3].ssa);
   tex->src[2] =
      nir_tex_src_for_ssa(nir_tex_src_texture_handle, intr->src[0].ssa);

   nir_def_init(&tex->instr, &tex->def, 4, intr->def.bit_size);
   nir_builder_instr_insert(b, &tex->instr);
   nir_def_replace(&intr->def,
                   nir_trim_vector(b, &tex->def, intr->def.num_components));

   b->cursor = nir_before_instr(&tex->instr);
   lower_sampler_g15(b, tex);
   return true;
}

static bool
lower_txf_sampler_g15(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   b->cursor = nir_before_instr(instr);

   if (instr->type == nir_instr_type_intrinsic)
      return lower_image_load_g15(b, nir_instr_as_intrinsic(instr));

   if (instr->type != nir_instr_type_tex)
      return false;

   return lower_sampler_g15(b, nir_instr_as_tex(instr));
}

/*
 * G15: agx_nir_lower_texture rewrites a buffer texture's fetch as a 2D fetch
 * without its sampler (G13 has a txf sampler state register); give it the
 * table's txf sampler again. Image loads become such fetches.
 */
bool
agx_nir_lower_txf_samplers_g15(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader, lower_txf_sampler_g15,
                                       nir_metadata_control_flow, NULL);
}

bool
agx_nir_lower_bindings(nir_shader *shader, bool *uses_bindless_samplers,
                       bool g15)
{
   struct lower_bindings_state state = {uses_bindless_samplers, g15};

   /* First lower index to offset so we can lower more naturally */
   bool progress = nir_lower_tex(
      shader, &(nir_lower_tex_options){.lower_index_to_offset = true});

   /* Next run constant folding so the constant optimizations above have a
    * chance.
    */
   progress |= nir_opt_constant_folding(shader);

   progress |= nir_shader_instructions_pass(shader, lower,
                                            nir_metadata_control_flow, &state);
   return progress;
}
