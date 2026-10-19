/*
 * Copyright 2023 Valve Corporation
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020 Collabora Ltd.
 * Copyright 2016 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "glsl_types.h"
#include "libagx.h"
#include "nir_builder_opcodes.h"
#include "nir_builtin_builder.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "shader_enums.h"

/* Residency flags are inverted from NIR */
#define AGX_RESIDENT (0)

static bool
fence_image(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_after_instr(&intr->instr);

   /* If the image is write-only, there is no fencing needed */
   if (nir_intrinsic_has_access(intr) &&
       (nir_intrinsic_access(intr) & ACCESS_NON_READABLE)) {
      return false;
   }

   switch (intr->intrinsic) {
   case nir_intrinsic_image_store:
   case nir_intrinsic_bindless_image_store:
      nir_fence_pbe_to_tex_agx(b);
      return true;

   case nir_intrinsic_image_atomic:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_image_atomic_swap:
   case nir_intrinsic_bindless_image_atomic_swap:
      nir_fence_mem_to_tex_agx(b);
      return true;

   default:
      return false;
   }
}

static nir_def *
texture_descriptor_ptr(nir_builder *b, nir_tex_instr *tex)
{
   int handle_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   assert(handle_idx >= 0 && "must be bindless");
   return nir_load_from_texture_handle_agx(b, tex->src[handle_idx].src.ssa);
}

static bool
has_nonzero_lod(nir_tex_instr *tex)
{
   int idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (idx < 0)
      return false;

   nir_src src = tex->src[idx].src;
   return !(nir_src_is_const(src) && nir_src_as_uint(src) == 0);
}

static bool
lower_tex_crawl(nir_builder *b, nir_instr *instr, void *data)
{
   bool g15 = *(bool *)data;
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   b->cursor = nir_before_instr(instr);

   if (tex->op != nir_texop_txs && tex->op != nir_texop_texture_samples &&
       tex->op != nir_texop_query_levels)
      return false;

   nir_def *ptr = texture_descriptor_ptr(b, tex);
   unsigned nr_comps = tex->def.num_components;
   assert(nr_comps <= 3);

   int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   nir_def *lod = lod_idx >= 0 ? nir_u2u16(b, tex->src[lod_idx].src.ssa)
                               : nir_imm_intN_t(b, 0, 16);

   nir_def *res;
   if (tex->op == nir_texop_txs) {
      nir_def *args[] = {
         nir_imm_int(b, nr_comps),
         nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_BUF),
         nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_1D),
         nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_2D),
         nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE),
         nir_imm_bool(b, tex->is_array),
      };
      res = g15 ? libagx_txs_g15(b, ptr, lod, args[0], args[1], args[2],
                                 args[3], args[4], args[5])
                : libagx_txs(b, ptr, lod, args[0], args[1], args[2], args[3],
                             args[4], args[5]);
   } else if (tex->op == nir_texop_query_levels) {
      res = g15 ? libagx_texture_levels_g15(b, ptr)
                : libagx_texture_levels(b, ptr);
   } else {
      res = g15 ? libagx_texture_samples_g15(b, ptr)
                : libagx_texture_samples(b, ptr);
   }

   nir_def_rewrite_uses(&tex->def, nir_trim_vector(b, res, nr_comps));
   nir_instr_remove(instr);
   return true;
}

/*
 * Given a 1D buffer texture coordinate, calculate the 2D coordinate vector that
 * will be used to access the linear 2D texture bound to the buffer.
 */
static nir_def *
coords_for_buffer_texture(nir_builder *b, nir_def *coord)
{
   return nir_vec2(b, nir_umod_imm(b, coord, AGX_TEXTURE_BUFFER_WIDTH),
                   nir_udiv_imm(b, coord, AGX_TEXTURE_BUFFER_WIDTH));
}

/*
 * Buffer textures are lowered to 2D (1024xN) textures in the driver to access
 * more storage. When lowering, we need to fix up the coordinate accordingly.
 *
 * Furthermore, RGB32 formats are emulated by lowering to global memory access,
 * so to read a buffer texture we generate code that looks like:
 *
 *    if (descriptor->format == RGB32)
 *       return ((uint32_t *) descriptor->address)[x];
 *    else
 *       return txf(texture_as_2d, vec2(x % 1024, x / 1024));
 */
static bool
lower_buffer_texture(nir_builder *b, nir_tex_instr *tex)
{
   nir_def *coord = nir_steal_tex_src(tex, nir_tex_src_coord);
   nir_def *size = nir_get_texture_size(b, tex);
   nir_def *oob = nir_uge(b, coord, size);

   /* Apply the buffer offset after calculating oob but before remapping */
   nir_def *desc = texture_descriptor_ptr(b, tex);
   coord = libagx_buffer_texture_offset(b, desc, coord);

   /* Map out-of-bounds indices to out-of-bounds coordinates for robustness2
    * semantics from the hardware.
    */
   coord = nir_bcsel(b, oob, nir_imm_int(b, -1), coord);

   bool is_float = nir_alu_type_get_base_type(tex->dest_type) == nir_type_float;

   /* Lower RGB32 reads if the format requires. If we are out-of-bounds, we use
    * the hardware path so we get a zero texel.
    */
   nir_if *nif = nir_push_if(
      b, nir_iand(b, libagx_texture_is_rgb32(b, desc), nir_inot(b, oob)));

   nir_def *rgb32 = nir_trim_vector(
      b, libagx_texture_load_rgb32(b, desc, coord, nir_imm_bool(b, is_float)),
      nir_tex_instr_result_size(tex));

   /* Raw loads do not return residency information, but residency queries are
    * supported on buffer textures. Fortunately, we do not need to support
    * sparse RGB32 buffers, so we simply claim all RGB32 loads were resident.
    * Nothing should hit this in practice, but if we don't do *something* here
    * we'll get vector size mismatches which blow up in vkd3d-proton.
    */
   if (tex->is_sparse) {
      rgb32 = nir_pad_vector_imm_int(b, rgb32, AGX_RESIDENT,
                                     rgb32->num_components + 1);
   }

   nir_push_else(b, nif);

   /* Otherwise, lower the texture instruction to read from 2D */
   assert(coord->num_components == 1 && "buffer textures are 1D");
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;

   nir_def *coord2d = coords_for_buffer_texture(b, coord);
   nir_instr_remove(&tex->instr);
   nir_builder_instr_insert(b, &tex->instr);
   nir_tex_instr_add_src(tex, nir_tex_src_backend1, coord2d);
   nir_steal_tex_src(tex, nir_tex_src_sampler_handle);
   nir_steal_tex_src(tex, nir_tex_src_sampler_offset);
   nir_block *else_block = nir_cursor_current_block(b->cursor);
   nir_pop_if(b, nif);

   /* Put it together with a phi */
   nir_def *phi = nir_if_phi(b, rgb32, &tex->def);
   nir_def_rewrite_uses(&tex->def, phi);
   nir_phi_instr *phi_instr = nir_def_as_phi(phi);
   nir_phi_src *else_src = nir_phi_get_src_from_block(phi_instr, else_block);
   nir_src_rewrite(&else_src->src, &tex->def);
   return true;
}

/*
 * NIR indexes into array textures with unclamped floats (integer for txf). AGX
 * requires the index to be a clamped integer. Lower tex_src_coord into
 * tex_src_backend1 for array textures by type-converting and clamping.
 */
static bool
lower_regular_texture(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   b->cursor = nir_before_instr(instr);

   if (nir_tex_instr_is_query(tex) && tex->op != nir_texop_lod)
      return false;

   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF)
      return lower_buffer_texture(b, tex);

   /* Don't lower twice */
   if (nir_tex_instr_src_index(tex, nir_tex_src_backend1) >= 0)
      return false;

   /* Get the coordinates */
   nir_def *coord = nir_steal_tex_src(tex, nir_tex_src_coord);
   nir_def *ms_idx = nir_steal_tex_src(tex, nir_tex_src_ms_index);

   /* Apply txf workaround, see libagx_lower_txf_robustness */
   bool is_txf = ((tex->op == nir_texop_txf) || (tex->op == nir_texop_txf_ms));

   if (is_txf &&
       (has_nonzero_lod(tex) || tex->is_array ||
        nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0) &&
       !(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP)) {

      int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      nir_def *lod =
         lod_idx >= 0 ? tex->src[lod_idx].src.ssa : nir_undef(b, 1, 16);

      nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);

      unsigned lidx = coord->num_components - 1;
      nir_def *layer = nir_channel(b, coord, lidx);

      nir_def *replaced = libagx_lower_txf_robustness(
         b, texture_descriptor_ptr(b, tex),
         nir_imm_bool(b, has_nonzero_lod(tex)), lod,
         nir_imm_bool(b, min_lod != NULL), min_lod ?: nir_undef(b, 1, 16),
         nir_imm_bool(b, tex->is_array), layer, nir_channel(b, coord, 0));

      coord = nir_vector_insert_imm(b, coord, replaced, 0);
   }

   /* The layer is always the last component of the NIR coordinate, split it off
    * because we'll need to swizzle.
    */
   nir_def *layer = NULL;

   if (tex->is_array && tex->op != nir_texop_lod) {
      unsigned lidx = coord->num_components - 1;
      nir_def *unclamped_layer = nir_channel(b, coord, lidx);
      coord = nir_trim_vector(b, coord, lidx);

      /* Round layer to nearest even */
      if (!is_txf) {
         unclamped_layer = nir_fround_even(b, unclamped_layer);

         /* Explicitly round negative to avoid undefined behaviour when constant
          * folding. This is load bearing on x86 builds.
          */
         unclamped_layer =
            nir_f2u32(b, nir_fmax(b, unclamped_layer, nir_imm_float(b, 0.0f)));
      }

      /* For a cube array, the layer is zero-indexed component 3 of the
       * coordinate but the number of layers is component 2 of the txs result.
       */
      if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
         assert(lidx == 3 && "4 components");
         lidx = 2;
      }

      /* Clamp to max layer = (# of layers - 1) for out-of-bounds handling.
       * Layer must be 16-bits for the hardware, drop top bits after clamping.
       *
       * For txf, we drop out-of-bounds components rather than clamp, see the
       * above txf robustness workaround.
       */
      if (!(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP) && !is_txf) {
         nir_def *txs = nir_get_texture_size(b, tex);
         nir_def *nr_layers = nir_channel(b, txs, lidx);
         nir_def *max_layer = nir_iadd_imm(b, nr_layers, -1);
         layer = nir_umin(b, unclamped_layer, max_layer);
      } else {
         layer = unclamped_layer;
      }

      layer = nir_u2u16(b, layer);
   }

   /* Combine layer and multisample index into 32-bit so we don't need a vec5 or
    * vec6 16-bit coordinate tuple, which would be inconvenient in NIR for
    * little benefit (a minor optimization, I guess).
    */
   nir_def *sample_array = (ms_idx && layer)
                              ? nir_pack_32_2x16_split(b, ms_idx, layer)
                           : ms_idx ? nir_u2u32(b, ms_idx)
                           : layer  ? nir_u2u32(b, layer)
                                    : NULL;

   /* Combine into the final 32-bit tuple */
   if (sample_array != NULL) {
      unsigned end = coord->num_components;
      coord = nir_pad_vector(b, coord, end + 1);
      coord = nir_vector_insert_imm(b, coord, sample_array, end);
   }

   nir_tex_instr_add_src(tex, nir_tex_src_backend1, coord);

   /* Furthermore, if there is an offset vector, it must be packed */
   nir_def *offset = nir_steal_tex_src(tex, nir_tex_src_offset);

   if (offset != NULL) {
      nir_def *packed = NULL;

      for (unsigned c = 0; c < offset->num_components; ++c) {
         nir_def *nibble = nir_iand_imm(b, nir_channel(b, offset, c), 0xF);
         nir_def *shifted = nir_ishl_imm(b, nibble, 4 * c);

         /* We pack with iadd instead of ior to let us fuse in the shift with an
          * iadd-lsl instruction.
          */
         if (packed != NULL)
            packed = nir_iadd(b, packed, shifted);
         else
            packed = shifted;
      }

      nir_tex_instr_add_src(tex, nir_tex_src_backend2, packed);
   }

   if (nir_tex_instr_src_index(tex, nir_tex_src_bias) >= 0 &&
       nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0) {

      nir_def *bias = nir_steal_tex_src(tex, nir_tex_src_bias);
      nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);
      nir_def *packed = nir_pack_32_2x16_split(b, bias, min_lod);
      nir_tex_instr_add_src(tex, nir_tex_src_lod_bias_min_agx, packed);
   }

   /* We reserve bound sampler #0, so we offset bound samplers by 1 and
    * otherwise map bound samplers as-is.
    */
   nir_def *sampler = nir_steal_tex_src(tex, nir_tex_src_sampler_offset);
   if (!sampler)
      sampler = nir_imm_intN_t(b, tex->sampler_index, 16);

   if (!is_txf &&
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) < 0) {

      nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle,
                            nir_iadd_imm(b, nir_u2u16(b, sampler), 1));
   }

   return true;
}

static bool
legalize_image_lod(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   nir_src *src;

#define CASE(op, idx)                                                          \
   case nir_intrinsic_##op:                                                    \
   case nir_intrinsic_bindless_##op:                                           \
      src = &intr->src[idx];                                                   \
      break;

   switch (intr->intrinsic) {
      CASE(image_load, 3)
      CASE(image_sparse_load, 3)
      CASE(image_store, 4)
      CASE(image_size, 1)
   default:
      return false;
   }

#undef CASE

   if (src->ssa->bit_size == 16)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(src, nir_i2i16(b, src->ssa));
   return true;
}

static nir_def *
txs_for_image(nir_builder *b, nir_intrinsic_instr *intr,
              unsigned num_components, unsigned bit_size, bool query_samples)
{
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   nir_def *lod = query_samples ? NULL : intr->src[1].ssa;
   nir_texop op = query_samples ? nir_texop_texture_samples : nir_texop_txs;

   nir_def *res =
      nir_build_tex(b, op, .texture_handle = intr->src[0].ssa, .lod = lod,
                    .dim = dim, .is_array = nir_intrinsic_image_array(intr),
                    .can_speculate = nir_instr_can_speculate(&intr->instr));

   /* Cube images are implemented as 2D arrays, so we need to divide here. */
   if (dim == GLSL_SAMPLER_DIM_CUBE && res->num_components > 2 &&
       !query_samples) {
      nir_def *divided = nir_udiv_imm(b, nir_channel(b, res, 2), 6);
      res = nir_vector_insert_imm(b, res, divided, 2);
   }

   return res;
}

static nir_def *
image_texel_address(nir_builder *b, nir_intrinsic_instr *intr,
                    bool return_index, bool g15)
{
   /* First, calculate the address of the PBE descriptor */
   nir_def *desc_address =
      nir_load_from_texture_handle_agx(b, intr->src[0].ssa);

   nir_def *coord = intr->src[1].ssa;

   /* For atomics, we always infer the format. We only go down this path with
    * formatless intrinsics when lowering multisampled image stores, but that
    * uses the return_index path that ignores the block size.
    */
   enum pipe_format format = nir_intrinsic_format(intr);
   assert(return_index || format != PIPE_FORMAT_NONE);

   nir_def *blocksize_B = nir_imm_int(b, util_format_get_blocksize(format));

   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool layered = nir_intrinsic_image_array(intr) ||
                  (dim == GLSL_SAMPLER_DIM_CUBE) ||
                  (dim == GLSL_SAMPLER_DIM_3D);

   if (dim == GLSL_SAMPLER_DIM_BUF && return_index) {
      return nir_channel(b, coord, 0);
   } else if (dim == GLSL_SAMPLER_DIM_BUF && g15) {
      return libagx_buffer_texel_address_g15(b, desc_address, coord,
                                             blocksize_B);
   } else if (dim == GLSL_SAMPLER_DIM_BUF) {
      return libagx_buffer_texel_address(b, desc_address, coord, blocksize_B);
   } else if (g15) {
      return libagx_image_texel_address_g15(
         b, desc_address, coord, nir_u2u32(b, intr->src[2].ssa), blocksize_B,
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_1D),
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_MS), nir_imm_bool(b, layered),
         nir_imm_bool(b, return_index));
   } else {
      return libagx_image_texel_address(
         b, desc_address, coord, nir_u2u32(b, intr->src[2].ssa), blocksize_B,
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_1D),
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_MS), nir_imm_bool(b, layered),
         nir_imm_bool(b, return_index));
   }
}

static void
lower_buffer_image(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *coord_vector = intr->src[1].ssa;
   nir_def *coord = nir_channel(b, coord_vector, 0);

   /* If we're not bindless, assume we don't need an offset (GL driver) */
   if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
       intr->intrinsic == nir_intrinsic_bindless_image_sparse_load) {

      nir_def *desc = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
      coord = libagx_buffer_texture_offset(b, desc, coord);
   } else if (intr->intrinsic == nir_intrinsic_bindless_image_store) {
      nir_def *desc = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
      coord = libagx_buffer_image_offset(b, desc, coord);
   }

   /* Lower the buffer load/store to a 2D image load/store, matching the 2D
    * texture/PBE descriptor the driver supplies for buffer images.
    */
   nir_def *coord2d = coords_for_buffer_texture(b, coord);
   nir_src_rewrite(&intr->src[1], nir_pad_vector(b, coord2d, 4));
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
}

static void
lower_1d_image(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *coord = intr->src[1].ssa;
   bool is_array = nir_intrinsic_image_array(intr);
   nir_def *zero = nir_imm_intN_t(b, 0, coord->bit_size);

   if (is_array) {
      assert(coord->num_components >= 2);
      coord =
         nir_vec3(b, nir_channel(b, coord, 0), zero, nir_channel(b, coord, 1));
   } else {
      assert(coord->num_components >= 1);
      coord = nir_vec2(b, coord, zero);
   }

   nir_src_rewrite(&intr->src[1], nir_pad_vector(b, coord, 4));
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
}

/*
 * Just like for txf, we need special handling around layers (and LODs, but we
 * don't support mipmapped images yet) for robust image_loads. See
 * libagx_lower_txf_robustness for more info.
 */
static bool
lower_image_load_robustness(nir_builder *b, nir_intrinsic_instr *intr)
{
   if (nir_intrinsic_access(intr) & ACCESS_IN_BOUNDS)
      return false;

   /* We only need to worry about array-like loads */
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   if (!nir_intrinsic_image_array(intr) && dim != GLSL_SAMPLER_DIM_CUBE)
      return false;

   /* Determine the coordinate component of the layer. Cubes and cube arrays
    * keep their array in their last non-array coordinate component, other
    * arrays are immediately after.
    */
   unsigned lidx = glsl_get_sampler_dim_coordinate_components(dim);
   if (dim == GLSL_SAMPLER_DIM_CUBE)
      lidx--;

   nir_def *coord = intr->src[1].ssa;
   nir_def *lod = nir_undef(b, 1, 16);
   nir_def *layer = nir_channel(b, coord, lidx);

   /* image_load is effectively the same as txf, reuse the txf lower */
   nir_def *replaced = libagx_lower_txf_robustness(
      b, nir_load_from_texture_handle_agx(b, intr->src[0].ssa),
      nir_imm_bool(b, false /* lower LOD */), lod,
      nir_imm_bool(b, false /* lower min LOD */), lod,
      nir_imm_bool(b, true /* lower layer */), layer, nir_channel(b, coord, 0));

   nir_src_rewrite(&intr->src[1], nir_vector_insert_imm(b, coord, replaced, 0));
   return true;
}

static bool
lower_images(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const bool *g15 = data;

   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_image_load:
   case nir_intrinsic_image_store:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_sparse_load:
   case nir_intrinsic_bindless_image_store: {
      /* Legalize MSAA index */
      nir_src_rewrite(&intr->src[2], nir_u2u16(b, intr->src[2].ssa));

      if (intr->intrinsic == nir_intrinsic_image_load ||
          intr->intrinsic == nir_intrinsic_bindless_image_load ||
          intr->intrinsic == nir_intrinsic_bindless_image_sparse_load) {
         lower_image_load_robustness(b, intr);
      }

      switch (nir_intrinsic_image_dim(intr)) {
      case GLSL_SAMPLER_DIM_1D:
         lower_1d_image(b, intr);
         return true;

      case GLSL_SAMPLER_DIM_BUF:
         lower_buffer_image(b, intr);
         return true;

      default:
         return true;
      }
   }

   case nir_intrinsic_bindless_image_size:
   case nir_intrinsic_bindless_image_samples:
      nir_def_rewrite_uses(
         &intr->def,
         txs_for_image(
            b, intr, intr->def.num_components, intr->def.bit_size,
            intr->intrinsic == nir_intrinsic_bindless_image_samples));
      return true;

   case nir_intrinsic_bindless_image_texel_address:
      nir_def_rewrite_uses(&intr->def,
                           image_texel_address(b, intr, false, *g15));
      return true;

   case nir_intrinsic_is_sparse_texels_resident:
      /* Residency information is in bit 0, so we need to mask. Unclear what's
       * in the upper bits. For now, let's match the blob.
       */
      nir_def_replace(
         &intr->def,
         nir_ieq_imm(b, nir_iand_imm(b, intr->src[0].ssa, 1), AGX_RESIDENT));
      return true;

   case nir_intrinsic_sparse_residency_code_and:
      /* ior because residency codes are inverted from NIR */
      nir_def_replace(&intr->def,
                      nir_ior(b, intr->src[0].ssa, intr->src[1].ssa));
      return true;

   case nir_intrinsic_image_size:
   case nir_intrinsic_image_texel_address:
      UNREACHABLE("should've been lowered");

   default:
      return false;
   }
}

/*
 * Map out-of-bounds storage texel buffer accesses and multisampled image stores
 * to -1 indices, which will become an out-of-bounds hardware access. This gives
 * cheap robustness2.
 */
static bool
lower_robustness(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_store:
      break;
   default:
      return false;
   }

   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool array = nir_intrinsic_image_array(intr);
   unsigned size_components = nir_image_intrinsic_coord_components(intr);

   nir_def *deref = intr->src[0].ssa;
   nir_def *coord = intr->src[1].ssa;

   if (dim != GLSL_SAMPLER_DIM_BUF &&
       !(dim == GLSL_SAMPLER_DIM_MS &&
         intr->intrinsic == nir_intrinsic_image_deref_store))
      return false;

   /* Bounds check the coordinate */
   nir_def *size =
      nir_image_deref_size(b, size_components, 32, deref, nir_imm_int(b, 0),
                           .image_dim = dim, .image_array = array);
   nir_def *oob = nir_bany(b, nir_uge(b, coord, size));

   /* Bounds check the sample */
   if (dim == GLSL_SAMPLER_DIM_MS) {
      nir_def *samples = nir_image_deref_samples(b, 32, deref, .image_dim = dim,
                                                 .image_array = array);

      oob = nir_ior(b, oob, nir_uge(b, intr->src[2].ssa, samples));
   }

   /* Replace the last coordinate component with a large coordinate for
    * out-of-bounds. We pick 0xFFF0 as it fits in 16-bit, and it is not signed
    * as 32-bit so we won't get in-bounds coordinates for arrays due to two's
    * complement wraparound. Additionally it still meets this requirement after
    * adding 0xF, the maximum tail offset.
    *
    * This ensures the resulting hardware coordinate is definitely
    * out-of-bounds, giving hardware-level robustness2 behaviour.
    */
   unsigned c = size_components - 1;
   nir_def *r =
      nir_bcsel(b, oob, nir_imm_int(b, 0xFFF0), nir_channel(b, coord, c));

   nir_src_rewrite(&intr->src[1], nir_vector_insert_imm(b, coord, r, c));
   return true;
}

/*
 * Early texture lowering passes, called by the driver before lowering
 * descriptor bindings. That means these passes operate on texture derefs. The
 * purpose is to make descriptor crawls explicit in the NIR, so that the driver
 * can accurately lower descriptors after this pass but before calling
 * the full agx_nir_lower_texture.
 */
bool
agx_nir_lower_texture_early(nir_shader *s, bool support_lod_bias)
{
   bool progress = false;

   NIR_PASS(progress, s, nir_shader_intrinsics_pass, lower_robustness,
            nir_metadata_control_flow, NULL);

   nir_lower_tex_options lower_tex_options = {
      .lower_txp = ~0,
      .lower_invalid_implicit_lod = true,
      .lower_tg4_offsets = true,
      .lower_index_to_offset = true,
      .lower_sampler_lod_bias = support_lod_bias,

      /* Unclear if/how mipmapped 1D textures work in the hardware. */
      .lower_1d = true,

      /* XXX: Metal seems to handle just like 3D txd, so why doesn't it work?
       * TODO: Stop using this lowering
       */
      .lower_txd_cube_map = true,
   };

   NIR_PASS(progress, s, nir_lower_tex, &lower_tex_options);
   return progress;
}

/*
 * Select the face of a cube direction and project the direction onto it, per
 * the GL rules (GL 4.6 table 8.19): returns the face-local (s, t) in [0, 1] and
 * the face index in *face. On a tie the major axis is z, then y, then x.
 */
static nir_def *
cube_face_coord(nir_builder *b, nir_def *dir, nir_def **face)
{
   nir_def *x = nir_channel(b, dir, 0);
   nir_def *y = nir_channel(b, dir, 1);
   nir_def *z = nir_channel(b, dir, 2);
   nir_def *ax = nir_fabs(b, x), *ay = nir_fabs(b, y), *az = nir_fabs(b, z);

   nir_def *is_z = nir_fge(b, az, nir_fmax(b, ax, ay));
   nir_def *is_y = nir_iand(b, nir_inot(b, is_z), nir_fge(b, ay, ax));

   nir_def *ma = nir_bcsel(b, is_z, z, nir_bcsel(b, is_y, y, x));
   nir_def *neg = nir_flt(b, ma, nir_imm_float(b, 0.0));

   /* +x (-z, -y), -x (z, -y), +y (x, z), -y (x, -z), +z (x, -y), -z (-x, -y) */
   nir_def *sc = nir_bcsel(b, nir_ior(b, is_z, is_y), x, nir_fneg(b, z));
   sc = nir_bcsel(b, nir_iand(b, neg, nir_inot(b, is_y)), nir_fneg(b, sc), sc);

   nir_def *tc_y = nir_bcsel(b, neg, nir_fneg(b, z), z);
   nir_def *tc = nir_bcsel(b, is_y, tc_y, nir_fneg(b, y));

   nir_def *half_rcp = nir_fmul_imm(b, nir_frcp(b, nir_fabs(b, ma)), 0.5);
   nir_def *s = nir_ffma(b, sc, half_rcp, nir_imm_float(b, 0.5));
   nir_def *t = nir_ffma(b, tc, half_rcp, nir_imm_float(b, 0.5));

   nir_def *axis =
      nir_bcsel(b, is_z, nir_imm_int(b, 4),
                nir_bcsel(b, is_y, nir_imm_int(b, 2), nir_imm_int(b, 0)));
   *face = nir_iadd(b, axis, nir_b2i32(b, neg));
   return nir_vec2(b, s, t);
}

static bool
lower_cube_g15(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE)
      return false;

   /* txd is a txl here (lower_txd_cube_map); LOD and size queries keep the
    * cube */
   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
       tex->op != nir_texop_txl && tex->op != nir_texop_tg4)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *coord = nir_f2fN(b, nir_steal_tex_src(tex, nir_tex_src_coord), 32);
   nir_def *face;
   nir_def *st = cube_face_coord(b, coord, &face);

   nir_def *cube = nir_imm_int(b, 0);
   if (tex->is_array) {
      cube = nir_fround_even(b, nir_channel(b, coord, 3));
      cube = nir_f2u32(b, nir_fmax_imm(b, cube, 0.0));
   }

   /* The LOD word: the cube at bits 0:10, the face at 12 */
   nir_def *word =
      nir_ior(b, nir_umin_imm(b, cube, 0x7ff), nir_ishl_imm(b, face, 12));

   tex->is_array = true;
   tex->coord_components = 4;
   tex->backend_flags |= AGX_TEXTURE_FLAG_NO_CLAMP;
   nir_tex_instr_add_src(
      tex, nir_tex_src_coord,
      nir_vec4(b, nir_channel(b, st, 0), nir_channel(b, st, 1),
               nir_imm_float(b, 0), nir_u2f32(b, word)));
   return true;
}

/*
 * G15 samples every cube in the hardware's cube array mode, whose descriptor
 * agx_pack_texture_g15 packs (the depth counts cubes): the face-local (s, t)
 * in cr0 and cr1 and the LOD word in cr3 with the cube at bits 0:10 and the
 * face at 12, as libLLVM 32023's cube array read sequence packs them
 * (sub_25092FC54: code 4 at 0, code 10 at 12); a cube is cube 0. The
 * direction is projected onto its face here, as Apple's compiler does for
 * cubedir. The hardware filters across the faces (seamless).
 */
/*
 * G15 has no known minimum-LOD form for a cube read: a fragment shader's
 * implicit-LOD cube sample with a minimum LOD becomes a sample with its
 * direction's derivatives, scaled by 2^bias, which lower_txd_cube_map turns
 * into an explicit LOD clamped to the minimum (dEQP-VK textureclamp cubes).
 */
static bool
cube_min_lod_to_txd(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb) ||
       nir_tex_instr_src_index(tex, nir_tex_src_min_lod) < 0 ||
       !nir_shader_supports_implicit_lod(b->shader))
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *dir = nir_trim_vector(b, nir_get_tex_src(tex, nir_tex_src_coord), 3);
   nir_def *dx = nir_ddx(b, dir), *dy = nir_ddy(b, dir);

   nir_def *bias = nir_steal_tex_src(tex, nir_tex_src_bias);
   if (bias) {
      nir_def *scale = nir_fexp2(b, nir_f2fN(b, bias, dir->bit_size));
      dx = nir_fmul(b, dx, scale);
      dy = nir_fmul(b, dy, scale);
   }

   nir_tex_instr_add_src(tex, nir_tex_src_ddx, dx);
   nir_tex_instr_add_src(tex, nir_tex_src_ddy, dy);
   tex->op = nir_texop_txd;
   return true;
}

bool
agx_nir_lower_cube_g15(nir_shader *s)
{
   bool progress = false;

   if (nir_shader_tex_pass(s, cube_min_lod_to_txd, nir_metadata_control_flow,
                           NULL)) {
      nir_lower_tex_options options = {.lower_txd_cube_map = true};
      NIR_PASS(_, s, nir_lower_tex, &options);
      progress = true;
   }

   progress |= nir_shader_tex_pass(s, lower_cube_g15,
                                   nir_metadata_control_flow, NULL);
   return progress;
}

/* g15: image atomics crawl G15's PBE (libagx_image_texel_address_g15);
 * g15_queries: size, level and sample queries read G15's texture descriptor
 */
static bool
lower_texture(nir_shader *s, bool g15, bool g15_queries)
{
   bool progress = false;

   nir_tex_src_type_constraints tex_constraints = {
      [nir_tex_src_lod] = {true, 16},
      [nir_tex_src_bias] = {true, 16},
      [nir_tex_src_ms_index] = {true, 16},
      [nir_tex_src_min_lod] = {true, 16},
      [nir_tex_src_texture_offset] = {true, 16},
      [nir_tex_src_sampler_offset] = {true, 16},
   };

   /* Insert fences before lowering image atomics, since image atomics need
    * different fencing than other image operations.
    */
   NIR_PASS(progress, s, nir_shader_intrinsics_pass, fence_image,
            nir_metadata_control_flow, NULL);

   NIR_PASS(progress, s, nir_lower_image_atomics_to_global, NULL, NULL);

   NIR_PASS(progress, s, nir_shader_intrinsics_pass, legalize_image_lod,
            nir_metadata_control_flow, NULL);
   NIR_PASS(progress, s, nir_shader_intrinsics_pass, lower_images,
            nir_metadata_control_flow, &g15);
   NIR_PASS(progress, s, nir_legalize_16bit_sampler_srcs, tex_constraints);

   /* Fold constants after nir_legalize_16bit_sampler_srcs so we can detect 0 in
    * lower_regular_texture. This is required for correctness.
    */
   NIR_PASS(progress, s, nir_opt_constant_folding);

   /* Lower texture sources after legalizing types (as the lowering depends on
    * 16-bit multisample indices) but before lowering queries (as the lowering
    * generates txs for array textures).
    */
   NIR_PASS(progress, s, nir_shader_instructions_pass, lower_regular_texture,
            nir_metadata_none, NULL);
   NIR_PASS(progress, s, nir_shader_instructions_pass, lower_tex_crawl,
            nir_metadata_control_flow, &g15_queries);

   return progress;
}

bool
agx_nir_lower_texture_gen(nir_shader *s, bool g15)
{
   return lower_texture(s, g15, g15);
}

bool
agx_nir_lower_texture(nir_shader *s)
{
   return lower_texture(s, false, false);
}

static bool
lower_multisampled_store(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const bool *g15 = data;
   b->cursor = nir_before_instr(&intr->instr);

   if (intr->intrinsic != nir_intrinsic_bindless_image_store)
      return false;

   if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_MS)
      return false;

   nir_def *index_px = nir_u2u32(b, image_texel_address(b, intr, true, *g15));
   nir_def *coord2d = coords_for_buffer_texture(b, index_px);

   nir_src_rewrite(&intr->src[1], nir_pad_vector(b, coord2d, 4));
   nir_src_rewrite(&intr->src[2], nir_imm_int(b, 0));
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
   nir_intrinsic_set_image_array(intr, false);
   return true;
}

bool
agx_nir_lower_multisampled_image_store(nir_shader *s, bool g15)
{
   return nir_shader_intrinsics_pass(s, lower_multisampled_store,
                                     nir_metadata_control_flow, &g15);
}

/*
 * Given a non-bindless instruction, return whether agx_nir_lower_texture will
 * lower it to something involving a descriptor crawl. This requires the driver
 * to lower the instruction to bindless before calling agx_nir_lower_texture.
 * The implementation just enumerates the cases handled in this file.
 */
bool
agx_nir_needs_texture_crawl(nir_instr *instr)
{
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

      switch (intr->intrinsic) {
      /* Queries, atomics always become a crawl */
      case nir_intrinsic_image_size:
      case nir_intrinsic_image_deref_size:
      case nir_intrinsic_image_samples:
      case nir_intrinsic_image_deref_samples:
      case nir_intrinsic_image_atomic:
      case nir_intrinsic_image_deref_atomic:
      case nir_intrinsic_image_atomic_swap:
      case nir_intrinsic_image_deref_atomic_swap:
         return true;

      /* Multisampled stores need a crawl, others do not */
      case nir_intrinsic_image_store:
      case nir_intrinsic_image_deref_store:
         return nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_MS;

      /* Array loads need a crawl, other load do not */
      case nir_intrinsic_image_load:
         return nir_intrinsic_image_array(intr) ||
                nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_CUBE;

      default:
         return false;
      }
   } else if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      /* Array textures get clamped to their size via txs */
      if (tex->is_array && !(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP))
         return true;

      switch (tex->op) {
      /* Queries always become a crawl */
      case nir_texop_txs:
      case nir_texop_texture_samples:
      case nir_texop_query_levels:
         return true;

      /* Buffer textures need their format read and txf needs its LOD/layer
       * clamped.  Buffer textures are only read through txf.
       */
      case nir_texop_txf:
      case nir_texop_txf_ms:
         return has_nonzero_lod(tex) || tex->is_array ||
                tex->sampler_dim == GLSL_SAMPLER_DIM_BUF;

      default:
         return false;
      }
   }

   return false;
}
