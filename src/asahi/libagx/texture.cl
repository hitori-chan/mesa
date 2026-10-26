/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "asahi/lib/agx_abi.h"
#include "compiler/libcl/libcl.h"
#include "libagx_intrinsics.h"
#include <agx_pack.h>

/* The fields the queries read. G15's descriptor keeps the size fields where
 * G13's does, the levels in the extended word (set with several levels or a
 * first level) and the sample count at bits 56:57, where G13 keeps its
 * levels.
 */
struct texture_query {
   uint64_t address;
   uint3 size;
   uint depth_2d_array;
   uint first_level, last_level;
   uint samples;
};

static struct texture_query
unpack_texture_query(constant struct agx_texture_packed *ptr, bool g15)
{
   struct texture_query q;

   if (g15) {
      agx_unpack(NULL, ptr, TEXTURE_G15, d);
      q.address = d.address;
      q.size = (uint3)(d.width, d.height, d.depth);
      q.depth_2d_array = d.depth;
      q.first_level = d.extended ? d.first_level : 0;
      q.last_level = d.extended ? d.last_level : 0;
      q.samples = d.samples;
   } else {
      agx_unpack(NULL, ptr, TEXTURE, d);
      q.address = d.address;
      q.size = (uint3)(d.width, d.height, d.depth);

      /* Linear 2D arrays are special.
       *
       * TODO: Optimize this, since linear 2D arrays aren't needed for APIs and
       * this just gets used internally for blits.
       */
      q.depth_2d_array =
         d.layout == AGX_LAYOUT_LINEAR ? d.depth_linear : d.depth;
      q.first_level = d.first_level;
      q.last_level = d.last_level;
      q.samples = d.samples;
   }

   return q;
}

uint3
libagx_txs(constant struct agx_texture_packed *ptr, uint16_t lod,
           unsigned nr_comps, bool is_buffer, bool is_1d, bool is_2d,
           bool is_cube, bool is_array, bool g15)
{
   struct texture_query d = unpack_texture_query(ptr, g15);

   /* From the Vulkan spec:
    *
    *    OpImageQuery*...  return 0 if the bound descriptor is a null descriptor
    */
   if (d.address == AGX_ZERO_PAGE_ADDRESS)
      return 0;

   /* Buffer textures are lowered to 2D so the original size is irrecoverable.
    * Instead, we stash it in the software-defined section, G13's bytes 16-23
    * on G15 too.
    */
   if (is_buffer) {
      agx_unpack(NULL, ptr, TEXTURE, sw);
      return sw.buffer_size_sw;
   }

   /* Load standard dimensions */
   uint3 size = d.size;
   lod += d.first_level;

   if (is_2d && is_array)
      size.z = d.depth_2d_array;

   /* 1D Arrays have their second component as the layer count */
   if (is_1d && is_array)
      size.y = size.z;

   /* Adjust for LOD, do not adjust array size */
   size.x = max(size.x >> lod, 1u);

   if (nr_comps - (uint)is_array >= 2)
      size.y = max(size.y >> lod, 1u);

   if (nr_comps - (uint)is_array >= 3)
      size.z = max(size.z >> lod, 1u);

   /* Cube maps have equal width and height, we save some instructions by only
    * reading one. Dead code elimination will remove the redundant instructions.
    */
   if (is_cube)
      size.y = size.x;

   return size;
}

uint
libagx_texture_samples(constant struct agx_texture_packed *ptr, bool g15)
{
   struct texture_query d = unpack_texture_query(ptr, g15);

   /* As above */
   if (d.address == AGX_ZERO_PAGE_ADDRESS)
      return 0;

   /* We may assume the input is multisampled, so just check the samples */
   return (d.samples == AGX_SAMPLE_COUNT_2) ? 2 : 4;
}

uint
libagx_texture_levels(constant struct agx_texture_packed *ptr, bool g15)
{
   struct texture_query d = unpack_texture_query(ptr, g15);

   if (d.address == AGX_ZERO_PAGE_ADDRESS)
      return 0;
   else
      return (d.last_level - d.first_level) + 1;
}

/*
 * Fix robustness behaviour of txf with out-of-bounds LOD. The hardware
 * returns the correct out-of-bounds colour for out-of-bounds coordinates,
 * just not LODs. So translate out-of-bounds LOD into an out-of-bounds
 * coordinate to get correct behaviour in 1 instruction.
 *
 * Returns the fixed X-coordinate.
 *
 * TODO: This looks like it might be an erratum workaround on G13 (Apple does
 * it), maybe check if G15 is affected.
 */
uint
libagx_lower_txf_robustness(constant struct agx_texture_packed *ptr,
                            bool check_lod, ushort lod, bool check_min_lod,
                            ushort min_lod, bool check_layer, uint layer,
                            uint x)
{
   agx_unpack(NULL, ptr, TEXTURE, d);

   bool valid = true;

   if (check_lod)
      valid &= lod <= (d.last_level - d.first_level);

   if (check_min_lod)
      valid &= lod >= min_lod;

   if (check_layer) {
      bool linear = (d.layout == AGX_LAYOUT_LINEAR);
      valid &= layer < (linear ? d.depth_linear : d.depth);
   }

   /* The maximum tail offset is 0xF so by returning 0xFFF0 for out-of-bounds we
    * stay under 0xFFFF and keep robustness after offsetting.
    */
   return valid ? x : 0xFFF0;
}

uint32_t
libagx_twiddle_coordinates(ushort2 coord, uint16_t tile_w_px,
                           uint16_t tile_h_px, uint32_t aligned_width_px)
{
   /* Modulo by the tile width/height to get the offsets within the tile */
   ushort2 tile_mask_vec = (ushort2)(tile_w_px - 1, tile_h_px - 1);
   uint32_t tile_mask = upsample(tile_mask_vec.y, tile_mask_vec.x);
   uint32_t coord_xy = upsample(coord.y, coord.x);
   ushort2 offs_px = as_ushort2(coord_xy & tile_mask);
   uint32_t offset_within_tile_px = nir_interleave(offs_px.x, offs_px.y);

   /* Get the coordinates of the corner of the tile */
   ushort2 tile_px = as_ushort2(coord_xy & ~tile_mask);

   /* tile row start (px) =
    *   (y // tile height) * (# of tiles/row) * (# of pix/tile) =
    *   align_down(y, tile height) / tile height * width_tl *tile width *
    *        tile height =
    *   align_down(y, tile height) * width_tl * tile width
    */
   uint32_t tile_row_start_px = tile_px.y * aligned_width_px;

   /* tile column start (px) =
    *   (x // tile width) * (# of pix/tile) =
    *   align_down(x, tile width) / tile width * tile width * tile height =
    *   align_down(x, tile width) * tile height
    */
   uint32_t tile_col_start_px = tile_px.x * tile_h_px;

   /* Get the total offset */
   return tile_row_start_px + tile_col_start_px + offset_within_tile_px;
}

/* The fields the texel address crawls read. G15's PBE holds the image's VA
 * (the one its texture descriptor holds; a buffer's offset is in it) and the
 * level in the extended word; the software words sit in the slot's fourth
 * word (PBE G15 software). A multisampled image is bound as a linear 2D view
 * of its samples, as on G13, with its own software words (PBE G15 software
 * MSAA). Word 4, which the hardware ignores in a buffer's PBE, holds the
 * elements Honeykrisp's view starts after its 16 B aligned address (Gallium's
 * is 0).
 */
struct pbe_query {
   uint64_t buffer;
   uint width, level;
   uint tile_width, tile_height, aligned_width_msaa;
   uint sample_count_log2, sample_size_log2;
   uint layer_stride, level_offset, buffer_offset;
};

static struct pbe_query
unpack_pbe_query(constant const struct agx_pbe_packed *ptr, bool g15)
{
   struct pbe_query q;

   if (g15) {
      constant uint8_t *sw_ptr = (constant uint8_t *)ptr + 24;
      agx_unpack(NULL, ptr, PBE_G15, d);
      agx_unpack(NULL, sw_ptr, PBE_G15_SOFTWARE, sw);
      agx_unpack(NULL, sw_ptr, PBE_G15_SOFTWARE_MSAA, ms);
      q.buffer = d.buffer;
      q.width = d.width;
      q.level = d.extended ? d.level : 0;
      q.tile_width = sw.tile_width;
      q.tile_height = sw.tile_height;
      q.aligned_width_msaa = ms.aligned_width;
      q.sample_count_log2 = ms.sample_count_log2;
      q.sample_size_log2 = ms.sample_size_log2;
      q.layer_stride = sw.layer_stride;
      q.level_offset = sw.level_offset;
      q.buffer_offset = ((constant uint32_t *)ptr)[4];
   } else {
      agx_unpack(NULL, ptr, PBE, d);
      q.buffer = d.buffer;
      q.width = d.width;
      q.level = d.level;
      q.tile_width = d.tile_width_sw;
      q.tile_height = d.tile_height_sw;
      q.aligned_width_msaa = d.aligned_width_msaa_sw;
      q.sample_count_log2 = d.sample_count_log2_sw;
      q.sample_size_log2 = 0;
      q.layer_stride = d.layer_stride_sw;
      q.level_offset = d.level_offset_sw;
      q.buffer_offset = d.buffer_offset_sw;
   }

   return q;
}

uint64_t
libagx_image_texel_address(constant const struct agx_pbe_packed *ptr,
                           uint4 coord, uint sample_idx,
                           uint bytes_per_sample_B, bool is_1d, bool is_msaa,
                           bool is_layered, bool return_index, bool g15)
{
   struct pbe_query d = unpack_pbe_query(ptr, g15);

   /* We do not allow atomics on linear 2D or linear 2D arrays, as there are no
    * known use cases. So we're twiddled in this path, unless we're handling a
    * 1D image which will be always linear, even if it uses a twiddled layout
    * degrading to linear-equivalent 1x1 tiles. (1D uses this path, not the
    * buffer path, for 1D arrays.)
    */
   uint total_px;
   if (is_1d) {
      total_px = coord.x;
   } else {
      uint aligned_width_px;
      if (is_msaa) {
         aligned_width_px = d.aligned_width_msaa;
      } else {
         uint width_px = max(d.width >> d.level, 1u);
         aligned_width_px = align(width_px, d.tile_width);
      }

      total_px =
         libagx_twiddle_coordinates(convert_ushort2(coord.xy), d.tile_width,
                                    d.tile_height, aligned_width_px);
   }

   uint samples_log2 = is_msaa ? d.sample_count_log2 : 0;

   /* G15: a formatless store passes a block size of 0 and steps a layer by
    * the view's texel size from the software words */
   if (is_layered) {
      uint layer_sa = bytes_per_sample_B ? d.layer_stride / bytes_per_sample_B
                                         : d.layer_stride >> d.sample_size_log2;
      total_px += coord[is_1d ? 1 : 2] * (layer_sa >> samples_log2);
   }

   uint total_sa = (total_px << samples_log2) + sample_idx;

   if (return_index)
      return total_sa;
   else
      return (d.buffer + (is_msaa ? 0 : d.level_offset)) +
             (uint64_t)total_sa * bytes_per_sample_B;
}

uint64_t
libagx_buffer_texel_address(constant const struct agx_pbe_packed *ptr,
                            uint4 coord, uint bytes_per_pixel_B, bool g15)
{
   struct pbe_query d = unpack_pbe_query(ptr, g15);

   uint32_t x_el = d.buffer_offset + coord.x;
   return d.buffer + (uint64_t)(x_el * bytes_per_pixel_B);
}

/* Buffer texture lowerings */
bool
libagx_texture_is_rgb32(constant struct agx_texture_packed *ptr)
{
   agx_unpack(NULL, ptr, TEXTURE, d);
   return d.channels == AGX_CHANNELS_R32G32B32_EMULATED;
}

uint4
libagx_texture_load_rgb32(constant struct agx_texture_packed *ptr, uint coord,
                          bool is_float)
{
   agx_unpack(NULL, ptr, TEXTURE, d);

   /* This is carefully written to let us do the * 3 with a 32-bit operation but
    * still use the free 64-bit add-extend-shift for the rest.
    */
   uint64_t addr = d.address + ((uint64_t)(coord * 3)) * 4;
   constant uint3 *data = (constant uint3 *)addr;

   return (uint4)(*data, is_float ? as_uint(1.0f) : 1);
}

uint
libagx_buffer_texture_offset(constant struct agx_texture_packed *ptr, uint x)
{
   agx_unpack(NULL, ptr, TEXTURE, d);

   return x + d.buffer_offset_sw;
}

uint
libagx_buffer_image_offset(constant struct agx_pbe_packed *ptr, uint x)
{
   agx_unpack(NULL, ptr, PBE, d);

   return x + d.buffer_offset_sw;
}
