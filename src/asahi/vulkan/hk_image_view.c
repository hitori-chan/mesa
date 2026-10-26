/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_image_view.h"
#include "util/format/u_format.h"
#include "vulkan/vulkan_core.h"

#include "agx_helpers.h"
#include "agx_nir_texture.h"
#include "agx_pack.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_physical_device.h"

#include "layout.h"
#include "vk_format.h"
#include "vk_meta.h"

enum hk_desc_usage {
   HK_DESC_USAGE_SAMPLED,
   HK_DESC_USAGE_STORAGE,
   HK_DESC_USAGE_INPUT,
   HK_DESC_USAGE_BG_EOT,
   HK_DESC_USAGE_LAYERED_BG_EOT,
   HK_DESC_USAGE_EMRT,
};

static bool
hk_image_view_type_is_array(VkImageViewType view_type)
{
   switch (view_type) {
   case VK_IMAGE_VIEW_TYPE_1D:
   case VK_IMAGE_VIEW_TYPE_2D:
   case VK_IMAGE_VIEW_TYPE_3D:
   case VK_IMAGE_VIEW_TYPE_CUBE:
      return false;

   case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
   case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
      return true;

   default:
      UNREACHABLE("Invalid image view type");
   }
}

static enum agx_texture_dimension
translate_image_view_type(VkImageViewType view_type, bool msaa, bool layered,
                          enum hk_desc_usage usage)
{
   if (usage == HK_DESC_USAGE_EMRT || usage == HK_DESC_USAGE_INPUT ||
       (usage == HK_DESC_USAGE_LAYERED_BG_EOT && layered)) {
      return msaa ? AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED
                  : AGX_TEXTURE_DIMENSION_2D_ARRAY;
   }

   /* For background/EOT, we ignore the application-provided view type */
   if (usage == HK_DESC_USAGE_BG_EOT || usage == HK_DESC_USAGE_LAYERED_BG_EOT) {
      return msaa ? AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED
                  : AGX_TEXTURE_DIMENSION_2D;
   }

   bool cubes_to_2d = usage != HK_DESC_USAGE_SAMPLED;

   switch (view_type) {
   case VK_IMAGE_VIEW_TYPE_1D:
   case VK_IMAGE_VIEW_TYPE_2D:
      return msaa ? AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED
                  : AGX_TEXTURE_DIMENSION_2D;

   case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
      return msaa ? AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED
                  : AGX_TEXTURE_DIMENSION_2D_ARRAY;

   case VK_IMAGE_VIEW_TYPE_3D:
      assert(!msaa);
      return AGX_TEXTURE_DIMENSION_3D;

   case VK_IMAGE_VIEW_TYPE_CUBE:
      assert(!msaa);
      return cubes_to_2d ? AGX_TEXTURE_DIMENSION_2D_ARRAY
                         : AGX_TEXTURE_DIMENSION_CUBE;

   case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
      assert(!msaa);
      return cubes_to_2d ? AGX_TEXTURE_DIMENSION_2D_ARRAY
                         : AGX_TEXTURE_DIMENSION_CUBE_ARRAY;

   default:
      UNREACHABLE("Invalid image view type");
   }
}

static enum pipe_swizzle
vk_swizzle_to_pipe(VkComponentSwizzle swizzle)
{
   switch (swizzle) {
   case VK_COMPONENT_SWIZZLE_R:
      return PIPE_SWIZZLE_X;
   case VK_COMPONENT_SWIZZLE_G:
      return PIPE_SWIZZLE_Y;
   case VK_COMPONENT_SWIZZLE_B:
      return PIPE_SWIZZLE_Z;
   case VK_COMPONENT_SWIZZLE_A:
      return PIPE_SWIZZLE_W;
   case VK_COMPONENT_SWIZZLE_ONE:
      return PIPE_SWIZZLE_1;
   case VK_COMPONENT_SWIZZLE_ZERO:
      return PIPE_SWIZZLE_0;
   default:
      UNREACHABLE("Invalid component swizzle");
   }
}

static enum pipe_format
get_stencil_format(enum pipe_format format)
{
   /* Note: R8 is not a stencil format, but our copy code will internally cast
    * between R8 and S8. It's simplest to handle here.
    */
   switch (format) {
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_S8_UINT:
      return PIPE_FORMAT_S8_UINT;
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
      return PIPE_FORMAT_X24S8_UINT;
   case PIPE_FORMAT_S8_UINT_Z24_UNORM:
      return PIPE_FORMAT_S8X24_UINT;
   case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
      return PIPE_FORMAT_X32_S8X24_UINT;
   default:
      UNREACHABLE("Unsupported depth/stencil format");
   }
}

struct hk_3d {
   unsigned x, y, z;
};

static struct hk_3d
view_denominator(struct hk_image_view *view)
{
   enum pipe_format view_format = hk_format_to_pipe_format(view->vk.format);
   enum pipe_format img_format =
      hk_format_to_pipe_format(view->vk.image->format);

   if (util_format_is_compressed(view_format)) {
      /*
       * We can do an uncompressed view of a compressed image but not the other
       * way around.
       */
      assert(util_format_is_compressed(img_format));
      assert(util_format_get_blockwidth(img_format) ==
             util_format_get_blockwidth(view_format));
      assert(util_format_get_blockheight(img_format) ==
             util_format_get_blockheight(view_format));
      assert(util_format_get_blockdepth(img_format) ==
             util_format_get_blockdepth(view_format));

      return (struct hk_3d){1, 1, 1};
   }

   if (!util_format_is_compressed(img_format)) {
      /* Both formats uncompressed */
      return (struct hk_3d){1, 1, 1};
   }

   /* Else, img is compressed but view is not */
   return (struct hk_3d){
      util_format_get_blockwidth(img_format),
      util_format_get_blockheight(img_format),
      util_format_get_blockdepth(img_format),
   };
}

enum pipe_format
hk_image_view_plane_format(struct hk_image_view *view, unsigned view_plane)
{
   const struct vk_format_ycbcr_info *ycbcr_info =
      vk_format_get_ycbcr_info(view->vk.format);

   assert(ycbcr_info || view_plane == 0);
   VkFormat plane_format =
      ycbcr_info ? ycbcr_info->planes[view_plane].format : view->vk.format;

   enum pipe_format p_format = hk_format_to_pipe_format(plane_format);
   if (view->vk.aspects == VK_IMAGE_ASPECT_STENCIL_BIT)
      p_format = get_stencil_format(p_format);

   return p_format;
}

/*
 * G15's texture descriptor (cmdbuf.xml's "Texture G15"), as Gallium packs it:
 * linear with one layer or twiddled, a multisampled one twiddled with
 * (samples >> 2) & 3, a compressed one with its metadata in the extended word,
 * the raw VA (texture.md). The rest fails here rather than reaching the
 * hardware misread.
 */
static void
pack_texture_g15(struct hk_image_view *view, unsigned view_plane,
                 enum hk_desc_usage usage, const uint8_t *swizzle,
                 enum pipe_format p_format, unsigned layers,
                 struct agx_texture_packed *out)
{
   struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   const uint8_t image_plane = view->planes[view_plane].image_plane;
   struct hk_image_plane *plane = &image->planes[image_plane];
   struct ail_layout *layout = &plane->layout;
   const struct util_format_description *desc =
      util_format_description(p_format);
   unsigned level = view->vk.base_mip_level;
   unsigned layer = view->vk.base_array_layer;
   struct hk_3d denom = view_denominator(view);

   bool ms = view->vk.image->samples > 1;

   if (layout->tiling == AIL_TILING_LINEAR && (layers > 1 || ms)) {
      mesa_loge("G15: a linear multisampled or linear layered texture is not "
                "supported yet");
      assert(0);
   }

   agx_pack(out, TEXTURE_G15, cfg) {
      cfg.dimension = translate_image_view_type(
         view->vk.view_type, view->vk.image->samples > 1, layers > 1, usage);

      /* A sampled cube is a cube array of one (agx_nir_lower_cube_g15); the
       * depth counts cubes */
      if (cfg.dimension == AGX_TEXTURE_DIMENSION_CUBE)
         cfg.dimension = AGX_TEXTURE_DIMENSION_CUBE_ARRAY;
      cfg.layout = agx_translate_layout(layout->tiling);
      cfg.channels = ail_pixel_format[p_format].channels;
      cfg.type = ail_pixel_format[p_format].type;
      cfg.swizzle_r = agx_channel_from_pipe(swizzle[0]);
      cfg.swizzle_g = agx_channel_from_pipe(swizzle[1]);
      cfg.swizzle_b = agx_channel_from_pipe(swizzle[2]);
      cfg.swizzle_a = agx_channel_from_pipe(swizzle[3]);

      /* The raw VA: G15 subtracts nothing */
      uint64_t base_addr = hk_image_base_address(image, image_plane);

      if (denom.x > 1) {
         assert(view->vk.level_count == 1);
         assert(view->vk.layer_count == 1);

         cfg.address = base_addr + ail_get_layer_level_B(layout, layer, level);
         cfg.width = DIV_ROUND_UP(u_minify(layout->width_px, level), denom.x);
         cfg.height = DIV_ROUND_UP(u_minify(layout->height_px, level), denom.y);
         cfg.first_level = 0;
         cfg.last_level = 0;
      } else {
         cfg.address = base_addr + ail_get_layer_offset_B(layout, layer);
         cfg.width = layout->width_px;
         cfg.height = layout->height_px;
         cfg.first_level = level;
         cfg.last_level = level + view->vk.level_count - 1;
      }

      /* A sparse resident image reads through its sparse page table, which
       * is G13's (pack_texture explains the mip tail and the level offset)
       */
      if (plane->sparse_map && level < layout->mip_tail_first_lod) {
         unsigned page = 0;
         if (denom.x > 1)
            page = ail_bytes_to_pages(layout->level_offsets_B[level]);

         cfg.mode = AGX_IMAGE_MODE_SPARSE;
         cfg.address = plane->sparse_map->va->addr +
                       ail_page_to_sparse_index_el(layout, layer, page) *
                          AIL_SPARSE_ELSIZE_B;
      }

      /* The level word comes with the mipmapped bit, as Apple's (texture.md) */
      cfg.mipmapped = layout->levels > 1;
      cfg.extended = cfg.mipmapped || cfg.first_level || cfg.last_level;
      cfg.srgb = (desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB);
      cfg.srgb_2_channel = cfg.srgb && util_format_colormask(desc) == 0x3;
      cfg.depth = layers;

      if (layout->tiling == AIL_TILING_LINEAR)
         cfg.stride = ail_get_linear_stride_B(layout, 0) - 16;
      else
         cfg.page_aligned_layers = layout->page_aligned_layers;

      /* (samples >> 2) & 3, the dimension says multisampled (texture.md) */
      if (ms)
         cfg.samples = agx_translate_sample_count(view->vk.image->samples);

      if (layout->compressed) {
         cfg.compressed = true;
         cfg.extended = true;
         cfg.acceleration_buffer = base_addr + layout->metadata_offset_B +
                                   (layer * layout->compression_layer_stride_B);
      }
   }
}

static void
pack_texture(struct hk_image_view *view, unsigned view_plane,
             enum hk_desc_usage usage, struct agx_texture_packed *out)
{
   struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   const uint8_t image_plane = view->planes[view_plane].image_plane;
   struct hk_image_plane *plane = &image->planes[image_plane];
   struct ail_layout *layout = &plane->layout;
   struct hk_device *dev = (struct hk_device *)view->vk.base.device;

   bool cubes_to_2d = usage != HK_DESC_USAGE_SAMPLED;

   unsigned level = view->vk.base_mip_level;
   unsigned layer = view->vk.base_array_layer;

   enum pipe_format p_format = hk_image_view_plane_format(view, view_plane);
   const struct util_format_description *desc =
      util_format_description(p_format);

   struct hk_3d denom = view_denominator(view);

   uint8_t format_swizzle[4] = {
      desc->swizzle[0],
      desc->swizzle[1],
      desc->swizzle[2],
      desc->swizzle[3],
   };

   /* Different APIs have different depth/stencil swizzle rules. Vulkan expects
    * R001 behaviour, override here because Mesa's format table is not that.
    */
   if (util_format_is_depth_or_stencil(p_format)) {
      format_swizzle[0] = PIPE_SWIZZLE_X;
      format_swizzle[1] = PIPE_SWIZZLE_0;
      format_swizzle[2] = PIPE_SWIZZLE_0;
      format_swizzle[3] = PIPE_SWIZZLE_1;
   }

   /* We only have a single swizzle for the user swizzle and the format
    * fixup, so compose them now.
    */
   uint8_t out_swizzle[4];
   uint8_t view_swizzle[4] = {
      vk_swizzle_to_pipe(view->vk.swizzle.r),
      vk_swizzle_to_pipe(view->vk.swizzle.g),
      vk_swizzle_to_pipe(view->vk.swizzle.b),
      vk_swizzle_to_pipe(view->vk.swizzle.a),
   };

   unsigned layers = view->vk.layer_count;
   if (view->vk.view_type == VK_IMAGE_VIEW_TYPE_3D) {
      layers = DIV_ROUND_UP(layout->depth_px, denom.z);
   } else if (!cubes_to_2d &&
              (view->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE ||
               view->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)) {

      layers /= 6;
   }

   util_format_compose_swizzles(format_swizzle, view_swizzle, out_swizzle);

   if (dev->dev.params.gpu_generation >= 15) {
      pack_texture_g15(view, view_plane, usage, out_swizzle, p_format, layers,
                       out);
      return;
   }

   uint64_t base_addr = hk_image_base_address(image, image_plane);

   agx_pack(out, TEXTURE, cfg) {
      cfg.dimension = translate_image_view_type(
         view->vk.view_type, view->vk.image->samples > 1, layers > 1, usage);
      cfg.layout = agx_translate_layout(layout->tiling);
      cfg.channels = ail_pixel_format[p_format].channels;
      cfg.type = ail_pixel_format[p_format].type;
      cfg.srgb = util_format_is_srgb(p_format);

      cfg.swizzle_r = agx_channel_from_pipe(out_swizzle[0]);
      cfg.swizzle_g = agx_channel_from_pipe(out_swizzle[1]);
      cfg.swizzle_b = agx_channel_from_pipe(out_swizzle[2]);
      cfg.swizzle_a = agx_channel_from_pipe(out_swizzle[3]);

      if (denom.x > 1) {
         assert(view->vk.level_count == 1);
         assert(view->vk.layer_count == 1);

         cfg.address = base_addr + ail_get_layer_level_B(layout, layer, level);
         cfg.width = DIV_ROUND_UP(u_minify(layout->width_px, level), denom.x);
         cfg.height = DIV_ROUND_UP(u_minify(layout->height_px, level), denom.y);
         cfg.first_level = 0;
         cfg.last_level = 1;
      } else {
         cfg.address = base_addr + ail_get_layer_offset_B(layout, layer);
         cfg.width = layout->width_px;
         cfg.height = layout->height_px;
         cfg.first_level = level;
         cfg.last_level = level + view->vk.level_count - 1;
      }

      /* To implement sparse resident textures, the hardware texture descriptor
       * can instead point to a secondary page table controlled in userspace.
       * This allows remapping pages and - crucially - disabling unmapped pages
       * to read zero and report non-resident with shader residency queries.
       * When we have a sparse map, we need to point to it here.
       *
       * However, there's a wrinkle: when handling uncompressed views of
       * compressed images in the above code, we need to offset the image
       * address to point to the specific mip level rather than use the hardware
       * "first level" field. This ensures the layouts are consistent despite us
       * munging the image dimensions. In that case, we need to also offset the
       * sparse page table accordingly. Of course, the sparse page table is in
       * terms of pages, so this trick only works when the mip level is
       * page-aligned.
       *
       * However, if the mip level is NOT page-aligned, it is in the mip tail by
       * definition. As the mip tail is always resident, there is no need for a
       * sparse page table. So either:
       *
       * 1. We are in the mip tail and don't need a sparse map, or
       * 2. We are not but the level is page-aligned in the sparse map.
       *
       * Either way we're okay.
       */
      if (plane->sparse_map && level < layout->mip_tail_first_lod) {
         unsigned page = 0;
         if (denom.x > 1) {
            page = ail_bytes_to_pages(layout->level_offsets_B[level]);
         }

         cfg.mode = AGX_IMAGE_MODE_SPARSE;
         cfg.address = plane->sparse_map->va->addr +
                       ail_page_to_sparse_index_el(layout, layer, page) *
                          AIL_SPARSE_ELSIZE_B;
      }

      cfg.srgb = (desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB);
      cfg.unk_mipmapped = layout->levels > 1;
      cfg.srgb_2_channel = cfg.srgb && util_format_colormask(desc) == 0x3;

      if (layout->compressed) {
         cfg.compressed = true;
         cfg.extended = true;
      }

      if (layout->compressed) {
         cfg.acceleration_buffer = base_addr + layout->metadata_offset_B +
                                   (layer * layout->compression_layer_stride_B);
      }

      if (layout->tiling == AIL_TILING_LINEAR &&
          (hk_image_view_type_is_array(view->vk.view_type))) {

         cfg.depth_linear = layers;
         cfg.layer_stride_linear = layout->layer_stride_B - 0x80;
         cfg.extended = true;
      } else {
         assert((layout->tiling != AIL_TILING_LINEAR) || (layers == 1));
         cfg.depth = layers;
      }

      if (view->vk.image->samples > 1) {
         cfg.samples = agx_translate_sample_count(view->vk.image->samples);
      }

      if (layout->tiling == AIL_TILING_LINEAR) {
         cfg.stride = ail_get_linear_stride_B(layout, 0) - 16;
      } else {
         cfg.page_aligned_layers = layout->page_aligned_layers;
      }
   }
}

/*
 * G15's PBE descriptor (cmdbuf.xml's "PBE G15"). What Apple's encoder and
 * noopwafel's demo15 settle is packed: images linear (the one the hardware
 * ran: a 2D RGBA8 target) or twiddled, one layer for linear, compressed with
 * the metadata in the extended word as Gallium packs it, multisampled (the
 * sample count and the multisampled dimensions) and 3D (the depth in the layer
 * count). A spilled render target's PBE (EMRT) writes the image uncompressed,
 * after its in-place decompression. Linear arrays are not decoded (a warning
 * says so).
 *
 * The address is the plane's VA as it is, 44 bits of it (G13's descriptor
 * has 36 and a level; demo15 sets "level 1" for the 1 TiB bit of its
 * target's address), the VA the texture descriptor holds: through another VA
 * of the same memory a fetch does not see an image write in the same launch.
 *
 * A multisampled image is stored to (an image store, an atomic or a spilled
 * render target) through a linear 2D view of its samples, as on G13: G15 has
 * no multisampled image write. sw gets the software words libagx's texel
 * addressing reads (PBE G15 software); the hardware reads that word of a
 * multisampled or compressed PBE, so the end-of-tile PBEs keep it zero.
 */
static void
pack_pbe_g15(struct hk_device *dev, struct hk_image_view *view,
             unsigned view_plane, enum hk_desc_usage usage,
             struct agx_pbe_packed *out, struct agx_pbe_g15_software_packed *sw)
{
   struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   const uint8_t image_plane = view->planes[view_plane].image_plane;
   struct ail_layout *layout = &image->planes[image_plane].layout;
   uint64_t base_addr = hk_image_base_address(image, image_plane);

   unsigned level = view->vk.base_mip_level;
   unsigned layer = view->vk.base_array_layer;

   enum pipe_format p_format = hk_image_view_plane_format(view, view_plane);

   bool eot =
      usage == HK_DESC_USAGE_BG_EOT || usage == HK_DESC_USAGE_LAYERED_BG_EOT;

   /* The tilebuffer is already in sRGB space if needed. Do not convert for
    * end-of-tile descriptors.
    */
   if (eot)
      p_format = util_format_linear(p_format);

   unsigned layers = view->vk.view_type == VK_IMAGE_VIEW_TYPE_3D
                        ? image->vk.extent.depth
                        : view->vk.layer_count;

   bool msaa = view->vk.image->samples > 1;
   bool msaa_view = msaa && !eot;
   /* Every view packs a storage PBE, but no storage image is compressed
    * (hk_can_compress): that one, like a spilled target's, is uncompressed */
   bool compressed = layout->compressed && usage != HK_DESC_USAGE_EMRT &&
                     usage != HK_DESC_USAGE_STORAGE;
   struct hk_3d denom = view_denominator(view);

   assert(!(compressed && msaa_view) && "no compressed multisampled view");

   /* What Apple's encoder does for these is not decoded: the descriptor is
    * packed as for a plain image */
   static bool warned;
   if (!warned && layout->tiling == AIL_TILING_LINEAR && layers > 1) {
      warned = true;
      fprintf(stderr, "G15: the PBE of a linear layered image is a guess\n");
   }

   agx_pack(out, PBE_G15, cfg) {
      cfg.dimension =
         translate_image_view_type(view->vk.view_type, msaa, layers > 1, usage);
      cfg.layout = agx_translate_layout(layout->tiling);
      agx_pbe_g15_format(&cfg, p_format);

      /* An uncompressed view of a block-compressed image, as G13 packs it:
       * the level's own address and size in blocks, at level 0 */
      if (denom.x > 1) {
         assert(denom.z == 1 && "todo how to handle?");
         assert(view->vk.level_count == 1);
         assert(view->vk.layer_count == 1);

         cfg.buffer = base_addr + ail_get_layer_level_B(layout, layer, level);
         cfg.width = DIV_ROUND_UP(u_minify(layout->width_px, level), denom.x);
         cfg.height = DIV_ROUND_UP(u_minify(layout->height_px, level), denom.y);
         level = 0;
      } else {
         cfg.buffer = base_addr + ail_get_layer_offset_B(layout, layer);
         cfg.width = layout->width_px;
         cfg.height = layout->height_px;
      }

      cfg.mipmapped = image->vk.mip_levels > 1;

      if (msaa && !msaa_view)
         cfg.samples = agx_translate_sample_count(view->vk.image->samples);

      if (msaa_view) {
         agx_pbe_g15_ms_view(&cfg, layout, layer, level, p_format);
      } else if (layout->tiling == AIL_TILING_LINEAR) {
         /* In 16-byte units, minus 1 (demo15's 0xF for a 256 B stride; the
          * rows' pitch is a multiple of 16 B) */
         assert((ail_get_linear_stride_B(layout, level) & 15) == 0);
         cfg.stride = (ail_get_linear_stride_B(layout, level) >> 4) - 1;
         assert(image->vk.mip_levels == 1);
      } else {
         cfg.layers = layers;
         cfg.page_aligned_layers = layout->page_aligned_layers;

         /* The level and the count of levels go in the extended word; Apple
          * sets it when either is not the default (not verified), and with
          * the metadata */
         if (level != 0 || image->vk.mip_levels > 1 || compressed) {
            cfg.extended = true;
            cfg.level = level;
            cfg.levels = image->vk.mip_levels;
         }

         if (compressed) {
            cfg.compressed = true;
            cfg.acceleration_buffer =
               base_addr + layout->metadata_offset_B +
               (layer * layout->compression_layer_stride_B);
         }
      }
   }

   memset(sw, 0, sizeof(*sw));
   if (eot)
      return;

   /* level is 0 for a block-compressed image's view, whose PBE has the
    * level's address */
   agx_pack_pbe_g15_software(sw, layout, view->vk.base_mip_level,
                             ail_get_level_offset_B(layout, level),
                             view->vk.image->samples, p_format);
}

static void
pack_pbe(struct hk_device *dev, struct hk_image_view *view, unsigned view_plane,
         enum hk_desc_usage usage, struct agx_pbe_packed *out,
         struct agx_pbe_g15_software_packed *g15_sw)
{
   if (dev->dev.params.gpu_generation >= 15) {
      pack_pbe_g15(dev, view, view_plane, usage, out, g15_sw);
      return;
   }

   struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   const uint8_t image_plane = view->planes[view_plane].image_plane;
   struct ail_layout *layout = &image->planes[image_plane].layout;
   uint64_t base_addr = hk_image_base_address(image, image_plane);

   unsigned level = view->vk.base_mip_level;
   unsigned layer = view->vk.base_array_layer;

   enum pipe_format p_format = hk_image_view_plane_format(view, view_plane);
   const struct util_format_description *desc =
      util_format_description(p_format);

   bool eot =
      usage == HK_DESC_USAGE_BG_EOT || usage == HK_DESC_USAGE_LAYERED_BG_EOT;

   /* The tilebuffer is already in sRGB space if needed. Do not convert for
    * end-of-tile descriptors.
    */
   if (eot)
      p_format = util_format_linear(p_format);

   bool msaa = view->vk.image->samples > 1;
   struct hk_3d denom = view_denominator(view);

   unsigned layers = view->vk.view_type == VK_IMAGE_VIEW_TYPE_3D
                        ? image->vk.extent.depth
                        : view->vk.layer_count;

   agx_pack(out, PBE, cfg) {
      cfg.dimension =
         translate_image_view_type(view->vk.view_type, msaa, layers > 1, usage);
      cfg.layout = agx_translate_layout(layout->tiling);
      cfg.channels = ail_pixel_format[p_format].channels;
      cfg.type = ail_pixel_format[p_format].type;
      cfg.srgb = util_format_is_srgb(p_format);

      cfg.swizzle_r = agx_pbe_swizzle(desc, 0);
      cfg.swizzle_g = agx_pbe_swizzle(desc, 1);
      cfg.swizzle_b = agx_pbe_swizzle(desc, 2);
      cfg.swizzle_a = agx_pbe_swizzle(desc, 3);

      cfg.buffer = base_addr + ail_get_layer_offset_B(layout, layer);
      cfg.unk_mipmapped = layout->levels > 1;

      if (msaa & !eot) {
         /* Multisampled images are bound like buffer textures, with
          * addressing arithmetic to determine the texel to write.
          *
          * Note that the end-of-tile program uses real multisample images
          * with image_write_block instructions.
          */
         unsigned blocksize_B = util_format_get_blocksize(p_format);
         unsigned size_px =
            (layout->size_B - layout->layer_stride_B * layer) / blocksize_B;

         cfg.dimension = AGX_TEXTURE_DIMENSION_2D;
         cfg.layout = AGX_LAYOUT_LINEAR;
         cfg.width = AGX_TEXTURE_BUFFER_WIDTH;
         cfg.height = DIV_ROUND_UP(size_px, cfg.width);
         cfg.stride = (cfg.width * blocksize_B) - 4;
         cfg.layers = 1;
         cfg.levels = 1;

         cfg.buffer += layout->level_offsets_B[level];
         cfg.level = 0;
      } else {
         if (denom.x > 1) {
            assert(denom.z == 1 && "todo how to handle?");
            assert(view->vk.level_count == 1);
            assert(view->vk.layer_count == 1);

            cfg.buffer =
               base_addr + ail_get_layer_level_B(layout, layer, level);
            cfg.width =
               DIV_ROUND_UP(u_minify(layout->width_px, level), denom.x);
            cfg.height =
               DIV_ROUND_UP(u_minify(layout->height_px, level), denom.y);
            cfg.level = 0;
         } else {
            cfg.buffer = base_addr + ail_get_layer_offset_B(layout, layer);
            cfg.width = layout->width_px;
            cfg.height = layout->height_px;
            cfg.level = level;
         }

         if (layout->tiling == AIL_TILING_LINEAR &&
             (hk_image_view_type_is_array(view->vk.view_type))) {

            cfg.depth_linear = layers;
            cfg.layer_stride_linear = (layout->layer_stride_B - 0x80);
            cfg.extended = true;
         } else {
            assert((layout->tiling != AIL_TILING_LINEAR) || (layers == 1));
            cfg.layers = layers;
         }

         cfg.levels = image->vk.mip_levels;

         if (layout->tiling == AIL_TILING_LINEAR) {
            cfg.stride = ail_get_linear_stride_B(layout, level) - 4;
            assert(cfg.levels == 1);
         } else {
            cfg.page_aligned_layers = layout->page_aligned_layers;
         }

         if (image->vk.samples > 1)
            cfg.samples = agx_translate_sample_count(image->vk.samples);
      }

      if (layout->compressed && usage != HK_DESC_USAGE_EMRT) {
         cfg.compressed = true;
         cfg.extended = true;

         cfg.acceleration_buffer = base_addr + layout->metadata_offset_B +
                                   (layer * layout->compression_layer_stride_B);
      }

      /* When the descriptor isn't extended architecturally, we use
       * the last 8 bytes as a sideband to accelerate image atomics.
       */
      if (!cfg.extended &&
          (layout->writeable_image || usage == HK_DESC_USAGE_EMRT)) {

         if (msaa) {
            assert(denom.x == 1 && "no MSAA of block-compressed");

            cfg.aligned_width_msaa_sw =
               align(u_minify(layout->width_px, level),
                     layout->tilesize_el[level].width_el);

            cfg.sample_count_log2_sw = util_logbase2(image->vk.samples);
         } else {
            cfg.level_offset_sw = ail_get_level_offset_B(layout, cfg.level);
         }

         if (layout->tiling != AIL_TILING_LINEAR) {
            struct ail_tile tile_size = layout->tilesize_el[level];
            cfg.tile_width_sw = tile_size.width_el;
            cfg.tile_height_sw = tile_size.height_el;

            cfg.layer_stride_sw = layout->layer_stride_B;
         }
      }
   };
}

static VkResult
hk_image_view_init(struct hk_device *dev, struct hk_image_view *view,
                   const VkImageViewCreateInfo *pCreateInfo)
{
   VK_FROM_HANDLE(hk_image, image, pCreateInfo->image);
   memset(view, 0, sizeof(*view));

   vk_image_view_init(&dev->vk, &view->vk, pCreateInfo);

   /* First, figure out which image planes we need. For depth/stencil, we only
    * have one aspect viewed at a time.
    */
   if (image->vk.aspects &
       (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {

      view->plane_count = 1;
      view->planes[0].image_plane =
         hk_image_aspects_to_plane(image, view->vk.aspects);
   } else {
      /* For other formats, retrieve the plane count from the aspect mask
       * and then walk through the aspect mask to map each image plane
       * to its corresponding view plane
       */
      assert(util_bitcount(view->vk.aspects) ==
             vk_format_get_plane_count(view->vk.format));
      view->plane_count = 0;
      u_foreach_bit(aspect_bit, view->vk.aspects) {
         view->planes[view->plane_count++].image_plane =
            hk_image_aspects_to_plane(image, 1u << aspect_bit);
      }
   }

   /* Fill in each view plane separately */
   for (unsigned p = 0; p < view->plane_count; p++) {
      pack_texture(view, p, HK_DESC_USAGE_SAMPLED, &view->planes[p].sampled);
      pack_texture(view, p, HK_DESC_USAGE_STORAGE, &view->planes[p].ro_storage);
      pack_texture(view, p, HK_DESC_USAGE_INPUT, &view->planes[p].ia);
      pack_texture(view, p, HK_DESC_USAGE_EMRT, &view->planes[p].emrt_texture);
      pack_texture(view, p, HK_DESC_USAGE_BG_EOT, &view->planes[p].background);
      pack_texture(view, p, HK_DESC_USAGE_LAYERED_BG_EOT,
                   &view->planes[p].layered_background);

      struct agx_pbe_g15_software_packed eot_sw;

      pack_pbe(dev, view, p, HK_DESC_USAGE_STORAGE, &view->planes[p].storage,
               &view->planes[p].storage_g15_sw);
      pack_pbe(dev, view, p, HK_DESC_USAGE_BG_EOT, &view->planes[p].eot,
               &eot_sw);
      pack_pbe(dev, view, p, HK_DESC_USAGE_EMRT, &view->planes[p].emrt_pbe,
               &view->planes[p].emrt_g15_sw);
      pack_pbe(dev, view, p, HK_DESC_USAGE_LAYERED_BG_EOT,
               &view->planes[p].layered_eot, &eot_sw);
   }

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyImageView(VkDevice _device, VkImageView imageView,
                    const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   VK_FROM_HANDLE(hk_image_view, view, imageView);

   if (!view)
      return;

   vk_image_view_finish(&view->vk);
   vk_free2(&dev->vk.alloc, pAllocator, view);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateImageView(VkDevice _device, const VkImageViewCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator, VkImageView *pView)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_image_view *view;
   VkResult result;

   view = vk_alloc2(&dev->vk.alloc, pAllocator, sizeof(*view), 8,
                    VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!view)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = hk_image_view_init(dev, view, pCreateInfo);
   if (result != VK_SUCCESS) {
      hk_DestroyImageView(_device, hk_image_view_to_handle(view), pAllocator);
      return result;
   }

   *pView = hk_image_view_to_handle(view);

   return VK_SUCCESS;
}
