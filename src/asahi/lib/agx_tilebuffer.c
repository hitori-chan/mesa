/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_tilebuffer.h"
#include <assert.h>
#include "util/format/u_format.h"
#include "layout.h"

/* Maximum number of bytes per tile on G13G. This may change in future versions
 * of the architecture.
 */
#define MAX_BYTES_PER_TILE (32768 - 1)

/* Maximum bytes per sample in the tilebuffer. Greater allocations require
 * spilling render targets to memory.
 */
#define MAX_BYTES_PER_SAMPLE (64)

/* G15 takes its tile from the sample count alone, as Metal's driver does for
 * up to 131 B a sample (FramebufferGen3_3, AGXMetalG15X_M1 0x21267b848):
 * eight 16 B render targets fit, so a colour pass never spills. That driver
 * budgets 256 KiB of tile memory a tile, which 128 B samples fill at 2 and 4
 * samples; its 64 KiB tile of two 2-sample RGBA32F targets ran right
 * (msmrt32).
 */
#define MAX_BYTES_PER_SAMPLE_G15 (128)
#define MAX_BYTES_PER_TILE_G15   (256 * 1024)

/* Metal's tiles past 32 KiB ran slower on the M3 Max than the same samples in
 * a tile halved to 32 KiB: eight RGBA16F targets at one sample (64 KiB) 1.26
 * times faster halved, eight RGBA32F (128 KiB) 2.9 times, five RGBA32F 1.3
 * times; elsewhere the two agree. So the tile halves while it is over
 * 32 KiB, down to the architectural 16x16, and the samples still fit.
 */
#define TILE_TARGET_B_G15 (32 * 1024)

/* Minimum tile size in pixels, architectural. */
#define MIN_TILE_SIZE_PX (16 * 16)

/* Select the largest tile size that fits */
static uint16_t
agx_select_tile_size(unsigned px_size_B, unsigned max_B)
{
   assert(px_size_B <= (max_B / MIN_TILE_SIZE_PX));

   return ((px_size_B * 32 * 32) <= max_B)   ? (32 * 32)
          : ((px_size_B * 32 * 16) <= max_B) ? (32 * 16)
                                             : MIN_TILE_SIZE_PX;
}

/* G15's tile for a sample count, Metal's: 32x32 at 1 and 2 samples, 32x16 at
 * 4, 16x16 at 8. A count not known yet (0) takes 32x32. */
static uint16_t
agx_select_tile_size_g15(unsigned nr_samples)
{
   return nr_samples <= 2 ? (32 * 32) : nr_samples <= 4 ? (32 * 16) : (16 * 16);
}

static inline unsigned
format_size_B(enum pipe_format format)
{
   /* For some reason util_format_get_blocksize(NONE) = 1 */
   if (format == PIPE_FORMAT_NONE)
      return 0;

   enum pipe_format phys = ail_pixel_format[format].renderable;
   unsigned size_B = util_format_get_blocksize(phys);
   if (util_format_get_nr_components(phys) == 1)
      size_B *= util_format_get_nr_components(format);

   return size_B;
}

static inline unsigned
format_align_B(enum pipe_format format)
{
   return MIN2(util_next_power_of_two(format_size_B(format)), 8);
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout(const enum pipe_format *formats, uint8_t nr_cbufs,
                            uint8_t nr_samples, bool layered, bool g15)
{
   struct agx_tilebuffer_layout tib = {
      .nr_samples = nr_samples,
      .layered = layered,
      .g15 = g15,
   };

   unsigned max_sample_B =
      g15 ? MAX_BYTES_PER_SAMPLE_G15 : MAX_BYTES_PER_SAMPLE;

   uint32_t offset_B = 0;
   uint8_t order[] = {0, 1, 2, 3, 4, 5, 6, 7};

   /* Sort render targets in descending order of alignment, eliminating padding
    * and giving the optimal order of render targets. We use insertion sort
    * because it is simple, stable, fast for small n, and free for n=1.
    */
   for (int i = 1; i < nr_cbufs; ++i) {
      for (int j = i; j > 0 && format_align_B(formats[order[j - 1]]) <
                                  format_align_B(formats[order[j]]);
           --j) {
         SWAP(order[j], order[j - 1]);
      }
   }

   for (unsigned i = 0; i < nr_cbufs; ++i) {
      unsigned rt = order[i];
      enum pipe_format format = formats[rt];
      tib.logical_format[rt] = format;

      unsigned size_B = format_size_B(format);
      unsigned start_B = ALIGN_POT(offset_B, MAX2(format_align_B(format), 1));

      /* If allocating this render target would exceed any tilebuffer limits, we
       * need to spill it to memory. Otherwise, allocate it to the tilebuffer.
       */
      unsigned new_offset_B = start_B + size_B;
      bool fits = (new_offset_B <= max_sample_B) &&
                  (g15 || (ALIGN_POT(new_offset_B, 8) * MIN_TILE_SIZE_PX *
                           nr_samples) <= MAX_BYTES_PER_TILE);
      if (fits) {
         tib._offset_B[rt] = start_B;
         offset_B = new_offset_B;
      } else {
         tib.spilled[rt] = true;
      }
   }

   assert(offset_B <= max_sample_B && "loop invariant");

   /* Multisampling needs a nonempty allocation.
    * XXX: Check this against hw
    */
   if (nr_samples > 1)
      offset_B = MAX2(offset_B, 1);

   tib.sample_size_B = align(offset_B, 8);

   if (g15) {
      tib.tile_size = agx_select_tile_size_g15(nr_samples);
      while (tib.tile_size > MIN_TILE_SIZE_PX &&
             agx_tilebuffer_total_size(&tib) > TILE_TARGET_B_G15)
         tib.tile_size /= 2;
      assert(agx_tilebuffer_total_size(&tib) <= MAX_BYTES_PER_TILE_G15);
   } else {
      tib.tile_size = agx_select_tile_size(tib.sample_size_B * nr_samples,
                                           MAX_BYTES_PER_TILE);
   }

   agx_tilebuffer_pack_usc(&tib);
   return tib;
}

/*
 * With attachmentless rendering in Vulkan, the sample count may not known until
 * draw-time. It's convenient to construct an agx_tilebuffer_layout anyway when
 * beginning rendering, updating the sample count later. This helper allows the
 * driver to set the sample count in a partial agx_tilebuffer_layout.
 *
 * When doing so, we need to rebuild entirely since e.g. tile size might change.
 */
void
agx_tilebuffer_set_samples(struct agx_tilebuffer_layout *tib,
                           unsigned nr_samples)
{
   assert(tib->nr_samples == 0 && "must not be initialized");

   *tib = agx_build_tilebuffer_layout(tib->logical_format,
                                      ARRAY_SIZE(tib->logical_format),
                                      nr_samples, tib->layered, tib->g15);
}

enum pipe_format
agx_tilebuffer_physical_format(struct agx_tilebuffer_layout *tib, unsigned rt)
{
   return ail_pixel_format[tib->logical_format[rt]].renderable;
}

bool
agx_tilebuffer_supports_mask(struct agx_tilebuffer_layout *tib, unsigned rt)
{
   /* We don't bother support masking with spilled render targets. This might be
    * optimized in the future but spilling is so rare anyway it's not worth it.
    */
   if (tib->spilled[rt])
      return false;

   enum pipe_format fmt = agx_tilebuffer_physical_format(tib, rt);
   return ail_isa_format_supports_mask((enum ail_isa_format)fmt);
}

uint32_t
agx_tilebuffer_total_size(struct agx_tilebuffer_layout *tib)
{
   return tib->sample_size_B * tib->nr_samples * tib->tile_size;
}

void
agx_tilebuffer_pack_usc(struct agx_tilebuffer_layout *tib)
{
   agx_pack(&tib->usc, USC_SHARED, cfg) {
      if (tib->nr_samples > 0) {
         cfg.uses_shared_memory = true;
         cfg.sample_count = tib->nr_samples;

         /* G15 runs no USC words: its entry program takes the imageblock from
          * the layout (agx_esl_code::tib), whose samples and tiles these
          * fields cannot hold past 120 B and 64 KiB */
         if (!tib->g15) {
            cfg.sample_stride_in_8_bytes = tib->sample_size_B / 8;
            cfg.bytes_per_threadgroup = agx_tilebuffer_total_size(tib);
         }

         if (tib->tile_size == 32 * 32)
            cfg.layout = AGX_SHARED_LAYOUT_32X32;
         else if (tib->tile_size == 32 * 16)
            cfg.layout = AGX_SHARED_LAYOUT_32X16;
         else if (tib->tile_size == 16 * 16)
            cfg.layout = AGX_SHARED_LAYOUT_16X16;
         else
            UNREACHABLE("Invalid tile size");
      } else {
         cfg.layout = AGX_SHARED_LAYOUT_VERTEX_COMPUTE;
         cfg.bytes_per_threadgroup = 65536;
      }
   }
}
