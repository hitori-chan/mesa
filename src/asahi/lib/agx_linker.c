/*
 * Copyright 2024 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_linker.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "util/ralloc.h"
#include "util/u_dynarray.h"
#include "agx_abi.h"
#include "agx_compile.h"
#include "agx_device.h"
#include "agx_pack.h"
#include "agx_scratch.h"

/*
 * When sample shading is used with a non-monolithic fragment shader, we
 * fast-link a program with the following structure:
 *
 *    Fragment prolog;
 *
 *    for (u16 sample_bit = 1; sample_bit < (1 << # of samples); ++sample_bit) {
 *       API fragment shader;
 *       Fragment epilog;
 *    }
 *
 * This means the prolog runs per-pixel but the fragment shader and epilog run
 * per-sample. To do this, we need to generate the loop on the fly. The
 * following binary sequences form the relevant loop.
 */

static_assert(AGX_ABI_FIN_SAMPLE_MASK == 2, "r1l known");

/* clang-format off */
static const uint8_t sample_loop_header[] = {
   /* mov_imm r0l, 0x0, 0b0 */
   0x62, 0x00, 0x00, 0x00,

   /* mov_imm r1l, 0x0, 0b0 */
   0x62, 0x04, 0x01, 0x00,
};

#define STOP                                                                   \
   /* stop */                                                                  \
   0x88, 0x00,                                                                 \
                                                                               \
   /* trap */                                                                  \
   0x08, 0x00, 0x08, 0x00, 0x08, 0x00, 0x08, 0x00,                             \
   0x08, 0x00, 0x08, 0x00, 0x08, 0x00, 0x08, 0x00,

static const uint8_t stop[] = {STOP};

static const uint8_t sample_loop_footer[] = {
   /* iadd r1l, 0, r1l, lsl 1 */
   0x0e, 0x04, 0x00, 0x20, 0x84, 0x00, 0x00, 0x00,

   /* while_icmp r0l, ult, r1h, 0, 1 */
   0x52, 0x2c, 0x42, 0x00, 0x00, 0x00,

   /* jmp_exec_any */
   0x00, 0xc0, 0x00, 0x00, 0x00, 0x00,

   /* pop_exec r0l, 1 */
   0x52, 0x0e, 0x00, 0x00, 0x00, 0x00,

   STOP
};

/* Offset in sample_loop_footer to the jmp_exec_any's target */
#define SAMPLE_LOOP_FOOTER_JMP_PATCH_OFFS (16)

/* Offset of the jmp_exec_any, for calculating the PC offsets */
#define SAMPLE_LOOP_FOOTER_JMP_OFFS (14)

/* Offset in sample_loop_footer to the while_icmp's sample count immediate. Bit
 * position in the byte given by the shift.
 */
#define SAMPLE_LOOP_FOOTER_COUNT_PATCH_OFFS (11)
#define SAMPLE_LOOP_FOOTER_COUNT_SHIFT (4)
/* clang-format on */

void
agx_fast_link(struct agx_linked_shader *linked, struct agx_device *dev,
              bool fragment, struct agx_shader_part *main,
              struct agx_shader_part *prolog, struct agx_shader_part *epilog,
              unsigned nr_samples_shaded)
{
   size_t size = 0;
   unsigned nr_gprs = 0, scratch_size = 0;
   bool reads_tib = false, writes_sample_mask = false,
        disable_tri_merging = false, tag_write_disable = true;
   bool g15 = dev->params.gpu_generation >= 15;

   /* G15: the loop header and footer in G15 instructions (agx_pack3.c) */
   struct util_dynarray loop_header;
   util_dynarray_init(&loop_header, NULL);
   if (g15 && nr_samples_shaded)
      agx3_emit_sample_loop_header(&loop_header);

   if (g15 && nr_samples_shaded) {
      size += loop_header.size;
   } else if (nr_samples_shaded) {
      size += sizeof(sample_loop_header);

      if (nr_samples_shaded > 1)
         size += sizeof(sample_loop_footer);
      else
         size += sizeof(stop);
   }

   struct agx_shader_part *parts[] = {prolog, main, epilog};

   for (unsigned i = 0; i < ARRAY_SIZE(parts); ++i) {
      struct agx_shader_part *part = parts[i];
      if (!part)
         continue;

      size += part->info.main_size;
      linked->pix_wait_mask |= part->info.pix_wait_mask;
      linked->pix_signal_mask |= part->info.pix_signal_mask;
      linked->writes_zs |= part->info.writes_zs;
      linked->cr_count = MAX2(linked->cr_count, part->info.cr_count);

      nr_gprs = MAX2(nr_gprs, part->info.nr_gprs);
      scratch_size = MAX2(scratch_size, part->info.scratch_size);
      reads_tib |= part->info.reads_tib;
      writes_sample_mask |= part->info.writes_sample_mask;
      disable_tri_merging |= part->info.disable_tri_merging;
      linked->uses_base_param |= part->info.uses_base_param;
      linked->uses_txf |= part->info.uses_txf;
      tag_write_disable &= part->info.tag_write_disable;
   }

   assert(size > 0 && "must stop");

   /* G15: the program releases the fences it waited on at its end, before
    * the last part's stop (agx3_emit_stop's 4 bytes), but those a part
    * released itself. In a sample loop the parts have no stop: the loop's
    * footer and the trailer end the program.
    */
   uint16_t pix_late = linked->pix_wait_mask & ~linked->pix_signal_mask;
   struct util_dynarray trailer;
   util_dynarray_init(&trailer, NULL);
   if (g15 && (pix_late || nr_samples_shaded)) {
      if (pix_late)
         agx3_emit_signal_pix(&trailer, pix_late);
      agx3_emit_stop(&trailer);
      size += trailer.size;
      if (!nr_samples_shaded)
         size -= 4;
   }

   /* The footer's size does not depend on its branch's target */
   struct util_dynarray loop_footer;
   util_dynarray_init(&loop_footer, NULL);
   if (g15 && nr_samples_shaded > 1) {
      agx3_emit_sample_loop_footer(&loop_footer, nr_samples_shaded, 0, 0);
      size += loop_footer.size;
   }

   linked->bo = agx_bo_create(dev, size, 0, AGX_BO_EXEC | AGX_BO_LOW_VA,
                              "Linked executable");
   uint8_t *linked_map = agx_bo_map(linked->bo);

   size_t offset = 0;

   /* FS prolog happens per-pixel, outside the sample loop */
   if (prolog) {
      size_t sz = prolog->info.main_size;
      memcpy(linked_map + offset, prolog->binary, sz);
      offset += sz;
   }

   if (g15 && nr_samples_shaded) {
      memcpy(linked_map + offset, loop_header.data, loop_header.size);
      offset += loop_header.size;
   } else if (nr_samples_shaded) {
      memcpy(linked_map + offset, sample_loop_header,
             sizeof(sample_loop_header));
      offset += sizeof(sample_loop_header);
   }

   size_t sample_loop_begin = offset;

   /* Main shader and epilog happen in the sample loop, so start from i=1 */
   for (unsigned i = 1; i < ARRAY_SIZE(parts); ++i) {
      struct agx_shader_part *part = parts[i];
      if (!part)
         continue;

      size_t sz = part->info.main_size;
      memcpy(linked_map + offset, part->binary + part->info.main_offset, sz);
      offset += sz;

      /* The last part ends the program: replace its stop (G15's, 4 bytes) */
      if (trailer.size && !nr_samples_shaded &&
          (i == ARRAY_SIZE(parts) - 1 || !parts[i + 1])) {
         assert(!memcmp(linked_map + offset - 4, "\x0e\0\0\0", 4));
         offset -= 4;
         memcpy(linked_map + offset, trailer.data, trailer.size);
         offset += trailer.size;
      }
   }

   if (g15 && nr_samples_shaded) {
      if (nr_samples_shaded > 1) {
         unsigned footer_size = loop_footer.size;
         util_dynarray_clear(&loop_footer);
         agx3_emit_sample_loop_footer(&loop_footer, nr_samples_shaded, offset,
                                      sample_loop_begin);
         assert(loop_footer.size == footer_size);
         memcpy(linked_map + offset, loop_footer.data, loop_footer.size);
         offset += loop_footer.size;
      }

      memcpy(linked_map + offset, trailer.data, trailer.size);
      offset += trailer.size;
   } else if (nr_samples_shaded > 1) {
      assert(sample_loop_footer[SAMPLE_LOOP_FOOTER_COUNT_PATCH_OFFS] == 0);

      /* Make a stack copy of the footer so we can efficiently patch it */
      uint8_t footer[sizeof(sample_loop_footer)];
      memcpy(footer, sample_loop_footer, sizeof(footer));

      /* Patch in sample end */
      uint8_t end = (1u << nr_samples_shaded) - 1;
      footer[SAMPLE_LOOP_FOOTER_COUNT_PATCH_OFFS] =
         end << SAMPLE_LOOP_FOOTER_COUNT_SHIFT;

      /* Patch in the branch target */
      int32_t loop_size = offset - sample_loop_begin;
      int32_t branch_offs = -(SAMPLE_LOOP_FOOTER_JMP_OFFS + loop_size);
      int32_t *target = (int32_t *)(footer + SAMPLE_LOOP_FOOTER_JMP_PATCH_OFFS);
      *target = branch_offs;

      /* Copy in the patched footer */
      memcpy(linked_map + offset, footer, sizeof(footer));
      offset += sizeof(footer);
   } else if (nr_samples_shaded) {
      /* Just end after the first sample, no need to loop for a single sample */
      memcpy(linked_map + offset, stop, sizeof(stop));
      offset += sizeof(stop);
   }

   assert(offset == size);
   util_dynarray_fini(&trailer);
   util_dynarray_fini(&loop_header);
   util_dynarray_fini(&loop_footer);
   linked->main_size = size;

   agx_pack(&linked->shader, USC_SHADER, cfg) {
      cfg.code = agx_usc_addr(dev, linked->bo->va->addr);
      cfg.unk_2 = fragment ? 2 : 3;

      if (fragment)
         cfg.loads_varyings = linked->cf.nr_bindings > 0;
   }

   agx_pack(&linked->regs, USC_REGISTERS, cfg) {
      cfg.register_count = nr_gprs;
      cfg.unk_1 = fragment;
      cfg.spill_size = scratch_size ? agx_scratch_get_bucket(scratch_size) : 0;
      cfg.unk_4 = 1;
   }

   if (fragment) {
      agx_pack(&linked->fragment_props, USC_FRAGMENT_PROPERTIES, cfg) {
         cfg.early_z_testing = !writes_sample_mask;
         cfg.unk_2 = true;
         cfg.unk_3 = 0xf;
         cfg.unk_4 = 0x2;
         cfg.unk_5 = 0x0;
      }

      /* If we skip the fragment shader, we can optimize the pass type */
      if (tag_write_disable) {
         reads_tib = false;
         writes_sample_mask = false;
         linked->no_op = true;
      }

      /* G15: Apple's discard and depth output disable triangle merging (their
       * fragment control 2, 0x44000200: draws.md)
       */
      if (g15 && writes_sample_mask)
         disable_tri_merging = true;

      agx_pack(&linked->fragment_control, FRAGMENT_CONTROL, cfg) {
         cfg.tag_write_disable = tag_write_disable;
         cfg.disable_tri_merging = disable_tri_merging;

         if (reads_tib && writes_sample_mask)
            cfg.pass_type = AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH;
         else if (reads_tib)
            cfg.pass_type = AGX_PASS_TYPE_TRANSLUCENT;
         else if (writes_sample_mask)
            cfg.pass_type = AGX_PASS_TYPE_PUNCH_THROUGH;
         else
            cfg.pass_type = AGX_PASS_TYPE_OPAQUE;
      }

      /* Merge the CF binding lists from the prolog to handle cull distance */
      memcpy(&linked->cf, &main->info.varyings.fs,
             sizeof(struct agx_varyings_fs));

      struct agx_varyings_fs *prolog_vary =
         prolog ? &prolog->info.varyings.fs : NULL;

      if (prolog_vary && prolog_vary->nr_bindings) {
         assert(!prolog_vary->reads_z);
         linked->cf.nr_cf = MAX2(linked->cf.nr_cf, prolog_vary->nr_cf);

         assert(linked->cf.nr_bindings + prolog_vary->nr_bindings <=
                   ARRAY_SIZE(linked->cf.bindings) &&
                "bounded by # of coeff registers");

         memcpy(linked->cf.bindings + linked->cf.nr_bindings,
                prolog_vary->bindings,
                sizeof(struct agx_cf_binding) * prolog_vary->nr_bindings);

         linked->cf.nr_bindings += prolog_vary->nr_bindings;
      }

      bool bary_persp = false, bary_linear = false;
      for (unsigned i = 0; i < linked->cf.nr_bindings; ++i) {
         gl_varying_slot slot = linked->cf.bindings[i].slot;
         bary_persp |= slot == AGX_VARYING_SLOT_BARY_PERSP;
         bary_linear |= slot == AGX_VARYING_SLOT_BARY_LINEAR;
      }

      agx_pack(&linked->osel, OUTPUT_SELECT, cfg) {
         cfg.varyings = linked->cf.nr_bindings > 0;
         cfg.frag_coord_z = linked->cf.reads_z;
         cfg.barycentric_coordinates = bary_persp;
         cfg.linear_barycentric_coordinates = bary_linear;
      }
   }
}
