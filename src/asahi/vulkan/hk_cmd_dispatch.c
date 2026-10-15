/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "libagx/query.h"
#include "vulkan/vulkan_core.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_pack.h"
#include "agx_scratch.h"
#include "agx_tilebuffer.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "pool.h"

#include <stdlib.h>

void
hk_cmd_buffer_begin_compute(struct hk_cmd_buffer *cmd,
                            const VkCommandBufferBeginInfo *pBeginInfo)
{
}

void
hk_cmd_invalidate_compute_state(struct hk_cmd_buffer *cmd)
{
   memset(&cmd->state.cs, 0, sizeof(cmd->state.cs));
}

void
hk_cmd_bind_compute_shader(struct hk_cmd_buffer *cmd,
                           struct hk_api_shader *shader)
{
   cmd->state.cs.shader = shader;
}

void
hk_cdm_cache_flush(struct hk_device *dev, struct hk_cs *cs)
{
   assert(cs->type == HK_CS_CDM);
   assert(cs->current + AGX_CDM_BARRIER_LENGTH < cs->end &&
          "caller must ensure space");

   if (dev->dev.params.gpu_generation >= 15) {
      uint32_t *out = cs->current;
      *(out++) = 0x60000160; /* G15C: Metal's barrier after a launch */
      cs->current = out;
   } else
      cs->current = agx_cdm_barrier(cs->current, dev->dev.chip);

   cs->stats.flushes++;
}

/* G15: the direct launch block (spec: metal-compute.md); the launch
 * points at the entry program, a full VA. Word 0 bits 14:3 and word 1 bits
 * 20:3 are the per-thread stack and spill counts, from Apple's
 * ComputeProgramVariant::finalize (decompiled 2026-07-21,
 * work/scratch/agx3/spill/cpv_finalize.c): word 0 = 8*(((stack_bytes+3)/4 +
 * 1) & ~1), word 1 = 8*((spill_bytes+3)/4). They tell the CDM the per-thread
 * private memory to carve from the UMA pool; a kernel returning a device
 * atomic's result (st0) needs a nonzero stack count, or the read of st0
 * returns 0 and the atomic is dropped (vkcompute dev 11/11 wrong). AGX_MESA_
 * G15_LAUNCH_W0 / _W1 override the two field values for the sweep. */
static uint32_t *
hk_cdm_launch_g15(struct hk_device *dev, uint32_t *out, struct agx_grid grid,
                  struct agx_workgroup wg, uint32_t usc, unsigned stack_size,
                  unsigned spill_size)
{
   uint64_t va = dev->dev.shader_base + usc;
   bool multi = grid.count[1] > 1 || grid.count[2] > 1;
   const char *e;

   uint32_t w0 = multi ? BITFIELD_BIT(19) : 0;
   if (stack_size)
      w0 |= (8u * (((stack_size + 3) / 4 + 1) & ~1u)) << 3;
   if ((e = getenv("AGX_MESA_G15_LAUNCH_W0")))
      w0 = (w0 & ~(BITFIELD_MASK(12) << 3)) |
           ((atoi(e) & BITFIELD_MASK(12)) << 3);
   *(out++) = w0;

   agx_g15_pipeline(out, va);
   if (spill_size)
      out[0] |= (8u * ((spill_size + 3) / 4)) << 3;
   if ((e = getenv("AGX_MESA_G15_LAUNCH_W1")))
      out[0] = (out[0] & ~(BITFIELD_MASK(18) << 3)) |
               ((atoi(e) & BITFIELD_MASK(18)) << 3);
   out += AGX_G15_PIPELINE_LENGTH / 4;
   *(out++) = 0x40000001;
   *(out++) = grid.count[0];
   *(out++) = grid.count[1];
   *(out++) = grid.count[2];
   *(out++) = wg.x & BITFIELD_MASK(11);
   *(out++) = wg.y & BITFIELD_MASK(11);
   *(out++) = wg.z & BITFIELD_MASK(11);
   return out;
}

void
hk_dispatch_with_usc_launch(struct hk_device *dev, struct hk_cs *cs,
                            struct agx_cdm_launch_word_0_packed launch,
                            uint32_t usc, struct agx_grid grid,
                            struct agx_workgroup wg, unsigned stack_size,
                            unsigned spill_size)
{
   hk_ensure_cs_has_space(cs->cmd, cs, 0x2000 /* TODO */);
   cs->stats.cmds++;

   if (dev->dev.params.gpu_generation >= 15) {
      if (!usc || agx_is_indirect(grid)) {
         fprintf(stderr, "G15: dispatch skipped (%s)\n",
                 usc ? "indirect dispatch not supported yet"
                     : "no entry program");
         return;
      }

      cs->current =
         hk_cdm_launch_g15(dev, cs->current, grid, wg, usc, stack_size,
                           spill_size);
   } else {
      cs->current =
         agx_cdm_launch(cs->current, dev->dev.chip, grid, wg, launch, usc);
   }

   hk_cdm_cache_flush(dev, cs);
}

void
hk_dispatch_with_usc(struct hk_device *dev, struct hk_cs *cs,
                     struct agx_shader_info *info, uint32_t usc,
                     struct agx_grid grid, struct agx_workgroup local_size)
{
   struct agx_cdm_launch_word_0_packed launch;
   agx_pack(&launch, CDM_LAUNCH_WORD_0, cfg) {
      cfg.texture_state_register_count = info->texture_state_count;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(info->sampler_state_count, false);
      cfg.uniform_register_count = info->push_count;
      cfg.preshader_register_count = info->nr_preamble_gprs;
   }

   hk_dispatch_with_usc_launch(dev, cs, launch, usc, grid, local_size,
                               info->stack_size,
                               0 /* G15: spilling is unsupported yet */);
}

static void
dispatch(struct hk_cmd_buffer *cmd, struct agx_grid grid)
{
   struct hk_shader *s = hk_only_variant(cmd->state.cs.shader);
   if (agx_is_shader_empty(&s->b))
      return;

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, true /* compute */);
   if (!cs)
      return;

   struct agx_workgroup local_size =
      agx_workgroup(s->b.info.workgroup_size[0], s->b.info.workgroup_size[1],
                    s->b.info.workgroup_size[2]);

   uint64_t stat = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT);

   if (hk_stat_enabled(stat)) {
      perf_debug(cmd, "CS invocation statistic");
      uint64_t grid = cmd->state.cs.descriptors.root.cs.group_count_addr;

      libagx_increment_cs_invocations(cmd, agx_1d(1), AGX_BARRIER_ALL, grid,
                                      stat, agx_workgroup_threads(local_size));
   }

   if (!agx_is_indirect(grid)) {
      grid.count[0] *= local_size.x;
      grid.count[1] *= local_size.y;
      grid.count[2] *= local_size.z;
   }

   hk_dispatch_with_local_size(cmd, cs, s, grid, local_size);
   cs->stats.calls++;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t baseGroupX,
                   uint32_t baseGroupY, uint32_t baseGroupZ,
                   uint32_t groupCountX, uint32_t groupCountY,
                   uint32_t groupCountZ)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   desc->root.cs.base_group[0] = baseGroupX;
   desc->root.cs.base_group[1] = baseGroupY;
   desc->root.cs.base_group[2] = baseGroupZ;

   /* We don't want to key the shader to whether we're indirectly dispatching,
    * so treat everything as indirect.
    */
   VkDispatchIndirectCommand group_count = {
      .x = groupCountX,
      .y = groupCountY,
      .z = groupCountZ,
   };

   desc->root.cs.group_count_addr =
      hk_pool_upload(cmd, &group_count, sizeof(group_count), 8);

   dispatch(cmd, agx_3d(groupCountX, groupCountY, groupCountZ));
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                       VkDeviceSize offset)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   desc->root.cs.base_group[0] = 0;
   desc->root.cs.base_group[1] = 0;
   desc->root.cs.base_group[2] = 0;

   uint64_t dispatch_addr = hk_buffer_address(buffer, offset, true);
   assert(dispatch_addr != 0);

   desc->root.cs.group_count_addr = dispatch_addr;
   dispatch(cmd, agx_grid_indirect(dispatch_addr));
}
