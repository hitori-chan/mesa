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

   cs->current = agx_cdm_barrier(cs->current, dev->dev.chip);

   cs->stats.flushes++;
}

void
hk_dispatch_with_usc_launch(struct hk_device *dev, struct hk_cs *cs,
                            struct agx_cdm_launch_word_0_packed launch,
                            uint32_t usc, struct agx_grid grid,
                            struct agx_workgroup wg)
{
   hk_ensure_cs_has_space(cs->cmd, cs, 0x2000 /* TODO */);
   cs->stats.cmds++;

   if (dev->dev.params.gpu_generation >= 15 && !usc) {
      fprintf(stderr, "G15: dispatch skipped (no entry program)\n");
      return;
   }

   cs->current = agx_cdm_launch(cs->current, dev->dev.chip, grid, wg, launch,
                                dev->dev.shader_base, usc);

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

   hk_dispatch_with_usc_launch(dev, cs, launch, usc, grid, local_size);
}

/* A direct grid counts workgroups, unless in_threads: invocations, the last
 * workgroup of each dimension partial.
 */
static void
dispatch(struct hk_cmd_buffer *cmd, struct agx_grid grid, bool in_threads)
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

   if (!agx_is_indirect(grid) && !in_threads) {
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

   dispatch(cmd, agx_3d(groupCountX, groupCountY, groupCountZ), false);
}

/* The common acceleration-structure builder's unaligned dispatch */
void
hk_cmd_dispatch_unaligned(VkCommandBuffer commandBuffer, uint32_t x,
                          uint32_t y, uint32_t z)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   const struct hk_shader *s = hk_only_variant(cmd->state.cs.shader);
   VkDispatchIndirectCommand group_count = {
      .x = DIV_ROUND_UP(x, s->b.info.workgroup_size[0]),
      .y = DIV_ROUND_UP(y, s->b.info.workgroup_size[1]),
      .z = DIV_ROUND_UP(z, s->b.info.workgroup_size[2]),
   };

   memset(desc->root.cs.base_group, 0, sizeof(desc->root.cs.base_group));
   desc->root.cs.group_count_addr =
      hk_pool_upload(cmd, &group_count, sizeof(group_count), 8);

   dispatch(cmd, agx_3d(x, y, z), true);
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

   /* The CDM fetches the group counts through the address it is given, and a
    * sparse buffer's read-only shadow of a page a GPU copy just wrote through
    * the read-write mapping can still read as stale: use the read-write one.
    */
   uint64_t dispatch_addr = hk_buffer_address(buffer, offset, false);
   assert(dispatch_addr != 0);

   desc->root.cs.group_count_addr = dispatch_addr;
   dispatch(cmd, agx_grid_indirect(dispatch_addr), false);
}

/* A ray-tracing pipeline's kernel (hk_rt_pipeline.c) reads its
 * VkTraceRaysIndirectCommand2KHR at params. A direct grid counts rays; an
 * indirect one is an indirect-local grid: rays, then the workgroup.
 */
static void
trace_rays(struct hk_cmd_buffer *cmd, uint64_t params, struct agx_grid grid)
{
   struct hk_descriptor_state *desc = &cmd->state.rt.descriptors;
   struct hk_shader *s = hk_only_variant(cmd->state.rt.shader);
   if (!s || agx_is_shader_empty(&s->b))
      return;

   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   desc->root.cs.rt_params = params;

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, true /* compute */);
   if (!cs)
      return;

   struct agx_workgroup local_size =
      agx_workgroup(s->b.info.workgroup_size[0], s->b.info.workgroup_size[1],
                    s->b.info.workgroup_size[2]);

   hk_dispatch_with_local_size(cmd, cs, s, grid, local_size);
   cs->stats.calls++;
}

static uint64_t
upload_rt_params(struct hk_cmd_buffer *cmd,
                 const VkStridedDeviceAddressRegionKHR *raygen,
                 const VkStridedDeviceAddressRegionKHR *miss,
                 const VkStridedDeviceAddressRegionKHR *hit,
                 const VkStridedDeviceAddressRegionKHR *callable,
                 uint32_t width, uint32_t height, uint32_t depth)
{
   VkTraceRaysIndirectCommand2KHR params = {
      .raygenShaderRecordAddress = raygen->deviceAddress,
      .raygenShaderRecordSize = raygen->size,
      .missShaderBindingTableAddress = miss->deviceAddress,
      .missShaderBindingTableSize = miss->size,
      .missShaderBindingTableStride = miss->stride,
      .hitShaderBindingTableAddress = hit->deviceAddress,
      .hitShaderBindingTableSize = hit->size,
      .hitShaderBindingTableStride = hit->stride,
      .callableShaderBindingTableAddress = callable->deviceAddress,
      .callableShaderBindingTableSize = callable->size,
      .callableShaderBindingTableStride = callable->stride,
      .width = width,
      .height = height,
      .depth = depth,
   };

   return hk_pool_upload(cmd, &params, sizeof(params), 16);
}

/* The grid of a dispatch whose size is at size: copied on the GPU */
static struct agx_grid
indirect_rt_grid(struct hk_cmd_buffer *cmd, uint64_t size)
{
   const struct hk_shader *s = hk_only_variant(cmd->state.rt.shader);
   uint32_t grid[6] = {
      0,
      0,
      0,
      s->b.info.workgroup_size[0],
      s->b.info.workgroup_size[1],
      s->b.info.workgroup_size[2],
   };

   uint64_t addr = hk_pool_upload(cmd, grid, sizeof(grid), 16);
   hk_cmd_copy(cmd, addr, size, 3 * sizeof(uint32_t));

   return agx_grid_indirect_local(addr);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdTraceRaysKHR(
   VkCommandBuffer commandBuffer,
   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
   uint32_t width, uint32_t height, uint32_t depth)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (!cmd->state.rt.shader || !width || !height || !depth)
      return;

   uint64_t params =
      upload_rt_params(cmd, pRaygenShaderBindingTable, pMissShaderBindingTable,
                       pHitShaderBindingTable, pCallableShaderBindingTable,
                       width, height, depth);

   trace_rays(cmd, params, agx_3d(width, height, depth));
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdTraceRaysIndirectKHR(
   VkCommandBuffer commandBuffer,
   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
   VkDeviceAddress indirectDeviceAddress)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (!cmd->state.rt.shader)
      return;

   /* The size is the indirect command's, a VkTraceRaysIndirectCommandKHR */
   uint64_t params = upload_rt_params(
      cmd, pRaygenShaderBindingTable, pMissShaderBindingTable,
      pHitShaderBindingTable, pCallableShaderBindingTable, 0, 0, 0);

   hk_cmd_copy(cmd, params + offsetof(VkTraceRaysIndirectCommand2KHR, width),
               indirectDeviceAddress, 3 * sizeof(uint32_t));

   trace_rays(cmd, params, indirect_rt_grid(cmd, indirectDeviceAddress));
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdTraceRaysIndirect2KHR(VkCommandBuffer commandBuffer,
                            VkDeviceAddress indirectDeviceAddress)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (!cmd->state.rt.shader)
      return;

   uint64_t size =
      indirectDeviceAddress + offsetof(VkTraceRaysIndirectCommand2KHR, width);

   trace_rays(cmd, indirectDeviceAddress, indirect_rt_grid(cmd, size));
}
