/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

/* Acceleration structures: Mesa's common builder (vk_acceleration_structure.c)
 * writes its intermediate tree, encode.comp turns it into Honeykrisp's
 * (bvh/hk_bvh_defines.h), which hk_nir_ray_tracing.c walks.
 */

#include "bvh/hk_bvh_defines.h"
#include "util/u_math.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "vk_acceleration_structure.h"
#include "vk_meta.h"

static const uint32_t encode_spv[] = {
#include "bvh/encode.spv.h"
};

static const uint32_t copy_spv[] = {
#include "bvh/copy.spv.h"
};

static_assert(sizeof(struct hk_bvh_header) == HK_BVH_ROOT_NODE_OFFSET,
              "the root follows the header");
static_assert(sizeof(struct hk_bvh_box_node) == HK_BVH_NODE_PREFETCH_SIZE,
              "the walk loads a box node");
static_assert(offsetof(struct hk_bvh_header, bounds) == 0,
              "the builder reads a bottom level's bounds at its address");
static_assert(sizeof(struct hk_bvh_instance_node) % 8 == 0 &&
                 sizeof(struct hk_bvh_box_node) % 8 == 0,
              "64-bit pointers stay aligned");

static const struct vk_acceleration_structure_build_args hk_build_args = {
   .subgroup_size = 32,
   .bvh_bounds_offset = offsetof(struct hk_bvh_header, bounds),
   .root_flags_offset = -1,
   .morton_sort_workgroup_size = 512,
   .morton_sort_kvs_per_thread = 2,
};

static uint32_t
hk_leaf_node_size(VkGeometryTypeKHR type)
{
   switch (type) {
   case VK_GEOMETRY_TYPE_TRIANGLES_KHR:
      return sizeof(struct hk_bvh_triangle_node);
   case VK_GEOMETRY_TYPE_AABBS_KHR:
      return sizeof(struct hk_bvh_aabb_node);
   default:
      return sizeof(struct hk_bvh_instance_node);
   }
}

/* The header, a box per internal node (at least the root), the leaves, and a
 * box node's size after them: the walk loads that much from every node.
 */
static uint64_t
hk_as_size(VkGeometryTypeKHR type, uint32_t leaf_count)
{
   uint64_t boxes = MAX2(leaf_count, 2) - 1;

   return sizeof(struct hk_bvh_header) +
          boxes * sizeof(struct hk_bvh_box_node) +
          (uint64_t)leaf_count * hk_leaf_node_size(type) +
          HK_BVH_NODE_PREFETCH_SIZE;
}

static VkDeviceSize
hk_get_as_size(VkDevice device,
               const struct vk_acceleration_structure_build_state *state)
{
   return hk_as_size(vk_get_as_geometry_type(state->build_info),
                     state->leaf_node_count);
}

/* LBVH only: PLOC's and H-PLOC's workgroups wait on each other's progress,
 * which G15 does not promise, and LBVH bounds the tree's depth (the walk's
 * stack, HK_BVH_STACK_SIZE).
 */
static void
hk_get_build_config(VkDevice device,
                    struct vk_acceleration_structure_build_state *state)
{
   if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE)
      state->config.internal_type = VK_INTERNAL_BUILD_TYPE_LBVH;
}

static VkResult
hk_get_pipeline(struct hk_device *dev, const char *name, const uint32_t *spv,
                uint32_t spv_size, uint32_t push_size, VkPipeline *pipeline,
                VkPipelineLayout *layout)
{
   const VkPushConstantRange range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = push_size,
   };

   VkResult result = vk_meta_get_pipeline_layout(
      &dev->vk, &dev->meta, NULL, &range, name, strlen(name), layout);
   if (result != VK_SUCCESS)
      return result;

   *pipeline = vk_meta_lookup_pipeline(&dev->meta, name, strlen(name));
   if (*pipeline != VK_NULL_HANDLE)
      return VK_SUCCESS;

   const VkShaderModuleCreateInfo module = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size,
      .pCode = spv,
   };

   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &module,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .pName = "main",
         },
      .layout = *layout,
   };

   return vk_meta_create_compute_pipeline(&dev->vk, &dev->meta, &info, name,
                                          strlen(name), pipeline);
}

static void
hk_encode(VkCommandBuffer commandBuffer, struct vk_device *vk_dev,
          struct vk_meta_device *meta,
          const struct vk_acceleration_structure_build_args *args,
          struct vk_acceleration_structure_build_state *states,
          uint32_t build_count, bool flushed_compute_after_init_update_scratch)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkPipeline pipeline;
   VkPipelineLayout layout;

   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);

   VkResult result =
      hk_get_pipeline(dev, "hk-bvh-encode", encode_spv, sizeof(encode_spv),
                      sizeof(struct hk_encode_args), &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   disp->CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                         pipeline);

   for (uint32_t i = 0; i < build_count; i++) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      VK_FROM_HANDLE(vk_acceleration_structure, dst,
                     state->build_info->dstAccelerationStructure);
      VkGeometryTypeKHR type = vk_get_as_geometry_type(state->build_info);
      uint64_t scratch = state->build_info->scratchData.deviceAddress;

      const struct hk_encode_args push = {
         .intermediate_bvh = scratch + state->scratch.ir_offset,
         .output_bvh = vk_acceleration_structure_get_va(dst),
         .header = scratch + state->scratch.header_offset,
         .size = hk_as_size(type, state->leaf_node_count),
         .leaf_node_count = state->leaf_node_count,
         .geometry_type = type,
      };

      disp->CmdPushConstants(commandBuffer, layout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                             &push);

      /* Invocation 0 writes the header, also of an empty structure */
      hk_cmd_dispatch_unaligned(commandBuffer,
                                MAX2(state->leaf_node_count, 1), 1, 1);
   }
}

static void
hk_write_buffer_cp(VkCommandBuffer commandBuffer, VkDeviceAddress addr,
                   void *data, uint32_t size)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   hk_cmd_copy(cmd, addr, hk_pool_upload(cmd, data, size, 4), size);
}

/* hk_cmd_copy's dispatch is a compute one: the pipeline barrier the builder
 * puts after these writes orders them.
 */
static void
hk_flush_buffer_write_cp(VkCommandBuffer commandBuffer)
{
}

static void
hk_cmd_fill_buffer_addr(VkCommandBuffer commandBuffer, VkDeviceAddress addr,
                        VkDeviceSize size, uint32_t data)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   hk_cmd_fill(cmd, addr, size, data);
}

static const struct vk_acceleration_structure_build_ops hk_build_ops = {
   .get_build_config = hk_get_build_config,
   .get_as_size = hk_get_as_size,
   .encode = hk_encode,
};

void
hk_device_init_accel_struct(struct hk_device *dev)
{
   dev->vk.as_build_ops = &hk_build_ops;
   dev->vk.write_buffer_cp = hk_write_buffer_cp;
   dev->vk.flush_buffer_write_cp = hk_flush_buffer_write_cp;
   dev->vk.cmd_dispatch_unaligned = hk_cmd_dispatch_unaligned;
   dev->vk.cmd_fill_buffer_addr = hk_cmd_fill_buffer_addr;
}

VKAPI_ATTR void VKAPI_CALL
hk_GetAccelerationStructureBuildSizesKHR(
   VkDevice device, VkAccelerationStructureBuildTypeKHR buildType,
   const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
   const uint32_t *pMaxPrimitiveCounts,
   VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo)
{
   vk_get_as_build_sizes(device, buildType, pBuildInfo, pMaxPrimitiveCounts,
                         pSizeInfo, &hk_build_args);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBuildAccelerationStructuresKHR(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_meta_save save;

   hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
   vk_cmd_build_acceleration_structures(commandBuffer, &dev->vk, &dev->meta,
                                        infoCount, pInfos, ppBuildRangeInfos,
                                        &hk_build_args);
   hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
}

/* The bytes of a structure are all it is (its node references are offsets),
 * and a compacted one is as big as its build wrote: the destination's size
 * covers it.
 */
VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyAccelerationStructureKHR(
   VkCommandBuffer commandBuffer, const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_acceleration_structure, src, pInfo->src);
   VK_FROM_HANDLE(vk_acceleration_structure, dst, pInfo->dst);

   hk_cmd_copy(cmd, vk_acceleration_structure_get_va(dst),
               vk_acceleration_structure_get_va(src),
               MIN2(src->size, dst->size));
}

static void
hk_cmd_serialize(struct hk_cmd_buffer *cmd, uint64_t src, uint64_t dst,
                 uint64_t max_size, uint32_t mode)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkCommandBuffer commandBuffer = hk_cmd_buffer_to_handle(cmd);
   struct hk_meta_save save;
   VkPipeline pipeline;
   VkPipelineLayout layout;

   VkResult result =
      hk_get_pipeline(dev, "hk-bvh-copy", copy_spv, sizeof(copy_spv),
                      sizeof(struct hk_copy_args), &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   struct hk_copy_args push = {.src = src, .dst = dst, .mode = mode};
   memcpy(push.uuid, pdev->vk.properties.driverUUID, VK_UUID_SIZE);
   memcpy(push.uuid + VK_UUID_SIZE / 4, pdev->vk.properties.pipelineCacheUUID,
          VK_UUID_SIZE);

   hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
   disp->CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                         pipeline);
   disp->CmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(push), &push);
   hk_cmd_dispatch_unaligned(commandBuffer, DIV_ROUND_UP(max_size, 4), 1, 1);
   hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
}

/* At most one instance per 120 bytes of the structure */
static uint64_t
hk_max_serialized_size(uint64_t as_size)
{
   return sizeof(struct vk_accel_struct_serialization_header) +
          as_size / sizeof(struct hk_bvh_instance_node) * sizeof(uint64_t) +
          as_size;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyAccelerationStructureToMemoryKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_acceleration_structure, src, pInfo->src);

   hk_cmd_serialize(cmd, vk_acceleration_structure_get_va(src),
                    pInfo->dst.deviceAddress,
                    hk_max_serialized_size(src->size),
                    HK_COPY_MODE_SERIALIZE);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyMemoryToAccelerationStructureKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_acceleration_structure, dst, pInfo->dst);

   hk_cmd_serialize(cmd, pInfo->src.deviceAddress,
                    vk_acceleration_structure_get_va(dst), dst->size,
                    HK_COPY_MODE_DESERIALIZE);
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDeviceAccelerationStructureCompatibilityKHR(
   VkDevice device, const VkAccelerationStructureVersionInfoKHR *pVersionInfo,
   VkAccelerationStructureCompatibilityKHR *pCompatibility)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_physical_device *pdev = hk_device_physical(dev);

   bool compat =
      memcmp(pVersionInfo->pVersionData, pdev->vk.properties.driverUUID,
             VK_UUID_SIZE) == 0 &&
      memcmp(pVersionInfo->pVersionData + VK_UUID_SIZE,
             pdev->vk.properties.pipelineCacheUUID, VK_UUID_SIZE) == 0;

   *pCompatibility = compat
                        ? VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR
                        : VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}
