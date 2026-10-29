/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_descriptor_set.h"
#include "asahi/lib/agx_bo.h"
#include "util/half_float.h"
#include "vulkan/vulkan_core.h"

#include "agx_helpers.h"
#include "hk_buffer.h"
#include "hk_buffer_view.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_physical_device.h"
#include "hk_sampler.h"
#include "vk_acceleration_structure.h"

static inline uint32_t
align_u32(uint32_t v, uint32_t a)
{
   assert(a != 0 && a == (a & -a));
   return (v + a - 1) & ~(a - 1);
}

static inline void *
desc_ubo_data(struct hk_descriptor_set *set, uint32_t binding, uint32_t elem,
              uint32_t *size_out)
{
   const struct hk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];

   uint32_t offset = binding_layout->offset + elem * binding_layout->stride;
   assert(offset < set->size);

   if (size_out != NULL)
      *size_out = set->size - offset;

   return (char *)set->mapped_ptr + offset;
}

static void
write_desc(struct hk_descriptor_set *set, uint32_t binding, uint32_t elem,
           const void *desc_data, size_t desc_size)
{
   ASSERTED uint32_t dst_size;
   void *dst = desc_ubo_data(set, binding, elem, &dst_size);
   assert(desc_size <= dst_size);
   memcpy(dst, desc_data, desc_size);
}

/* The format a G15 custom border is packed for: the view's (a combined
 * descriptor), else the sampler's. A stencil view reads its S8 plane, also the
 * stencil of a combined format. A sampler made without one, written alone,
 * gets the border's 32-bit words as they are (not verified).
 */
static enum pipe_format
g15_border_format(const struct hk_sampler *sampler, struct hk_image_view *view,
                  unsigned plane)
{
   if (view && view->vk.aspects == VK_IMAGE_ASPECT_STENCIL_BIT)
      return PIPE_FORMAT_S8_UINT;

   if (view)
      return hk_image_view_plane_format(view, plane);

   if (sampler->vk.format != VK_FORMAT_UNDEFINED)
      return hk_format_to_pipe_format(sampler->vk.format);

   return vk_border_color_is_int(sampler->vk.border_color)
             ? PIPE_FORMAT_R32G32B32A32_UINT
             : PIPE_FORMAT_R32G32B32A32_FLOAT;
}

/* G15's opaque black puts its 1 in the format's last channel, the red of an
 * RGBA4 view (A4B4G4R4 in memory) and the blue of a BGRA4 one (dEQP-VK
 * sampler clamp_to_border_opaque_black). With the view known, a format whose
 * alpha is an earlier channel takes it as a custom border, packed for it.
 */
static void
g15_opaque_black_as_custom(struct hk_sampled_image_descriptor_g15 *desc,
                           struct hk_image_view *view, unsigned plane)
{
   enum pipe_format format = hk_image_view_plane_format(view, plane);
   if (util_format_description(format)->swizzle[3] >= PIPE_SWIZZLE_W)
      return;

   agx_unpack(NULL, &desc->sampler, SAMPLER, state);
   if (state.border_colour != AGX_BORDER_COLOUR_OPAQUE_BLACK)
      return;

   agx_pack(&desc->sampler, SAMPLER, cfg) {
      cfg = state;
      cfg.border_colour = AGX_BORDER_COLOUR_CUSTOM;
   }

   bool integer = util_format_is_pure_integer(format);
   const uint32_t black[4] = {0, 0, 0, integer ? 1 : fui(1.0)};
   agx_pack_border(&desc->border, black, format);
}

/* A view plane's texture and minimum LOD, as both descriptor layouts hold
 * them */
static void
desc_view_plane(const struct hk_image_view *view, uint8_t plane, bool ia,
                struct agx_texture_packed *tex, uint16_t *min_lod_fp16,
                uint16_t *min_lod_uint16)
{
   *tex = ia ? view->planes[plane].ia : view->planes[plane].sampled;

   float min_lod = MAX2(view->vk.min_lod - view->vk.base_mip_level, 0.0);

   *min_lod_fp16 = _mesa_float_to_half(min_lod);
   *min_lod_uint16 = min_lod;
}

/* The sampler a descriptor binds: its binding's immutable one, else info's */
static struct hk_sampler *
desc_sampler(struct hk_descriptor_set *set, uint32_t binding, uint32_t elem,
             const VkDescriptorImageInfo *info)
{
   const struct hk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];

   if (binding_layout->immutable_samplers)
      return binding_layout->immutable_samplers[elem];
   else
      return hk_sampler_from_handle(info->sampler);
}

/* G15's layout of a sampled image descriptor (hk_descriptor_set.h) */
static void
write_sampled_image_view_desc_g15(struct hk_descriptor_set *set,
                                  const VkDescriptorImageInfo *const info,
                                  uint32_t binding, uint32_t elem,
                                  VkDescriptorType descriptor_type)
{
   struct hk_sampled_image_descriptor_g15 desc[3] = {};

   for (unsigned i = 0; i < 3; ++i) {
      agx_set_null_texture(&desc[i].tex, true);
      desc[i].clamp_0_sampler_index_or_negative = -1;
   }

   uint8_t plane_count = 1;
   bool ia = (descriptor_type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
   struct hk_image_view *view = NULL;

   /* An image without a sampler fetches (txf) with the txf sampler */
   if (descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || ia) {
      for (unsigned i = 0; i < 3; ++i)
         agx_pack_txf_sampler(&desc[i].sampler);
   }

   if (descriptor_type != VK_DESCRIPTOR_TYPE_SAMPLER && info &&
       info->imageView != VK_NULL_HANDLE) {
      view = hk_image_view_from_handle(info->imageView);

      plane_count = view->plane_count;
      for (uint8_t plane = 0; plane < plane_count; plane++) {
         desc_view_plane(view, plane, ia, &desc[plane].tex,
                         &desc[plane].min_lod_fp16,
                         &desc[plane].min_lod_uint16);
      }
   }

   if (descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLER ||
       descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      struct hk_sampler *sampler = desc_sampler(set, binding, elem, info);

      plane_count = MAX2(plane_count, sampler->plane_count);

      for (uint8_t plane = 0; plane < plane_count; plane++) {
         uint8_t sampler_plane = MIN2(plane, sampler->plane_count - 1);
         desc[plane].sampler = sampler->planes[sampler_plane].hw->key;
         desc[plane].lod_bias_fp16 = sampler->lod_bias_fp16;

         /* The custom border, read by the hardware after the state */
         if (sampler->has_border) {
            agx_pack_border(&desc[plane].border, sampler->custom_border.uint32,
                            g15_border_format(sampler, view, plane));
         } else if (view) {
            g15_opaque_black_as_custom(&desc[plane], view, plane);
         }
      }
   }
   write_desc(set, binding, elem, desc, sizeof(desc[0]) * plane_count);
}

static void
write_sampled_image_view_desc(struct hk_descriptor_set *set,
                              const VkDescriptorImageInfo *const info,
                              uint32_t binding, uint32_t elem,
                              VkDescriptorType descriptor_type)
{
   /* The layout's device: a push descriptor set has no object of its own */
   struct hk_device *dev = (struct hk_device *)set->layout->vk.base.device;
   if (dev->dev.params.gpu_generation >= 15) {
      write_sampled_image_view_desc_g15(set, info, binding, elem,
                                        descriptor_type);
      return;
   }

   struct hk_sampled_image_descriptor desc[3] = {};

   /* "Zero" initialize */
   for (unsigned i = 0; i < 3; ++i) {
      agx_set_null_texture(&desc[i].tex, false);
   }

   uint8_t plane_count = 1;
   bool ia = (descriptor_type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);

   if (descriptor_type != VK_DESCRIPTOR_TYPE_SAMPLER && info &&
       info->imageView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(hk_image_view, view, info->imageView);

      plane_count = view->plane_count;
      for (uint8_t plane = 0; plane < plane_count; plane++) {
         desc_view_plane(view, plane, ia, &desc[plane].tex,
                         &desc[plane].min_lod_fp16,
                         &desc[plane].min_lod_uint16);
      }
   }

   if (descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLER ||
       descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      struct hk_sampler *sampler = desc_sampler(set, binding, elem, info);

      if (sampler->has_border)
         assert(plane_count == 1);
      else
         plane_count = MAX2(plane_count, sampler->plane_count);

      for (uint8_t plane = 0; plane < plane_count; plane++) {
         /* We need to replicate the last sampler plane out to all image
          * planes due to sampler table entry limitations. See
          * hk_CreateSampler in hk_sampler.c for more details.
          */
         uint8_t sampler_plane = MIN2(plane, sampler->plane_count - 1);
         assert(sampler->planes[sampler_plane].hw->index < (1 << 12));

         /* All bindless samplers are indexed from 28 in hardware, add here so
          * we don't have to care in the shader.
          */
         desc[plane].sampler_index =
            sampler->planes[sampler_plane].hw->index + 28;
         desc[plane].sampler = sampler->planes[sampler_plane].hw->key;
         desc[plane].lod_bias_fp16 = sampler->lod_bias_fp16;
         desc[plane].clamp_0_sampler_index_or_negative = -1;
      }

      if (sampler->has_border) {
         assert(sampler->plane_count == 2);
         desc[0].clamp_0_sampler_index_or_negative =
            sampler->planes[1].hw->index + 28;

         assert(desc[0].clamp_0_sampler_index_or_negative >= 0 &&
                "we have a border colour");

         static_assert(sizeof(desc[0].border) == sizeof(sampler->custom_border),
                       "fixed format");

         memcpy(desc[0].border, sampler->custom_border.uint32,
                sizeof(sampler->custom_border));
      }
   }
   write_desc(set, binding, elem, desc, sizeof(desc[0]) * plane_count);
}

static bool
set_is_g15(struct hk_descriptor_set *set)
{
   /* The layout's device: a push descriptor set has no object of its own */
   struct hk_device *dev = (struct hk_device *)set->layout->vk.base.device;
   return dev->dev.params.gpu_generation >= 15;
}

/* G15's storage image and texel buffer layout (hk_descriptor_set.h) */
static void
write_storage_desc_g15(struct hk_descriptor_set *set, uint32_t binding,
                       uint32_t elem, const struct agx_texture_packed *tex,
                       const struct agx_pbe_packed *pbe,
                       const struct agx_pbe_g15_software_packed *sw)
{
   struct hk_storage_image_descriptor_g15 desc = {};

   if (tex) {
      desc.tex = *tex;
      desc.pbe = *pbe;

      if (sw)
         memcpy(desc.pbe_pad, sw, sizeof(desc.pbe_pad));
   } else {
      agx_set_null_texture(&desc.tex, true);
      agx_set_null_pbe(&desc.pbe, true);
   }

   agx_pack_txf_sampler(&desc.txf_sampler);
   write_desc(set, binding, elem, &desc, sizeof(desc));
}

static void
write_storage_image_view_desc(struct hk_descriptor_set *set,
                              const VkDescriptorImageInfo *const info,
                              uint32_t binding, uint32_t elem)
{
   struct hk_storage_image_descriptor desc = {};

   if (set_is_g15(set)) {
      if (info && info->imageView != VK_NULL_HANDLE) {
         VK_FROM_HANDLE(hk_image_view, view, info->imageView);
         assert(view->plane_count == 1);
         write_storage_desc_g15(set, binding, elem, &view->planes[0].ro_storage,
                                &view->planes[0].storage,
                                &view->planes[0].storage_g15_sw);
      } else {
         write_storage_desc_g15(set, binding, elem, NULL, NULL, NULL);
      }
      return;
   }

   if (info && info->imageView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(hk_image_view, view, info->imageView);

      /* Storage images are always single plane */
      assert(view->plane_count == 1);
      uint8_t plane = 0;

      desc.tex = view->planes[plane].ro_storage;
      desc.pbe = view->planes[plane].storage;
   } else {
      agx_set_null_texture(&desc.tex, false);
      agx_set_null_pbe(&desc.pbe, false);
   }

   write_desc(set, binding, elem, &desc, sizeof(desc));
}

static struct hk_buffer_address
get_buffer_address(const VkDescriptorBufferInfo *const info, bool read_only)
{
   VK_FROM_HANDLE(hk_buffer, buffer, info->buffer);
   const struct hk_addr_range addr_range =
      hk_buffer_addr_range(buffer, info->offset, info->range, read_only);
   assert(addr_range.range <= UINT32_MAX);

   return (struct hk_buffer_address){
      .base_addr = addr_range.addr,
      .size = addr_range.range,
   };
}

static void
write_buffer_desc(struct hk_descriptor_set *set,
                  const VkDescriptorBufferInfo *const info, uint32_t binding,
                  uint32_t elem, bool read_only)
{
   const struct hk_buffer_address desc = get_buffer_address(info, read_only);

   write_desc(set, binding, elem, &desc, sizeof(desc));
}

static void
write_dynamic_buffer_desc(struct hk_descriptor_set *set,
                          const VkDescriptorBufferInfo *const info,
                          uint32_t binding, uint32_t elem, bool read_only)
{
   const struct hk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];

   struct hk_buffer_address *desc =
      &set->dynamic_buffers[binding_layout->dynamic_buffer_index + elem];

   *desc = get_buffer_address(info, read_only);
}

static void
write_buffer_view_desc(struct hk_descriptor_set *set,
                       const VkBufferView bufferView, uint32_t binding,
                       uint32_t elem)
{
   struct hk_storage_image_descriptor desc = {};

   if (set_is_g15(set)) {
      if (bufferView != VK_NULL_HANDLE) {
         VK_FROM_HANDLE(hk_buffer_view, view, bufferView);
         write_storage_desc_g15(set, binding, elem, &view->tex, &view->pbe,
                                NULL);
      } else {
         write_storage_desc_g15(set, binding, elem, NULL, NULL, NULL);
      }
      return;
   }

   if (bufferView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(hk_buffer_view, view, bufferView);

      desc.tex = view->tex;
      desc.pbe = view->pbe;
   } else {
      agx_set_null_texture(&desc.tex, false);
      agx_set_null_pbe(&desc.pbe, false);
   }

   write_desc(set, binding, elem, &desc, sizeof(desc));
}

static void
write_accel_struct_desc(struct hk_descriptor_set *set,
                        VkAccelerationStructureKHR handle, uint32_t binding,
                        uint32_t elem)
{
   VK_FROM_HANDLE(vk_acceleration_structure, accel, handle);
   const uint64_t desc = accel ? vk_acceleration_structure_get_va(accel) : 0;

   write_desc(set, binding, elem, &desc, sizeof(desc));
}

static void
write_accel_struct_descs(struct hk_descriptor_set *set,
                         const VkWriteDescriptorSet *write)
{
   const VkWriteDescriptorSetAccelerationStructureKHR *info =
      vk_find_struct_const(write->pNext,
                           WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR);

   for (uint32_t j = 0; j < write->descriptorCount; j++) {
      write_accel_struct_desc(set, info->pAccelerationStructures[j],
                              write->dstBinding, write->dstArrayElement + j);
   }
}

static void
write_inline_uniform_data(struct hk_descriptor_set *set,
                          const VkWriteDescriptorSetInlineUniformBlock *info,
                          uint32_t binding, uint32_t offset)
{
   assert(set->layout->binding[binding].stride == 1);
   write_desc(set, binding, offset, info->pData, info->dataSize);
}

VKAPI_ATTR void VKAPI_CALL
hk_UpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                        const VkWriteDescriptorSet *pDescriptorWrites,
                        uint32_t descriptorCopyCount,
                        const VkCopyDescriptorSet *pDescriptorCopies)
{
   for (uint32_t w = 0; w < descriptorWriteCount; w++) {
      const VkWriteDescriptorSet *write = &pDescriptorWrites[w];
      VK_FROM_HANDLE(hk_descriptor_set, set, write->dstSet);

      switch (write->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_sampled_image_view_desc(
               set, write->pImageInfo + j, write->dstBinding,
               write->dstArrayElement + j, write->descriptorType);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_storage_image_view_desc(set, write->pImageInfo + j,
                                          write->dstBinding,
                                          write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_view_desc(set, write->pTexelBufferView[j],
                                   write->dstBinding,
                                   write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_desc(
               set, write->pBufferInfo + j, write->dstBinding,
               write->dstArrayElement + j,
               write->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_dynamic_buffer_desc(
               set, write->pBufferInfo + j, write->dstBinding,
               write->dstArrayElement + j,
               write->descriptorType ==
                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC);
         }
         break;

      case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
         write_accel_struct_descs(set, write);
         break;

      case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK: {
         const VkWriteDescriptorSetInlineUniformBlock *write_inline =
            vk_find_struct_const(write->pNext,
                                 WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK);
         assert(write_inline->dataSize == write->descriptorCount);
         write_inline_uniform_data(set, write_inline, write->dstBinding,
                                   write->dstArrayElement);
         break;
      }

      default:
         break;
      }
   }

   for (uint32_t i = 0; i < descriptorCopyCount; i++) {
      const VkCopyDescriptorSet *copy = &pDescriptorCopies[i];
      VK_FROM_HANDLE(hk_descriptor_set, src, copy->srcSet);
      VK_FROM_HANDLE(hk_descriptor_set, dst, copy->dstSet);

      const struct hk_descriptor_set_binding_layout *src_binding_layout =
         &src->layout->binding[copy->srcBinding];
      const struct hk_descriptor_set_binding_layout *dst_binding_layout =
         &dst->layout->binding[copy->dstBinding];

      if (dst_binding_layout->stride > 0 && src_binding_layout->stride > 0) {
         for (uint32_t j = 0; j < copy->descriptorCount; j++) {
            ASSERTED uint32_t dst_max_size, src_max_size;
            void *dst_map = desc_ubo_data(
               dst, copy->dstBinding, copy->dstArrayElement + j, &dst_max_size);
            const void *src_map = desc_ubo_data(
               src, copy->srcBinding, copy->srcArrayElement + j, &src_max_size);
            const uint32_t copy_size =
               MIN2(dst_binding_layout->stride, src_binding_layout->stride);
            assert(copy_size <= dst_max_size && copy_size <= src_max_size);
            memcpy(dst_map, src_map, copy_size);
         }
      }

      if (vk_descriptor_type_is_dynamic(src_binding_layout->type)) {
         const uint32_t dst_dyn_start =
            dst_binding_layout->dynamic_buffer_index + copy->dstArrayElement;
         const uint32_t src_dyn_start =
            src_binding_layout->dynamic_buffer_index + copy->srcArrayElement;
         typed_memcpy(&dst->dynamic_buffers[dst_dyn_start],
                      &src->dynamic_buffers[src_dyn_start],
                      copy->descriptorCount);
      }
   }
}

void
hk_push_descriptor_set_update(struct hk_push_descriptor_set *push_set,
                              struct hk_descriptor_set_layout *layout,
                              uint32_t write_count,
                              const VkWriteDescriptorSet *writes)
{
   assert(layout->non_variable_descriptor_buffer_size < sizeof(push_set->data));
   struct hk_descriptor_set set = {
      .layout = layout,
      .size = sizeof(push_set->data),
      .mapped_ptr = push_set->data,
   };

   for (uint32_t w = 0; w < write_count; w++) {
      const VkWriteDescriptorSet *write = &writes[w];
      assert(write->dstSet == VK_NULL_HANDLE);

      switch (write->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_sampled_image_view_desc(
               &set, write->pImageInfo + j, write->dstBinding,
               write->dstArrayElement + j, write->descriptorType);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_storage_image_view_desc(&set, write->pImageInfo + j,
                                          write->dstBinding,
                                          write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_view_desc(&set, write->pTexelBufferView[j],
                                   write->dstBinding,
                                   write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_desc(
               &set, write->pBufferInfo + j, write->dstBinding,
               write->dstArrayElement + j,
               write->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
         }
         break;

      case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
         write_accel_struct_descs(&set, write);
         break;

      default:
         break;
      }
   }
}

static void hk_descriptor_pool_free(struct hk_descriptor_pool *pool,
                                    uint64_t addr, uint64_t size);

static void
hk_descriptor_set_destroy(struct hk_device *dev,
                          struct hk_descriptor_pool *pool,
                          struct hk_descriptor_set *set)
{
   list_del(&set->link);
   if (set->size > 0)
      hk_descriptor_pool_free(pool, set->addr, set->size);
   vk_descriptor_set_layout_unref(&dev->vk, &set->layout->vk);

   vk_object_free(&dev->vk, NULL, set);
}

static void
hk_destroy_descriptor_pool(struct hk_device *dev,
                           const VkAllocationCallbacks *pAllocator,
                           struct hk_descriptor_pool *pool)
{
   list_for_each_entry_safe(struct hk_descriptor_set, set, &pool->sets, link)
      hk_descriptor_set_destroy(dev, pool, set);

   util_vma_heap_finish(&pool->heap);
   agx_bo_unreference(&dev->dev, pool->bo);

   vk_object_free(&dev->vk, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateDescriptorPool(VkDevice _device,
                        const VkDescriptorPoolCreateInfo *pCreateInfo,
                        const VkAllocationCallbacks *pAllocator,
                        VkDescriptorPool *pDescriptorPool)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct hk_descriptor_pool *pool;

   pool = vk_object_zalloc(&dev->vk, pAllocator, sizeof(*pool),
                           VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   list_inithead(&pool->sets);

   const VkMutableDescriptorTypeCreateInfoEXT *mutable_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT);

   uint32_t max_align = 0;
   for (unsigned i = 0; i < pCreateInfo->poolSizeCount; ++i) {
      const VkMutableDescriptorTypeListEXT *type_list = NULL;
      if (pCreateInfo->pPoolSizes[i].type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
          mutable_info && i < mutable_info->mutableDescriptorTypeListCount)
         type_list = &mutable_info->pMutableDescriptorTypeLists[i];

      uint32_t stride, alignment;
      hk_descriptor_stride_align_for_type(pdev, pCreateInfo->pPoolSizes[i].type,
                                          type_list, &stride, &alignment);
      max_align = MAX2(max_align, alignment);
   }

   uint64_t bo_size = 0;
   for (unsigned i = 0; i < pCreateInfo->poolSizeCount; ++i) {
      const VkMutableDescriptorTypeListEXT *type_list = NULL;
      if (pCreateInfo->pPoolSizes[i].type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
          mutable_info && i < mutable_info->mutableDescriptorTypeListCount)
         type_list = &mutable_info->pMutableDescriptorTypeLists[i];

      uint32_t stride, alignment;
      hk_descriptor_stride_align_for_type(pdev, pCreateInfo->pPoolSizes[i].type,
                                          type_list, &stride, &alignment);
      bo_size +=
         MAX2(stride, max_align) * pCreateInfo->pPoolSizes[i].descriptorCount;
   }

   /* Individual descriptor sets are aligned to the min UBO alignment to
    * ensure that we don't end up with unaligned data access in any shaders.
    * This means that each descriptor buffer allocated may burn up to 16B of
    * extra space to get the right alignment.  (Technically, it's at most 28B
    * because we're always going to start at least 4B aligned but we're being
    * conservative here.)  Allocate enough extra space that we can chop it
    * into maxSets pieces and align each one of them to 32B.
    */
   bo_size += HK_MIN_UBO_ALIGNMENT * pCreateInfo->maxSets;

   if (bo_size) {
      pool->bo = agx_bo_create(&dev->dev, bo_size, 0, 0, "Descriptor pool");
      if (!pool->bo) {
         hk_destroy_descriptor_pool(dev, pAllocator, pool);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      pool->mapped_ptr = agx_bo_map(pool->bo);

      /* The BO may be larger thanks to GPU page alignment.  We may as well
       * make that extra space available to the client.
       */
      assert(pool->bo->size >= bo_size);
      util_vma_heap_init(&pool->heap, pool->bo->va->addr, pool->bo->size);
   } else {
      util_vma_heap_init(&pool->heap, 0, 0);
   }

   *pDescriptorPool = hk_descriptor_pool_to_handle(pool);
   return VK_SUCCESS;
}

static VkResult
hk_descriptor_pool_alloc(struct hk_descriptor_pool *pool, uint64_t size,
                         uint64_t alignment, uint64_t *addr_out, void **map_out)
{
   assert(size > 0);
   uint64_t addr = util_vma_heap_alloc(&pool->heap, size, alignment);
   if (addr == 0)
      return VK_ERROR_OUT_OF_POOL_MEMORY;

   assert(addr >= pool->bo->va->addr);
   assert(addr + size <= pool->bo->va->addr + pool->bo->size);
   uint64_t offset = addr - pool->bo->va->addr;

   *addr_out = addr;
   *map_out = pool->mapped_ptr + offset;

   return VK_SUCCESS;
}

static void
hk_descriptor_pool_free(struct hk_descriptor_pool *pool, uint64_t addr,
                        uint64_t size)
{
   assert(size > 0);
   assert(addr >= pool->bo->va->addr);
   assert(addr + size <= pool->bo->va->addr + pool->bo->size);
   util_vma_heap_free(&pool->heap, addr, size);
}

static VkResult
hk_descriptor_set_create(struct hk_device *dev, struct hk_descriptor_pool *pool,
                         struct hk_descriptor_set_layout *layout,
                         uint32_t variable_count,
                         struct hk_descriptor_set **out_set)
{
   struct hk_descriptor_set *set;
   VkResult result;

   uint32_t mem_size =
      sizeof(struct hk_descriptor_set) +
      layout->vk.dynamic_descriptor_count * sizeof(struct hk_buffer_address);

   set =
      vk_object_zalloc(&dev->vk, NULL, mem_size, VK_OBJECT_TYPE_DESCRIPTOR_SET);
   if (!set)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   set->size = layout->non_variable_descriptor_buffer_size;

   if (layout->binding_count > 0 &&
       (layout->binding[layout->binding_count - 1].flags &
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)) {
      uint32_t stride = layout->binding[layout->binding_count - 1].stride;

      /* Off by one so we don't underallocate the end. Otherwise vkd3d-proton
       * descriptor-performance underallocates.
       */
      set->size += stride * (variable_count + 1);
   }

   set->size = align64(set->size, HK_MIN_UBO_ALIGNMENT);

   if (set->size > 0) {
      result = hk_descriptor_pool_alloc(pool, set->size, HK_MIN_UBO_ALIGNMENT,
                                        &set->addr, &set->mapped_ptr);
      if (result != VK_SUCCESS) {
         vk_object_free(&dev->vk, NULL, set);
         return result;
      }
   }

   vk_descriptor_set_layout_ref(&layout->vk);
   set->layout = layout;

   for (uint32_t b = 0; b < layout->binding_count; b++) {
      if (layout->binding[b].type != VK_DESCRIPTOR_TYPE_SAMPLER &&
          layout->binding[b].type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
         continue;

      if (layout->binding[b].immutable_samplers == NULL)
         continue;

      uint32_t array_size = layout->binding[b].array_size;
      if (layout->binding[b].flags &
          VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)
         array_size = variable_count;

      for (uint32_t j = 0; j < array_size; j++) {
         write_sampled_image_view_desc(set, NULL, b, j,
                                       layout->binding[b].type);
      }
   }

   list_addtail(&set->link, &pool->sets);
   *out_set = set;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_AllocateDescriptorSets(VkDevice device,
                          const VkDescriptorSetAllocateInfo *pAllocateInfo,
                          VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_descriptor_pool, pool, pAllocateInfo->descriptorPool);

   VkResult result = VK_SUCCESS;
   uint32_t i;

   struct hk_descriptor_set *set = NULL;

   const VkDescriptorSetVariableDescriptorCountAllocateInfo *var_desc_count =
      vk_find_struct_const(
         pAllocateInfo->pNext,
         DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);

   /* allocate a set of buffers for each shader to contain descriptors */
   for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
      VK_FROM_HANDLE(hk_descriptor_set_layout, layout,
                     pAllocateInfo->pSetLayouts[i]);
      /* If descriptorSetCount is zero or this structure is not included in
       * the pNext chain, then the variable lengths are considered to be zero.
       */
      const uint32_t variable_count =
         var_desc_count && var_desc_count->descriptorSetCount > 0
            ? var_desc_count->pDescriptorCounts[i]
            : 0;

      result =
         hk_descriptor_set_create(dev, pool, layout, variable_count, &set);
      if (result != VK_SUCCESS)
         break;

      pDescriptorSets[i] = hk_descriptor_set_to_handle(set);
   }

   if (result != VK_SUCCESS) {
      hk_FreeDescriptorSets(device, pAllocateInfo->descriptorPool, i,
                            pDescriptorSets);
      for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
         pDescriptorSets[i] = VK_NULL_HANDLE;
      }
   }
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_FreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool,
                      uint32_t descriptorSetCount,
                      const VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_descriptor_pool, pool, descriptorPool);

   for (uint32_t i = 0; i < descriptorSetCount; i++) {
      VK_FROM_HANDLE(hk_descriptor_set, set, pDescriptorSets[i]);

      if (set)
         hk_descriptor_set_destroy(dev, pool, set);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyDescriptorPool(VkDevice device, VkDescriptorPool _pool,
                         const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_descriptor_pool, pool, _pool);

   if (!_pool)
      return;

   hk_destroy_descriptor_pool(dev, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_ResetDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                       VkDescriptorPoolResetFlags flags)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_descriptor_pool, pool, descriptorPool);

   list_for_each_entry_safe(struct hk_descriptor_set, set, &pool->sets, link)
      hk_descriptor_set_destroy(dev, pool, set);

   return VK_SUCCESS;
}

static void
hk_descriptor_set_write_template(
   struct hk_descriptor_set *set,
   const struct vk_descriptor_update_template *template, const void *data)
{
   for (uint32_t i = 0; i < template->entry_count; i++) {
      const struct vk_descriptor_template_entry *entry = &template->entries[i];

      switch (entry->type) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorImageInfo *info =
               data + entry->offset + j * entry->stride;

            write_sampled_image_view_desc(set, info, entry->binding,
                                          entry->array_element + j,
                                          entry->type);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorImageInfo *info =
               data + entry->offset + j * entry->stride;

            write_storage_image_view_desc(set, info, entry->binding,
                                          entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkBufferView *bview =
               data + entry->offset + j * entry->stride;

            write_buffer_view_desc(set, *bview, entry->binding,
                                   entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorBufferInfo *info =
               data + entry->offset + j * entry->stride;

            write_buffer_desc(set, info, entry->binding,
                              entry->array_element + j,
                              entry->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorBufferInfo *info =
               data + entry->offset + j * entry->stride;

            write_dynamic_buffer_desc(
               set, info, entry->binding, entry->array_element + j,
               entry->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC);
         }
         break;

      case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
         write_desc(set, entry->binding, entry->array_element,
                    data + entry->offset, entry->array_count);
         break;

      case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkAccelerationStructureKHR *accel =
               data + entry->offset + j * entry->stride;

            write_accel_struct_desc(set, *accel, entry->binding,
                                    entry->array_element + j);
         }
         break;

      default:
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_UpdateDescriptorSetWithTemplate(
   VkDevice device, VkDescriptorSet descriptorSet,
   VkDescriptorUpdateTemplate descriptorUpdateTemplate, const void *pData)
{
   VK_FROM_HANDLE(hk_descriptor_set, set, descriptorSet);
   VK_FROM_HANDLE(vk_descriptor_update_template, template,
                  descriptorUpdateTemplate);

   hk_descriptor_set_write_template(set, template, pData);
}

void
hk_push_descriptor_set_update_template(
   struct hk_push_descriptor_set *push_set,
   struct hk_descriptor_set_layout *layout,
   const struct vk_descriptor_update_template *template, const void *data)
{
   struct hk_descriptor_set tmp_set = {
      .layout = layout,
      .size = sizeof(push_set->data),
      .mapped_ptr = push_set->data,
   };
   hk_descriptor_set_write_template(&tmp_set, template, data);
}
