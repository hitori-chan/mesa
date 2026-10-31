/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

/* Ray-tracing pipelines, tier 1. The common runtime (vk_pipeline.c) compiles
 * each stage to a holder of its NIR and links a pipeline's groups, its
 * libraries' included, through link_rt_pipeline: one compute kernel
 * (hk_nir_rt_pipeline.c), bound at the ray-tracing bind point. A group
 * handle names its shaders by content, so a handle is the same in every
 * pipeline and library, and replays.
 */

#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_nir_ray_tracing.h"
#include "hk_shader.h"

#include "util/blob.h"
#include "util/mesa-blake3.h"
#include "nir_serialize.h"
#include "vk_log.h"
#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"

struct hk_rt_stage_shader {
   struct vk_shader vk;

   /* Its NIR's content, never 0: a null group's */
   uint64_t id;
   struct vk_pipeline_robustness_state rs;

   void *nir;
   uint32_t nir_size;
};

static void
hk_rt_stage_destroy(struct vk_device *vk_dev, struct vk_shader *vk_shader,
                    const VkAllocationCallbacks *pAllocator)
{
   struct hk_rt_stage_shader *s =
      container_of(vk_shader, struct hk_rt_stage_shader, vk);

   free(s->nir);
   vk_shader_free(vk_dev, pAllocator, &s->vk);
}

static bool
hk_rt_stage_serialize(struct vk_device *vk_dev,
                      const struct vk_shader *vk_shader, struct blob *blob)
{
   const struct hk_rt_stage_shader *s =
      container_of(vk_shader, struct hk_rt_stage_shader, vk);

   /* The stage first, as hk_api_shader_serialize */
   blob_write_uint8(blob, vk_shader->stage);
   blob_write_uint64(blob, s->id);
   blob_write_bytes(blob, &s->rs, sizeof(s->rs));
   blob_write_uint32(blob, s->nir_size);
   blob_write_bytes(blob, s->nir, s->nir_size);

   return !blob->out_of_memory;
}

static VkResult
hk_rt_stage_get_executable_properties(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t *executable_count, VkPipelineExecutablePropertiesKHR *properties)
{
   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutablePropertiesKHR, out, properties,
                          executable_count);

   vk_outarray_append_typed(VkPipelineExecutablePropertiesKHR, &out, props)
   {
      props->stages = mesa_to_vk_shader_stage(vk_shader->stage);
      props->subgroupSize = 32;
      VK_COPY_STR(props->name, _mesa_shader_stage_to_string(vk_shader->stage));
      VK_PRINT_STR(props->description,
                   "%s shader, inlined into its pipeline's kernel",
                   _mesa_shader_stage_to_string(vk_shader->stage));
   }

   return vk_outarray_status(&out);
}

static VkResult
hk_rt_stage_get_executable_statistics(
   UNUSED struct vk_device *device, UNUSED const struct vk_shader *vk_shader,
   UNUSED uint32_t executable_index, uint32_t *statistic_count,
   VkPipelineExecutableStatisticKHR *statistics)
{
   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableStatisticKHR, out, statistics,
                          statistic_count);
   return vk_outarray_status(&out);
}

static VkResult
hk_rt_stage_get_executable_internal_representations(
   UNUSED struct vk_device *device, UNUSED const struct vk_shader *vk_shader,
   UNUSED uint32_t executable_index, uint32_t *internal_representation_count,
   VkPipelineExecutableInternalRepresentationKHR *internal_representations)
{
   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableInternalRepresentationKHR, out,
                          internal_representations,
                          internal_representation_count);
   return vk_outarray_status(&out);
}

static const struct vk_shader_ops hk_rt_stage_ops = {
   .destroy = hk_rt_stage_destroy,
   .serialize = hk_rt_stage_serialize,
   .get_executable_properties = hk_rt_stage_get_executable_properties,
   .get_executable_statistics = hk_rt_stage_get_executable_statistics,
   .get_executable_internal_representations =
      hk_rt_stage_get_executable_internal_representations,
};

static VkResult
hk_rt_stage_init(struct hk_device *dev, mesa_shader_stage stage, uint64_t id,
                 const struct vk_pipeline_robustness_state *rs, const void *nir,
                 uint32_t nir_size, const VkAllocationCallbacks *pAllocator,
                 struct vk_shader **shader_out)
{
   struct hk_rt_stage_shader *s = vk_shader_zalloc(
      &dev->vk, &hk_rt_stage_ops, stage, pAllocator, sizeof(*s));
   if (s == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   s->nir = malloc(nir_size);
   if (s->nir == NULL) {
      vk_shader_free(&dev->vk, pAllocator, &s->vk);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   memcpy(s->nir, nir, nir_size);
   s->nir_size = nir_size;
   s->id = id;
   s->rs = *rs;

   /* The kernel's scratch is sized when it is linked: no stack */
   s->vk.stack_size = 0;

   *shader_out = &s->vk;
   return VK_SUCCESS;
}

static VkResult
hk_rt_stage_create(struct hk_device *dev,
                   const struct vk_shader_compile_info *info,
                   const VkAllocationCallbacks *pAllocator,
                   struct vk_shader **shader_out)
{
   struct blob blob;
   blob_init(&blob);
   nir_serialize(&blob, info->nir, false);

   if (blob.out_of_memory) {
      blob_finish(&blob);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   uint8_t stage = info->stage;
   struct mesa_blake3 ctx;
   blake3_hash hash;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, &stage, sizeof(stage));
   _mesa_blake3_update(&ctx, blob.data, blob.size);
   _mesa_blake3_final(&ctx, hash);

   uint64_t id;
   memcpy(&id, hash, sizeof(id));
   if (id == 0)
      id = 1;

   VkResult result =
      hk_rt_stage_init(dev, info->stage, id, info->robustness, blob.data,
                       blob.size, pAllocator, shader_out);
   blob_finish(&blob);
   return result;
}

VkResult
hk_compile_rt_stages(struct hk_device *dev, uint32_t shader_count,
                     struct vk_shader_compile_info *infos,
                     const VkAllocationCallbacks *pAllocator,
                     struct vk_shader **shaders_out)
{
   VkResult result = VK_SUCCESS;
   uint32_t i;

   for (i = 0; i < shader_count && result == VK_SUCCESS; i++) {
      result = hk_rt_stage_create(dev, &infos[i], pAllocator, &shaders_out[i]);
      ralloc_free(infos[i].nir);
   }

   if (result != VK_SUCCESS) {
      for (uint32_t j = i; j < shader_count; j++)
         ralloc_free(infos[j].nir);

      for (uint32_t j = 0; j + 1 < i; j++)
         vk_shader_destroy(&dev->vk, shaders_out[j], pAllocator);

      memset(shaders_out, 0, shader_count * sizeof(*shaders_out));
   }

   return result;
}

VkResult
hk_rt_stage_deserialize(struct hk_device *dev, struct blob_reader *blob,
                        mesa_shader_stage stage,
                        const VkAllocationCallbacks *pAllocator,
                        struct vk_shader **shader_out)
{
   uint64_t id = blob_read_uint64(blob);

   struct vk_pipeline_robustness_state rs;
   blob_copy_bytes(blob, &rs, sizeof(rs));

   uint32_t nir_size = blob_read_uint32(blob);
   const void *nir = blob_read_bytes(blob, nir_size);

   if (blob->overrun || id == 0)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   return hk_rt_stage_init(dev, stage, id, &rs, nir, nir_size, pAllocator,
                           shader_out);
}

/* {general or closest hit, any hit, intersection, 0} */
static void
hk_rt_group_handle(const struct vk_shader **shaders, uint32_t shader_count,
                   void *output)
{
   uint64_t ids[HK_RT_HANDLE_SIZE / 8] = {0};

   for (uint32_t i = 0; i < shader_count; i++) {
      const struct hk_rt_stage_shader *s =
         container_of(shaders[i], struct hk_rt_stage_shader, vk);

      switch (s->vk.stage) {
      case MESA_SHADER_ANY_HIT:
         ids[1] = s->id;
         break;
      case MESA_SHADER_INTERSECTION:
         ids[2] = s->id;
         break;
      default:
         ids[0] = s->id;
         break;
      }
   }

   memcpy(output, ids, sizeof(ids));
}

void
hk_write_rt_shader_group(UNUSED struct vk_device *device,
                         UNUSED VkRayTracingShaderGroupTypeKHR type,
                         const struct vk_shader **shaders,
                         uint32_t shader_count, void *output)
{
   hk_rt_group_handle(shaders, shader_count, output);
}

void
hk_write_rt_shader_group_replay_handle(UNUSED struct vk_device *device,
                                       const struct vk_shader **shaders,
                                       uint32_t shader_count, void *output)
{
   hk_rt_group_handle(shaders, shader_count, output);
}

/* A handle is its shaders' contents: replaying needs nothing */
void
hk_replay_rt_shader_group(UNUSED struct vk_device *device,
                          UNUSED VkRayTracingShaderGroupTypeKHR type,
                          UNUSED uint32_t shader_count,
                          UNUSED struct vk_shader **shaders,
                          UNUSED const void *replay_data)
{
}

static void
hk_merge_robustness(struct vk_pipeline_robustness_state *rs,
                    const struct vk_pipeline_robustness_state *stage)
{
   /* Each behaviour's enum grows with its strength */
   rs->storage_buffers = MAX2(rs->storage_buffers, stage->storage_buffers);
   rs->uniform_buffers = MAX2(rs->uniform_buffers, stage->uniform_buffers);
   rs->vertex_inputs = MAX2(rs->vertex_inputs, stage->vertex_inputs);
   rs->images = MAX2(rs->images, stage->images);
   rs->null_uniform_buffer_descriptor |= stage->null_uniform_buffer_descriptor;
   rs->null_storage_buffer_descriptor |= stage->null_storage_buffer_descriptor;
}

VkResult
hk_link_rt_pipeline(struct vk_device *vk_dev,
                    const struct vk_rt_link_info *info,
                    const VkAllocationCallbacks *pAllocator,
                    struct vk_shader **shader_out)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);
   const nir_shader_compiler_options *options =
      hk_get_nir_options(vk_dev->physical, MESA_SHADER_COMPUTE, NULL);
   VkResult result = VK_SUCCESS;

   void *mem_ctx = ralloc_context(NULL);
   uint32_t max_stages = MAX2(info->group_count * 3, 1);
   struct hk_rt_stage *stages =
      rzalloc_array(mem_ctx, struct hk_rt_stage, max_stages);
   const struct hk_rt_stage_shader **holders =
      rzalloc_array(mem_ctx, const struct hk_rt_stage_shader *, max_stages);
   struct hk_rt_group *groups =
      rzalloc_array(mem_ctx, struct hk_rt_group, MAX2(info->group_count, 1));
   uint32_t stage_count = 0;

   struct vk_pipeline_robustness_state rs = vk_robustness_disabled;

   for (uint32_t g = 0; g < info->group_count; g++) {
      const struct vk_rt_link_group *group = &info->groups[g];
      groups[g] = (struct hk_rt_group){
         .general = HK_RT_UNUSED,
         .closest_hit = HK_RT_UNUSED,
         .any_hit = HK_RT_UNUSED,
         .intersection = HK_RT_UNUSED,
      };

      for (uint32_t i = 0; i < group->shader_count; i++) {
         const struct hk_rt_stage_shader *s =
            container_of(group->shaders[i], struct hk_rt_stage_shader, vk);

         /* Equal IDs are the same shader: inlined once a ladder */
         uint32_t index = 0;
         while (index < stage_count && stages[index].id != s->id)
            index++;

         if (index == stage_count) {
            stages[stage_count].id = s->id;
            holders[stage_count++] = s;
            hk_merge_robustness(&rs, &s->rs);
         }

         switch (s->vk.stage) {
         case MESA_SHADER_CLOSEST_HIT:
            groups[g].closest_hit = index;
            break;
         case MESA_SHADER_ANY_HIT:
            groups[g].any_hit = index;
            break;
         case MESA_SHADER_INTERSECTION:
            groups[g].intersection = index;
            break;
         default:
            groups[g].general = index;
            break;
         }
      }
   }

   for (uint32_t i = 0; i < stage_count; i++) {
      struct blob_reader blob;
      blob_reader_init(&blob, holders[i]->nir, holders[i]->nir_size);
      stages[i].nir = nir_deserialize(mem_ctx, options, &blob);

      if (stages[i].nir == NULL) {
         result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         goto out;
      }
   }

   const struct hk_rt_kernel_info kernel_info = {
      .stages = stages,
      .stage_count = stage_count,
      .groups = groups,
      .group_count = info->group_count,
      .skip_triangles =
         info->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR,
      .skip_aabbs =
         info->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR,
   };

   if (!hk_nir_rt_kernel_supported(&kernel_info)) {
      result = vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                         "a callable shader calls one: not inlined yet");
      goto out;
   }

   nir_shader *nir = hk_nir_build_rt_kernel(options, &kernel_info);

   hk_preprocess_nir_internal(vk_dev->physical, nir);
   const nir_lower_compute_system_values_options csv_options = {0};
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);

   struct vk_pipeline_layout *layout = info->layout;
   struct vk_shader_compile_info compile_info = {
      .stage = MESA_SHADER_COMPUTE,
      .nir = nir,
      .robustness = &rs,
      .set_layout_count = layout != NULL ? layout->set_count : 0,
      .set_layouts = layout != NULL ? layout->set_layouts : NULL,
   };

   struct hk_api_shader *obj;
   result = hk_compile_rt_kernel(dev, &compile_info, pAllocator, &obj);
   if (result == VK_SUCCESS) {
      hk_only_variant(obj)->info.cs.ray_tracing = true;
      *shader_out = &obj->vk;
   }

out:
   ralloc_free(mem_ctx);
   return result;
}

void
hk_cmd_set_rt_state(UNUSED struct vk_command_buffer *vk_cmd,
                    UNUSED struct vk_pipeline_layout *layout,
                    UNUSED VkDeviceSize scratch_size,
                    UNUSED uint32_t ray_queries,
                    UNUSED const uint8_t *dynamic_descriptor_offsets)
{
   /* The kernel carries its scratch, its descriptors are the bind point's */
}

void
hk_cmd_set_rt_stack_size(struct vk_command_buffer *vk_cmd,
                         VkDeviceSize stack_size)
{
   struct hk_cmd_buffer *cmd = container_of(vk_cmd, struct hk_cmd_buffer, vk);

   /* Inlined stages have no stack: kept for tier 2 */
   cmd->state.rt.stack_size = stack_size;
}

void
hk_cmd_bind_rt_shader(struct vk_command_buffer *vk_cmd,
                      struct vk_shader *shader)
{
   struct hk_cmd_buffer *cmd = container_of(vk_cmd, struct hk_cmd_buffer, vk);

   cmd->state.rt.shader =
      shader != NULL ? container_of(shader, struct hk_api_shader, vk) : NULL;
}
