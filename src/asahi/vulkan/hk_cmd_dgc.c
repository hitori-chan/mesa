/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

/* VK_EXT_device_generated_commands. The tokens stay in the application's buffer
 * and are read at execution, so preprocessing does nothing. Execution records a
 * draw or dispatch for each possible sequence as an indirect one, each on its
 * own root table, and kernels fill those root tables and the indirect arguments
 * from the sequences before the first of them runs. Sequences past the count
 * draw nothing. The recording costs as much as maxSequenceCount draws.
 *
 * A sequence's index buffer has no place in a root table: for a hardware
 * vertex shader a kernel writes the draw's VDM words, for tessellation and
 * geometry the setup kernel reads the token. Those take 32-bit indices: their
 * software vertex fetch is keyed on the size.
 */

#include "agx_nir_lower_vbo.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "vk_device_generated_commands.h"

struct hk_indirect_commands_layout {
   struct vk_indirect_command_layout vk;
   VkIndirectCommandsLayoutTokenEXT tokens[0];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_indirect_commands_layout, vk.base,
                               VkIndirectCommandsLayoutEXT,
                               VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_EXT)

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateIndirectCommandsLayoutEXT(
   VkDevice _device, const VkIndirectCommandsLayoutCreateInfoEXT *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkIndirectCommandsLayoutEXT *pIndirectCommandsLayout)
{
   VK_FROM_HANDLE(hk_device, dev, _device);

   struct hk_indirect_commands_layout *layout =
      vk_indirect_command_layout_create(&dev->vk, pCreateInfo, pAllocator,
                                        sizeof(*layout));
   if (!layout)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pIndirectCommandsLayout = hk_indirect_commands_layout_to_handle(layout);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyIndirectCommandsLayoutEXT(
   VkDevice _device, VkIndirectCommandsLayoutEXT indirectCommandsLayout,
   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   VK_FROM_HANDLE(hk_indirect_commands_layout, layout, indirectCommandsLayout);

   if (layout)
      vk_indirect_command_layout_destroy(&dev->vk, pAllocator, &layout->vk);
}

VKAPI_ATTR void VKAPI_CALL
hk_GetGeneratedCommandsMemoryRequirementsEXT(
   VkDevice _device, const VkGeneratedCommandsMemoryRequirementsInfoEXT *pInfo,
   VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_physical_device *pdev = hk_device_physical(dev);

   /* Nothing is preprocessed into it */
   pMemoryRequirements->memoryRequirements = (VkMemoryRequirements){
      .size = 64,
      .alignment = 4,
      .memoryTypeBits = BITFIELD_MASK(pdev->mem_type_count),
   };
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdPreprocessGeneratedCommandsEXT(
   VkCommandBuffer commandBuffer,
   const VkGeneratedCommandsInfoEXT *pGeneratedCommandsInfo,
   VkCommandBuffer stateCommandBuffer)
{
}

static struct libagx_dgc_params
hk_dgc_params(struct hk_cmd_buffer *cmd,
              struct hk_indirect_commands_layout *layout,
              const VkGeneratedCommandsInfoEXT *info, uint64_t roots,
              uint64_t args, unsigned views, unsigned draw_src_B,
              unsigned draw_words)
{
   struct libagx_dgc_params p = {
      .stream = info->indirectAddress,
      .count = info->sequenceCountAddress,
      .roots = roots,
      .args = args,
      .cond = cmd->state.cond.addr,
      .cond_inv = cmd->state.cond.inverted,
      .stride_B = layout->vk.stride,
      .max_sequences = info->maxSequenceCount,
      .views = views,
      .root_size_B = sizeof(struct hk_root_descriptor_table),
      .push_B = hk_root_descriptor_offset(push),
      .sequence_index_B = -1,
      .draw_src_B = draw_src_B,
      .draw_words = draw_words,
      .index_src_B = layout->vk.index_src_offset_B,
      .index_dx = layout->vk.index_mode_is_dx,
      .attrib_base_B = hk_root_descriptor_offset(draw.attrib_base),
      .attrib_clamps_B = hk_root_descriptor_offset(draw.attrib_clamps),
      .attrib_strides_B = hk_root_descriptor_offset(draw.attrib_strides),
   };

   assert(layout->vk.n_pc_layouts <= ARRAY_SIZE(p.copies));

   for (unsigned i = 0; i < layout->vk.n_pc_layouts; ++i) {
      const struct vk_indirect_command_push_constant_layout *pc =
         &layout->vk.pc_layouts[i];

      p.copies[p.nr_copies++] = (struct libagx_dgc_copy){
         .dst_B = pc->dst_offset_B,
         .src_B = pc->src_offset_B,
         .size_B = pc->size_B,
      };
   }

   if (layout->vk.dgc_info & BITFIELD_BIT(MESA_VK_DGC_SI))
      p.sequence_index_B = layout->vk.si_layout.dst_offset_B;

   return p;
}

/* The vertex attributes that read a buffer of the sequence */
static void
hk_dgc_attribs(struct hk_cmd_buffer *cmd,
               struct hk_indirect_commands_layout *layout,
               struct libagx_dgc_params *p)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   struct hk_shader *sw_vs = hk_bound_sw_vs(gfx);

   unsigned slot = 0;
   u_foreach_bit(a, sw_vs->info.vs.attribs_read) {
      if (dyn->vi->attributes_valid & BITFIELD_BIT(a)) {
         struct vk_vertex_attribute_state attr = dyn->vi->attributes[a];

         for (unsigned i = 0; i < layout->vk.n_vb_layouts; ++i) {
            if (layout->vk.vb_layouts[i].binding != attr.binding)
               continue;

            enum pipe_format fmt = hk_format_to_pipe_format(attr.format);
            enum pipe_format interchange = agx_vbo_internal_format(fmt);

            p->attribs[p->nr_attribs++] = (struct libagx_dgc_attrib){
               .src_B = layout->vk.vb_layouts[i].src_offset_B,
               .slot = slot,
               .offset_B = attr.offset,
               .element_B = util_format_get_blocksize(fmt),
               .interchange_B = util_format_get_blocksize(interchange),
            };
         }
      }

      ++slot;
   }
}

static void
hk_dgc_graphics(struct hk_cmd_buffer *cmd,
                struct hk_indirect_commands_layout *layout,
                const VkGeneratedCommandsInfoEXT *info)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   uint32_t dgc = layout->vk.dgc_info;
   uint32_t max = info->maxSequenceCount;

   bool indexed = dgc & BITFIELD_BIT(MESA_VK_DGC_DRAW_INDEXED);
   bool mesh = dgc & BITFIELD_BIT(MESA_VK_DGC_DRAW_MESH);
   bool index_token = dgc & BITFIELD_BIT(MESA_VK_DGC_IB);
   bool vb_token = dgc & BITFIELD_BIT(MESA_VK_DGC_VB);

   unsigned views = MAX2(util_bitcount(gfx->render.view_mask), 1);
   unsigned words = indexed ? 5 : mesh ? 3 : 4;

   struct agx_ptr roots =
      hk_pool_alloc(cmd, sizeof(struct hk_root_descriptor_table) * views * max, 8);
   struct agx_ptr args = hk_pool_alloc(cmd, words * 4 * max, 4);
   if (!roots.gpu || !args.gpu)
      return;

   struct libagx_dgc_params p =
      hk_dgc_params(cmd, layout, info, roots.gpu, args.gpu, views,
                    layout->vk.draw_src_offset_B, words);

   if (vb_token)
      hk_dgc_attribs(cmd, layout, &p);

   uint64_t params = hk_pool_upload(cmd, &p, sizeof(p), 8);

   libagx_dgc_args(cmd, agx_1d(max), AGX_BARRIER_ALL | AGX_PREGFX, params);

   if (p.nr_copies || p.nr_attribs || p.sequence_index_B >= 0) {
      libagx_dgc_roots(cmd, agx_1d(max * views), AGX_BARRIER_ALL | AGX_PREGFX,
                       params);
   }

   struct util_dynarray vdm;
   util_dynarray_init(&vdm, NULL);

   gfx->dgc.dynamic_strides = vb_token;
   gfx->dgc.index_token = index_token;
   gfx->dgc.index_dx = layout->vk.index_mode_is_dx;
   gfx->dgc.vdm = &vdm;

   /* What draws read as the index size. A sequence's own size overrides it in
    * the tessellation and geometry paths, which then read it from their
    * vertex parameters.
    */
   if (index_token) {
      gfx->index = (struct hk_index_buffer_state){.size = AGX_INDEX_SIZE_U32};
      vk_cmd_set_index_buffer_type(&cmd->vk, VK_INDEX_TYPE_UINT32);
   }

   cmd->state.dgc_roots = roots;
   cmd->state.dgc_roots_left = max * views;

   for (uint32_t seq = 0; seq < max; ++seq) {
      uint64_t addr = args.gpu + (seq * words * 4);

      gfx->dgc.sequence = seq;
      gfx->dgc.index_addr = info->indirectAddress + (seq * layout->vk.stride) +
                            layout->vk.index_src_offset_B;
      gfx->descriptors.root_dirty = true;

      if (mesh) {
         hk_draw_mesh(cmd, 0, agx_grid_indirect(addr));
      } else if (indexed) {
         struct hk_addr_range ib = gfx->index.buffer;
         hk_draw(cmd, 0, agx_draw_indexed_indirect(addr, ib.addr, ib.range, 0, 0));
      } else {
         hk_draw(cmd, 0, agx_draw_indirect(addr));
      }
   }

   cmd->state.dgc_roots_left = 0;

   if (util_dynarray_num_elements(&vdm, struct libagx_dgc_vdm)) {
      uint64_t table = hk_pool_upload(cmd, vdm.data, vdm.size, 8);

      libagx_dgc_vdm(cmd,
                     agx_1d(util_dynarray_num_elements(
                        &vdm, struct libagx_dgc_vdm)),
                     AGX_BARRIER_ALL | AGX_PREGFX, params, table,
                     dyn->ia.primitive_restart_enable, dev->dev.chip);
   }

   util_dynarray_fini(&vdm);

   /* The VDM words set the restart index behind hk_flush_index */
   struct hk_cs *cs = cmd->current_cs.gfx;
   if (cs && index_token)
      cs->restart_index = 0;

   /* gfx->root is a sequence's. The buffers, strides and index buffer the
    * sequences bound leave the state undefined: bind what was bound.
    */
   memset(&gfx->dgc, 0, sizeof(gfx->dgc));
   gfx->descriptors.root_dirty = true;

   if (vb_token)
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_VI_BINDING_STRIDES);
}

static void
hk_dgc_compute(struct hk_cmd_buffer *cmd,
               struct hk_indirect_commands_layout *layout,
               const VkGeneratedCommandsInfoEXT *info)
{
   uint32_t max = info->maxSequenceCount;

   struct agx_ptr roots =
      hk_pool_alloc(cmd, sizeof(struct hk_root_descriptor_table) * max, 8);
   struct agx_ptr args = hk_pool_alloc(cmd, sizeof(VkDispatchIndirectCommand) * max, 4);
   if (!roots.gpu || !args.gpu)
      return;

   struct libagx_dgc_params p =
      hk_dgc_params(cmd, layout, info, roots.gpu, args.gpu, 1,
                    layout->vk.dispatch_src_offset_B, 3);

   uint64_t params = hk_pool_upload(cmd, &p, sizeof(p), 8);

   libagx_dgc_args(cmd, agx_1d(max), AGX_BARRIER_ALL, params);

   if (p.nr_copies || p.sequence_index_B >= 0)
      libagx_dgc_roots(cmd, agx_1d(max), AGX_BARRIER_ALL, params);

   cmd->state.dgc_roots = roots;
   cmd->state.dgc_roots_left = max;

   for (uint32_t seq = 0; seq < max; ++seq) {
      hk_cmd_dispatch_dgc(cmd, args.gpu + seq * sizeof(VkDispatchIndirectCommand));
   }

   cmd->state.dgc_roots_left = 0;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdExecuteGeneratedCommandsEXT(
   VkCommandBuffer commandBuffer, VkBool32 isPreprocessed,
   const VkGeneratedCommandsInfoEXT *pGeneratedCommandsInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_indirect_commands_layout, layout,
                  pGeneratedCommandsInfo->indirectCommandsLayout);

   if (!pGeneratedCommandsInfo->maxSequenceCount)
      return;

   if (layout->vk.dgc_info & BITFIELD_BIT(MESA_VK_DGC_DISPATCH))
      hk_dgc_compute(cmd, layout, pGeneratedCommandsInfo);
   else
      hk_dgc_graphics(cmd, layout, pGeneratedCommandsInfo);
}

/* Execution sets: supportedIndirectCommandsShaderStages*Binding are zero */
VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateIndirectExecutionSetEXT(
   VkDevice _device, const VkIndirectExecutionSetCreateInfoEXT *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkIndirectExecutionSetEXT *pIndirectExecutionSet)
{
   VK_FROM_HANDLE(hk_device, dev, _device);

   return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyIndirectExecutionSetEXT(
   VkDevice _device, VkIndirectExecutionSetEXT indirectExecutionSet,
   const VkAllocationCallbacks *pAllocator)
{
}

VKAPI_ATTR void VKAPI_CALL
hk_UpdateIndirectExecutionSetPipelineEXT(
   VkDevice _device, VkIndirectExecutionSetEXT indirectExecutionSet,
   uint32_t executionSetWriteCount,
   const VkWriteIndirectExecutionSetPipelineEXT *pExecutionSetWrites)
{
}

VKAPI_ATTR void VKAPI_CALL
hk_UpdateIndirectExecutionSetShaderEXT(
   VkDevice _device, VkIndirectExecutionSetEXT indirectExecutionSet,
   uint32_t executionSetWriteCount,
   const VkWriteIndirectExecutionSetShaderEXT *pExecutionSetWrites)
{
}
