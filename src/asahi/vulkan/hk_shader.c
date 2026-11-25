/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2023 Advanced Micro Devices, Inc.
 * Copyright 2018 Intel Corporation
 * SPDX-License-Identifier: MIT
 */
#include "hk_shader.h"
#include "util/disk_cache.h"
#include "util/u_cpu_detect.h"
#include "util/u_debug.h"
#include "util/u_queue.h"

#include "poly/nir/poly_nir.h"
#include "agx_debug.h"
#include "agx_device.h"
#include "agx_helpers.h"
#include "agx_nir_lower_vbo.h"
#include "glsl_types.h"
#include "hk_instance.h"
#include "hk_nir_ray_tracing.h"
#include "libagx.h"
#include "nir.h"
#include "nir_builder.h"

#include "agx_bo.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_physical_device.h"
#include "hk_sampler.h"
#include "hk_shader.h"

#include "nir_builder_opcodes.h"
#include "nir_builtin_builder.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "nir_xfb_info.h"
#include "shader_enums.h"
#include "vk_graphics_state.h"
#include "vk_nir_convert_ycbcr.h"
#include "vk_physical_device_features.h"
#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"
#include "vk_shader.h"
#include "vk_shader_module.h"
#include "vk_ycbcr_conversion.h"

#include "asahi/compiler/agx_compiler.h"
#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_nir.h"
#include "asahi/compiler/agx_nir_texture.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx_linker.h"
#include "asahi/lib/agx_tilebuffer.h"
#include "asahi/lib/agx_uvs.h"
#include "asahi/isa/disasm.h"
#include "compiler/spirv/nir_spirv.h"

#include "util/blob.h"
#include "util/hash_table.h"
#include "util/macros.h"
#include "util/mesa-blake3.h"
#include "nir_serialize.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "vulkan/vulkan_core.h"

struct hk_fs_key {
   bool zs_self_dep;

   /** True if sample shading is forced on via an API knob such as
    * VkPipelineMultisampleStateCreateInfo::minSampleShading
    */
   bool force_sample_shading;

   /* The pipeline draws one sample with alpha to coverage off, both static:
    * a main shader that builds a coverage word is built for its one-sample
    * target (fs.nr_samples 1) and runs the tests unless the epilog writes
    * depth or stencil (hk_compile_nir), as Apple's program knows its
    * pipeline's sample count
    */
   bool one_sample;

   uint8_t pad[1];
};
static_assert(sizeof(struct hk_fs_key) == 4, "packed");

struct hk_vs_key {
   struct agx_velem_key attribs[32];
   bool skip_prolog;
   bool static_strides;
   bool kill_psiz;
   bool pad[1];
};
static_assert(sizeof(struct hk_vs_key) == 260, "packed");

union hk_key {
   struct hk_vs_key vs;
   struct hk_fs_key fs;
};

static void
shared_var_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size =
      glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length, *align = comp_size;
}

uint64_t
hk_physical_device_compiler_flags(const struct hk_physical_device *pdev)
{
   /* This could be optimized but it doesn't matter. The device's debug flags
    * take the low 32 bits and the compiler's (AGX_MESA_DEBUG, many of which
    * change the code, bits 0-22) the next ones, so that a cache never hands a
    * shader compiled under other flags; HK_PERFTEST's that change the code
    * take bits 56-59 and a driconf option that changes the code a high bit */
   const struct hk_instance *instance =
      hk_physical_device_instance((struct hk_physical_device *)pdev);
   const uint32_t perf = pdev->perftest & HK_PERF_SHADER_CODE;
   const uint64_t perf_key = (uint64_t)!!(perf & HK_PERF_NOBORDER) |
                             ((uint64_t)!!(perf & HK_PERF_NOROBUST) << 1) |
                             ((uint64_t)!!(perf & HK_PERF_HWTESS) << 2) |
                             ((uint64_t)!!(perf & HK_PERF_HWTESS32) << 3);

   return pdev->dev.debug | (agx_get_compiler_debug() << 32) |
          (perf_key << 56) |
          ((uint64_t)pdev->sampled_views_as_arrays << 63) |
          ((uint64_t)instance->drirc.misc.d3d_nearest_mip << 62);
}

const nir_shader_compiler_options *
hk_get_nir_options(struct vk_physical_device *vk_pdev, mesa_shader_stage stage,
                   UNUSED const struct vk_pipeline_robustness_state *rs)
{
   return &agx_nir_options;
}

static struct spirv_to_nir_options
hk_get_spirv_options(struct vk_physical_device *vk_pdev,
                     UNUSED mesa_shader_stage stage,
                     const struct vk_pipeline_robustness_state *rs)
{
   return (struct spirv_to_nir_options){
      .ssbo_addr_format = hk_buffer_addr_format(rs->storage_buffers),
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .ubo_addr_format = hk_buffer_addr_format(rs->uniform_buffers),
      .shared_addr_format = nir_address_format_32bit_offset,
      .min_ssbo_alignment = HK_MIN_SSBO_ALIGNMENT,
      .min_ubo_alignment = HK_MIN_UBO_ALIGNMENT,
   };
}

void
hk_preprocess_nir_internal(struct vk_physical_device *vk_pdev, nir_shader *nir)
{
   /* Must lower before io to temps */
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_lower_terminate_to_demote);
      NIR_PASS(_, nir, nir_lower_halt_to_return);
      NIR_PASS(_, nir, nir_lower_returns);
   }

   /* Unroll loops before lowering indirects via
    * nir_lower_io_vars_to_temporaries */
   UNUSED bool progress = false;
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);

   struct set *skip = _mesa_pointer_set_create(NULL);
   do {
      progress = false;
      NIR_LOOP_PASS(progress, skip, nir, nir_lower_vars_to_ssa);
      NIR_LOOP_PASS(progress, skip, nir, nir_opt_copy_prop);
      NIR_LOOP_PASS(progress, skip, nir, nir_opt_dce);
      NIR_LOOP_PASS(progress, skip, nir, nir_opt_constant_folding);
      NIR_LOOP_PASS_NOT_IDEMPOTENT(progress, skip, nir, nir_opt_loop);
      NIR_LOOP_PASS_NOT_IDEMPOTENT(progress, skip, nir, nir_opt_loop_unroll);
   } while (progress);
   _mesa_set_destroy(skip, NULL);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      struct nir_lower_sysvals_to_varyings_options sysvals_opts = {
         .point_coord = true,
         .primitive_id = nir->info.stage == MESA_SHADER_FRAGMENT,
      };

      nir_lower_sysvals_to_varyings(nir, &sysvals_opts);
   }

   NIR_PASS(_, nir, nir_lower_system_values);

   /* Gather info before preprocess_nir but after some general lowering, so
    * inputs_read and system_values_read are accurately set.
    */
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
            nir_shader_get_entrypoint(nir), nir_var_shader_out);

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);

   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_split_struct_vars, nir_var_function_temp);

   /* Into a walk whose state and stack are local variables, which
    * agx_preprocess_nir puts in registers or scratch.
    */
   NIR_PASS(_, nir, hk_nir_lower_ray_queries);

   /* Optimize but allow copies because we haven't lowered them yet */
   agx_preprocess_nir(nir);

   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_memcpy);
}

/* A ray-tracing stage stays NIR until its pipeline is linked
 * (hk_rt_pipeline.c), lowered only as far as inlining needs; its ray queries
 * are lowered in the kernel.
 */
static void
hk_preprocess_rt_stage(nir_shader *nir)
{
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_system_values);

   /* Into variables of plain values, before they go to SSA */
   NIR_PASS(_, nir, hk_nir_lower_hit_objects);

   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);

   /* The shader record: a cast of load_shader_record_ptr */
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_constant,
            nir_address_format_64bit_global);

   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
   } while (progress);

   NIR_PASS(_, nir, nir_remove_dead_variables,
            nir_var_function_temp | nir_var_shader_temp, NULL);
}

/* The hardware's global ID runs x-fastest over the dispatch, so a compute
 * shader with quad derivatives builds it from the local ID, which
 * nir_lower_compute_system_values then lays out in 2x2 quads. The base
 * workgroup is not in load_workgroup_id.
 */
static bool
lower_global_id_for_quads(nir_builder *b, nir_intrinsic_instr *intr,
                          UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_global_invocation_id)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   unsigned bit_size = intr->def.bit_size;
   nir_def *group = nir_iadd(b, nir_u2uN(b, nir_load_workgroup_id(b), bit_size),
                             nir_load_base_workgroup_id(b, bit_size));
   nir_def *size = nir_u2uN(b, nir_load_workgroup_size(b), bit_size);
   nir_def *local = nir_u2uN(b, nir_load_local_invocation_id(b), bit_size);

   nir_def_replace(&intr->def, nir_iadd(b, nir_imul(b, group, size), local));
   return true;
}

static void
hk_preprocess_nir(struct vk_physical_device *vk_pdev, nir_shader *nir,
                  UNUSED const struct vk_pipeline_robustness_state *rs)
{
   if (mesa_shader_stage_is_rt(nir->info.stage)) {
      hk_preprocess_rt_stage(nir);
      return;
   }

   hk_preprocess_nir_internal(vk_pdev, nir);
   nir_lower_compute_system_values_options csv_options = {
      /* Task and mesh shaders use the graphics root, without a base */
      .has_base_workgroup_id = !mesa_shader_stage_is_mesh(nir->info.stage),
      .shuffle_local_ids_for_quad_derivatives = true,
      .lower_local_invocation_index =
         nir->info.derivative_group == DERIVATIVE_GROUP_QUADS,
   };
   if (nir->info.derivative_group == DERIVATIVE_GROUP_QUADS) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_global_id_for_quads,
               nir_metadata_control_flow, NULL);
   }

   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);
}

static void
hk_populate_vs_key(struct hk_vs_key *key,
                   const struct vk_graphics_pipeline_state *state)
{
   memset(key, 0, sizeof(*key));

   if (state && state->ia &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY)) {

      key->kill_psiz = state->ia->primitive_topology != MESA_PRIM_POINTS;
   }

   if (state && state->vi && !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI) &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI_BINDINGS_VALID)) {

      agx_fill_velem_keys(state->vi, ~0 /* compacted on use */, key->attribs);
      key->skip_prolog = true;
      key->static_strides =
         !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI_BINDING_STRIDES);

      if (!key->static_strides) {
         for (unsigned i = 0; i < ARRAY_SIZE(key->attribs); ++i) {
            key->attribs[i].stride = 0;
         }
      }
   }
}

static void
hk_populate_fs_key(struct hk_fs_key *key,
                   const struct vk_graphics_pipeline_state *state)
{
   memset(key, 0, sizeof(*key));

   if (state == NULL)
      return;

   if (state->pipeline_flags &
       VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT)
      key->zs_self_dep = true;

   /* We force per-sample interpolation whenever sampleShadingEnable is set
    * regardless of minSampleShading or rasterizationSamples.
    *
    * When sampleShadingEnable is set, few guarantees are made about the
    * location of interpolation of the inputs.  The only real guarantees are
    * that the inputs are interpolated within the pixel and that you get at
    * least `rasterizationSamples * minSampleShading` unique positions.
    * Importantly, it does not require that when `rasterizationSamples *
    * minSampleShading <= 1.0` that those positions are at the fragment
    * center.  Therefore, it's valid to just always do per-sample all the time.
    *
    * The one caveat here is that we have to be careful about gl_SampleMaskIn.
    * When `hk_fs_key::force_sample_shading = true` we also turn any reads of
    * gl_SampleMaskIn into `1 << gl_SampleID` because the hardware sample mask
    * is actually per-fragment, not per-pass.  We handle this by smashing
    * minSampleShading to 1.0 whenever gl_SampleMaskIn is read.
    */
   const struct vk_multisample_state *ms = state->ms;
   if (ms != NULL && ms->sample_shading_enable)
      key->force_sample_shading = true;

   if (ms != NULL &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES) &&
       !BITSET_TEST(state->dynamic,
                    MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE) &&
       ms->rasterization_samples <= VK_SAMPLE_COUNT_1_BIT &&
       !ms->alpha_to_coverage_enable &&
       !(agx_get_compiler_debug() & AGX_DBG_OLDCOVWORD))
      key->one_sample = true;
}

enum hk_feature_key {
   HK_FEAT_CUSTOM_BORDER = BITFIELD_BIT(0),
   HK_FEAT_BORDER_WITHOUT_FORMAT = BITFIELD_BIT(1),
};

static enum hk_feature_key
hk_make_feature_key(const struct vk_features *feats)
{
   enum hk_feature_key key = 0;

   if (!feats || feats->customBorderColors)
      key |= HK_FEAT_CUSTOM_BORDER;

   if (!feats || (feats->customBorderColors &&
                  feats->customBorderColorWithoutFormat))
      key |= HK_FEAT_BORDER_WITHOUT_FORMAT;

   return key;
}

/*
 * Conservative rasterization runs the vertex shader (or TES) as compute before
 * a passthrough geometry shader, so it needs the software variant. Shader
 * objects take the mode dynamically whatever the pipeline state says.
 */
static bool
hk_may_rasterize_conservative(const struct hk_device *dev,
                              const struct vk_graphics_pipeline_state *state)
{
   if (state == NULL)
      return dev && dev->vk.enabled_extensions.EXT_conservative_rasterization;

   return state->rs &&
          (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_CONSERVATIVE_MODE) ||
           state->rs->conservative_mode !=
              VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
}

static void
hk_hash_graphics_state(struct vk_physical_device *device,
                       const struct vk_graphics_pipeline_state *state,
                       const struct vk_features *features,
                       VkShaderStageFlags stages, blake3_hash blake3_out)
{
   struct mesa_blake3 blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);
   if (state && (stages & VK_SHADER_STAGE_VERTEX_BIT)) {
      struct hk_vs_key key;
      hk_populate_vs_key(&key, state);
      _mesa_blake3_update(&blake3_ctx, &key, sizeof(key));
   } else if (state && (stages & VK_SHADER_STAGE_FRAGMENT_BIT)) {
      struct hk_fs_key key;
      hk_populate_fs_key(&key, state);
      _mesa_blake3_update(&blake3_ctx, &key, sizeof(key));
   }

   if (state &&
       (stages & (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
                  VK_SHADER_STAGE_MESH_BIT_EXT))) {
      const bool is_multiview = state->mv->view_mask != 0;
      _mesa_blake3_update(&blake3_ctx, &is_multiview, sizeof(is_multiview));
   }

   if (state && (stages & (VK_SHADER_STAGE_VERTEX_BIT |
                           VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
                           VK_SHADER_STAGE_MESH_BIT_EXT))) {
      const bool conservative = hk_may_rasterize_conservative(NULL, state);
      _mesa_blake3_update(&blake3_ctx, &conservative, sizeof(conservative));
   }

   enum hk_feature_key feature_key = hk_make_feature_key(features);
   _mesa_blake3_update(&blake3_ctx, &feature_key, sizeof(feature_key));

   _mesa_blake3_final(&blake3_ctx, blake3_out);
}

static nir_def *
bounds_check(nir_builder *b, nir_def *data, nir_def *offs, nir_def *bound)
{
   if (data->bit_size == 32) {
      /* One icmpsel a word, where a bcsel of the vector adds the compare */
      unsigned n = data->num_components;
      return nir_bounds_agx(b, data, nir_replicate(b, offs, n),
                            nir_replicate(b, bound, n));
   } else {
      /* TODO: Optimize */
      return nir_bcsel(b, nir_uge(b, bound, offs), data,
                       nir_imm_zero(b, data->num_components, data->bit_size));
   }
}

static bool
is_op(nir_scalar s, nir_op op)
{
   return nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == op;
}

static nir_def *
check_in_bounds(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *offset = intr->src[1].ssa;
   nir_def *bound = intr->src[2].ssa;

   unsigned bit_size = intr->def.bit_size;
   assert(bit_size >= 8 && bit_size % 8 == 0);
   unsigned byte_size = bit_size / 8;
   unsigned load_size = byte_size * intr->num_components;

   /* Try to bounds check in terms of elements */
   nir_scalar offset_s = nir_scalar_resolved(offset, 0);
   if (is_op(offset_s, nir_op_amul)) {
      nir_scalar srcs[] = {nir_scalar_chase_alu_src(offset_s, 0),
                           nir_scalar_chase_alu_src(offset_s, 1)};
      unsigned i = nir_scalar_is_const(srcs[0]) ? 1 : 0;
      if (nir_scalar_is_const(srcs[1 - i]) &&
          nir_scalar_as_uint(srcs[1 - i]) == load_size) {

         nir_def *index = nir_mov_scalar(b, srcs[i]);
         return nir_ult(b, index, nir_udiv_imm(b, bound, load_size));
      }
   }

   /* TODO: handle also the iadd(amul) pattern, this is important */

   /* Otherwise bounds check in bytes */
   nir_def *sat_offset = nir_umin_imm(b, offset, UINT32_MAX - (load_size - 1));
   return nir_ult(b, nir_iadd_imm(b, sat_offset, load_size - 1), bound);
}

static nir_def *
bound_offset(nir_builder *b, nir_def *valid, nir_scalar offset)
{
   /* If we can, clamp the source of an amul instead of the result, to remain
    * compatible with the hardware address mode.
    */
   if (is_op(offset, nir_op_amul)) {
      nir_scalar srcs[] = {
         nir_scalar_chase_alu_src(offset, 0),
         nir_scalar_chase_alu_src(offset, 1),
      };
      unsigned i = nir_scalar_is_const(srcs[0]) ? 1 : 0;
      nir_def *x = nir_mov_scalar(b, srcs[i]);
      nir_def *y = nir_mov_scalar(b, srcs[1 - i]);

      return nir_amul(b, nir_bcsel(b, valid, x, nir_imm_int(b, 0)), y);
   }

   /* Similar case for when there's an addition (chain) in the way */
   if (is_op(offset, nir_op_iadd)) {
      nir_scalar x = nir_scalar_chase_alu_src(offset, 0);
      nir_scalar y = nir_scalar_chase_alu_src(offset, 1);

      if (is_op(x, nir_op_amul) || is_op(y, nir_op_amul)) {
         return nir_iadd(b, bound_offset(b, valid, x),
                         bound_offset(b, valid, y));
      }
   }

   nir_def *def = nir_mov_scalar(b, offset);

   /* If the offset fits within the zero page, clamping is pointless */
   if (nir_scalar_is_const(offset) &&
       (nir_scalar_as_uint(offset) + 16) < AGX_ZERO_PAGE_SIZE)
      return def;

   /* Otherwise, fallback on clamping the offset */
   return nir_bcsel(b, valid, def, nir_imm_int(b, 0));
}

static void
lower_load_global_bounded(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *base = intr->src[0].ssa;
   nir_def *offset = intr->src[1].ssa;

   /* Bounds check compatibly with the hardware address mode */
   nir_def *valid = check_in_bounds(b, intr);
   base = nir_bcsel(b, valid, base, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
   offset = bound_offset(b, valid, nir_scalar_resolved(offset, 0));

   enum gl_access_qualifier access = nir_intrinsic_access(intr);

   if (intr->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      access |= ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   }

   nir_def *val = nir_load_global(
      b, intr->def.num_components, intr->def.bit_size,
      nir_iadd(b, base, nir_u2u64(b, offset)),
      .align_mul = nir_intrinsic_align_mul(intr),
      .align_offset = nir_intrinsic_align_offset(intr), .access = access);

   nir_def_replace(&intr->def, val);
}

/* A bounded constant-offset load's check covers the 64-B granule of its last
 * byte. Where the loads of one granule of one buffer select three or more
 * words, the granule's base address is selected instead, out of bounds the
 * zero page: two selects (a 64-bit one) and shared, where each word took one.
 * The heaviest of Grim Dawn's light shaders select 40 words from three
 * granules, all in the entry program (e4cap, 2026-11-20).
 */
struct hk_ubo_granule {
   nir_def *base, *bound;
   unsigned first, words;
};

struct hk_ubo_lower {
   bool soft_fault;
   struct util_dynarray granules;
};

static unsigned
hk_ubo_granule_first(nir_intrinsic_instr *intrin)
{
   nir_scalar offs = nir_scalar_resolved(intrin->src[1].ssa, 0);
   if (!nir_scalar_is_const(offs))
      return 0;

   /* An offset whose last byte wraps (a constant negative index) is left to
    * the per-load path */
   uint64_t last = nir_scalar_as_uint(offs) +
                   (intrin->def.bit_size / 8) * intrin->num_components - 1;
   if (last > UINT32_MAX)
      return 0;

   return (unsigned)last & ~63u;
}

static struct hk_ubo_granule *
hk_ubo_granule_get(struct hk_ubo_lower *ctx, nir_intrinsic_instr *intrin,
                   bool add)
{
   unsigned first = hk_ubo_granule_first(intrin);
   if (!first)
      return NULL;

   nir_def *base = intrin->src[0].ssa, *bound = intrin->src[2].ssa;
   util_dynarray_foreach(&ctx->granules, struct hk_ubo_granule, g) {
      if (g->base == base && g->bound == bound && g->first == first)
         return g;
   }

   if (!add)
      return NULL;

   struct hk_ubo_granule g = {base, bound, first, 0};
   util_dynarray_append(&ctx->granules, g);
   return util_dynarray_top_ptr(&ctx->granules, struct hk_ubo_granule);
}

/* nir_lower_explicit_io keeps UBO vector loads whole
 * (has_load_global_constant_bounded_vectors), and one check covers a vector
 * only within one 64-B granule: the check is on the granule of its last byte.
 * A vector that may straddle one, and with exact bounds (no soft faults) any
 * vector, is split into the per-component loads explicit IO made before. A
 * scalar that straddles is out of bounds as a whole, as robustness2 allows.
 */
static bool
hk_ubo_load_in_one_granule(nir_intrinsic_instr *intrin)
{
   unsigned size = (intrin->def.bit_size / 8) * intrin->num_components;
   unsigned m = MIN2(nir_intrinsic_align_mul(intrin), HK_MIN_UBO_ALIGNMENT);
   return (nir_intrinsic_align_offset(intrin) % m) + size <= m;
}

static bool
split_ubo_vector(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   const bool *soft_fault = data;
   if (intrin->intrinsic != nir_intrinsic_load_global_constant_bounded ||
       intrin->num_components == 1 ||
       (*soft_fault && hk_ubo_load_in_one_granule(intrin)))
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   unsigned bit_size = intrin->def.bit_size;
   unsigned align_mul = nir_intrinsic_align_mul(intrin);
   unsigned align_offset = nir_intrinsic_align_offset(intrin);
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];

   for (unsigned c = 0; c < intrin->num_components; ++c) {
      unsigned off = c * (bit_size / 8);
      comps[c] = nir_load_global_constant_bounded(
         b, 1, bit_size, intrin->src[0].ssa,
         nir_iadd_imm(b, intrin->src[1].ssa, off), intrin->src[2].ssa,
         .access = nir_intrinsic_access(intrin), .align_mul = align_mul,
         .align_offset = (align_offset + off) % align_mul);
   }

   nir_def_replace(&intrin->def, nir_vec(b, comps, intrin->num_components));
   return true;
}

static void
hk_ubo_count_granules(nir_shader *nir, struct hk_ubo_lower *ctx)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic != nir_intrinsic_load_global_constant_bounded)
               continue;

            struct hk_ubo_granule *g = hk_ubo_granule_get(ctx, intrin, true);
            if (g) {
               g->words += intrin->num_components *
                           DIV_ROUND_UP(intrin->def.bit_size, 32);
            }
         }
      }
   }
}

static bool
lower_load_global_constant_offset_instr(nir_builder *b,
                                        nir_intrinsic_instr *intrin, void *data)
{
   struct hk_ubo_lower *ctx = data;
   bool *has_soft_fault = &ctx->soft_fault;

   if (intrin->intrinsic == nir_intrinsic_load_global_bounded ||
       (intrin->intrinsic == nir_intrinsic_load_global_constant_bounded &&
        !(*has_soft_fault))) {

      lower_load_global_bounded(b, intrin);
      return true;
   }

   if (intrin->intrinsic != nir_intrinsic_load_global_constant_offset &&
       intrin->intrinsic != nir_intrinsic_load_global_constant_bounded)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *base_addr = intrin->src[0].ssa;
   nir_def *offset = intrin->src[1].ssa;
   nir_def *bound = NULL;

   unsigned bit_size = intrin->def.bit_size;
   assert(bit_size >= 8 && bit_size % 8 == 0);
   unsigned byte_size = bit_size / 8;
   unsigned load_size = byte_size * intrin->num_components;

   if (intrin->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      bound = intrin->src[2].ssa;
   }

   unsigned align_mul = nir_intrinsic_align_mul(intrin);
   unsigned align_offset = nir_intrinsic_align_offset(intrin);

   /* The granule's base selected, its words loaded as they are */
   struct hk_ubo_granule *g =
      bound ? hk_ubo_granule_get(ctx, intrin, false) : NULL;
   bool select_base =
      g && g->words >= 3 && g->first <= AGX_ZERO_PAGE_SIZE - 64;

   if (select_base) {
      assert(align_mul == 64);
      nir_def *valid = nir_uge(b, bound, nir_imm_int(b, g->first + 1));
      base_addr = nir_bcsel(b, valid, base_addr,
                            nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
   }

   nir_def *val = nir_load_global_constant(
      b, intrin->def.num_components, intrin->def.bit_size,
      nir_iadd(b, base_addr, nir_u2u64(b, offset)), .align_mul = align_mul,
      .align_offset = align_offset, .access = nir_intrinsic_access(intrin));

   if (intrin->intrinsic == nir_intrinsic_load_global_constant_bounded &&
       !select_base) {
      nir_scalar offs = nir_scalar_resolved(offset, 0);
      if (nir_scalar_is_const(offs)) {
         /* The last byte loaded */
         unsigned last = nir_scalar_as_uint(offs) + load_size - 1;

         /* Simplify the bounds check. Uniform buffers are bounds checked at
          * K = 64B granularity (robustUniformBufferAccessSizeAlignment), so a
          * load may return data while last < round_up(bound, K). Then
          *
          * last < round_up(bound, K) <==> round_down(last, K) < bound. Proof:
          *
          *      Let a, c be integers s.t. round_down(last, K) = Ka and
          *      round_up(bound, K) = Kc, so Ka <= last < K(a + 1).
          *
          *      last < Kc <==> a < c, since Kc is a multiple of K.
          *      a < c = ceil(bound / K) <==> Ka < bound.
          *
          * The end of the load, last + 1, does not work in place of last: a
          * load of bytes [64, 72) would pass with bound = 64.
          */
         assert(align_mul == 64);
         unsigned first = last & ~(align_mul - 1);

         /* Bounds checks are `offset > bound ? 0 : val`, so the check is
          * first + 1 <= bound. With first = 0 it holds for any descriptor but
          * a null one, whose zero page reads 0, so it is useless.
          */
         if (first) {
            val = bounds_check(b, val, nir_imm_int(b, first + 1), bound);
         }
      } else {
         /* Against the range rounded up to the 64-B granule, as the constant
          * case's check: robustness2 rounds it so for its bounds checks
          * (robustUniformBufferAccessSizeAlignment), and a vector the
          * vectorizer merged then reads what its components did. Once a
          * buffer after CSE; a range that would wrap stays as it is.
          */
         nir_def *rounded = nir_iand_imm(
            b, nir_iadd_imm(b, bound, HK_MIN_UBO_ALIGNMENT - 1),
            ~(HK_MIN_UBO_ALIGNMENT - 1));
         rounded = nir_umax(b, rounded, bound);

         /* An element of an array, amul(x, stride) as check_in_bounds',
          * checks its index: x * stride + load_size <= R, amul not wrapping,
          * is x < (R - load_size) / stride + 1, and no x where R < load_size.
          * The limit is uniform and x the load's own, so the end offset's
          * arithmetic a load goes (Direct3D 9 rows i, i + 1, i + 2).
          */
         nir_scalar offs = nir_scalar_resolved(offset, 0);
         nir_scalar srcs[2];
         unsigned i = 0;
         if (is_op(offs, nir_op_amul)) {
            srcs[0] = nir_scalar_chase_alu_src(offs, 0);
            srcs[1] = nir_scalar_chase_alu_src(offs, 1);
            i = nir_scalar_is_const(srcs[0]) ? 1 : 0;
         }

         if (is_op(offs, nir_op_amul) && nir_scalar_is_const(srcs[1 - i]) &&
             nir_scalar_as_uint(srcs[1 - i]) != 0) {
            uint32_t stride = nir_scalar_as_uint(srcs[1 - i]);
            nir_def *x = nir_mov_scalar(b, srcs[i]);
            nir_def *lim = nir_iadd_imm(
               b, nir_udiv_imm(b, nir_iadd_imm(b, rounded, -load_size), stride),
               1);
            lim = nir_bcsel(b, nir_uge_imm(b, rounded, load_size), lim,
                            nir_imm_int(b, 0));
            val = nir_bcsel(b, nir_ult(b, x, lim), val,
                            nir_imm_zero(b, val->num_components, bit_size));
         } else {
            offset = nir_iadd_imm(b, offset, load_size);
            val = bounds_check(b, val, offset, rounded);
         }
      }
   }

   nir_def_replace(&intrin->def, val);
   return true;
}

struct lower_ycbcr_state {
   uint32_t set_layout_count;
   struct vk_descriptor_set_layout *const *set_layouts;
};

static const struct vk_ycbcr_conversion_state *
lookup_ycbcr_conversion(const void *_state, uint32_t set, uint32_t binding,
                        uint32_t array_index)
{
   const struct lower_ycbcr_state *state = _state;
   assert(set < state->set_layout_count);
   assert(state->set_layouts[set] != NULL);
   const struct hk_descriptor_set_layout *set_layout =
      vk_to_hk_descriptor_set_layout(state->set_layouts[set]);
   assert(binding < set_layout->binding_count);

   const struct hk_descriptor_set_binding_layout *bind_layout =
      &set_layout->binding[binding];

   /* The layout's copy: the samplers may be destroyed by now */
   if (bind_layout->ycbcr == NULL)
      return NULL;

   array_index = MIN2(array_index, bind_layout->array_size - 1);

   const struct vk_ycbcr_conversion_state *ycbcr =
      &bind_layout->ycbcr[array_index];

   return ycbcr->format != VK_FORMAT_UNDEFINED ? ycbcr : NULL;
}

static unsigned
glsl_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/*
 * This is the world's worst multiview implementation. We simply duplicate each
 * draw on the CPU side, changing a uniform in between, and then plumb the view
 * index into the layer ID here. Whatever, it works.
 *
 * The "proper" implementation on AGX would use vertex amplification, but a
 * MacBook is not a VR headset.
 */
static void
hk_lower_multiview(nir_shader *nir)
{
   /* If there's an existing layer ID write, ignore it. This avoids validation
    * splat with vk_meta.
    */
   nir_variable *existing = nir_find_variable_with_location(
      nir, nir_var_shader_out, VARYING_SLOT_LAYER);

   if (existing) {
      existing->data.mode = nir_var_shader_temp;
      existing->data.location = 0;
      nir_fixup_deref_modes(nir);
   }

   /* Now write the view index as the layer */
   nir_builder b =
      nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));

   nir_variable *layer =
      nir_variable_create(nir, nir_var_shader_out, glsl_uint_type(), NULL);

   layer->data.location = VARYING_SLOT_LAYER;

   nir_store_var(&b, layer, nir_load_view_index(&b), nir_component_mask(1));
   b.shader->info.outputs_written |= VARYING_BIT_LAYER;
}

static nir_def *
query_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex, nir_texop_custom_border_color_agx, 4,
                                  tex->dest_type, false, false);
}

static nir_def *
has_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex, nir_texop_has_custom_border_color_agx,
                                  1, nir_type_bool1, false, false);
}

static bool
lower(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (!nir_tex_instr_need_sampler(tex) || nir_tex_instr_is_query(tex))
      return false;

   /* XXX: this is a really weird edge case, is this even well-defined? */
   if (tex->is_shadow)
      return false;

   b->cursor = nir_after_instr(&tex->instr);
   nir_def *has_custom = has_custom_border(b, tex);

   nir_instr *orig = nir_instr_clone(b->shader, &tex->instr);
   nir_builder_instr_insert(b, orig);
   nir_def *clamp_to_1 = &nir_instr_as_tex(orig)->def;

   nir_push_if(b, has_custom);
   nir_def *replaced = NULL;
   {
      /* Sample again, this time with clamp-to-0 instead of clamp-to-1 */
      nir_instr *clone_instr = nir_instr_clone(b->shader, &tex->instr);
      nir_builder_instr_insert(b, clone_instr);

      nir_tex_instr *tex_0 = nir_instr_as_tex(clone_instr);
      nir_def *clamp_to_0 = &tex_0->def;

      tex_0->backend_flags |= AGX_TEXTURE_FLAG_CLAMP_TO_0;

      /* Grab the border colour */
      nir_def *border = query_custom_border(b, tex_0);

      if (tex->op == nir_texop_tg4) {
         border = nir_replicate(b, nir_channel(b, border, tex->component), 4);
      }

      /* Combine together with the border */
      if (nir_alu_type_get_base_type(tex->dest_type) == nir_type_float &&
          tex->op != nir_texop_tg4) {

         /* For floats, lerp together:
          *
          * For border texels:  (1 * border) + (0 * border      ) = border
          * For regular texels: (x * border) + (x * (1 - border)) = x.
          *
          * Linear filtering is linear (duh), so lerping is compatible.
          */
         replaced = nir_flrp(b, clamp_to_0, clamp_to_1, border);
         b->shader->info.flrp_lowered = false;
      } else {
         /* For integers, just select componentwise since there is no linear
          * filtering. Gathers also use this path since they are unfiltered in
          * each component.
          */
         replaced = nir_bcsel(b, nir_ieq(b, clamp_to_0, clamp_to_1), clamp_to_0,
                              border);
      }
   }
   nir_pop_if(b, NULL);

   /* Put it together with a phi */
   nir_def *phi = nir_if_phi(b, replaced, clamp_to_1);
   nir_def_replace(&tex->def, phi);
   return true;
}

static bool
agx_nir_lower_custom_border(nir_shader *nir)
{
   return nir_shader_tex_pass(nir, lower, nir_metadata_none, NULL);
}

static nir_def *
query_min_lod(nir_builder *b, nir_tex_instr *tex, bool int_coords)
{
   nir_alu_type T = int_coords ? nir_type_uint16 : nir_type_float16;
   return nir_build_texture_query(b, tex, nir_texop_image_min_lod_agx, 1, T,
                                  false, false);
}

static bool
lower_min_lod(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (nir_tex_instr_is_query(tex))
      return false;

   /* Buffer textures don't have levels-of-detail */
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF)
      return false;

   if (tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP)
      return false;

   bool int_coords = tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms ||
                     tex->op == nir_texop_tg4;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *min_lod = query_min_lod(b, tex, int_coords);
   nir_def *other_min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);

   if (tex->op == nir_texop_tg4) {
      b->cursor = nir_after_instr(&tex->instr);

      /* The Vulkan spec section "Texel Gathering" says:
       *
       *    If levelbase < minLodIntegerimageView, then any values fetched are
       *    zero if the robustImageAccess2 feature is enabled.
       *
       * We currently always enable robustImageAccess2, so implement that
       * semantic here.
       *
       * We could probably optimize this with a special descriptor for this case
       * but tg4 is rare enough I'm not bothered.
       */
      nir_def *old = &tex->def;
      nir_def *oob = nir_ine_imm(b, min_lod, 0);
      nir_def *zero = nir_imm_zero(b, old->num_components, old->bit_size);
      nir_def *new_ = nir_bcsel(b, oob, zero, old);
      nir_def_rewrite_uses_after(old, new_);
   } else if (tex->op == nir_texop_txl) {
      assert(other_min_lod == NULL && "txl doesn't have an API min lod");

      nir_def *lod = nir_steal_tex_src(tex, nir_tex_src_lod);
      if (lod) {
         min_lod = nir_fmax(b, nir_f2fN(b, lod, min_lod->bit_size), min_lod);
      }

      nir_tex_instr_add_src(tex, nir_tex_src_lod, min_lod);
   } else {
      if (other_min_lod) {
         assert(!int_coords && "no API min lod");
         min_lod = nir_fmax(b, min_lod, nir_f2f16(b, other_min_lod));
      }

      nir_tex_instr_add_src(tex, nir_tex_src_min_lod, min_lod);
   }

   return true;
}

static bool
agx_nir_lower_image_view_min_lod(nir_shader *nir)
{
   return nir_shader_tex_pass(nir, lower_min_lod, nir_metadata_none, NULL);
}

/*
 * In Vulkan, the VIEWPORT should read 0 in the fragment shader if it is not
 * written by the vertex shader, but in our implementation, the varying would
 * otherwise be undefined. This small pass predicates VIEWPORT reads based on
 * whether the hardware vertex shader writes the VIEWPORT (nonzero UVS index).
 */
static bool
lower_viewport_fs(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_VIEWPORT)
      return false;

   b->cursor = nir_after_instr(&intr->instr);
   nir_def *orig = &intr->def;

   nir_def *uvs = nir_load_uvs_index_agx(b, .io_semantics = sem);
   nir_def *def = nir_bcsel(b, nir_ine_imm(b, uvs, 0), orig, nir_imm_int(b, 0));

   nir_def_rewrite_uses_after(orig, def);
   return true;
}

static bool
lower_subpass_dim(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS)
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   else if (tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS_MS)
      tex->sampler_dim = GLSL_SAMPLER_DIM_MS;
   else
      return false;

   return true;
}

static bool
should_lower_robust(const nir_intrinsic_instr *intr, const void *_)
{
   /* The hardware is robust, but our software image atomics are not. Unlike the
    * GL driver, we don't use the common buffer image lowering, using the
    * agx_nir_lower_texture lowering for robustImageAccess2 semantics.
    */
   return intr->intrinsic == nir_intrinsic_image_deref_atomic ||
          intr->intrinsic == nir_intrinsic_image_deref_atomic_swap;
}

/*
 * G15 reads every level of a texture as level 0, fetched or sampled, when the
 * texture instruction and the descriptor disagree on being an array
 * (vkdimprobe), and Direct3D reads a plain view through an array declaration
 * and the reverse. So for Direct3D 12 (driconf hk_sampled_views_as_arrays) a
 * sampled 1D or 2D view is an array of one on G15 (pack_texture_g15), and a
 * plain read of one becomes an array read of layer 0, as a cube is a cube
 * array of one.
 */
static bool
lower_2d_as_array(nir_builder *b, nir_tex_instr *tex, void *data)
{
   if (tex->is_array || (tex->sampler_dim != GLSL_SAMPLER_DIM_1D &&
                         tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
                         tex->sampler_dim != GLSL_SAMPLER_DIM_MS))
      return false;

   switch (tex->op) {
   case nir_texop_lod:
      /* An array's LOD query takes no layer */
      tex->is_array = true;
      return true;

   case nir_texop_tex:
   case nir_texop_txb:
   case nir_texop_txl:
   case nir_texop_txd:
   case nir_texop_txf:
   case nir_texop_txf_ms:
   case nir_texop_tg4:
      break;

   default:
      return false;
   }

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *coord = nir_steal_tex_src(tex, nir_tex_src_coord);
   bool integer = tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms;
   nir_def *zero = integer ? nir_imm_intN_t(b, 0, coord->bit_size)
                           : nir_imm_floatN_t(b, 0.0, coord->bit_size);
   nir_def *comps[4];

   for (unsigned i = 0; i < coord->num_components; ++i)
      comps[i] = nir_channel(b, coord, i);

   comps[coord->num_components] = zero;
   nir_tex_instr_add_src(tex, nir_tex_src_coord,
                         nir_vec(b, comps, coord->num_components + 1));
   tex->coord_components++;
   tex->is_array = true;
   tex->backend_flags |= AGX_TEXTURE_FLAG_LAYER_ZERO;
   return true;
}

static void
hk_lower_nir(struct hk_device *dev, nir_shader *nir,
             const struct vk_pipeline_robustness_state *rs, bool is_multiview,
             uint32_t set_layout_count,
             struct vk_descriptor_set_layout *const *set_layouts,
             enum hk_feature_key features)
{
   if (HK_PERF(dev, NOROBUST)) {
      rs = &vk_robustness_disabled;
   }

   const nir_opt_access_options access_options = {
      .is_vulkan = true,
   };
   NIR_PASS(_, nir, nir_opt_access, &access_options);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_lower_input_attachments,
               &(nir_input_attachment_options){
                  .use_view_id_for_layer = is_multiview,
               });

      NIR_PASS(_, nir, nir_shader_tex_pass, lower_subpass_dim, nir_metadata_all,
               NULL);
      NIR_PASS(_, nir, nir_lower_wpos_center);
   }

   /* XXX: should be last geometry stage, how do I get to that? */
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      if (is_multiview)
         hk_lower_multiview(nir);
   }

   if (nir->info.stage == MESA_SHADER_TESS_EVAL) {
      NIR_PASS(_, nir, nir_lower_patch_vertices,
               nir->info.tess.tcs_vertices_out, NULL);
   }

   const struct lower_ycbcr_state ycbcr_state = {
      .set_layout_count = set_layout_count,
      .set_layouts = set_layouts,
   };
   NIR_PASS(_, nir, nir_vk_lower_ycbcr_tex, lookup_ycbcr_conversion,
            &ycbcr_state);

   /* Lower push constants before lower_descriptors */
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);

   // NIR_PASS(_, nir, nir_opt_large_constants, NULL, 32);

   /* Turn cache flushes into image coherency bits while we still have derefs */
   NIR_PASS(_, nir, nir_lower_memory_model);

   NIR_PASS(_, nir, nir_lower_robust_access, should_lower_robust, NULL);

   /* We must do early lowering before hk_nir_lower_descriptors, since this will
    * create lod_bias instructions.
    */
   NIR_PASS(_, nir, agx_nir_lower_texture_early, true /* support_lod_bias */);

   /* G15 samples a cube in cube array mode (pack_texture_g15), seamless */
   if (dev->dev.params.gpu_generation >= 15)
      NIR_PASS(_, nir, agx_nir_lower_cube_g15);

   /* ... and with Direct3D 12 a plain 1D or 2D view as an array of one */
   if (hk_device_physical(dev)->sampled_views_as_arrays) {
      NIR_PASS(_, nir, nir_shader_tex_pass, lower_2d_as_array,
               nir_metadata_control_flow, NULL);
   }

   struct hk_instance *instance = hk_physical_device_instance(
      (struct hk_physical_device *)dev->vk.physical);

   if (instance->drirc.misc.image_view_min_lod) {
      NIR_PASS(_, nir, agx_nir_lower_image_view_min_lod);
   }

   /* G15 takes custom borders from the descriptor (hk_descriptor_set.h) */
   if ((features & HK_FEAT_CUSTOM_BORDER) && !HK_PERF(dev, NOBORDER) &&
       dev->dev.params.gpu_generation < 15) {
      NIR_PASS(_, nir, agx_nir_lower_custom_border);
   }

   NIR_PASS(_, nir, hk_nir_lower_descriptors, rs, set_layout_count, set_layouts,
            dev->dev.params.gpu_generation >= 15,
            instance->drirc.misc.d3d_nearest_mip,
            features & HK_FEAT_BORDER_WITHOUT_FORMAT);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            hk_buffer_addr_format(rs->storage_buffers));
   /* One load a vector before the bounds go in: Direct3D 9 shaders through
    * DXVK read a constant a component at a time, and each component's load
    * took its own address, base and bounds check (Grim Dawn's skinned vertex
    * shader: 48 loads for 12 rows, e4cap 2026-11-20).
    */
   bool g15 = dev->dev.params.gpu_generation >= 15;
   bool robust_ubo = rs->uniform_buffers !=
                     VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT;

   /* Robust, only where the bounds are checked on the rounded range (the
    * soft-fault path below): with check_in_bounds's exact range
    * split_ubo_vector splits every vector again */
   if (!robust_ubo || agx_has_soft_fault(&dev->dev)) {
      NIR_PASS(_, nir, nir_opt_load_store_vectorize,
               &(const nir_load_store_vectorize_options){
                  .modes = nir_var_mem_ubo,
                  .callback = agx_mem_vectorize_cb,
                  .cb_data = &g15,
                  .robust_modes = robust_ubo ? nir_var_mem_ubo : 0,
               });
   }

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            hk_buffer_addr_format(rs->uniform_buffers));

   /* Before inserting bounds checks, we want to do a fair bit of optimization.
    * lower_load_global_constant_offset_instr has special optimizations for
    * constant offsets, so we want as many offsets to be constant as possible.
    */
   struct set *skip = _mesa_pointer_set_create(NULL);
   bool progress;
   do {
      progress = false;
      NIR_LOOP_PASS(progress, skip, nir, nir_opt_constant_folding);
      NIR_LOOP_PASS_NOT_IDEMPOTENT(progress, skip, nir, nir_opt_algebraic);
      NIR_LOOP_PASS(progress, skip, nir, nir_opt_copy_prop);
      NIR_LOOP_PASS(progress, skip, nir, nir_opt_dce);
   } while (progress);
   _mesa_set_destroy(skip, NULL);

   nir_move_options opts = nir_move_load_ssbo | nir_move_load_ubo;
   NIR_PASS(progress, nir, nir_opt_sink, opts);
   NIR_PASS(progress, nir, nir_opt_move, opts);

   struct hk_ubo_lower ubo = {.soft_fault = agx_has_soft_fault(&dev->dev)};
   util_dynarray_init(&ubo.granules, NULL);
   bool split = false;
   NIR_PASS(split, nir, nir_shader_intrinsics_pass, split_ubo_vector,
            nir_metadata_control_flow, &ubo.soft_fault);
   if (split)
      NIR_PASS(_, nir, nir_opt_constant_folding);

   if (ubo.soft_fault) {
      /* nir_lower_explicit_io gives each load its own base and bound: one
       * buffer's loads share them only after CSE, which the granules key on */
      NIR_PASS(_, nir, nir_opt_cse);
      hk_ubo_count_granules(nir, &ubo);
   }

   NIR_PASS(_, nir, nir_shader_intrinsics_pass,
            lower_load_global_constant_offset_instr, nir_metadata_none, &ubo);
   util_dynarray_fini(&ubo.granules);

   assert(nir->info.shared_size == 0);

   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
            shared_var_info);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
            nir_address_format_32bit_offset);
   if (dev->dev.params.gpu_generation >= 15)
      NIR_PASS(_, nir, hk_nir_lower_shared_atomic64);

   /* The payload lives in global memory (poly_nir_lower_mesh.c) */
   if (mesa_shader_stage_is_mesh(nir->info.stage)) {
      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
               nir_var_mem_task_payload, shared_var_info);
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_task_payload,
               nir_address_format_32bit_offset);
   }

   if (nir->info.stage == MESA_SHADER_TASK) {
      NIR_PASS(_, nir, nir_lower_task_shader,
               (nir_lower_task_shader_options){0});
   }

   if (nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
      /* Align everything up to 16B so we can write whole vec4s. */
      nir->info.shared_size = align(nir->info.shared_size, 16);
      NIR_PASS(_, nir, nir_zero_initialize_shared_memory, nir->info.shared_size,
               16);

      /* We need to call lower_compute_system_values again because
       * nir_zero_initialize_shared_memory generates load_invocation_id which
       * has to be lowered to load_invocation_index.
       */
      NIR_PASS(_, nir, nir_lower_compute_system_values, NULL);
   }

   /* TODO: we can do indirect VS output */
   nir_variable_mode lower_indirect_modes = 0;
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      lower_indirect_modes |= nir_var_shader_out;

      /* G15's iter takes its coefficient register as an immediate only */
      if (dev->dev.params.gpu_generation >= 15)
         lower_indirect_modes |= nir_var_shader_in;
   } else if (nir->info.stage == MESA_SHADER_VERTEX)
      lower_indirect_modes |= nir_var_shader_in | nir_var_shader_out;

   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
            lower_indirect_modes, UINT32_MAX);

   /* Mesh outputs stay indirect: a vector component must not read as a slot */
   if (nir->info.stage == MESA_SHADER_MESH) {
      NIR_PASS(_, nir, nir_lower_array_deref_of_vec, nir_var_shader_out, NULL,
               nir_lower_direct_array_deref_of_vec_load |
                  nir_lower_indirect_array_deref_of_vec_load |
                  nir_lower_direct_array_deref_of_vec_store |
                  nir_lower_indirect_array_deref_of_vec_store);
   }

   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            glsl_type_size,
            nir_lower_io_lower_64bit_to_32 |
               nir_lower_io_use_interpolated_input_intrinsics);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_viewport_fs,
               nir_metadata_control_flow, NULL);
      NIR_PASS(_, nir, hk_nir_lower_fully_covered);
   }

   if (dev->dev.params.gpu_generation >= 15) {
      NIR_PASS(_, nir, hk_nir_lower_image_r64);
      NIR_PASS(_, nir, hk_nir_lower_sparse_texel_buffers_g15);
   }
   NIR_PASS(_, nir, agx_nir_lower_texture_gen,
            dev->dev.params.gpu_generation >= 15);
   if (dev->dev.params.gpu_generation >= 15)
      NIR_PASS(_, nir, hk_nir_lower_image_loads_g15);
   NIR_PASS(_, nir, hk_nir_lower_atomic64);
   NIR_PASS(_, nir, agx_nir_lower_multisampled_image_store,
            dev->dev.params.gpu_generation >= 15);

   agx_preprocess_nir(nir);

   nir_opt_peephole_select_options peephole_select_options = {
      .limit = 0,
      .discard_ok = true,
   };
   NIR_PASS(_, nir, nir_opt_peephole_select, &peephole_select_options);
   NIR_PASS(_, nir, nir_opt_if,
            nir_opt_if_optimize_phi_true_false | nir_opt_if_avoid_64bit_phis);
}

static void
hk_upload_shader(struct hk_device *dev, struct hk_shader *shader)
{
   if (shader->b.info.has_preamble || shader->b.info.rodata.size_16) {
      /* TODO: Do we wnat to compact? Revisit when we rework prolog/epilogs. */
      size_t size = shader->b.info.binary_size;
      assert(size > 0);

      shader->bo = agx_bo_create(&dev->dev, size, 0,
                                 AGX_BO_EXEC | AGX_BO_LOW_VA, "Preamble");
      memcpy(agx_bo_map(shader->bo), shader->b.binary, size);
      shader->preamble_addr =
         shader->bo->va->addr + shader->b.info.preamble_offset;
   }

   if (!shader->linked.ht) {
      /* If we only have a single variant, link now. */
      shader->only_linked = hk_fast_link(dev, false, shader, NULL, NULL, 0, 1);
   }

   if (shader->info.stage == MESA_SHADER_FRAGMENT) {
      agx_pack_fragment_face_2(&shader->frag_face, 0, &shader->b.info);
   }

   agx_pack(&shader->counts, COUNTS, cfg) {
      cfg.uniform_register_count = shader->b.info.push_count;
      cfg.preshader_register_count = shader->b.info.nr_preamble_gprs;
      cfg.sampler_state_register_count = agx_translate_sampler_state_count(
         shader->b.info.sampler_state_count, false);
      cfg.texture_state_register_count = shader->b.info.texture_state_count;
   }
}

DERIVE_HASH_TABLE(hk_fast_link_key_vs);
DERIVE_HASH_TABLE(hk_fast_link_key_fs);

static VkResult
hk_init_link_ht(struct hk_shader *shader, mesa_shader_stage sw_stage)
{
   simple_mtx_init(&shader->linked.lock, mtx_plain);

   bool multiple_variants =
      sw_stage == MESA_SHADER_VERTEX || sw_stage == MESA_SHADER_FRAGMENT;

   if (!multiple_variants)
      return VK_SUCCESS;

   if (sw_stage == MESA_SHADER_VERTEX)
      shader->linked.ht = hk_fast_link_key_vs_table_create(NULL);
   else
      shader->linked.ht = hk_fast_link_key_fs_table_create(NULL);

   return (shader->linked.ht == NULL) ? VK_ERROR_OUT_OF_HOST_MEMORY
                                      : VK_SUCCESS;
}

static bool
lower_uniforms(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   /* Root is first, descriptor sets follow immediately. */
   unsigned *root_ = data;
   unsigned root = *root_;
   unsigned sets = root + 4;

   if (intr->intrinsic == nir_intrinsic_bindless_image_agx ||
       intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
      /* Change of units from sets to uniforms */
      nir_intrinsic_set_desc_set(intr,
                                 sets + (nir_intrinsic_desc_set(intr) * 4));
      return true;
   }

   if (intr->intrinsic != nir_intrinsic_load_root_agx &&
       intr->intrinsic != nir_intrinsic_load_descriptor_set_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *rep;

   if (intr->intrinsic == nir_intrinsic_load_descriptor_set_agx) {
      unsigned s = nir_intrinsic_desc_set(intr);
      rep = nir_load_preamble(b, 1, 64, .base = sets + (4 * s));
   } else {
      rep = nir_load_preamble(b, 1, 64, .base = root);
   }

   nir_def_replace(&intr->def, rep);
   return true;
}

static bool
kill_psiz_write(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_PSIZ)
      return false;

   if (nir_remove_sysval_output(intr, MESA_SHADER_FRAGMENT)) {
      b->shader->info.outputs_written &= ~VARYING_BIT_PSIZ;
   }

   return true;
}

/* G15 clips at the viewport and gives the new vertices the provoking vertex's
 * flat values, so a clipped primitive's fragments read wrong barycentrics and
 * per-vertex inputs. Every draw gets a guard band: gl_Position.xy over
 * HK_GUARD_BAND here and the viewport's scale times it (hk_flush_vp_state), so
 * the hardware clips that much further out and the implicit scissor keeps the
 * viewport. Exact, the factor being a power of two. Every draw, not only those
 * that read their vertices: the same primitive must cover the same pixels
 * whatever its fragment shader.
 */
static bool
lower_guard_band(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_POS ||
       nir_intrinsic_component(intr) > 1)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(&intr->src[0],
                   nir_fmul_imm(b, intr->src[0].ssa, 1.0 / HK_GUARD_BAND));
   return true;
}

static void
hk_lower_hw_vs(nir_shader *nir, struct hk_shader *shader, bool kill_psiz,
               bool g15)
{
   /* Point size must be clamped, excessively large points don't render
    * properly on G13.
    *
    * Must be synced with pointSizeRange.
    */
   NIR_PASS(_, nir, nir_lower_point_size, 1.0f, 511.95f);

   if (kill_psiz) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, kill_psiz_write,
               nir_metadata_control_flow, NULL);
   }

   NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_out, NULL, NULL);
   if (g15) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_guard_band,
               nir_metadata_control_flow, NULL);
   }

   NIR_PASS(_, nir, agx_nir_lower_cull_distance_vs);

   NIR_PASS(_, nir, agx_nir_lower_uvs, &shader->info.uvs);
}

static bool
remove_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const gl_varying_slot *slot = data;

   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location != *slot)
      return false;

   nir_instr_remove(&intr->instr);
   return true;
}

/*
 * The UVS holds 128 words. The conservative variant writes 7 more than the
 * rasterization variant: the fragment test's bounding box and edges. Drop the
 * edges, then the box, when they do not fit.
 */
static void
hk_lower_conservative_budget(nir_shader *nir, struct hk_shader *shader,
                             unsigned rast_uvs_size)
{
   static const gl_varying_slot box = POLY_VARYING_SLOT_CONSERVATIVE_BOX;
   static const gl_varying_slot edges = POLY_VARYING_SLOT_CONSERVATIVE_EDGES;

   if (rast_uvs_size + 7 <= 128) {
      shader->info.conservative = AGX_CONSERVATIVE_BOX_EDGES;
      return;
   }

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, remove_output,
            nir_metadata_control_flow, (void *)&edges);
   nir->info.outputs_written &= ~BITFIELD64_BIT(edges);
   shader->info.conservative = AGX_CONSERVATIVE_BOX;

   if (rast_uvs_size + 4 > 128) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, remove_output,
               nir_metadata_control_flow, (void *)&box);
      nir->info.outputs_written &= ~BITFIELD64_BIT(box);
      shader->info.conservative = AGX_CONSERVATIVE_NONE;
   }
}

/*
 * A mesh shader's per-primitive outputs reach the rasterizer on every corner
 * (poly_nir_lower_mesh.c), so they read as flat inputs.
 */
static bool
lower_per_primitive_input(nir_builder *b, nir_intrinsic_instr *intr,
                          UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_per_primitive_input)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);

   nir_def *repl = nir_load_input(
      b, intr->def.num_components, intr->def.bit_size, intr->src[0].ssa,
      .base = nir_intrinsic_base(intr), .range = sem.num_slots,
      .component = nir_intrinsic_component(intr),
      .dest_type = nir_intrinsic_dest_type(intr), .io_semantics = sem);

   nir_def_replace(&intr->def, repl);
   return true;
}

/* Whether the shader has any texture or image instruction */
static bool
hk_accesses_images(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex)
               return true;

            if (instr->type == nir_instr_type_intrinsic &&
                nir_intrinsic_has_image_dim(nir_instr_as_intrinsic(instr)))
               return true;
         }
      }
   }

   return false;
}

/* Results by the hash of their input. A pipeline's stage often comes in as
 * another pipeline's did: pipelines that share a vertex shader but not their
 * fragment shader (the runtime keys a stage on every stage it is linked
 * with), or modules that differ in nothing the compile reads. Elden Ring's
 * 8,825 startup pipelines lower 9,939 stages from 2,744 inputs and reach the
 * backend 10,248 times with 3,484. The lowering and the backend are functions
 * of their input, so the device keeps each result, up to
 * HK_RESULT_CACHE_BYTES a cache (then starts again). Asserts-on builds
 * compute a hit anyway and check that both agree.
 */
#define HK_RESULT_CACHE_BYTES (64 << 20)

struct hk_result_entry {
   blake3_hash hash;
   size_t size;
   uint8_t data[];
};

static uint32_t
hk_result_entry_hash(const void *key)
{
   uint32_t h;
   memcpy(&h, key, sizeof(h));
   return h;
}

static bool
hk_result_entry_equal(const void *a, const void *b)
{
   return memcmp(a, b, sizeof(blake3_hash)) == 0;
}

void
hk_result_cache_init(struct hk_result_cache *cache)
{
   simple_mtx_init(&cache->lock, mtx_plain);
   cache->ht = _mesa_hash_table_create(NULL, hk_result_entry_hash,
                                       hk_result_entry_equal);
}

static void
hk_result_cache_clear(struct hk_result_cache *cache)
{
   hash_table_foreach(cache->ht, entry)
      free(entry->data);

   _mesa_hash_table_clear(cache->ht, NULL);
   cache->bytes = 0;
}

void
hk_result_cache_finish(struct hk_result_cache *cache)
{
   if (cache->ht) {
      hk_result_cache_clear(cache);
      _mesa_hash_table_destroy(cache->ht, NULL);
   }

   simple_mtx_destroy(&cache->lock);
}

/* A copy of the result kept for the hash, or NULL */
static void *
hk_result_cache_get(struct hk_result_cache *cache, const blake3_hash hash,
                    size_t *size)
{
   void *data = NULL;

   simple_mtx_lock(&cache->lock);
   struct hash_entry *he = _mesa_hash_table_search(cache->ht, hash);
   if (he) {
      const struct hk_result_entry *e = he->data;
      data = malloc(e->size);
      if (data) {
         memcpy(data, e->data, e->size);
         *size = e->size;
      }
   }
   simple_mtx_unlock(&cache->lock);

   return data;
}

/* Keeps A followed by B as the result for the hash */
static void
hk_result_cache_put(struct hk_result_cache *cache, const blake3_hash hash,
                    const void *a, size_t a_size, const void *b, size_t b_size)
{
   size_t size = sizeof(struct hk_result_entry) + a_size + b_size;
   struct hk_result_entry *e = malloc(size);
   if (!e)
      return;

   memcpy(e->hash, hash, sizeof(e->hash));
   e->size = a_size + b_size;
   memcpy(e->data, a, a_size);
   if (b_size)
      memcpy(e->data + a_size, b, b_size);

   simple_mtx_lock(&cache->lock);
   if (cache->bytes + size > HK_RESULT_CACHE_BYTES)
      hk_result_cache_clear(cache);

   /* Another thread may have put the same result meanwhile */
   if (_mesa_hash_table_search(cache->ht, e->hash)) {
      free(e);
   } else {
      _mesa_hash_table_insert(cache->ht, e->hash, e);
      cache->bytes += size;
   }
   simple_mtx_unlock(&cache->lock);
}

/* The NIR as a pass reads it: no names or debug information, no source hash
 * (modules that preprocess alike share it), with the locations that
 * stripping drops */
static void
hk_hash_nir(struct mesa_blake3 *ctx, nir_shader *nir)
{
   blake3_hash source;
   memcpy(source, nir->info.source_blake3, sizeof(source));
   memset(nir->info.source_blake3, 0, sizeof(nir->info.source_blake3));

   struct blob blob;
   blob_init(&blob);
   nir_serialize(&blob, nir, true);
   memcpy(nir->info.source_blake3, source, sizeof(source));

   _mesa_blake3_update(ctx, blob.data, blob.size);
   blob_finish(&blob);

   nir_foreach_variable_in_shader(var, nir) {
      const int32_t loc[2] = {var->data.mode, var->data.location};
      _mesa_blake3_update(ctx, loc, sizeof(loc));
   }
}

/* The backend's input: the NIR and the key's fields, its padding left out.
 * A hit leaves the NIR as the backend found it, so nothing after the backend
 * may read what the backend changes in it. */
static void
hk_backend_hash(nir_shader *nir, const struct agx_shader_key *key,
                blake3_hash out)
{
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   hk_hash_nir(&ctx, nir);

   /* A field added to the key is added here too */
   STATIC_ASSERT(sizeof(struct agx_shader_key) == 24);
   const uint32_t k[] = {
      key->dev.needs_g13x_coherency,
      key->dev.soft_fault,
      key->dev.gpu_generation,
      key->reserved_preamble,
      key->has_scratch,
      key->is_helper,
      key->promote_constants,
      key->promote_textures,
      key->no_stop,
      key->secondary,
      key->vs_prolog,
      key->fs.ignore_tib_dependencies,
      key->fs.inside_sample_loop,
      key->fs.cf_base,
      key->fs.nr_samples,
   };
   _mesa_blake3_update(&ctx, k, sizeof(k));
   _mesa_blake3_final(&ctx, out);
}

static bool
hk_backend_cache_get(struct hk_device *dev, const blake3_hash hash,
                     struct agx_shader_part *out)
{
   size_t size;
   uint8_t *data = hk_result_cache_get(&dev->backend_cache, hash, &size);
   if (!data)
      return false;

   memcpy(&out->info, data, sizeof(out->info));
   assert(size == sizeof(out->info) + out->info.binary_size);
   out->binary = malloc(out->info.binary_size);
   if (out->binary)
      memcpy(out->binary, data + sizeof(out->info), out->info.binary_size);

   free(data);
   return out->binary != NULL;
}

static void
hk_backend_cache_put(struct hk_device *dev, const blake3_hash hash,
                     const struct agx_shader_part *part)
{
   hk_result_cache_put(&dev->backend_cache, hash, &part->info,
                       sizeof(part->info), part->binary,
                       part->info.binary_size);
}

/* hk_lower_nir's input: the stage's NIR and the rest the lowering reads */
static void
hk_lower_hash(nir_shader *nir, const struct vk_pipeline_robustness_state *rs,
              bool is_multiview, uint32_t set_layout_count,
              struct vk_descriptor_set_layout *const *set_layouts,
              enum hk_feature_key features, blake3_hash out)
{
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   hk_hash_nir(&ctx, nir);

   const uint32_t k[] = {
      rs->storage_buffers,
      rs->uniform_buffers,
      rs->vertex_inputs,
      rs->images,
      rs->null_uniform_buffer_descriptor,
      rs->null_storage_buffer_descriptor,
      is_multiview,
      features,
      set_layout_count,
   };
   _mesa_blake3_update(&ctx, k, sizeof(k));

   for (uint32_t i = 0; i < set_layout_count; i++) {
      static const blake3_hash none = {0};
      _mesa_blake3_update(&ctx, set_layouts[i] ? set_layouts[i]->blake3 : none,
                          sizeof(blake3_hash));
   }

   _mesa_blake3_final(&ctx, out);
}

/* hk_lower_nir, or its result for the same input. Either way the stage goes
 * on from the lowered NIR as serialized, so that a hit compiles as a miss
 * does. Returns the lowered NIR, which replaces (and frees) the input. */
static nir_shader *
hk_lower_nir_cached(struct hk_device *dev, nir_shader *nir,
                    const struct vk_pipeline_robustness_state *rs,
                    bool is_multiview, uint32_t set_layout_count,
                    struct vk_descriptor_set_layout *const *set_layouts,
                    enum hk_feature_key features)
{
   if (HK_PERF(dev, NOLOWERCACHE) || !dev->lower_cache.ht) {
      hk_lower_nir(dev, nir, rs, is_multiview, set_layout_count, set_layouts,
                   features);
      return nir;
   }

   blake3_hash hash;
   hk_lower_hash(nir, rs, is_multiview, set_layout_count, set_layouts,
                 features, hash);

   struct blob blob;
   blob_init(&blob);

   size_t size;
   void *data = hk_result_cache_get(&dev->lower_cache, hash, &size);
   bool hit = data != NULL;

   if (!hit) {
      hk_lower_nir(dev, nir, rs, is_multiview, set_layout_count, set_layouts,
                   features);
      nir_serialize(&blob, nir, false);
      if (blob.out_of_memory) {
         blob_finish(&blob);
         return nir;
      }

      hk_result_cache_put(&dev->lower_cache, hash, blob.data, blob.size,
                          NULL, 0);
      data = blob.data;
      size = blob.size;
   }

   struct blob_reader reader;
   blob_reader_init(&reader, data, size);
   nir_shader *lowered = nir_deserialize(NULL, nir->options, &reader);
   memcpy(lowered->info.source_blake3, nir->info.source_blake3,
          sizeof(lowered->info.source_blake3));

#ifndef NDEBUG
   if (hit) {
      hk_lower_nir(dev, nir, rs, is_multiview, set_layout_count, set_layouts,
                   features);
      blake3_hash a, b;
      struct mesa_blake3 ctx;
      _mesa_blake3_init(&ctx);
      hk_hash_nir(&ctx, nir);
      _mesa_blake3_final(&ctx, a);
      _mesa_blake3_init(&ctx);
      hk_hash_nir(&ctx, lowered);
      _mesa_blake3_final(&ctx, b);
      assert(memcmp(a, b, sizeof(a)) == 0);
   }
#endif

   if (hit)
      free(data);
   else
      blob_finish(&blob);

   ralloc_free(nir);
   return lowered;
}

/* agx_compile_shader_nir, or its result for the same input */
static void
hk_compile_backend(struct hk_device *dev, nir_shader *nir,
                   struct agx_shader_key *key, struct agx_shader_part *out)
{
   if (HK_PERF(dev, NOBACKENDCACHE) || !dev->backend_cache.ht) {
      agx_compile_shader_nir(nir, key, out);
      return;
   }

   blake3_hash hash;
   hk_backend_hash(nir, key, hash);

   if (hk_backend_cache_get(dev, hash, out)) {
#ifndef NDEBUG
      /* On a clone: a hit leaves the NIR as the backend found it, here too */
      nir_shader *clone = nir_shader_clone(NULL, nir);
      struct agx_shader_part check;
      agx_compile_shader_nir(clone, key, &check);
      ralloc_free(clone);
      assert(memcmp(&check.info, &out->info, sizeof(check.info)) == 0);
      assert(check.info.binary_size == 0 ||
             memcmp(check.binary, out->binary, check.info.binary_size) == 0);
      free(check.binary);
#endif
      return;
   }

   agx_compile_shader_nir(nir, key, out);
   hk_backend_cache_put(dev, hash, out);
}

/* The main shader of a pipeline drawn with alpha to coverage off and no depth
 * or stencil output in the epilog runs the tests itself (no_epilog_discard ~0
 * at every draw, hk_flush_dynamic_state): the switch folds, and a discard's
 * coverage word targets every sample.
 */
static bool
lower_tests_zs_main(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_shader_part_tests_zs_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_intN_t(b, 0xffff, 16));
   return true;
}

static bool
writes_coverage_intr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic == nir_intrinsic_sample_mask_agx ||
       intr->intrinsic == nir_intrinsic_store_zs_agx)
      *(bool *)data = true;

   return false;
}

/* Whether the main shader builds a coverage word (a discard, a sample mask, or
 * depth and stencil output it emits itself): only such a shader's code depends
 * on the sample count, so only it is built for its pipeline's, and the others
 * keep their backend cache entries across pipelines.
 */
static bool
writes_coverage(nir_shader *nir)
{
   bool found = false;
   nir_shader_intrinsics_pass(nir, writes_coverage_intr, nir_metadata_all,
                              &found);
   return found;
}

static VkResult
hk_compile_nir(struct hk_device *dev, const VkAllocationCallbacks *pAllocator,
               nir_shader *nir, VkShaderCreateFlagsEXT shader_flags,
               const struct vk_pipeline_robustness_state *rs,
               const union hk_key *key, enum hk_feature_key features,
               struct hk_shader *shader, mesa_shader_stage sw_stage, bool hw,
               bool hw_tess, nir_xfb_info *xfb_info, unsigned set_count)
{
   unsigned nr_vbos = 0;
   bool kill_psiz = false;

   /* For now, only shader objects are supported */
   if (sw_stage == MESA_SHADER_VERTEX) {
      nr_vbos = DIV_ROUND_UP(
         BITSET_LAST_BIT(shader->info.vs.attrib_components_read), 4);
      kill_psiz = key->vs.kill_psiz;
      shader->info.vs.accesses_images = hk_accesses_images(nir);
   } else if (sw_stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_per_primitive_input,
               nir_metadata_control_flow, NULL);

      shader->info.fs.writes_memory = nir->info.writes_memory;
      shader->info.fs.interp.linear = nir->info.linear_varyings;
      shader->info.fs.interp.flat =
         ~(nir->info.linear_varyings | nir->info.perspective_varyings);

      if (dev->dev.params.gpu_generation >= 15)
         NIR_PASS(_, nir, agx_nir_lower_stencil_only_g15);

      /* Discards must be lowering before lowering MSAA to handle discards */
      NIR_PASS(_, nir, agx_nir_lower_discard_zs_emit);
      NIR_PASS(_, nir, agx_nir_lower_fs_output_to_epilog,
               &shader->info.fs.epilog_key);
      NIR_PASS(_, nir, agx_nir_lower_sample_mask);

      if (key && key->fs.one_sample && writes_coverage(nir)) {
         shader->info.fs.one_sample = true;

         if (!shader->info.fs.epilog_key.write_z &&
             !shader->info.fs.epilog_key.write_s) {
            NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_tests_zs_main,
                     nir_metadata_control_flow, NULL);
         }
      }

      if (nir->info.fs.uses_sample_shading) {
         /* Ensure the sample mask is preserved in register */
         nir_builder b =
            nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));

         nir_def *mask =
            nir_load_exported_agx(&b, 1, 16, .base = AGX_ABI_FIN_SAMPLE_MASK);

         nir_export_agx(&b, mask, .base = AGX_ABI_FOUT_SAMPLE_MASK);

         NIR_PASS(_, nir, agx_nir_lower_to_per_sample);
      }

      NIR_PASS(_, nir, agx_nir_lower_fs_active_samples_to_register);
      NIR_PASS(_, nir, agx_nir_lower_interpolation,
               dev->dev.params.gpu_generation >= 15);
   } else if (sw_stage == MESA_SHADER_TESS_EVAL ||
              sw_stage == MESA_SHADER_TESS_CTRL) {

      shader->info.tess.info.ccw = nir->info.tess.ccw;
      shader->info.tess.info.points = nir->info.tess.point_mode;
      shader->info.tess.info.spacing = nir->info.tess.spacing;
      shader->info.tess.info.mode = nir->info.tess._primitive_mode;

      if (sw_stage == MESA_SHADER_TESS_CTRL) {
         shader->info.tess.tcs_output_patch_size =
            nir->info.tess.tcs_vertices_out;
         shader->info.tess.tcs_per_vertex_outputs =
            poly_tcs_per_vertex_outputs(nir);
         shader->info.tess.tcs_nr_patch_outputs =
            util_last_bit(nir->info.patch_outputs_written);
         shader->info.tess.tcs_output_stride = poly_tcs_output_stride(nir);
      } else if (hw_tess) {
         /* As below, for the hardware tessellator */
         NIR_PASS(_, nir, poly_nir_lower_tes_hw);
      } else {
         /* This destroys info so it needs to happen after the gather */
         NIR_PASS(_, nir, poly_nir_lower_tes, hw);
      }
   }

   /* Normally, vertex shaders need to write a default point size. However, if
    * we have a geometry/tessellation shader, the hardware vertex (software
    * GS/TES) will handle this itself instead.
    */
   if (sw_stage == MESA_SHADER_VERTEX && hw && !kill_psiz) {
      NIR_PASS(_, nir, nir_lower_default_point_size);
   }

   uint64_t outputs = nir->info.outputs_written;
   if (sw_stage == MESA_SHADER_VERTEX || sw_stage == MESA_SHADER_TESS_EVAL) {
      if (hw) {
         hk_lower_hw_vs(nir, shader, kill_psiz,
                        dev->dev.params.gpu_generation >= 15);
      } else {
         NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);
         nir->info.stage = MESA_SHADER_COMPUTE;
         memset(&nir->info.cs, 0, sizeof(nir->info.cs));
         nir->xfb_info = NULL;
      }
   }

   unsigned root = 0;
   if (sw_stage == MESA_SHADER_FRAGMENT)
      root = AGX_ABI_FUNI_ROOT;
   else if (sw_stage == MESA_SHADER_VERTEX)
      root = AGX_ABI_VUNI_COUNT_VK(nr_vbos);

   shader->info.set_count = set_count;

   /* XXX: rename */
   NIR_PASS(_, nir, hk_lower_uvs_index, sw_stage, nr_vbos,
            dev->dev.params.gpu_generation >= 15);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_uniforms,
            nir_metadata_control_flow, &root);

#if 0
   /* TODO */
   nir_variable_mode robust2_modes = 0;
   if (rs->uniform_buffers == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT)
      robust2_modes |= nir_var_mem_ubo;
   if (rs->storage_buffers == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT)
      robust2_modes |= nir_var_mem_ssbo;
#endif

   struct agx_shader_key backend_key = {
      .reserved_preamble = root + (4 * (1 + set_count)),
      .dev = agx_gather_device_key(&dev->dev),
      .no_stop = nir->info.stage == MESA_SHADER_FRAGMENT,
      .vs_prolog = sw_stage == MESA_SHADER_VERTEX && hw,
      .has_scratch = !nir->info.internal,
      .promote_constants = true,
      .promote_textures = true,
   };

   /* For now, sample shading is always dynamic. Indicate that. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       nir->info.fs.uses_sample_shading)
      backend_key.fs.inside_sample_loop = true;

   /* The pipeline's one sample, which the coverage words and tile accesses
    * take literally */
   if (nir->info.stage == MESA_SHADER_FRAGMENT && shader->info.fs.one_sample)
      backend_key.fs.nr_samples = 1;

   /* The epilog reads the colours where the main shader leaves them */
   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       !backend_key.fs.inside_sample_loop &&
       !(agx_get_compiler_debug() & AGX_DBG_OLDEPILOG))
      backend_key.fs.float_colour_exports = true;

   simple_mtx_t *lock = NULL;
   if (agx_get_compiler_debug())
      lock = &hk_device_physical(dev)->debug_compile_lock;

   if (lock)
      simple_mtx_lock(lock);

   /* A pipeline that captures its internal representations prints the NIR as
    * the backend left it: compile it every time */
   assert(nir->info.io_lowered);
   if (lock || (shader_flags & VK_SHADER_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_MESA))
      agx_compile_shader_nir(nir, &backend_key, &shader->b);
   else
      hk_compile_backend(dev, nir, &backend_key, &shader->b);

   if (lock)
      simple_mtx_unlock(lock);

   shader->code_ptr = shader->b.binary;
   shader->code_size = shader->b.info.binary_size;

   shader->info.stage = sw_stage;
   shader->info.clip_distance_array_size = nir->info.clip_distance_array_size;
   shader->info.cull_distance_array_size = nir->info.cull_distance_array_size;
   shader->b.info.outputs = outputs;

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      typed_memcpy(shader->info.fs.epilog_key.colour_src,
                   shader->b.info.colour_src,
                   ARRAY_SIZE(shader->info.fs.epilog_key.colour_src));
   }

   if (xfb_info) {
      assert(xfb_info->output_count < ARRAY_SIZE(shader->info.xfb_outputs));

      memcpy(&shader->info.xfb_info, xfb_info,
             nir_xfb_info_size(xfb_info->output_count));

      typed_memcpy(shader->info.xfb_stride, nir->info.xfb_stride, 4);
   }

   if (nir->constant_data_size > 0) {
      uint32_t data_size = align(nir->constant_data_size, HK_MIN_UBO_ALIGNMENT);

      void *data = malloc(data_size);
      if (data == NULL) {
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }

      memcpy(data, nir->constant_data, nir->constant_data_size);

      assert(nir->constant_data_size <= data_size);
      memset(data + nir->constant_data_size, 0,
             data_size - nir->constant_data_size);

      shader->data_ptr = data;
      shader->data_size = data_size;
   }

   VkResult result = hk_init_link_ht(shader, sw_stage);
   if (result != VK_SUCCESS)
      return vk_error(dev, result);

   hk_upload_shader(dev, shader);
   return VK_SUCCESS;
}

static const struct vk_shader_ops hk_shader_ops;

static void
hk_destroy_linked_shader(struct hk_device *dev, struct hk_linked_shader *linked)
{
   agx_bo_unreference(&dev->dev, linked->b.bo);
   ralloc_free(linked);
}

static void
hk_shader_destroy(struct hk_device *dev, struct hk_shader *s)
{
   free((void *)s->code_ptr);
   free((void *)s->data_ptr);
   agx_bo_unreference(&dev->dev, s->bo);

   simple_mtx_destroy(&s->linked.lock);

   if (s->only_linked)
      hk_destroy_linked_shader(dev, s->only_linked);

   if (s->linked.ht) {
      hash_table_foreach(s->linked.ht, entry) {
         hk_destroy_linked_shader(dev, entry->data);
      }
      _mesa_hash_table_destroy(s->linked.ht, NULL);
   }
}

static struct hk_api_shader *
hk_api_shader_alloc(struct hk_device *dev, mesa_shader_stage stage,
                    const VkAllocationCallbacks *pAllocator)
{
   size_t size = sizeof(struct hk_api_shader) +
                 sizeof(struct hk_shader) * hk_num_variants(stage);
   struct hk_api_shader *obj =
      vk_shader_zalloc(&dev->vk, &hk_shader_ops, stage, pAllocator, size);

   if (obj != NULL) {
      util_queue_fence_init(&obj->ready);
      simple_mtx_init(&obj->later_lock, mtx_plain);
   }

   return obj;
}

void
hk_api_shader_destroy(struct vk_device *vk_dev, struct vk_shader *vk_shader,
                      const VkAllocationCallbacks *pAllocator)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   /* A pipeline destroyed while its stages compile waits for them */
   hk_api_shader_wait(obj);

   hk_foreach_variant(obj, shader) {
      hk_shader_destroy(dev, shader);
   }

   free(obj->debug_nir_dump);
   free(obj->debug_agx_dump);

   util_queue_fence_destroy(&obj->ready);
   simple_mtx_destroy(&obj->later_lock);
   vk_shader_free(&dev->vk, pAllocator, &obj->vk);
}

static void dump_shader_ir(
      struct hk_api_shader *obj, struct hk_shader* shader, nir_shader* nir,
      FILE *debug_nir_stream, FILE *debug_agx_stream);

VkResult
hk_compile_shader(struct hk_device *dev, struct vk_shader_compile_info *info,
                  const struct vk_graphics_pipeline_state *state,
                  const struct vk_features *vk_features,
                  const VkAllocationCallbacks *pAllocator,
                  struct hk_api_shader **shader_out)
{
   VkResult result;
   enum hk_feature_key features = hk_make_feature_key(vk_features);

   /* We consume the NIR, regardless of success or failure */
   nir_shader *nir = info->nir;

   struct hk_api_shader *obj = hk_api_shader_alloc(dev, info->stage, pAllocator);

   if (obj == NULL) {
      ralloc_free(nir);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   if (!nir->info.io_lowered) {
      hk_lower_nir(dev, nir, info->robustness, false, info->set_layout_count,
                   info->set_layouts, features);
   }

   mesa_shader_stage sw_stage = nir->info.stage;

   union hk_key key_tmp, *key = NULL;
   if (sw_stage == MESA_SHADER_FRAGMENT) {
      hk_populate_fs_key(&key_tmp.fs, state);
      key = &key_tmp;

      nir->info.fs.uses_sample_shading |= key->fs.force_sample_shading;

      /* Force late-Z for Z/S self-deps. TODO: There's probably a less silly way
       * to do this.
       */
      if (key->fs.zs_self_dep) {
         nir_builder b =
            nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
         nir_discard_if(&b, nir_imm_false(&b));
         nir->info.fs.uses_discard = true;
      }

      NIR_PASS(_, nir, agx_nir_lower_bary_coord);
      NIR_PASS(_, nir, agx_nir_lower_sample_intrinsics, false);
   } else if (sw_stage == MESA_SHADER_VERTEX) {
      hk_populate_vs_key(&key_tmp.vs, state);
      key = &key_tmp;
   } else if (sw_stage == MESA_SHADER_TESS_CTRL) {
      /* With the hardware tessellator, every TCS also writes the levels where
       * the hardware reads them, for the draws that use it
       */
      if (HK_PERF(dev, HWTESS)) {
         NIR_PASS(_, nir, poly_nir_lower_tcs_hw_factors);
         obj->variants[0].info.tess.hw_factors = true;
      }

      NIR_PASS(_, nir, poly_nir_lower_tcs, true);
   } else if (sw_stage == MESA_SHADER_TASK) {
      NIR_PASS(_, nir, poly_nir_lower_task, &obj->variants[0].info.mesh);
      agx_preprocess_nir(nir);
   }

   const bool dump_vk =
      info->flags & VK_SHADER_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_MESA;

   size_t debug_agx_size = 0;
   char *debug_agx_buffer = NULL;
   FILE *debug_agx_stream = NULL;
   size_t debug_nir_size = 0;
   char *debug_nir_buffer = NULL;
   FILE *debug_nir_stream = NULL;

   if (dump_vk) {
      debug_agx_stream = open_memstream(&debug_agx_buffer, &debug_agx_size);
      debug_nir_stream = open_memstream(&debug_nir_buffer, &debug_nir_size);
   }

   /* Compile all variants up front */
   if (sw_stage == MESA_SHADER_GEOMETRY) {
      struct hk_shader *main_variant = hk_main_gs_variant(obj);
      struct hk_shader *count_variant = hk_count_gs_variant(obj);

      nir_shader *count = NULL, *rast = NULL, *rast_cons = NULL;
      nir_shader *pre_gs = NULL;

      /* Only the passthrough geometry shader rasterizes conservatively */
      bool conservative =
         nir->info.internal &&
         dev->vk.enabled_extensions.EXT_conservative_rasterization;

      poly_nir_lower_gs(nir, &count, &rast, conservative ? &rast_cons : NULL,
                        &pre_gs, &count_variant->info.gs);

      agx_preprocess_nir(count);
      agx_preprocess_nir(rast);
      agx_preprocess_nir(pre_gs);

      struct hk_shader *shader = &obj->variants[HK_GS_VARIANT_RAST];
      hk_lower_hw_vs(rast, shader, false, dev->dev.params.gpu_generation >= 15);
      shader->info.gs = count_variant->info.gs;
      main_variant->info.gs = count_variant->info.gs;

      struct hk_shader *cons_variant =
         &obj->variants[HK_GS_VARIANT_RAST_CONSERVATIVE];

      if (rast_cons) {
         agx_preprocess_nir(rast_cons);
         hk_lower_conservative_budget(rast_cons, cons_variant,
                                      shader->info.uvs.size);
         hk_lower_hw_vs(rast_cons, cons_variant, false,
                        dev->dev.params.gpu_generation >= 15);
         cons_variant->info.gs = count_variant->info.gs;
      }

      struct {
         nir_shader *in;
         struct hk_shader *out;
      } variants[] = {
         {nir, hk_main_gs_variant(obj)},
         {pre_gs, hk_pre_gs_variant(obj)},
         {count, count_variant},
         {rast, &obj->variants[HK_GS_VARIANT_RAST]},
         {rast_cons, cons_variant},
      };

      for (unsigned v = 0; v < ARRAY_SIZE(variants); ++v) {
         if (variants[v].in) {
            result =
               hk_compile_nir(dev, pAllocator, variants[v].in, info->flags,
                              info->robustness, NULL, features, variants[v].out,
                              sw_stage, true, false, NULL,
                              info->set_layout_count);

            if (result != VK_SUCCESS) {
               ralloc_free(variants[v].in);
               hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
               ralloc_free(nir);
               ralloc_free(pre_gs);
               ralloc_free(count);
               ralloc_free(rast);
               ralloc_free(rast_cons);
               return result;
            }

            if (dump_vk) {
               dump_shader_ir(obj, variants[v].out, variants[v].in, debug_nir_stream, debug_agx_stream);
            }

            ralloc_free(variants[v].in);
         }
      }
   } else if (sw_stage == MESA_SHADER_MESH) {
      struct hk_shader *main_variant = &obj->variants[HK_MESH_VARIANT_MAIN];
      struct hk_shader *rast_variant = &obj->variants[HK_MESH_VARIANT_RAST];
      struct hk_shader *cons_variant =
         &obj->variants[HK_MESH_VARIANT_RAST_CONSERVATIVE];
      nir_shader *rast = NULL, *rast_cons = NULL;

      bool task = !(info->flags & VK_SHADER_CREATE_NO_TASK_SHADER_BIT_EXT);
      bool multiview = state && state->mv && state->mv->view_mask;
      bool conservative = hk_may_rasterize_conservative(dev, state);

      NIR_PASS(_, nir, poly_nir_lower_mesh, task, multiview, &rast,
               conservative ? &rast_cons : NULL, &main_variant->info.mesh);

      agx_preprocess_nir(nir);
      agx_preprocess_nir(rast);

      rast_variant->info.mesh = main_variant->info.mesh;
      hk_lower_hw_vs(rast, rast_variant,
                     main_variant->info.mesh.prim != MESA_PRIM_POINTS,
                     dev->dev.params.gpu_generation >= 15);

      if (rast_cons) {
         agx_preprocess_nir(rast_cons);
         hk_lower_conservative_budget(rast_cons, cons_variant,
                                      rast_variant->info.uvs.size);
         hk_lower_hw_vs(rast_cons, cons_variant, true,
                        dev->dev.params.gpu_generation >= 15);
         cons_variant->info.mesh = main_variant->info.mesh;
      }

      struct {
         nir_shader *in;
         struct hk_shader *out;
      } variants[] = {
         {nir, main_variant},
         {rast, rast_variant},
         {rast_cons, cons_variant},
      };

      for (unsigned v = 0; v < ARRAY_SIZE(variants); ++v) {
         if (!variants[v].in)
            continue;

         result =
            hk_compile_nir(dev, pAllocator, variants[v].in, info->flags,
                           info->robustness, NULL, features, variants[v].out,
                           sw_stage, true, false, NULL, info->set_layout_count);

         if (result != VK_SUCCESS) {
            hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
            ralloc_free(nir);
            ralloc_free(rast);
            ralloc_free(rast_cons);
            return result;
         }

         if (dump_vk) {
            dump_shader_ir(obj, variants[v].out, variants[v].in,
                           debug_nir_stream, debug_agx_stream);
         }
      }

      ralloc_free(nir);
      ralloc_free(rast);
      ralloc_free(rast_cons);
   } else if (sw_stage == MESA_SHADER_VERTEX ||
              sw_stage == MESA_SHADER_TESS_EVAL) {

      VkShaderStageFlags next_stage = info->next_stage_mask;

      /* Transform feedback and conservative rasterization are layered on top
       * of geometry shaders. If there is not a geometry shader in the
       * pipeline, we will compile a geometry shader for the purpose. Update
       * the next_stage mask accordingly.
       */
      if (nir->xfb_info != NULL || hk_may_rasterize_conservative(dev, state) ||
          (dev->dev.debug & AGX_DBG_PASSTHROUGH_GS)) {
         next_stage |= VK_SHADER_STAGE_GEOMETRY_BIT;
      }

      if (sw_stage == MESA_SHADER_VERTEX) {
         assert(
            !(nir->info.inputs_read & BITFIELD64_MASK(VERT_ATTRIB_GENERIC0)) &&
            "Fixed-function attributes not used in Vulkan");

         NIR_PASS(_, nir, nir_recompute_io_bases, nir_var_shader_in);
      }

      /* Fold constant offset srcs for IO. */
      NIR_PASS(_, nir, nir_opt_constant_folding);

      /* Only compile the software variant if we might use this shader with
       * geometry/tessellation. We need to compile the hardware variant
       * unconditionally to handle the VS -> null FS case, which does not
       * require setting the FRAGMENT bit.
       */
      bool compile_sw =
         next_stage & (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                       VK_SHADER_STAGE_GEOMETRY_BIT);

      /* G15's hardware tessellator runs the TES as a hardware vertex shader
       * of its own. The domain and points mode may be the TCS's alone, which
       * the draw checks.
       */
      bool compile_hw_tess =
         sw_stage == MESA_SHADER_TESS_EVAL && HK_PERF(dev, HWTESS) &&
         !nir->info.tess.point_mode &&
         nir->info.tess._primitive_mode != TESS_PRIMITIVE_ISOLINES;

      unsigned last_variant = compile_hw_tess ? HK_TES_VARIANT_HW_TESS
                              : compile_sw    ? HK_VS_VARIANT_SW
                                              : HK_VS_VARIANT_HW;

      for (unsigned v = 0; v < hk_num_variants(sw_stage); ++v) {
         if (v == HK_VS_VARIANT_SW && !compile_sw)
            continue;

         if (v == HK_TES_VARIANT_HW_TESS && !compile_hw_tess)
            continue;

         struct hk_shader *shader = &obj->variants[v];
         bool hw = v != HK_VS_VARIANT_SW;
         bool hw_tess = v == HK_TES_VARIANT_HW_TESS;
         bool last = v == last_variant;

         /* Each variant gets its own NIR. To save an extra clone, we use the
          * original NIR for the last variant that is compiled, so that it is
          * freed exactly once, as that variant's clone.
          */
         nir_shader *clone = last ? nir : nir_shader_clone(NULL, nir);

         NIR_PASS(_, clone, agx_nir_gather_vs_inputs,
                  shader->info.vs.attrib_components_read);

         if (sw_stage == MESA_SHADER_VERTEX) {
            shader->info.vs.use_prolog = !(key && key->vs.skip_prolog);
            shader->info.vs.attribs_read =
               nir->info.inputs_read >> VERT_ATTRIB_GENERIC0;

            if (shader->info.vs.use_prolog) {
               NIR_PASS(_, clone, agx_nir_lower_vs_input_to_prolog);
            } else {
               struct agx_velem_key attribs[AGX_MAX_ATTRIBS];
               for (unsigned a = 0; a < AGX_MAX_ATTRIBS; ++a) {
                  if (key->vs.attribs[a].format) {
                     unsigned slot = util_bitcount64(
                        shader->info.vs.attribs_read & BITFIELD_MASK(a));

                     attribs[slot] = key->vs.attribs[a];
                  }
               }

               struct agx_robustness agx_rs = {
                  .soft_fault = agx_has_soft_fault(&dev->dev),

                  /* Correctly handling GPL + pipeline-robustness requires
                   * runtime changes, and I don't care enough to optimize this.
                   */
                  .level = AGX_ROBUSTNESS_D3D,
               };

               agx_nir_lower_vbo(clone, attribs, agx_rs,
                                 !key->vs.static_strides);

               unsigned nr = DIV_ROUND_UP(
                  BITSET_LAST_BIT(shader->info.vs.attrib_components_read), 4);
               agx_nir_lower_non_monolithic_uniforms(clone, nr);
            }
         }

         /* hk_compile_nir does not free the NIR, even on failure */
         result =
            hk_compile_nir(dev, pAllocator, clone, info->flags,
                           info->robustness, key, features, shader, sw_stage,
                           hw, hw_tess, nir->xfb_info, info->set_layout_count);
         if (result != VK_SUCCESS) {
            ralloc_free(clone);
            hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);

            /* For the last variant, clone is nir and was freed above */
            if (!last)
               ralloc_free(nir);

            return result;
         }


         if (dump_vk) {
            dump_shader_ir(obj, shader, clone, debug_nir_stream, debug_agx_stream);
         }

         ralloc_free(clone);
      }
   } else {
      struct hk_shader *shader = hk_only_variant(obj);

      /* hk_compile_nir does not free the NIR, even on failure */
      result = hk_compile_nir(dev, pAllocator, nir, info->flags,
                              info->robustness, key, features, shader, sw_stage,
                              true, false, NULL, info->set_layout_count);
      if (result != VK_SUCCESS) {
         ralloc_free(nir);
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return result;
      }

      if (dump_vk) {
         dump_shader_ir(obj, shader, nir, debug_nir_stream, debug_agx_stream);
      }

      ralloc_free(nir);
   }

   if (dump_vk) {
      fflush(debug_agx_stream);
      fclose(debug_agx_stream);
      obj->debug_agx_dump = debug_agx_buffer;
      fflush(debug_nir_stream);
      fclose(debug_nir_stream);
      obj->debug_nir_dump = debug_nir_buffer;
   }

   *shader_out = obj;
   return VK_SUCCESS;
}

static void
nir_opts(nir_shader *nir, void *data)
{
   /* A pass runs again only once another has changed the shader */
   struct set *skip = _mesa_pointer_set_create(NULL);
   bool progress;

   do {
      progress = false;

      NIR_LOOP_PASS_NOT_IDEMPOTENT(progress, skip, nir, nir_opt_loop);
      NIR_LOOP_PASS(progress, skip, nir, nir_opt_copy_prop);
      NIR_LOOP_PASS(progress, skip, nir, nir_opt_remove_phis);
      NIR_LOOP_PASS(progress, skip, nir, nir_opt_dce);

      NIR_LOOP_PASS_NOT_IDEMPOTENT(progress, skip, nir, nir_opt_if, 0);
      NIR_LOOP_PASS(progress, skip, nir, nir_opt_dead_cf);
      NIR_LOOP_PASS(progress, skip, nir, nir_opt_cse);

      NIR_LOOP_PASS(progress, skip, nir, nir_opt_peephole_select,
                    &(nir_opt_peephole_select_options){
                       .limit = 8,
                       .expensive_alu_ok = true,
                       .discard_ok = true,
                    });

      NIR_LOOP_PASS(progress, skip, nir, nir_opt_phi_precision);
      NIR_LOOP_PASS_NOT_IDEMPOTENT(progress, skip, nir, nir_opt_algebraic);
      NIR_LOOP_PASS(progress, skip, nir, nir_opt_constant_folding);

      NIR_LOOP_PASS(progress, skip, nir, nir_opt_undef);
      NIR_LOOP_PASS_NOT_IDEMPOTENT(progress, skip, nir, nir_opt_loop_unroll);
   } while (progress);

   _mesa_set_destroy(skip, NULL);
}

/* A pipeline's stages are compiled beside each other once linked: an
 * application that creates its pipelines on one thread (Elden Ring's startup
 * set) otherwise waits for each stage in turn. hk_compile_shader is already
 * called from several threads at once by applications that compile in
 * parallel, so a stage is safe on another thread. Four threads take a
 * pipeline's stages but the last, which the calling thread compiles meanwhile;
 * a stage that finds them all busy is compiled on the calling thread, so
 * applications that compile on many threads themselves do not queue behind
 * them.
 */
#define HK_COMPILE_THREADS 4

struct hk_compile_job {
   struct hk_device *dev;
   struct vk_shader_compile_info *info;
   const struct vk_graphics_pipeline_state *state;
   const struct vk_features *features;
   const VkAllocationCallbacks *pAllocator;
   struct hk_api_shader **out;
   struct util_queue_fence fence;
   bool queued;
   VkResult result;
};

static void
hk_compile_queue_init(const void *data)
{
   struct hk_device *dev = (struct hk_device *)data;

   dev->compile.ok =
      util_queue_init(&dev->compile.queue, "hk_compile", 32,
                      HK_COMPILE_THREADS,
                      UTIL_QUEUE_INIT_RESIZE_IF_FULL |
                         UTIL_QUEUE_INIT_SET_FULL_THREAD_AFFINITY,
                      NULL);
}

static void
hk_compile_job_execute(void *data, UNUSED void *gdata, UNUSED int thread_index)
{
   struct hk_compile_job *job = data;

   job->result = hk_compile_shader(job->dev, job->info, job->state,
                                   job->features, job->pAllocator, job->out);
   p_atomic_dec(&job->dev->compile.busy);
}

/* SERIAL keeps every stage on the calling thread (a pool job's) */
static VkResult
hk_compile_shaders_now(struct hk_device *dev, uint32_t shader_count,
                       struct vk_shader_compile_info *infos,
                       const struct vk_graphics_pipeline_state *state,
                       const struct vk_features *features,
                       const VkAllocationCallbacks *pAllocator,
                       struct vk_shader **shaders_out, bool serial)
{
   /* The runtime compiles ray-tracing stages one by one, without
    * get_rt_group_linking
    */
   if (shader_count > 0 && mesa_shader_stage_is_rt(infos[0].stage)) {
      return hk_compile_rt_stages(dev, shader_count, infos, pAllocator,
                                  shaders_out);
   }

   nir_shader *shaders[shader_count];
   bool mesh = false;

   /* Lower shaders, notably lowering I/O. This is a prerequisite for
    * intershader optimization.
    */
   for (uint32_t i = 0; i < shader_count; i++) {
      const struct vk_shader_compile_info *info = &infos[i];
      /* TODO: Multiview with ESO */
      const bool is_multiview = state && state->mv->view_mask != 0;
      enum hk_feature_key hk_features = hk_make_feature_key(features);
      nir_shader *nir = info->nir;

      mesh |= nir->info.stage == MESA_SHADER_MESH;

      /* As NVK: after a mesh shader these inputs are per-primitive, which
       * nir_opt_varyings needs both sides to agree on.
       */
      if (nir->info.stage == MESA_SHADER_FRAGMENT && mesh) {
         nir_foreach_shader_in_variable(var, nir) {
            if (var->data.location == VARYING_SLOT_PRIMITIVE_ID ||
                var->data.location == VARYING_SLOT_VIEWPORT ||
                var->data.location == VARYING_SLOT_LAYER)
               var->data.per_primitive = true;
         }
      }

      /* A captured stage prints its own NIR, names included */
      if (info->flags & VK_SHADER_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_MESA) {
         hk_lower_nir(dev, nir, info->robustness, is_multiview,
                      info->set_layout_count, info->set_layouts, hk_features);
      } else {
         nir = hk_lower_nir_cached(dev, nir, info->robustness, is_multiview,
                                   info->set_layout_count, info->set_layouts,
                                   hk_features);
         infos[i].nir = nir;
      }

      if (nir->xfb_info) {
         /* Fold constant offset srcs for IO. */
         NIR_PASS(_, nir, nir_opt_constant_folding);

         nir_io_add_intrinsic_xfb_info(nir);
      }

      shaders[i] = nir;
   }

   nir_opt_varyings_bulk(shaders, shader_count, true, UINT32_MAX, UINT32_MAX,
                         nir_opts, NULL, NULL);

   if (shader_count > 1 && !serial && !HK_PERF(dev, SERIALCOMPILE)) {
      util_call_once_data(&dev->compile.once, hk_compile_queue_init, dev);

      if (dev->compile.ok) {
         struct hk_compile_job jobs[shader_count];
         VkResult result = VK_SUCCESS;

         /* The last stage (the fragment shader, usually the largest) stays on
          * this thread, the others go to the queue while it has room */
         for (uint32_t i = 0; i < shader_count; i++) {
            jobs[i] = (struct hk_compile_job){
               .dev = dev,
               .info = &infos[i],
               .state = state,
               .features = features,
               .pAllocator = pAllocator,
               .out = (struct hk_api_shader **)&shaders_out[i],
            };

            if (i + 1 < shader_count &&
                p_atomic_inc_return(&dev->compile.busy) <= HK_COMPILE_THREADS) {
               jobs[i].queued = true;
               util_queue_fence_init(&jobs[i].fence);
               util_queue_add_job(&dev->compile.queue, &jobs[i], &jobs[i].fence,
                                  hk_compile_job_execute, NULL, 0);
            } else if (i + 1 < shader_count) {
               p_atomic_dec(&dev->compile.busy);
            }
         }

         for (uint32_t i = 0; i < shader_count; i++) {
            if (!jobs[i].queued) {
               jobs[i].result =
                  hk_compile_shader(dev, jobs[i].info, state, features,
                                    pAllocator, jobs[i].out);
            }
         }

         for (uint32_t i = 0; i < shader_count; i++) {
            if (jobs[i].queued) {
               util_queue_fence_wait(&jobs[i].fence);
               util_queue_fence_destroy(&jobs[i].fence);
            }

            if (jobs[i].result != VK_SUCCESS && result == VK_SUCCESS)
               result = jobs[i].result;
         }

         /* Each stage consumed its NIR: on a failure, destroy the ones built */
         if (result != VK_SUCCESS) {
            for (uint32_t i = 0; i < shader_count; i++) {
               if (jobs[i].result == VK_SUCCESS)
                  hk_api_shader_destroy(&dev->vk, shaders_out[i], pAllocator);
            }

            memset(shaders_out, 0, shader_count * sizeof(*shaders_out));
         }

         return result;
      }
   }

   for (uint32_t i = 0; i < shader_count; i++) {
      VkResult result =
         hk_compile_shader(dev, &infos[i], state, features, pAllocator,
                           (struct hk_api_shader **)&shaders_out[i]);
      if (result != VK_SUCCESS) {
         /* Clean up all the shaders before this point */
         for (uint32_t j = 0; j < i; j++)
            hk_api_shader_destroy(&dev->vk, shaders_out[j], pAllocator);

         /* Clean up all the NIR after this point */
         for (uint32_t j = i + 1; j < shader_count; j++)
            ralloc_free(infos[j].nir);

         /* Memset the output array */
         memset(shaders_out, 0, shader_count * sizeof(*shaders_out));

         return result;
      }
   }

   return VK_SUCCESS;
}

/* A graphics pipeline's stages compile on the device's pool after its creation
 * returns: an application that creates its pipelines on one thread (Elden
 * Ring's startup set, 8,825 pipelines) otherwise waits for each in turn, and
 * Elden Ring letterboxes only once its set is created. The objects returned
 * are allocated now and filled by the job; each reader of their variants
 * waits for it (hk_api_shader_wait: binding, serializing, the executables'
 * queries, destroying), and the runtime's disk cache takes a shader once it is
 * compiled (put_when_compiled). The queue holds as many jobs waiting as it has
 * threads: a full queue makes the creating thread wait, which bounds the
 * memory the jobs hold and the wait of a bind. The inputs that outlive the
 * call only as the pipeline's (its state, robustness, set layouts, samplers)
 * are copied or referenced.
 */
static bool hk_api_shader_serialize_compiled(struct vk_device *vk_dev,
                                             struct hk_api_shader *obj,
                                             struct blob *blob);

struct hk_later_job {
   struct hk_device *dev;
   uint32_t count;
   struct vk_shader_compile_info infos[MESA_VK_MAX_GRAPHICS_PIPELINE_STAGES];
   struct vk_pipeline_robustness_state rs[MESA_VK_MAX_GRAPHICS_PIPELINE_STAGES];
   VkPushConstantRange push[MESA_VK_MAX_GRAPHICS_PIPELINE_STAGES];
   struct vk_sampler_state *samplers[MESA_VK_MAX_GRAPHICS_PIPELINE_STAGES];
   uint32_t set_layout_count;
   struct vk_descriptor_set_layout *set_layouts[MESA_VK_MAX_DESCRIPTOR_SETS];
   struct vk_graphics_pipeline_state state;
   void *state_mem;
   struct hk_api_shader *out[MESA_VK_MAX_GRAPHICS_PIPELINE_STAGES];
};

static void
hk_later_queue_init(const void *data)
{
   struct hk_device *dev = (struct hk_device *)data;
   int threads = debug_get_num_option(
      "HK_LATER_THREADS", MAX2(util_get_cpu_caps()->nr_cpus - 2, 1));

   /* u_queue starts one thread and adds the others from the thread that
    * queues a job, which a game may pin (Elden Ring pins the one creating its
    * pipelines to a core): each worker takes the process's CPUs instead */
   dev->later.ok = threads > 0 &&
                   util_queue_init(&dev->later.queue, "hk_later", threads,
                                   threads,
                                   UTIL_QUEUE_INIT_SET_FULL_THREAD_AFFINITY,
                                   NULL);
}

/* An hk_shader holds no pointer to itself or to its api shader: the
 * variants move by copy */
static void
hk_api_shader_take(struct hk_device *dev, struct hk_api_shader *obj,
                   struct hk_api_shader *built)
{
   obj->is_passthrough = built->is_passthrough;
   obj->debug_nir_dump = built->debug_nir_dump;
   obj->debug_agx_dump = built->debug_agx_dump;
   memcpy(obj->variants, built->variants,
          sizeof(struct hk_shader) * hk_num_variants(obj->vk.stage));

   util_queue_fence_destroy(&built->ready);
   simple_mtx_destroy(&built->later_lock);
   vk_shader_free(&dev->vk, NULL, &built->vk);
}

/* The shader is compiled: it goes to the disk cache the runtime gave it, then
 * its readers go on */
static void
hk_api_shader_compiled(struct hk_device *dev, struct hk_api_shader *obj)
{
   simple_mtx_lock(&obj->later_lock);
   obj->compiling = false;
   struct disk_cache *disk_cache = obj->disk_cache;
   obj->disk_cache = NULL;
   simple_mtx_unlock(&obj->later_lock);

   if (disk_cache != NULL && obj->result == VK_SUCCESS) {
      struct blob blob;
      blob_init(&blob);

      if (hk_api_shader_serialize_compiled(&dev->vk, obj, &blob) &&
          !blob.out_of_memory)
         disk_cache_put(disk_cache, obj->disk_key, blob.data, blob.size, NULL);

      blob_finish(&blob);
   }

   util_queue_fence_signal(&obj->ready);
}

static void
hk_later_job_execute(void *data, UNUSED void *gdata, UNUSED int thread_index)
{
   struct hk_later_job *job = data;
   struct hk_device *dev = job->dev;
   struct vk_shader *built[MESA_VK_MAX_GRAPHICS_PIPELINE_STAGES];

   VkResult result =
      hk_compile_shaders_now(dev, job->count, job->infos, &job->state,
                             &dev->vk.enabled_features, NULL, built, true);

   for (uint32_t i = 0; i < job->count; i++) {
      struct hk_api_shader *obj = job->out[i];

      if (result == VK_SUCCESS) {
         hk_api_shader_take(dev, obj,
                            container_of(built[i], struct hk_api_shader, vk));
      } else {
         mesa_loge("hk: a pipeline's %s shader failed to compile after its "
                   "creation returned (%d)",
                   _mesa_shader_stage_to_string(obj->vk.stage), result);
      }

      obj->result = result;
      hk_api_shader_compiled(dev, obj);
   }
}

static void
hk_later_job_cleanup(void *data, UNUSED void *gdata, UNUSED int thread_index)
{
   struct hk_later_job *job = data;
   struct hk_device *dev = job->dev;

   for (uint32_t i = 0; i < job->set_layout_count; i++) {
      if (job->set_layouts[i])
         vk_descriptor_set_layout_unref(&dev->vk, job->set_layouts[i]);
   }

   for (uint32_t i = 0; i < job->count; i++)
      free(job->samplers[i]);

   vk_free(&dev->vk.alloc, job->state_mem);
   free(job);
}

static bool
hk_compile_later_ok(struct hk_device *dev, uint32_t shader_count,
                    const struct vk_shader_compile_info *infos,
                    const struct vk_graphics_pipeline_state *state)
{
   /* Monolithic graphics pipelines only (may_complete_later): shader objects'
    * binaries are read at once, compute and ray tracing pipelines are few */
   if (state == NULL || shader_count == 0 || HK_PERF(dev, SYNCCOMPILE) ||
       shader_count > MESA_VK_MAX_GRAPHICS_PIPELINE_STAGES)
      return false;

   for (uint32_t i = 0; i < shader_count; i++) {
      if (!infos[i].may_complete_later ||
          mesa_shader_stage_is_rt(infos[i].stage) ||
          mesa_shader_stage_is_compute(infos[i].stage) ||
          infos[i].nir->info.internal ||
          infos[i].set_layout_count > MESA_VK_MAX_DESCRIPTOR_SETS ||
          infos[i].push_constant_range_count > 1 ||
          infos[i].set_layouts != infos[0].set_layouts ||
          infos[i].set_layout_count != infos[0].set_layout_count ||
          (infos[i].flags &
           VK_SHADER_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_MESA))
         return false;
   }

   util_call_once_data(&dev->later.once, hk_later_queue_init, dev);
   return dev->later.ok;
}

static VkResult
hk_compile_shaders_later(struct hk_device *dev, uint32_t shader_count,
                         struct vk_shader_compile_info *infos,
                         const struct vk_graphics_pipeline_state *state,
                         const VkAllocationCallbacks *pAllocator,
                         struct vk_shader **shaders_out)
{
   struct hk_later_job *job = calloc(1, sizeof(*job));
   if (job == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   job->dev = dev;
   job->count = shader_count;

   if (vk_graphics_pipeline_state_copy(&dev->vk, &job->state, state, NULL,
                                       VK_SYSTEM_ALLOCATION_SCOPE_OBJECT,
                                       &job->state_mem) != VK_SUCCESS) {
      free(job);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   for (uint32_t i = 0; i < shader_count; i++) {
      job->out[i] = hk_api_shader_alloc(dev, infos[i].stage, pAllocator);

      if (job->out[i] != NULL && infos[i].embedded_sampler_count > 0) {
         size_t size =
            infos[i].embedded_sampler_count * sizeof(*infos[i].embedded_samplers);
         job->samplers[i] = malloc(size);

         if (job->samplers[i] != NULL)
            memcpy(job->samplers[i], infos[i].embedded_samplers, size);
      }

      if (job->out[i] == NULL || (infos[i].embedded_sampler_count > 0 &&
                                  job->samplers[i] == NULL)) {
         for (uint32_t j = 0; j <= i; j++) {
            if (job->out[j] != NULL)
               hk_api_shader_destroy(&dev->vk, &job->out[j]->vk, pAllocator);
         }

         job->count = i + 1;
         hk_later_job_cleanup(job, NULL, 0);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
   }

   job->set_layout_count = infos[0].set_layout_count;
   for (uint32_t i = 0; i < job->set_layout_count; i++) {
      job->set_layouts[i] = infos[0].set_layouts[i];
      if (job->set_layouts[i])
         vk_descriptor_set_layout_ref(job->set_layouts[i]);
   }

   for (uint32_t i = 0; i < shader_count; i++) {
      struct hk_api_shader *obj = job->out[i];

      job->infos[i] = infos[i];
      job->rs[i] = *infos[i].robustness;
      job->infos[i].robustness = &job->rs[i];
      job->infos[i].set_layouts = job->set_layouts;
      job->infos[i].embedded_samplers = job->samplers[i];

      if (infos[i].push_constant_range_count > 0) {
         job->push[i] = infos[i].push_constant_ranges[0];
         job->infos[i].push_constant_ranges = &job->push[i];
      }

      obj->compiling = true;
      util_queue_fence_reset(&obj->ready);
      shaders_out[i] = &obj->vk;
   }

   util_queue_add_job(&dev->later.queue, job, NULL, hk_later_job_execute,
                      hk_later_job_cleanup, 0);
   return VK_SUCCESS;
}

static VkResult
hk_compile_shaders(struct vk_device *vk_dev, uint32_t shader_count,
                   struct vk_shader_compile_info *infos,
                   const struct vk_graphics_pipeline_state *state,
                   const struct vk_features *features,
                   const VkAllocationCallbacks *pAllocator,
                   struct vk_shader **shaders_out)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);

   if (hk_compile_later_ok(dev, shader_count, infos, state) &&
       hk_compile_shaders_later(dev, shader_count, infos, state, pAllocator,
                                shaders_out) == VK_SUCCESS)
      return VK_SUCCESS;

   return hk_compile_shaders_now(dev, shader_count, infos, state, features,
                                 pAllocator, shaders_out, false);
}

/* A ray-tracing pipeline's kernel (hk_rt_pipeline.c), lowered as an API
 * compute shader
 */
VkResult
hk_compile_rt_kernel(struct hk_device *dev, struct vk_shader_compile_info *info,
                     const VkAllocationCallbacks *pAllocator,
                     struct hk_api_shader **shader_out)
{
   const struct vk_features *features = &dev->vk.enabled_features;

   hk_lower_nir(dev, info->nir, info->robustness, false, info->set_layout_count,
                info->set_layouts, hk_make_feature_key(features));

   return hk_compile_shader(dev, info, NULL, features, pAllocator, shader_out);
}

static VkResult
hk_deserialize_shader(struct hk_device *dev, struct blob_reader *blob,
                      struct hk_shader *shader)
{
   struct hk_shader_info info;
   blob_copy_bytes(blob, &info, sizeof(info));

   struct agx_shader_info b_info;
   blob_copy_bytes(blob, &b_info, sizeof(b_info));

   const uint32_t code_size = blob_read_uint32(blob);
   const uint32_t data_size = blob_read_uint32(blob);
   if (blob->overrun)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   VkResult result = hk_init_link_ht(shader, info.stage);
   if (result != VK_SUCCESS)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   simple_mtx_init(&shader->linked.lock, mtx_plain);

   shader->b.info = b_info;
   shader->info = info;
   shader->code_size = code_size;
   shader->data_size = data_size;
   shader->b.info.binary_size = code_size;

   shader->code_ptr = malloc(code_size);
   if (shader->code_ptr == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   shader->data_ptr = malloc(data_size);
   if (shader->data_ptr == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   blob_copy_bytes(blob, (void *)shader->code_ptr, shader->code_size);
   blob_copy_bytes(blob, (void *)shader->data_ptr, shader->data_size);
   if (blob->overrun)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   shader->b.binary = (void *)shader->code_ptr;
   hk_upload_shader(dev, shader);
   return VK_SUCCESS;
}

static VkResult
hk_deserialize_api_shader(struct vk_device *vk_dev, struct blob_reader *blob,
                          uint32_t binary_version,
                          const VkAllocationCallbacks *pAllocator,
                          struct vk_shader **shader_out)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);

   mesa_shader_stage stage = blob_read_uint8(blob);
   if (blob->overrun)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   if (mesa_shader_stage_is_rt(stage))
      return hk_rt_stage_deserialize(dev, blob, stage, pAllocator, shader_out);

   struct hk_api_shader *obj = hk_api_shader_alloc(dev, stage, pAllocator);

   if (obj == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   hk_foreach_variant(obj, shader) {
      VkResult result = hk_deserialize_shader(dev, blob, shader);

      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return result;
      }
   }

   *shader_out = &obj->vk;
   return VK_SUCCESS;
}

static void
hk_shader_serialize(struct vk_device *vk_dev, const struct hk_shader *shader,
                    struct blob *blob)
{
   blob_write_bytes(blob, &shader->info, sizeof(shader->info));
   blob_write_bytes(blob, &shader->b.info, sizeof(shader->b.info));

   blob_write_uint32(blob, shader->code_size);
   blob_write_uint32(blob, shader->data_size);
   blob_write_bytes(blob, shader->code_ptr, shader->code_size);
   blob_write_bytes(blob, shader->data_ptr, shader->data_size);
}

static bool
hk_api_shader_serialize_compiled(struct vk_device *vk_dev,
                                 struct hk_api_shader *obj, struct blob *blob)
{
   blob_write_uint8(blob, obj->vk.stage);

   hk_foreach_variant(obj, shader) {
      hk_shader_serialize(vk_dev, shader, blob);
   }

   return !blob->out_of_memory;
}

static bool
hk_api_shader_serialize(struct vk_device *vk_dev,
                        const struct vk_shader *vk_shader, struct blob *blob)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   hk_api_shader_wait(obj);
   return hk_api_shader_serialize_compiled(vk_dev, obj, blob);
}

static bool
hk_api_shader_put_when_compiled(UNUSED struct vk_device *vk_dev,
                                struct vk_shader *vk_shader,
                                struct disk_cache *disk_cache,
                                const unsigned char *cache_key)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   simple_mtx_lock(&obj->later_lock);
   bool compiling = obj->compiling;
   if (compiling) {
      obj->disk_cache = disk_cache;
      memcpy(obj->disk_key, cache_key, sizeof(obj->disk_key));
   }
   simple_mtx_unlock(&obj->later_lock);

   return compiling;
}

static VkResult
hk_shader_get_executable_properties(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t *executable_count, VkPipelineExecutablePropertiesKHR *properties)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutablePropertiesKHR, out, properties,
                          executable_count);

   vk_outarray_append_typed(VkPipelineExecutablePropertiesKHR, &out, props)
   {
      props->stages = mesa_to_vk_shader_stage(obj->vk.stage);
      props->subgroupSize = 32;
      VK_COPY_STR(props->name, _mesa_shader_stage_to_string(obj->vk.stage));
      VK_PRINT_STR(props->description, "%s shader",
                   _mesa_shader_stage_to_string(obj->vk.stage));
   }

   return vk_outarray_status(&out);
}

static VkResult
hk_shader_get_executable_statistics(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *statistic_count,
   VkPipelineExecutableStatisticKHR *statistics)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   hk_api_shader_wait(obj);

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableStatisticKHR, out, statistics,
                          statistic_count);

   assert(executable_index == 0);

   /* TODO: find a sane way to report multiple variants and have that play nice
    * with zink.
    */
   struct hk_shader *shader = hk_any_variant(obj);
   vk_add_agx2_stats(out, &shader->b.info.stats);
   return vk_outarray_status(&out);
}

static void dump_shader_ir(
   struct hk_api_shader *obj, struct hk_shader* shader, nir_shader* nir,
   FILE *debug_nir_stream, FILE *debug_agx_stream)
{
   const char* shader_name = hk_variant_name(obj, shader);
   if (!shader_name) shader_name = "";

   fprintf(debug_nir_stream, "nir %s\n", shader_name);
   nir_print_shader_annotated(nir, debug_nir_stream, NULL);
   fprintf(debug_nir_stream, "\n");

   fprintf(debug_agx_stream, "agx %s\n", shader_name);

   unsigned preamble_size = 0;
   if (shader->b.info.has_preamble) {
      if (shader->b.info.preamble_offset > shader->b.info.main_offset) {
         preamble_size = shader->b.info.binary_size - shader->b.info.preamble_offset;
      } else {
         preamble_size = shader->b.info.main_offset - shader->b.info.preamble_offset;
      }
      fprintf(debug_agx_stream, "preamble:\n");
      agx2_disassemble((uint8_t*)shader->code_ptr + shader->b.info.preamble_offset, preamble_size, debug_agx_stream);
   }

   fprintf(debug_agx_stream, "entrypoint:\n");
   agx2_disassemble((uint8_t*)shader->code_ptr + shader->b.info.main_offset, shader->b.info.main_size, debug_agx_stream);
   fprintf(debug_agx_stream, "\n");
}

static bool
write_ir_text(VkPipelineExecutableInternalRepresentationKHR *ir,
              const char *data)
{
   ir->isText = VK_TRUE;

   size_t data_len = strlen(data) + 1;

   if (ir->pData == NULL) {
      ir->dataSize = data_len;
      return true;
   }

   strncpy(ir->pData, data, ir->dataSize);
   if (ir->dataSize < data_len)
      return false;

   ir->dataSize = data_len;
   return true;
}

static VkResult
hk_shader_get_executable_internal_representations(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *internal_representation_count,
   VkPipelineExecutableInternalRepresentationKHR *internal_representations)
{
   assert(executable_index == 0);

   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   hk_api_shader_wait(obj);

   VkPipelineExecutableInternalRepresentationKHR *p = internal_representations;
   VkPipelineExecutableInternalRepresentationKHR *end =
      p + (internal_representations ? *internal_representation_count : 0);

   VkResult result = VK_SUCCESS;
   if (p < end) {
      VK_COPY_STR(p->name, "NIR Shader(s)");
      VK_COPY_STR(p->description, "The optimized NIR shader(s)");
      if (obj->debug_nir_dump) {
         if (!write_ir_text(p, obj->debug_nir_dump))
            result = VK_INCOMPLETE;
      } else {
         if (!write_ir_text(p, "Set AGX_MESA_DEBUG=dump_vk to dump NIR shaders\n")) 
            result = VK_INCOMPLETE;
      }
   }
   p++;

   if (p < end) {
      VK_COPY_STR(p->name, "AGX Shader(s)");
      VK_COPY_STR(p->description, "The final optimized AGX shader(s)");
      if (obj->debug_nir_dump) {
         if (!write_ir_text(p, obj->debug_agx_dump))
            result = VK_INCOMPLETE;
      } else {
         if (!write_ir_text(p, "Set AGX_MESA_DEBUG=dump_vk to dump AGX shaders\n")) 
            result = VK_INCOMPLETE;
      }
   }
   p++;

   if (!internal_representations)
      *internal_representation_count = p - internal_representations;
   else if (p > end) {
      result = VK_INCOMPLETE;
      *internal_representation_count = end - internal_representations;
   } else {
      *internal_representation_count = p - internal_representations;
   }

   return result;
}

static const struct vk_shader_ops hk_shader_ops = {
   .destroy = hk_api_shader_destroy,
   .serialize = hk_api_shader_serialize,
   .put_when_compiled = hk_api_shader_put_when_compiled,
   .get_executable_properties = hk_shader_get_executable_properties,
   .get_executable_statistics = hk_shader_get_executable_statistics,
   .get_executable_internal_representations =
      hk_shader_get_executable_internal_representations,
};

const struct vk_device_shader_ops hk_device_shader_ops = {
   .get_nir_options = hk_get_nir_options,
   .get_spirv_options = hk_get_spirv_options,
   .preprocess_nir = hk_preprocess_nir,
   .hash_state = hk_hash_graphics_state,
   .compile = hk_compile_shaders,
   .deserialize = hk_deserialize_api_shader,
   .cmd_set_dynamic_graphics_state = vk_cmd_set_dynamic_graphics_state,
   .cmd_bind_shaders = hk_cmd_bind_shaders,
   .write_rt_shader_group = hk_write_rt_shader_group,
   .write_rt_shader_group_replay_handle =
      hk_write_rt_shader_group_replay_handle,
   .replay_rt_shader_group = hk_replay_rt_shader_group,
   .cmd_set_rt_state = hk_cmd_set_rt_state,
   .cmd_set_stack_size = hk_cmd_set_rt_stack_size,
   .link_rt_pipeline = hk_link_rt_pipeline,
   .cmd_bind_rt_shader = hk_cmd_bind_rt_shader,
};

struct hk_linked_shader *
hk_fast_link(struct hk_device *dev, bool fragment, struct hk_shader *main,
             struct agx_shader_part *prolog, struct agx_shader_part *epilog,
             unsigned nr_samples_shaded, unsigned nr_samples)
{
   struct hk_linked_shader *s = rzalloc(NULL, struct hk_linked_shader);
   agx_fast_link(&s->b, &dev->dev, fragment, &main->b, prolog, epilog,
                 nr_samples_shaded, nr_samples);

   if (fragment) {
      unsigned samplers = main->b.info.sampler_state_count;
      if (s->b.uses_txf)
         samplers = MAX2(samplers, 1);

      agx_pack(&s->fs_counts, FRAGMENT_SHADER_WORD_0, cfg) {
         cfg.cf_binding_count = s->b.cf.nr_bindings;
         cfg.uniform_register_count = main->b.info.push_count;
         cfg.preshader_register_count = main->b.info.nr_preamble_gprs;
         cfg.texture_state_register_count = main->b.info.texture_state_count;
         cfg.sampler_state_register_count =
            agx_translate_sampler_state_count(samplers, false);
      }
   }

   /* Now that we've linked, bake the USC words to bind this program */
   struct agx_usc_builder b = agx_usc_builder(s->usc.data, sizeof(s->usc.data));

   if (main && main->b.info.rodata.size_16) {
      agx_usc_immediates(&b, &main->b.info.rodata, main->bo->va->addr);
   }

   /* G15 has no sampler state registers: a fetch's sampler is a descriptor
    * entry (hk_descriptor_set.h), and an entry program has no sampler word */
   if (s->b.uses_txf && dev->dev.params.gpu_generation < 15)
      agx_usc_push_packed(&b, SAMPLER, dev->dev.txf_sampler);

   agx_usc_shared_non_fragment(&b, &main->b.info, 0);
   agx_usc_push_packed(&b, SHADER, s->b.shader);
   agx_usc_push_packed(&b, REGISTERS, s->b.regs);

   if (fragment)
      agx_usc_push_packed(&b, FRAGMENT_PROPERTIES, s->b.fragment_props);

   if (main && main->b.info.has_preamble) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code = agx_usc_addr(&dev->dev, main->preamble_addr);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   s->usc.size = b.head - s->usc.data;
   return s;
}
