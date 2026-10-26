/*
 * Copyright 2024 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx_bo.h"
#include "agx_compile.h"
#include "agx_esl.h"
#include "agx_nir_lower_vbo.h"
#include "agx_pack.h"
#include "nir_lower_blend.h"
#include "util/macros.h"

struct agx_linked_shader {
   /* Mapped executable memory */
   struct agx_bo *bo;

   /* Set if the linked SW vertex shader reads base vertex/instance. The VS
    * prolog can read base instance even when the API VS does not, which is why
    * this needs to be aggregated in the linker.
    */
   bool uses_base_param;

   /* Set if the linked shader uses txf. The epilog may even if the main shader
    * does not, in the case of spilled render targets.
    */
   bool uses_txf;

   /* Whether the program is a no-op and can be skipped. */
   bool no_op;

   /* G15: the pixel fences the program waits on (it signals them at its end;
    * its entry program releases the others)
    */
   uint16_t pix_wait_mask;

   /* G15: the fences of pix_wait_mask a part releases itself before the end */
   uint16_t pix_signal_mask;

   /* Whether the program writes depth or stencil (G15's fragment state) */
   bool writes_zs;

   /* Size in bytes of the linked program (G15's entry program wants it) */
   unsigned main_size;

   /* G15: coefficient registers the linked program writes (specTPR) */
   unsigned cr_count;

   /* Coefficient register bindings */
   struct agx_varyings_fs cf;

   /* Data structures packed for the linked program */
   struct agx_usc_shader_packed shader;
   struct agx_usc_registers_packed regs;
   struct agx_usc_fragment_properties_packed fragment_props;
   struct agx_output_select_packed osel;
   struct agx_fragment_control_packed fragment_control;
};

void agx_fast_link(struct agx_linked_shader *linked, struct agx_device *dev,
                   bool fragment, struct agx_shader_part *main,
                   struct agx_shader_part *prolog,
                   struct agx_shader_part *epilog, unsigned nr_samples_shaded);

/* The code of a shader at va, its binary at cpu. A shader linked into a
 * program (linked, NULL for none) loads the program's main part. */
static inline struct agx_esl_code
agx_esl_code_for_shader(uint64_t va, const uint8_t *cpu,
                        const struct agx_shader_info *info,
                        const struct agx_linked_shader *linked)
{
   bool l = linked && linked->bo;

   return (struct agx_esl_code){
      .va = va,
      .cpu = cpu,
      .size = info->binary_size,
      .preamble_offset = info->preamble_offset,
      .preamble_size = info->preamble_size,
      .main_size = l ? linked->main_size : info->main_size,
      .stack_size = info->stack_size,
      .cr_count = l ? MAX2(linked->cr_count, info->cr_count) : info->cr_count,
      .pix_wait_mask = linked ? linked->pix_wait_mask : 0,
   };
}

/* The code of a libagx kernel at va, bin as asahi_clc stores it */
static inline struct agx_esl_code
agx_esl_code_for_precomp(uint64_t va, const uint32_t *bin)
{
   const struct agx_precompiled_kernel_info *info = (const void *)bin;

   return (struct agx_esl_code){
      .va = va,
      .cpu = (const uint8_t *)(info + 1),
      .size = info->binary_size,
      .preamble_offset = info->preamble_offset,
      .preamble_size = info->preamble_size,
      .main_size = info->main_size,
      .stack_size = info->stack_size,
      .cr_count = info->cr_count,
   };
}

/* The hardware stage of a shader: geometry and tessellation run as compute
 * kernels and a hardware vertex shader */
static inline enum agx_esl_stage
agx_esl_stage_for(mesa_shader_stage stage)
{
   return stage == MESA_SHADER_FRAGMENT ? AGX_ESL_FRAGMENT
          : stage == MESA_SHADER_VERTEX ? AGX_ESL_VERTEX
                                        : AGX_ESL_COMPUTE;
}

/* These parts of the vertex element affect the generated code */
struct agx_velem_key {
   uint32_t divisor;
   uint16_t stride;
   uint8_t format;
   bool instanced;
};

struct agx_vs_prolog_key {
   struct agx_velem_key attribs[AGX_MAX_VBUFS];

   /* Bit mask of attribute components to load */
   BITSET_DECLARE(component_mask, AGX_MAX_ATTRIBS * 4);

   /* Whether running as a hardware vertex shader (versus compute) */
   bool hw;

   /* Whether the shader uses static vertex input and the prolog just fixes up
    * the vertex ID.
    */
   bool static_vi;

   /* If !hw and the draw call is indexed, the index size */
   uint8_t sw_index_size_B;

   /* Adjacency primitive to emulate */
   enum mesa_prim adjacency;

   /* Robustness settings for the vertex fetch */
   struct agx_robustness robustness;
};

struct agx_fs_prolog_key {
   /* glSampleMask() mask */
   uint8_t api_sample_mask;

   /* Number of cull planes requiring lowering */
   uint8_t cull_distance_size;

   /* Need to count FRAGMENT_SHADER_INVOCATIONS */
   bool statistics;

   /* Need to lower desktop OpenGL polygon stipple */
   bool polygon_stipple;

   /* If we discard, whether we need to run Z/S tests */
   bool run_zs_tests;

   /* If we emulate cull distance, the base offset for our allocated coefficient
    * registers so we don't interfere with the main shader.
    */
   unsigned cf_base;
};

struct agx_blend_rt_key {
   unsigned colormask      : 4;
   unsigned advanced_blend : 1;
   unsigned pad            : 1;
   unsigned mode           : 26;
};
static_assert(sizeof(struct agx_blend_rt_key) == 4, "packed");

struct agx_blend_standard {
   enum pipe_blend_func rgb_func          : 3;
   enum pipe_blendfactor rgb_src_factor   : 5;
   enum pipe_blendfactor rgb_dst_factor   : 5;
   enum pipe_blend_func alpha_func        : 3;
   enum pipe_blendfactor alpha_src_factor : 5;
   enum pipe_blendfactor alpha_dst_factor : 5;
   unsigned pad                           : 6;
};
static_assert(sizeof(struct agx_blend_standard) == 4, "packed");

static inline unsigned
agx_pack_blend_standard(enum pipe_blend_func rgb_func,
                        enum pipe_blendfactor rgb_src_factor,
                        enum pipe_blendfactor rgb_dst_factor,
                        enum pipe_blend_func alpha_func,
                        enum pipe_blendfactor alpha_src_factor,
                        enum pipe_blendfactor alpha_dst_factor)
{
   struct agx_blend_standard blend = {
      .rgb_func = rgb_func,
      .rgb_src_factor = rgb_src_factor,
      .rgb_dst_factor = rgb_dst_factor,
      .alpha_func = alpha_func,
      .alpha_src_factor = alpha_src_factor,
      .alpha_dst_factor = alpha_dst_factor,
   };
   unsigned val;
   typed_memcpy(&val, &blend, 1);
   return val;
}

static inline struct agx_blend_standard
agx_unpack_blend_standard(unsigned mode)
{
   struct agx_blend_standard blend;
   typed_memcpy(&blend, &mode, 1);
   return blend;
}

struct agx_blend_advanced {
   enum pipe_advanced_blend_mode op     : 8;
   enum pipe_blend_overlap_mode overlap : 2;
   unsigned src_premultiplied           : 1;
   unsigned dst_premultiplied           : 1;
   unsigned clamp_results               : 1;
   unsigned pad                         : 19;
};
static_assert(sizeof(struct agx_blend_advanced) == 4, "packed");

static inline unsigned
agx_pack_blend_advanced(enum pipe_advanced_blend_mode op,
                        enum pipe_blend_overlap_mode overlap,
                        bool src_premultiplied, bool dst_premultiplied,
                        bool clamp_results)
{
   struct agx_blend_advanced blend = {
      .op = op,
      .overlap = overlap,
      .src_premultiplied = src_premultiplied,
      .dst_premultiplied = dst_premultiplied,
      .clamp_results = clamp_results,
   };
   unsigned val;
   typed_memcpy(&val, &blend, 1);
   return val;
}

static inline struct agx_blend_advanced
agx_unpack_blend_advanced(unsigned mode)
{
   struct agx_blend_advanced blend;
   typed_memcpy(&blend, &mode, 1);
   return blend;
}

struct agx_blend_key {
   struct agx_blend_rt_key rt[8];
   uint8_t logicop_func;
   bool alpha_to_coverage, alpha_to_one;
   bool logicop_enable;
};
static_assert(sizeof(struct agx_blend_key) == 36, "packed");

struct agx_fs_epilog_link_info {
   /* Base index of spilled render targets in the binding table */
   uint8_t rt_spill_base;

   /* Bit mask of the bit size written to each render target. Bit i set if RT i
    * uses 32-bit registers, else 16-bit registers.
    */
   uint8_t size_32;

   /* Mask of locations written by the main shader */
   uint8_t loc_written;

   /* If set, the API fragment shader uses sample shading. This means the epilog
    * will be invoked per-sample as well.
    */
   unsigned sample_shading : 1;

   /* If set, broadcast location #0 value to all render targets. This
    * implements gl_FragColor semantics. This tells the driver to set remap
    * appropriately.
    */
   unsigned broadcast_rt0 : 1;

   /* If set, force location 0's W channel to 1.0. This optimizes blending
    * calculations in some applications.
    */
   unsigned loc0_w_1 : 1;

   /* If set, the API fragment shader wants to write depth/stencil respectively.
    * This happens in the epilog for correctness when the epilog discards.
    */
   unsigned write_z : 1;
   unsigned write_s : 1;

   /* Whether the fragment prolog or main fragment shader already ran tests due
    * to early_fragment_tests. In this case, the epilog must not run tests.
    */
   unsigned already_ran_zs : 1;

   /* Whether the main fragment shader ran tests before discards due to
    * early_fragment_tests. In this case, the epilog must mask the stores in
    * software instead.
    */
   bool sample_mask_after_force_early : 1;

   unsigned padding : 1;
};
static_assert(sizeof(struct agx_fs_epilog_link_info) == 4, "packed");

struct agx_fs_epilog_key {
   struct agx_fs_epilog_link_info link;

   /* Blend state. Blending happens in the epilog. */
   struct agx_blend_key blend;

   /* Colour attachment remapping for Vulkan. Negative values indicate that an
    * attachment is discarded. Positive values indicate the output location we
    * want to store at the indexed colour attachment.
    */
   int8_t remap[8];

   /* Tilebuffer configuration */
   enum pipe_format rt_formats[8];
   uint8_t nr_samples;
   bool force_small_tile;

   /* G15: texture handles count 32 B entries, spilled render targets are
    * read by a fetch (agx_tilebuffer_layout::g15) */
   bool g15;
};

void agx_nir_vs_prolog(struct nir_builder *b, const void *key_);
void agx_nir_fs_epilog(struct nir_builder *b, const void *key_);
void agx_nir_fs_prolog(struct nir_builder *b, const void *key_);

bool agx_nir_gather_vs_inputs(nir_shader *s,
                              BITSET_WORD *attrib_components_read);

bool agx_nir_lower_vs_input_to_prolog(nir_shader *s);

bool agx_nir_lower_fs_output_to_epilog(nir_shader *s,
                                       struct agx_fs_epilog_link_info *out);

bool agx_nir_lower_fs_active_samples_to_register(nir_shader *s);

bool agx_nir_lower_non_monolithic_uniforms(nir_shader *nir, unsigned nr);
