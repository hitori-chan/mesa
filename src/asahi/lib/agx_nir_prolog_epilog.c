/*
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "gallium/include/pipe/p_defines.h"
#include "poly/cl/libpoly.h"
#include "poly/nir/poly_nir.h"
#include "util/format/u_formats.h"
#include "agx_abi.h"
#include "agx_compile.h"
#include "agx_debug.h"
#include "agx_linker.h"
#include "agx_nir.h"
#include "agx_nir_lower_vbo.h"
#include "agx_pack.h"
#include "agx_tilebuffer.h"
#include "libagx.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_lower_blend.h"
#include "shader_enums.h"

/*
 * Insert code into a fragment shader to lower polygon stipple. The stipple is
 * passed in a sideband, rather than requiring a texture binding. This is
 * simpler for drivers to integrate and might be more efficient.
 */
static bool
agx_nir_lower_poly_stipple(nir_shader *s)
{
   assert(s->info.stage == MESA_SHADER_FRAGMENT);

   /* Insert at the beginning for performance. */
   nir_builder b_ =
      nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(s)));
   nir_builder *b = &b_;

   /* The stipple coordinate is defined at the window coordinate mod 32. It's
    * reversed along the X-axis to simplify the driver, hence the NOT.
    */
   nir_def *raw = nir_u2u32(b, nir_load_pixel_coord(b));
   nir_def *coord = nir_umod_imm(
      b,
      nir_vec2(b, nir_inot(b, nir_channel(b, raw, 0)), nir_channel(b, raw, 1)),
      32);

   /* Extract the column from the packed bitfield */
   nir_def *pattern = nir_load_polygon_stipple_agx(b, nir_channel(b, coord, 1));
   nir_def *bit = nir_ubitfield_extract(b, pattern, nir_channel(b, coord, 0),
                                        nir_imm_int(b, 1));

   /* Discard fragments where the pattern is 0 */
   nir_demote_if(b, nir_ieq_imm(b, bit, 0));
   s->info.fs.uses_discard = true;

   return nir_progress(true, b->impl, nir_metadata_control_flow);
}

static int
map_vs_part_uniform(nir_intrinsic_instr *intr, unsigned nr_attribs)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_vbo_base_agx:
      return AGX_ABI_VUNI_VBO_BASE(nir_src_as_uint(intr->src[0]));

   case nir_intrinsic_load_attrib_clamp_agx:
      return AGX_ABI_VUNI_VBO_CLAMP(nr_attribs, nir_src_as_uint(intr->src[0]));

   case nir_intrinsic_load_first_vertex:
      return AGX_ABI_VUNI_FIRST_VERTEX(nr_attribs);

   case nir_intrinsic_load_base_instance:
      return AGX_ABI_VUNI_BASE_INSTANCE(nr_attribs);

   case nir_intrinsic_load_vertex_param_buffer_poly:
      return AGX_ABI_VUNI_VERTEX_PARAMS(nr_attribs);

   default:
      return -1;
   }
}

static int
map_fs_part_uniform(nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_blend_const_color_r_float:
      return AGX_ABI_FUNI_BLEND_R;

   case nir_intrinsic_load_blend_const_color_g_float:
      return AGX_ABI_FUNI_BLEND_G;

   case nir_intrinsic_load_blend_const_color_b_float:
      return AGX_ABI_FUNI_BLEND_B;

   case nir_intrinsic_load_blend_const_color_a_float:
      return AGX_ABI_FUNI_BLEND_A;

   default:
      return -1;
   }
}

static bool
lower_non_monolithic_uniforms(nir_builder *b, nir_intrinsic_instr *intr,
                              void *data)
{
   int unif;
   if (b->shader->info.stage == MESA_SHADER_VERTEX) {
      unsigned *nr_attribs = data;
      unif = map_vs_part_uniform(intr, *nr_attribs);
   } else {
      unif = map_fs_part_uniform(intr);
   }

   if (unif >= 0) {
      b->cursor = nir_instr_remove(&intr->instr);
      nir_def *load = nir_load_preamble(b, 1, intr->def.bit_size, .base = unif);
      nir_def_rewrite_uses(&intr->def, load);
      return true;
   } else if (intr->intrinsic == nir_intrinsic_load_texture_handle_agx) {
      /* G15 counts 32 B entries, not bytes */
      bool g15 =
         b->shader->info.stage == MESA_SHADER_FRAGMENT && data && *(bool *)data;

      b->cursor = nir_instr_remove(&intr->instr);
      nir_def *offs = nir_u2u32(b, intr->src[0].ssa);
      if (!g15)
         offs = nir_imul_imm(b, offs, AGX_TEXTURE_LENGTH);
      nir_def_rewrite_uses(&intr->def,
                           nir_bindless_image_agx(b, offs, .desc_set = 0));
      return true;
   } else {
      return false;
   }
}

bool
agx_nir_lower_non_monolithic_uniforms(nir_shader *nir, unsigned nr)
{
   return nir_shader_intrinsics_pass(nir, lower_non_monolithic_uniforms,
                                     nir_metadata_control_flow, &nr);
}

static bool
lower_dynamic_stride(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_vs_prolog_key *key = data;

   if (intr->intrinsic != nir_intrinsic_load_vbo_stride_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   unsigned nr = DIV_ROUND_UP(BITSET_LAST_BIT(key->component_mask), 4);
   nir_def *root =
      nir_load_preamble(b, 1, 64, .base = AGX_ABI_VUNI_COUNT_VK(nr));
   nir_def *offset_B = nir_iadd_imm(b, nir_imul_imm(b, intr->src[0].ssa, 4),
                                    key->dynamic_strides_B);

   nir_def_replace(&intr->def,
                   nir_load_global_constant(
                      b, 1, 32, nir_iadd(b, root, nir_u2u64(b, offset_B))));
   return true;
}

static bool
lower_adjacency(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_vs_prolog_key *key = data;
   b->cursor = nir_before_instr(&intr->instr);

   if (intr->intrinsic != nir_intrinsic_load_vertex_id)
      return false;

   nir_def *id = nir_load_vertex_id(b);

   if (key->adjacency == MESA_PRIM_LINES_ADJACENCY) {
      id = poly_map_to_line_adj(b, id);
   } else if (key->adjacency == MESA_PRIM_TRIANGLE_STRIP_ADJACENCY) {
      id = poly_map_to_tri_strip_adj(b, id,
                                     nir_imm_bool(b, key->flatshade_first));
   } else if (key->adjacency == MESA_PRIM_LINE_STRIP_ADJACENCY) {
      id = poly_map_to_line_strip_adj(b, id);
   } else if (key->adjacency == MESA_PRIM_TRIANGLES_ADJACENCY) {
      /* Sequence (0, 2, 4), (6, 8, 10), ... */
      id = nir_imul_imm(b, id, 2);
   } else {
      UNREACHABLE("unknown");
   }

   id = poly_nir_load_vertex_id(b, id);

   nir_def_replace(&intr->def, id);
   return true;
}

/*
 * An index size read at run time puts control flow around the vertex ID's
 * load, and exports belong in the last block. Their sources, written at the top
 * level, dominate it.
 */
static void
sink_exports(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_block *last = nir_impl_last_block(impl);

   nir_foreach_block(block, impl) {
      if (block == last)
         continue;

      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_export_agx) {

            nir_instr_move(nir_after_block(last), instr);
         }
      }
   }

   nir_progress(true, impl, nir_metadata_control_flow);
}

void
agx_nir_vs_prolog(nir_builder *b, const void *key_)
{
   const struct agx_vs_prolog_key *key = key_;
   b->shader->info.stage = MESA_SHADER_VERTEX;
   b->shader->info.name = "VS prolog";

   /* First, construct a passthrough shader reading each attribute and exporting
    * the value. We also need to export vertex/instance ID in their usual regs.
    */
   if (!key->static_vi) {
      unsigned i = 0;
      nir_def *vec = NULL;
      unsigned vec_idx = ~0;
      BITSET_FOREACH_SET(i, key->component_mask, AGX_MAX_ATTRIBS * 4) {
         unsigned a = i / 4;
         unsigned c = i % 4;

         if (vec_idx != a) {
            vec = nir_load_input(b, 4, 32, nir_imm_int(b, 0), .base = a);
            vec_idx = a;
         }

         nir_export_agx(b, nir_channel(b, vec, c),
                        .base = AGX_ABI_VIN_ATTRIB(i));
      }
   }

   if (!key->hw) {
      nir_export_agx(b, nir_channel(b, nir_load_global_invocation_id(b, 32), 0),
                     .base = AGX_ABI_VIN_VERTEX_ID_ZERO_BASE);
   }

   nir_export_agx(b, nir_load_vertex_id(b), .base = AGX_ABI_VIN_VERTEX_ID);
   nir_export_agx(b, nir_load_instance_id(b), .base = AGX_ABI_VIN_INSTANCE_ID);

   /* Now lower the resulting program using the key */
   if (!key->static_vi) {
      agx_nir_lower_vbo(b->shader, key->attribs, key->robustness,
                        key->dynamic_strides_B != 0);

      if (key->dynamic_strides_B) {
         nir_shader_intrinsics_pass(b->shader, lower_dynamic_stride,
                                    nir_metadata_control_flow, (void *)key);
      }

      /* Clean up redundant vertex ID loads */
      if (!key->hw || key->adjacency) {
         NIR_PASS(_, b->shader, nir_opt_cse);
         NIR_PASS(_, b->shader, nir_opt_dce);
      }
   }

   if (!key->hw) {
      b->cursor = nir_before_impl(nir_shader_get_entrypoint(b->shader));
      poly_nir_lower_sw_vs(b->shader);
   } else if (key->adjacency) {
      nir_shader_intrinsics_pass(b->shader, lower_adjacency,
                                 nir_metadata_control_flow, (void *)key);
   }
   if (key->sw_index_size_B == AGX_SW_INDEX_SIZE_DYNAMIC) {
      poly_nir_lower_sysvals(b->shader);
      sink_exports(b->shader);
   } else {
      nir_inline_sysval(b->shader, nir_intrinsic_load_index_size_poly,
                        key->sw_index_size_B);
   }

   /* Finally, lower uniforms according to our ABI */
   unsigned nr = DIV_ROUND_UP(BITSET_LAST_BIT(key->component_mask), 4);
   agx_nir_lower_non_monolithic_uniforms(b->shader, nr);
   b->shader->info.io_lowered = true;
}

static bool
gather_inputs(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   unsigned idx = nir_src_as_uint(intr->src[0]) + nir_intrinsic_base(intr);
   unsigned comp = nir_intrinsic_component(intr);

   assert(intr->def.bit_size == 32 && "todo: push conversions up?");
   unsigned base = 4 * idx + comp;

   b->cursor = nir_before_instr(&intr->instr);
   BITSET_WORD *comps_read = data;
   nir_component_mask_t mask = nir_def_components_read(&intr->def);

   u_foreach_bit(c, mask) {
      BITSET_SET(comps_read, base + c);
   }

   return false;
}

bool
agx_nir_gather_vs_inputs(nir_shader *s, BITSET_WORD *attrib_components_read)
{
   return nir_shader_intrinsics_pass(
      s, gather_inputs, nir_metadata_control_flow, attrib_components_read);
}

static bool
lower_input_to_prolog(nir_builder *b, nir_intrinsic_instr *intr, void *_data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   unsigned idx = nir_src_as_uint(intr->src[0]) + nir_intrinsic_base(intr);
   unsigned comp = nir_intrinsic_component(intr);

   assert(intr->def.bit_size == 32 && "todo: push conversions up?");
   unsigned base = 4 * idx + comp;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *val =
      nir_load_exported_agx(b, intr->def.num_components, intr->def.bit_size,
                            .base = AGX_ABI_VIN_ATTRIB(base));

   nir_def_replace(&intr->def, val);
   return true;
}

bool
agx_nir_lower_vs_input_to_prolog(nir_shader *s)
{
   return nir_shader_intrinsics_pass(s, lower_input_to_prolog,
                                     nir_metadata_control_flow, NULL);
}

static bool
lower_active_samples_to_register(nir_builder *b, nir_intrinsic_instr *intr,
                                 void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_active_samples_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *id =
      nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FIN_SAMPLE_MASK);

   nir_def_replace(&intr->def, id);
   return true;
}

static bool
lower_tests_zs_intr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   bool *value = data;
   if (intr->intrinsic != nir_intrinsic_load_shader_part_tests_zs_agx)
      return false;

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def_rewrite_uses(&intr->def, nir_imm_intN_t(b, *value ? 0xFF : 0, 16));
   return true;
}

static bool
lower_tests_zs(nir_shader *s, bool value)
{
   if (!s->info.fs.uses_discard)
      return false;

   return nir_shader_intrinsics_pass(s, lower_tests_zs_intr,
                                     nir_metadata_control_flow, &value);
}

static inline bool
blend_uses_2src(struct agx_blend_rt_key rt)
{
   assert(rt.advanced_blend == false);
   const struct agx_blend_standard blend = agx_unpack_blend_standard(rt.mode);

   enum pipe_blendfactor factors[] = {
      blend.rgb_src_factor,
      blend.rgb_dst_factor,
      blend.alpha_src_factor,
      blend.alpha_dst_factor,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(factors); ++i) {
      switch (factors[i]) {
      case PIPE_BLENDFACTOR_SRC1_COLOR:
      case PIPE_BLENDFACTOR_SRC1_ALPHA:
      case PIPE_BLENDFACTOR_INV_SRC1_COLOR:
      case PIPE_BLENDFACTOR_INV_SRC1_ALPHA:
         return true;
      default:
         break;
      }
   }

   return false;
}

/* One load of an exported channel, shared by every location that reads it */
static nir_def *
load_channel(nir_builder *b, const struct agx_fs_epilog_key *key,
             unsigned ch, nir_def **channels)
{
   if (!channels[ch]) {
      unsigned loc = ch / 4;
      unsigned size = (key->link.size_32 & BITFIELD_BIT(loc)) ? 32 : 16;
      unsigned src = key->link.colour_src[ch];

      if (src == AGX_COLOUR_SRC_ABI) {
         channels[ch] = nir_load_exported_agx(
            b, 1, size,
            .base = AGX_ABI_FOUT_COLOUR(loc) + (ch % 4) * (size / 16));
      } else if (src & AGX_COLOUR_SRC_UNIFORM) {
         channels[ch] = nir_load_preamble(b, 1, size,
                                          .base = src & AGX_COLOUR_SRC_INDEX);
      } else {
         channels[ch] = nir_load_exported_agx(
            b, 1, size, .base = src & AGX_COLOUR_SRC_INDEX);
      }
   }

   return channels[ch];
}

static void
copy_colour(nir_builder *b, const struct agx_fs_epilog_key *key,
            unsigned out_rt, unsigned in_loc, bool dual_src,
            nir_def **channels)
{
   unsigned size = (key->link.size_32 & BITFIELD_BIT(in_loc)) ? 32 : 16;

   /* The channels at their export registers in one load, as before; the
    * others each where the main shader left it */
   nir_def *value =
      nir_load_exported_agx(b, 4, size, .base = AGX_ABI_FOUT_COLOUR(in_loc));

   for (unsigned c = 0; c < 4; ++c) {
      if (key->link.colour_src[4 * in_loc + c] != AGX_COLOUR_SRC_ABI) {
         value = nir_vector_insert_imm(
            b, value, load_channel(b, key, 4 * in_loc + c, channels), c);
      }
   }

   u_foreach_bit(c, (key->link.zero_channels >> (4 * in_loc)) & 0xf) {
      value = nir_vector_insert_imm(b, value, nir_imm_zero(b, 1, size), c);
   }

   /* An aliased channel reads its source's load, and a source the load every
    * alias of it reads, so that the blends' clamp, products and 1 - alpha of
    * one value are computed once.
    */
   uint32_t sources = 0;
   u_foreach_bit(ch, key->link.alias_channels) {
      sources |= BITFIELD_BIT(key->link.alias_of[ch]);
   }

   for (unsigned c = 0; c < 4; ++c) {
      unsigned ch = 4 * in_loc + c;

      if (key->link.alias_channels & BITFIELD_BIT(ch)) {
         value = nir_vector_insert_imm(
            b, value, load_channel(b, key, key->link.alias_of[ch], channels),
            c);
      } else if (sources & BITFIELD_BIT(ch)) {
         value = nir_vector_insert_imm(b, value,
                                       load_channel(b, key, ch, channels), c);
      }
   }

   if (key->link.loc0_w_1 && in_loc == 0) {
      value =
         nir_vector_insert_imm(b, value, nir_imm_floatN_t(b, 1.0, size), 3);
   }

   nir_store_output(b, value, nir_imm_int(b, 0),
                    .io_semantics.location = FRAG_RESULT_DATA0 + out_rt,
                    .io_semantics.dual_source_blend_index = dual_src,
                    .src_type = nir_type_float | size);
}

static bool
is_fmul_uses(nir_src *src, bool once)
{
   nir_alu_instr *alu = nir_def_as_alu_or_null(src->ssa);
   return alu && alu->op == nir_op_fmul &&
          list_is_singular(&alu->def.uses) == once;
}

/*
 * The fma fusion takes the first product of a sum that only sums use: for a
 * blend's s * a + d * (1 - a) that is s * a, multiplied again in every channel
 * where one value (an aliased channel, the shared alpha's a * a) is shared.
 * Put the destination's product first, used once, so the fma takes it and the
 * shared product is computed once.
 */
static bool
shared_product_last(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if (alu->op != nir_op_fadd || alu->def.num_components != 1 ||
       !is_fmul_uses(&alu->src[0].src, false) ||
       !is_fmul_uses(&alu->src[1].src, true))
      return false;

   nir_def *s0 = alu->src[0].src.ssa, *s1 = alu->src[1].src.ssa;
   uint8_t sw0 = alu->src[0].swizzle[0], sw1 = alu->src[1].swizzle[0];

   nir_src_rewrite(&alu->src[0].src, s1);
   nir_src_rewrite(&alu->src[1].src, s0);
   alu->src[0].swizzle[0] = sw1;
   alu->src[1].swizzle[0] = sw0;
   return true;
}

void
agx_nir_fs_epilog(nir_builder *b, const void *key_)
{
   const struct agx_fs_epilog_key *key = key_;
   b->shader->info.stage = MESA_SHADER_FRAGMENT;
   b->shader->info.name = "FS epilog";

   /* First, construct a passthrough shader reading each colour and outputting
    * the value.
    */
   nir_def *channels[32] = {NULL};

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->remap); ++rt) {
      int location = key->remap[rt];

      /* Negative remaps indicate the attachment isn't written. */
      if (location >= 0 && key->link.loc_written & BITFIELD_BIT(location)) {
         copy_colour(b, key, rt, location, false, channels);

         /* If this render target uses dual source blending, also copy the dual
          * source colour. While the copy_colour above is needed even for
          * missing attachments to handle alpha-to-coverage, this copy is only
          * for blending so should be suppressed for missing attachments to keep
          * the assert from blowing up on OpenGL.
          */
         if (!key->blend.rt[rt].advanced_blend &&
             blend_uses_2src(key->blend.rt[rt]) &&
             key->rt_formats[rt] != PIPE_FORMAT_NONE) {

            assert(location == 0);
            copy_colour(b, key, rt, 1, true, channels);
         }
      }
   }

   /* Grab registers early, this has to happen in the first block. */
   nir_def *sample_id = NULL, *write_samples = NULL;
   if (key->link.sample_shading) {
      sample_id =
         nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FOUT_SAMPLE_MASK);
   }

   if (key->link.sample_mask_after_force_early) {
      write_samples =
         nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FOUT_WRITE_SAMPLES);
   }

   /* Now lower the resulting program using the key */
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(key->rt_formats, ARRAY_SIZE(key->rt_formats),
                                  key->nr_samples, true, key->g15);

   if (key->force_small_tile)
      tib.tile_size = 16 * 16;

   bool force_translucent = false;
   nir_lower_blend_options opts = {
      .scalar_blend_const = true,
      .logicop_enable = key->blend.logicop_enable,
      .logicop_func = key->blend.logicop_func,
   };

   static_assert(ARRAY_SIZE(opts.rt) == 8, "max RTs out of sync");

   for (unsigned i = 0; i < 8; ++i) {
      if (key->blend.rt[i].advanced_blend) {
         const struct agx_blend_advanced blend =
            agx_unpack_blend_advanced(key->blend.rt[i].mode);

         opts.rt[i] = (nir_lower_blend_rt){
            .format = key->rt_formats[i],
            .advanced_blend = true,
            .colormask = key->blend.rt[i].colormask,
            .blend_mode = blend.op,
            .src_premultiplied = blend.src_premultiplied,
            .dst_premultiplied = blend.dst_premultiplied,
            .overlap = blend.overlap,
         };
      } else {
         const struct agx_blend_standard blend =
            agx_unpack_blend_standard(key->blend.rt[i].mode);

         opts.rt[i] = (nir_lower_blend_rt){
            .format = key->rt_formats[i],

            .rgb.src_factor = blend.rgb_src_factor,
            .rgb.dst_factor = blend.rgb_dst_factor,
            .rgb.func = blend.rgb_func,

            .alpha.src_factor = blend.alpha_src_factor,
            .alpha.dst_factor = blend.alpha_dst_factor,
            .alpha.func = blend.alpha_func,

            .colormask = key->blend.rt[i].colormask,
         };
      }
   }

   /* It's more efficient to use masked stores (with
    * agx_nir_lower_tilebuffer) than to emulate colour masking with
    * nir_lower_blend.
    */
   uint8_t colormasks[8] = {0};

   for (unsigned i = 0; i < 8; ++i) {
      if (key->rt_formats[i] == PIPE_FORMAT_NONE)
         continue;

      if (agx_tilebuffer_supports_mask(&tib, i)) {
         colormasks[i] = key->blend.rt[i].colormask;
         opts.rt[i].colormask = (uint8_t)BITFIELD_MASK(4);
      } else {
         colormasks[i] = (uint8_t)BITFIELD_MASK(4);
      }

      /* If not all bound RTs are fully written to, we need to force
       * translucent pass type. agx_nir_lower_tilebuffer will take
       * care of this for its own colormasks input.
       */
      unsigned comps = util_format_get_nr_components(key->rt_formats[i]);
      if ((opts.rt[i].colormask & BITFIELD_MASK(comps)) !=
          BITFIELD_MASK(comps)) {
         force_translucent = true;
      }
   }

   /* Alpha-to-coverage must be lowered before alpha-to-one */
   if (key->blend.alpha_to_coverage && key->nr_samples == 1)
      NIR_PASS(_, b->shader, nir_lower_alpha_to_coverage_single_sample, NULL);
   else if (key->blend.alpha_to_coverage)
      NIR_PASS(_, b->shader, nir_lower_alpha_to_coverage, false,
               NULL);

   /* Depth/stencil writes must be deferred until after all discards,
    * particularly alpha-to-coverage.
    */
   if (key->link.write_z || key->link.write_s) {
      nir_store_zs_agx(
         b, nir_imm_intN_t(b, 0xFF, 16),
         nir_load_exported_agx(b, 1, 32, .base = AGX_ABI_FOUT_Z),
         nir_load_exported_agx(b, 1, 16, .base = AGX_ABI_FOUT_S),
         .base = (key->link.write_z ? 1 : 0) | (key->link.write_s ? 2 : 0));

      if (key->link.write_z)
         b->shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DEPTH);

      if (key->link.write_s)
         b->shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_STENCIL);
   }

   /* Alpha-to-one must be lowered before blending */
   if (key->blend.alpha_to_one)
      NIR_PASS(_, b->shader, nir_lower_alpha_to_one);

   NIR_PASS(_, b->shader, nir_lower_blend, &opts);

   /* Merge the blends' clamps and products of one value (an aliased channel,
    * the shared alpha's a * a) before the optimizer fuses each into its own
    * ffma: a product used more than once stays, and the fma takes the
    * destination's, d * (1 - a) + (s * a) with s * a computed once.
    * AGX_MESA_DEBUG=oldepilog leaves it to the optimizer, as before.
    */
   if (!(agx_get_compiler_debug() & AGX_DBG_OLDEPILOG)) {
      NIR_PASS(_, b->shader, nir_lower_alu_to_scalar, NULL, NULL);
      NIR_PASS(_, b->shader, nir_opt_copy_prop);
      NIR_PASS(_, b->shader, nir_opt_cse);
      nir_shader_alu_pass(b->shader, shared_product_last, nir_metadata_all,
                          NULL);
   }

   unsigned rt_spill = key->link.rt_spill_base;
   NIR_PASS(_, b->shader, agx_nir_lower_tilebuffer, &tib, colormasks, &rt_spill,
            write_samples, &force_translucent);
   NIR_PASS(_, b->shader, agx_nir_lower_texture_gen, key->g15);
   NIR_PASS(_, b->shader, agx_nir_lower_multisampled_image_store, key->g15);

   /* If the API shader runs once per sample, then the epilog runs once per
    * sample as well, so we need to lower our code to run for a single sample.
    *
    * If the API shader runs once per pixel, then the epilog runs once per
    * pixel. So we run through the monolithic MSAA lowering, which wraps the
    * epilog in the sample loop if needed. This localizes sample shading
    * to the epilog, when sample shading is not used but blending is.
    */
   if (key->link.sample_shading) {
      /* Lower the resulting discards. Done in agx_nir_lower_monolithic_msaa for
       * the pixel shaded path. Must be done before agx_nir_lower_to_per_sample
       * to avoid duplicating tests.
       */
      if (key->blend.alpha_to_coverage) {
         NIR_PASS(_, b->shader, agx_nir_lower_sample_mask);
      }

      NIR_PASS(_, b->shader, agx_nir_lower_to_per_sample);
      NIR_PASS(_, b->shader, agx_nir_lower_fs_active_samples_to_register);

      /* Ensure the sample ID is preserved in register. We do this late since it
       * has to go in the last block, and the above passes might add control
       * flow when lowering.
       */
      b->cursor = nir_after_impl(b->impl);
      nir_export_agx(b, sample_id, .base = AGX_ABI_FIN_SAMPLE_MASK);
   } else {
      NIR_PASS(_, b->shader, agx_nir_lower_monolithic_msaa, key->nr_samples);
   }

   /* Finally, lower uniforms according to our ABI */
   bool g15 = key->g15;
   nir_shader_intrinsics_pass(b->shader, lower_non_monolithic_uniforms,
                              nir_metadata_control_flow, &g15);

   /* There is no shader part after the epilog, so we're always responsible for
    * running our own tests, unless the fragment shader forced early tests.
    */
   NIR_PASS(_, b->shader, lower_tests_zs, !key->link.already_ran_zs);

   b->shader->info.io_lowered = true;
   b->shader->info.fs.uses_fbfetch_output |= force_translucent;
   b->shader->info.fs.uses_sample_shading = key->link.sample_shading;
}

/*
 * G15: the tile buffer words (32-bit, of one sample) that a fragment program
 * linked with this epilog overwrites in full, as Apple's occlusion query 2
 * word gives them (agxsGetShaderPassInfo): bit w for word w, the top word's
 * bit repeated up to bit 16. Under an opaque draw the hardware drops earlier
 * work only for the words it marks, the background's load among them; 0 marks
 * none. A word is written when each render target byte in it belongs to a
 * target the epilog stores in full: its location written, every component in
 * the colour mask, blending off, not spilled. A program that reads the tile
 * buffer, discards, writes memory or runs a logic op or alpha-to-coverage
 * marks none. At 2 and 4 samples the hardware hides the background only in
 * the samples the draw covers, and the background program stores only the
 * samples its coverage names (agx_nir_lower_tile_raw), as Metal's, which
 * marks at any sample count.
 */
uint32_t
agx_fs_epilog_tib_written(const struct agx_fs_epilog_key *key,
                          const struct agx_linked_shader *linked)
{
   if (linked->no_op || linked->writes_memory || key->blend.logicop_enable ||
       key->blend.alpha_to_coverage)
      return 0;

   agx_unpack(NULL, &linked->fragment_control, FRAGMENT_CONTROL, fc);
   if (fc.pass_type != AGX_PASS_TYPE_OPAQUE)
      return 0;

   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(key->rt_formats, ARRAY_SIZE(key->rt_formats),
                                  key->nr_samples, true, key->g15);
   unsigned words = DIV_ROUND_UP(tib.sample_size_B, 4);
   if (words > 17)
      return 0;

   const unsigned replace = agx_pack_blend_standard(
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
      PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
   uint64_t held = 0, written = 0;

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->rt_formats); ++rt) {
      enum pipe_format format = key->rt_formats[rt];
      if (format == PIPE_FORMAT_NONE || tib.spilled[rt])
         continue;

      unsigned offset = agx_tilebuffer_offset_B(&tib, rt);
      unsigned size =
         util_format_get_blocksize(agx_tilebuffer_physical_format(&tib, rt));
      uint64_t bytes = BITFIELD64_RANGE(offset, size);
      held |= bytes;

      struct agx_blend_rt_key blend = key->blend.rt[rt];
      unsigned need = BITFIELD_MASK(util_format_get_nr_components(format));
      int loc = key->remap[rt];

      if (loc >= 0 && (key->link.loc_written & BITFIELD_BIT(loc)) &&
          !blend.advanced_blend && blend.mode == replace &&
          (blend.colormask & need) == need)
         written |= bytes;
   }

   /* No target held: nothing to drop */
   if (!held)
      return 0;

   uint32_t out = 0;
   for (unsigned w = 0; w < words; ++w) {
      uint64_t word = BITFIELD64_RANGE(4 * w, 4) & held;
      if ((written & word) == word)
         out |= BITFIELD_BIT(w);
   }

   if (out & BITFIELD_BIT(words - 1))
      out |= BITFIELD_MASK(17) & ~BITFIELD_MASK(words);

   return out;
}

struct lower_epilog_ctx {
   struct agx_fs_epilog_link_info *info;
   nir_variable *masked_samples;
};

static bool
lower_output_to_epilog(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower_epilog_ctx *ctx = data;
   struct agx_fs_epilog_link_info *info = ctx->info;

   if (intr->intrinsic == nir_intrinsic_store_zs_agx) {
      assert(nir_src_as_uint(intr->src[0]) == 0xff && "msaa not yet lowered");
      b->cursor = nir_instr_remove(&intr->instr);

      unsigned base = nir_intrinsic_base(intr);
      info->write_z = !!(base & 1);
      info->write_s = !!(base & 2);

      if (info->write_z)
         nir_export_agx(b, intr->src[1].ssa, .base = AGX_ABI_FOUT_Z);

      if (info->write_s)
         nir_export_agx(b, intr->src[2].ssa, .base = AGX_ABI_FOUT_S);

      return true;
   }

   if (intr->intrinsic == nir_intrinsic_demote_samples &&
       b->shader->info.fs.early_fragment_tests) {

      if (!ctx->masked_samples) {
         b->cursor = nir_before_impl(nir_shader_get_entrypoint(b->shader));

         ctx->masked_samples =
            nir_local_variable_create(b->impl, glsl_uint16_t_type(), NULL);

         nir_store_var(b, ctx->masked_samples, nir_imm_intN_t(b, 0xFF, 16),
                       nir_component_mask(1));
      }

      b->cursor = nir_before_instr(&intr->instr);

      nir_def *mask = nir_load_var(b, ctx->masked_samples);
      nir_def *mask_2 =
         nir_ixor(b, intr->src[0].ssa, nir_imm_intN_t(b, 0xff, 16));

      mask = nir_iand(b, mask, mask_2);
      nir_store_var(b, ctx->masked_samples, mask, nir_component_mask(1));

      nir_instr_remove(&intr->instr);
      return true;
   }

   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);

   /* Fix up gl_FragColor */
   if (sem.location == FRAG_RESULT_COLOR) {
      sem.location = FRAG_RESULT_DATA0;
      info->broadcast_rt0 = true;
   }

   /* We don't use the epilog for sample mask writes */
   if (sem.location < FRAG_RESULT_DATA0)
      return false;

   /* Determine the ABI location. Dual source blending aliases a second
    * render target, so get that out of the way now.
    */
   unsigned loc = sem.location - FRAG_RESULT_DATA0;
   loc += nir_src_as_uint(intr->src[1]);

   if (sem.dual_source_blend_index) {
      assert(loc == 0);
      loc = 1;
   }

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def *vec = intr->src[0].ssa;

   info->loc_written |= BITFIELD_BIT(loc);

   if (vec->bit_size == 32)
      info->size_32 |= BITFIELD_BIT(loc);
   else
      assert(vec->bit_size == 16);

   uint32_t one_f = (vec->bit_size == 32 ? fui(1.0) : _mesa_float_to_half(1.0));
   unsigned comp = nir_intrinsic_component(intr);

   u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
      nir_scalar s = nir_scalar_resolved(vec, c);
      if (loc == 0 && c == 3 && nir_scalar_is_const(s) &&
          nir_scalar_as_uint(s) == one_f) {

         info->loc0_w_1 = true;
      } else if ((info->zero_channels | info->alias_channels) &
                 BITFIELD_BIT(4 * loc + comp + c)) {
         /* The epilog knows this channel's value */
      } else {
         unsigned stride = vec->bit_size / 16;

         nir_export_agx(b, nir_channel(b, vec, c),
                        .base = AGX_ABI_FOUT_COLOUR(loc) + (comp + c) * stride);
      }
   }

   return true;
}

static bool
same_scalar(nir_scalar a, nir_scalar b)
{
   if (a.def->bit_size != b.def->bit_size)
      return false;

   if (nir_scalar_is_const(a) && nir_scalar_is_const(b))
      return nir_scalar_as_uint(a) == nir_scalar_as_uint(b);

   return nir_scalar_equal(a, b);
}

/*
 * What the epilog can know of the colours from the main shader alone: channels
 * written as zero, and channels holding the value of an earlier one (a
 * G-buffer's shared alpha, a grey in three channels, a uniform in two
 * targets). Only outputs written once, in the last block, count: their value
 * is the value at the end of the shader whatever path was taken.
 */
static void
gather_known_outputs(nir_shader *s, struct agx_fs_epilog_link_info *info)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_block *last = nir_impl_last_block(impl);
   uint32_t written = 0, bad = 0, zero = 0;
   nir_scalar value[32];

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;

         nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
         if (sem.location == FRAG_RESULT_COLOR || sem.dual_source_blend_index)
            return;

         if (sem.location < FRAG_RESULT_DATA0)
            continue;

         unsigned loc = sem.location - FRAG_RESULT_DATA0;
         loc += nir_src_as_uint(intr->src[1]);
         unsigned comp = nir_intrinsic_component(intr);

         u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
            unsigned ch = 4 * loc + comp + c;
            uint32_t bit = BITFIELD_BIT(ch);
            nir_scalar v = nir_scalar_resolved(intr->src[0].ssa, c);

            if ((written & bit) || block != last)
               bad |= bit;

            written |= bit;
            value[ch] = v;

            if (nir_scalar_is_const(v) && nir_scalar_as_uint(v) == 0)
               zero |= bit;
         }
      }
   }

   info->zero_channels = zero & ~bad;

   /* Location 0's constant alpha may not be exported (loc0_w_1) */
   uint32_t known = written & ~bad & ~info->zero_channels;
   if ((known & BITFIELD_BIT(3)) && nir_scalar_is_const(value[3]))
      known &= ~BITFIELD_BIT(3);

   /* AGX_MESA_DEBUG=oldepilog: alphas alone, as before (r309) */
   if (agx_get_compiler_debug() & AGX_DBG_OLDEPILOG)
      known &= 0x88888888;

   uint32_t sources = 0;
   u_foreach_bit(ch, known) {
      u_foreach_bit(src, sources) {
         if (same_scalar(value[src], value[ch])) {
            info->alias_channels |= BITFIELD_BIT(ch);
            info->alias_of[ch] = src;
            break;
         }
      }

      if (!(info->alias_channels & BITFIELD_BIT(ch)))
         sources |= BITFIELD_BIT(ch);
   }
}

bool
agx_nir_lower_fs_output_to_epilog(nir_shader *s,
                                  struct agx_fs_epilog_link_info *out)
{
   struct lower_epilog_ctx ctx = {.info = out};

   gather_known_outputs(s, out);

   nir_shader_intrinsics_pass(s, lower_output_to_epilog,
                              nir_metadata_control_flow, &ctx);

   if (ctx.masked_samples) {
      nir_builder b =
         nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(s)));

      nir_export_agx(&b, nir_load_var(&b, ctx.masked_samples),
                     .base = AGX_ABI_FOUT_WRITE_SAMPLES);
      out->sample_mask_after_force_early = true;

      bool progress;
      do {
         progress = false;
         NIR_PASS(progress, s, nir_lower_vars_to_ssa);
         NIR_PASS(progress, s, nir_opt_dce);
      } while (progress);
   }

   out->sample_shading = s->info.fs.uses_sample_shading;
   return true;
}

bool
agx_nir_lower_fs_active_samples_to_register(nir_shader *s)
{
   return nir_shader_intrinsics_pass(s, lower_active_samples_to_register,
                                     nir_metadata_control_flow, NULL);
}

static bool
agx_nir_lower_stats_fs(nir_shader *s)
{
   assert(s->info.stage == MESA_SHADER_FRAGMENT);
   nir_builder b_ =
      nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(s)));
   nir_builder *b = &b_;

   nir_push_if(b, nir_inot(b, nir_load_helper_invocation(b, 1)));
   nir_def *samples = nir_bit_count(b, nir_load_sample_mask_in(b));
   unsigned query = PIPE_STAT_QUERY_PS_INVOCATIONS;

   nir_def *addr = nir_load_stat_query_address_poly(b, .base = query);
   nir_global_atomic(b, 32, addr, samples, .atomic_op = nir_atomic_op_iadd);

   nir_pop_if(b, NULL);
   return nir_progress(true, b->impl, nir_metadata_control_flow);
}

/*
 * Conservative rasterization: the triangle was widened, so keep only the
 * pixels whose square touches it. The vertex shader wrote its bounding box and
 * its edge functions (>= -0.5 at the centre iff the square touches the edge's
 * half-plane) as linear varyings, which are tested at the pixel centre.
 */
static bool
agx_nir_lower_conservative_fs(nir_shader *s, enum agx_conservative_test test)
{
   assert(s->info.stage == MESA_SHADER_FRAGMENT);
   nir_builder b_ =
      nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(s)));
   nir_builder *b = &b_;

   nir_def *bary = nir_load_barycentric_pixel(
      b, 32, .interp_mode = INTERP_MODE_NOPERSPECTIVE);
   nir_def *centre =
      nir_fadd_imm(b, nir_u2f32(b, nir_load_pixel_coord(b)), 0.5);

   nir_def *box = nir_load_interpolated_input(
      b, 4, 32, bary, nir_imm_int(b, 0), .dest_type = nir_type_float32,
      .io_semantics.location = POLY_VARYING_SLOT_CONSERVATIVE_BOX,
      .io_semantics.num_slots = 1);

   nir_def *touches =
      nir_iand(b,
               nir_ball(b, nir_fge(b, nir_fadd_imm(b, centre, 0.5),
                                   nir_trim_vector(b, box, 2))),
               nir_ball(b, nir_fge(b, nir_channels(b, box, 0xc),
                                   nir_fadd_imm(b, centre, -0.5))));

   if (test == AGX_CONSERVATIVE_BOX_EDGES) {
      nir_def *edges = nir_load_interpolated_input(
         b, 3, 32, bary, nir_imm_int(b, 0), .dest_type = nir_type_float32,
         .io_semantics.location = POLY_VARYING_SLOT_CONSERVATIVE_EDGES,
         .io_semantics.num_slots = 1);

      touches = nir_iand(b, touches, nir_ball(b, nir_fge_imm(b, edges, -0.5)));
      s->info.inputs_read |=
         BITFIELD64_BIT(POLY_VARYING_SLOT_CONSERVATIVE_EDGES);
   }

   nir_demote_if(b, nir_inot(b, touches));

   s->info.inputs_read |= BITFIELD64_BIT(POLY_VARYING_SLOT_CONSERVATIVE_BOX);
   s->info.fs.uses_discard = true;
   return nir_progress(true, b->impl, nir_metadata_control_flow);
}

void
agx_nir_fs_prolog(nir_builder *b, const void *key_)
{
   const struct agx_fs_prolog_key *key = key_;
   b->shader->info.stage = MESA_SHADER_FRAGMENT;
   b->shader->info.name = "FS prolog";

   /* First, insert code for any emulated features */
   if (key->api_sample_mask != 0xff) {
      /* Kill samples that are NOT covered by the mask */
      nir_demote_samples(b, nir_imm_intN_t(b, key->api_sample_mask ^ 0xff, 16));
      b->shader->info.fs.uses_discard = true;
   }

   if (key->statistics) {
      NIR_PASS(_, b->shader, agx_nir_lower_stats_fs);
   }

   if (key->cull_distance_size) {
      NIR_PASS(_, b->shader, agx_nir_lower_cull_distance_fs,
               key->cull_distance_size);
   }

   if (key->polygon_stipple) {
      NIR_PASS(_, b->shader, agx_nir_lower_poly_stipple);
   }

   /* Last, so it runs first: it stands in for rasterization */
   if (key->conservative) {
      NIR_PASS(_, b->shader, agx_nir_lower_conservative_fs, key->conservative);
   }

   /* Then, lower the prolog */
   NIR_PASS(_, b->shader, agx_nir_lower_discard_zs_emit);
   NIR_PASS(_, b->shader, agx_nir_lower_sample_mask);
   NIR_PASS(_, b->shader, nir_shader_intrinsics_pass,
            lower_non_monolithic_uniforms, nir_metadata_control_flow, NULL);
   NIR_PASS(_, b->shader, lower_tests_zs, key->run_zs_tests);

   b->shader->info.io_lowered = true;
}
