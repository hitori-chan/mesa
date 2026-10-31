/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/shader_enums.h"
#include "agx_compile.h"
#include "agx_debug.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"

/*
 * In AGX, the values of fragment shader inputs are represented as coefficient
 * vectors <A, B, C>, which are dotted with <x, y, 1> to perform interpolation.
 * x and y are relative to the tile. In other words, A and B are the
 * screen-space partial derivatives of the input, and C is the value at the
 * corner of the tile.
 *
 * For some interpolation modes, the dot product happens in the iterator
 * hardware. Other modes are implemented in this file, by lowering to math on
 * the coefficient vectors.
 */

/* XXX: It's not clear what this is for, but seems necessary */
static nir_def *
cf_valid(nir_builder *b, nir_def *cf, bool g15)
{
   nir_def *bit = nir_ieq_imm(b, nir_iand_imm(b, nir_channel(b, cf, 0), 1), 0);

   /* G15 (perspprobe, #338): a constant plane has A's bit 0 set and neither A
    * nor B with an exponent, and iterates to the value itself rather than the
    * value over W.
    */
   if (g15) {
      nir_def *ab = nir_ior(b, nir_channel(b, cf, 0), nir_channel(b, cf, 1));
      return nir_ior(b, bit,
                     nir_ine_imm(b, nir_iand_imm(b, ab, 0x7f800000), 0));
   }

   /* XXX: Apple's compiler actually checks that the significand is nonzero and
    * the exponent is 0 or 1. This is probably a typo -- it doesn't make any
    * logical sense.  Presumably they just meant to check for denorms, so let's
    * do that. Either way the tests pass.
    */
   nir_def *cf01 = nir_trim_vector(b, cf, 2);
   return nir_ior(b, bit, nir_fisnormal(b, cf01));
}

static nir_def *
load_rhw(nir_builder *b, nir_def *bary)
{
   return nir_load_interpolated_input(b, 1, 32, bary, nir_imm_int(b, 0),
                                      .component = 3,
                                      .io_semantics = {
                                         .location = VARYING_SLOT_POS,
                                         .num_slots = 1,
                                      });
}

static nir_def *
interpolate_at_offset(nir_builder *b, nir_def *cf, nir_def *offset,
                      bool perspective, bool g15)
{
   /* Get the coordinate of the pixel within the tile */
   nir_def *pixel_coords = nir_load_pixel_coord(b);
   nir_def *tile_offs = nir_umod_imm(b, pixel_coords, 32);

   /* Convert to float, getting the center of the pixel */
   nir_def *center = nir_fadd_imm(b, nir_u2f32(b, tile_offs), 0.5);

   /* Calculate the location to interpolate. offset is defined relative to the
    * center of the pixel and is a float.
    */
   nir_def *pos = nir_fadd(b, center, nir_f2f32(b, offset));

   /* Interpolate with the given coefficients */
   nir_def *interp = nir_ffma_weak(b, nir_channel(b, pos, 1), nir_channel(b, cf, 1),
                                   nir_channel(b, cf, 2));

   interp = nir_ffma_weak(b, nir_channel(b, pos, 0), nir_channel(b, cf, 0), interp);

   /* Divide by RHW. This load will be lowered recursively. */
   if (perspective) {
      nir_def *bary = nir_load_barycentric_at_offset(
         b, 32, offset, .interp_mode = INTERP_MODE_NOPERSPECTIVE);

      interp = nir_fdiv(b, interp, load_rhw(b, bary));
   }

   /* Replace invalid interpolations with the constant channel */
   return nir_bcsel(b, cf_valid(b, cf, g15), interp, nir_channel(b, cf, 2));
}

static nir_def *
interpolate_flat(nir_builder *b, nir_def *coefficients)
{
   /* Same value anywhere, so just take the constant (affine) component. For
    * triangle fans with the first provoking vertex, the CF layout is slightly
    * different. I am unsure why, but Apple does the same and the bcsel is
    * required for corrctness.
    */
   return nir_bcsel(b, nir_load_is_first_fan_agx(b),
                    nir_channel(b, coefficients, 1),
                    nir_channel(b, coefficients, 2));
}

static enum glsl_interp_mode
interp_mode_for_load(nir_intrinsic_instr *load)
{
   if (load->intrinsic == nir_intrinsic_load_input)
      return INTERP_MODE_FLAT;
   else
      return nir_intrinsic_interp_mode(nir_src_as_intrinsic(load->src[0]));
}

static bool
needs_lower(const nir_instr *instr, const void *data)
{
   const bool *g15 = data;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   const nir_intrinsic_instr *load = nir_instr_as_intrinsic(instr);

   /* at_offset barycentrics need to be lowered, and on G15 every perspective
    * load but those at the pixel centre and a constant offset, which the
    * backend projects (agx_emit_load_vary; all with AGX_MESA_DEBUG=noperspmul)
    */
   if (load->intrinsic == nir_intrinsic_load_interpolated_input) {
      nir_intrinsic_instr *bary = nir_src_as_intrinsic(load->src[0]);

      if (bary->intrinsic == nir_intrinsic_load_barycentric_at_offset)
         return true;

      if (!*g15 || nir_intrinsic_interp_mode(bary) == INTERP_MODE_NOPERSPECTIVE)
         return false;

      return (agx_get_compiler_debug() & AGX_DBG_NOPERSPMUL) ||
             bary->intrinsic != nir_intrinsic_load_barycentric_pixel ||
             !nir_src_is_const(
                *nir_get_io_offset_src((nir_intrinsic_instr *)load));
   }

   /* Flat shading always lowered */
   return (load->intrinsic == nir_intrinsic_load_input);
}

static nir_def *
interpolate_channel(nir_builder *b, nir_intrinsic_instr *load, unsigned channel,
                    bool g15)
{
   nir_def *coefficients = nir_load_coefficients_agx(
      b, nir_get_io_offset_src(load)->ssa,
      .component = nir_intrinsic_component(load) + channel,
      .interp_mode = interp_mode_for_load(load),
      .io_semantics = nir_intrinsic_io_semantics(load));

   if (load->intrinsic == nir_intrinsic_load_input) {
      assert(load->def.bit_size == 32);

      if (nir_intrinsic_io_semantics(load).location == VARYING_SLOT_LAYER)
         return nir_load_layer_id(b);
      else
         return interpolate_flat(b, coefficients);
   } else {
      nir_intrinsic_instr *bary = nir_src_as_intrinsic(load->src[0]);

      nir_def *interp = interpolate_at_offset(
         b, coefficients, bary->src[0].ssa,
         nir_intrinsic_interp_mode(bary) != INTERP_MODE_NOPERSPECTIVE, g15);

      return nir_f2fN(b, interp, load->def.bit_size);
   }
}

/*
 * G15 has no projected iteration: its iteration of a perspective input gives
 * the value over W, which this divides by the iterated 1/W (W's noperspective
 * load at the same location), as interpolate_at_offset does. Apple's compiler
 * sets the perspective flag on W's iteration, but then its reciprocal and
 * multiplies name the coefficient registers (forms the packer does not have);
 * with plain ones the flagged iteration reads 0 (boots #274-#282). The load
 * stays, now at 32 bits; the caller has already moved its uses to the result.
 */
static nir_def *
project_g15(nir_builder *b, nir_intrinsic_instr *load)
{
   nir_intrinsic_instr *bary = nir_src_as_intrinsic(load->src[0]);
   unsigned bit_size = load->def.bit_size;
   load->def.bit_size = 32;

   nir_intrinsic_instr *linear =
      nir_instr_as_intrinsic(nir_instr_clone(b->shader, &bary->instr));
   nir_intrinsic_set_interp_mode(linear, INTERP_MODE_NOPERSPECTIVE);
   nir_builder_instr_insert(b, &linear->instr);
   nir_def *rhw = load_rhw(b, &linear->def);

   nir_def *values[NIR_MAX_VEC_COMPONENTS] = {NULL};
   for (unsigned i = 0; i < load->def.num_components; ++i) {
      nir_def *cf = nir_load_coefficients_agx(
         b, nir_get_io_offset_src(load)->ssa,
         .component = nir_intrinsic_component(load) + i,
         .interp_mode = nir_intrinsic_interp_mode(bary),
         .io_semantics = nir_intrinsic_io_semantics(load));

      nir_def *v = nir_channel(b, &load->def, i);
      v = nir_bcsel(b, cf_valid(b, cf, true), nir_fdiv(b, v, rhw), v);
      values[i] = nir_f2fN(b, v, bit_size);
   }

   return nir_vec(b, values, load->def.num_components);
}

static nir_def *
lower(nir_builder *b, nir_instr *instr, void *data)
{
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   const bool *g15 = data;

   if (*g15 && intr->intrinsic == nir_intrinsic_load_interpolated_input &&
       nir_src_as_intrinsic(intr->src[0])->intrinsic !=
          nir_intrinsic_load_barycentric_at_offset)
      return project_g15(b, intr);

   /* Each component is loaded separated */
   nir_def *values[NIR_MAX_VEC_COMPONENTS] = {NULL};
   for (unsigned i = 0; i < intr->def.num_components; ++i) {
      values[i] = interpolate_channel(b, intr, i, *g15);
   }

   return nir_vec(b, values, intr->def.num_components);
}

bool
agx_nir_lower_interpolation(nir_shader *s, bool g15)
{
   assert(s->info.stage == MESA_SHADER_FRAGMENT);

   return nir_shader_lower_instructions(s, needs_lower, lower, &g15);
}
