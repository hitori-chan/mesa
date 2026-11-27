/*
 * Copyright 2026 hitori1403
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "compiler/nir/nir_builder.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "util/u_math.h"
#include "agx_compile.h"
#include "agx_uvs.h"
#include "hk_private.h"
#include "hk_shader.h"
#include "nir.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "shader_enums.h"

/*
 * G15's mesh stage in hardware, as Apple's compiler builds it
 * (spec/agx-g15c/mesh.md). Both stages are compute-form programs the VDM
 * runs; the outputs go through no memory of the driver's, and the TA takes
 * each mesh threadgroup's vertices once, its primitives indexed.
 *
 * The mesh program keeps its outputs in threadgroup memory, after the
 * shader's own: the u8 indices, then each UVS slot as an array of
 * max_vertices dwords rounded up to 16 B, as each uvst's address is (see
 * pad_threadgroup), the slots in the order of the vertex shader's UVS
 * (agx_nir_lower_uvs: position, the varyings, the point size, the clip
 * distances). Its epilogue, Apple's agc.mesh.write_uvb (AGXCompilerCore_272
 * 0x22928b618), runs on SIMD group 0 after a barrier: it clamps the indices,
 * puts the position through G15's guard band and clamps the point size,
 * allocates the vertex buffer (allocate_uvb) and moves the indices and every
 * slot there (uvst), the group's lanes (32, or the threadgroup's size under
 * 32) at a time.
 *
 * Without a task shader, Metal's passthrough object program of three threads
 * writes the draw's grid, thread t its component t (a uvsw with no slot).
 *
 * Not covered, so drawn as compute: a task shader (its payload needs the
 * kernel's buffer, and a task draw of many groups loses some of them:
 * work/scratch/agents/mesh-hw.md), per-primitive outputs (user ones, the
 * primitive ID, layer, viewport, culling: Apple's go through memory the
 * fragment shader reads, mesh.md), cull distances, outputs other than 32-bit,
 * more than Metal's 256 vertices or 512 primitives, threadgroup memory past
 * 32 KiB.
 */

static nir_def *
load_param(nir_builder *b, unsigned offset, unsigned components,
           unsigned bit_size)
{
   nir_def *p = nir_load_mesh_param_buffer_poly(b);
   return nir_load_global_constant(b, components, bit_size,
                                   nir_iadd_imm(b, p, offset), .align_mul = 4);
}

#define LOAD_PARAM(b, field, components, bit_size)                             \
   load_param(b, offsetof(struct hk_mesh_hw_params, field), components,        \
              bit_size)

/* The draw's grid */
static nir_def *
load_grid(nir_builder *b)
{
   return nir_load_global_constant(b, 3, 32, LOAD_PARAM(b, grid, 1, 64),
                                   .align_mul = 4);
}

struct mesh_hw_state {
   struct hk_mesh_hw_info *info;
   struct agx_unlinked_uvs_layout *uvs;
   uint8_t components[VARYING_SLOT_MAX];
   unsigned vpp;
   unsigned idx_base, vtx_base;

   /* A slot's array in vertices: max_vertices, 16 B aligned */
   unsigned vtx_stride;
   bool points, single_vary, psiz_written, unsupported;
   nir_variable *vertex_count, *primitive_count;
};

/* Gather the per-vertex outputs' components, as agx_nir_lower_uvs, and what
 * the hardware path does not cover */
static bool
gather(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct mesh_hw_state *s = data;
   bool store = false;

   switch (intr->intrinsic) {
   case nir_intrinsic_store_per_primitive_output:
   case nir_intrinsic_load_per_primitive_output:
      if (nir_intrinsic_io_semantics(intr).location !=
          VARYING_SLOT_PRIMITIVE_INDICES)
         s->unsupported = true;
      return false;

   /* A payload without a task shader */
   case nir_intrinsic_load_task_payload:
      s->unsupported = true;
      return false;

   case nir_intrinsic_store_per_vertex_output:
      store = true;
      break;
   case nir_intrinsic_load_per_vertex_output:
      break;
   default:
      return false;
   }

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   unsigned bit_size =
      store ? nir_src_bit_size(intr->src[0]) : intr->def.bit_size;

   if (bit_size != 32 || sem.location == VARYING_SLOT_LAYER ||
       sem.location == VARYING_SLOT_VIEWPORT ||
       sem.location == VARYING_SLOT_PRIMITIVE_ID ||
       sem.location == VARYING_SLOT_CULL_DIST0 ||
       sem.location == VARYING_SLOT_CULL_DIST1 ||
       sem.location + sem.num_slots > 64) {
      s->unsupported = true;
      return false;
   }

   if (sem.location == VARYING_SLOT_PSIZ) {
      s->psiz_written |= store;
      return false;
   }

   unsigned components = nir_intrinsic_component(intr) +
                         (store ? util_last_bit(nir_intrinsic_write_mask(intr))
                                : intr->def.num_components);

   nir_src *offset = nir_get_io_offset_src(intr);
   if (nir_src_is_const(*offset)) {
      unsigned loc = sem.location + nir_src_as_uint(*offset);
      s->components[loc] = MAX2(s->components[loc], components);
   } else {
      for (unsigned i = 0; i < sem.num_slots; ++i)
         s->components[sem.location + i] = 4;
   }

   return false;
}

/* The UVS layout, as agx_nir_lower_uvs computes a vertex shader's */
static void
layout_uvs(nir_shader *nir, struct mesh_hw_state *s)
{
   struct agx_unlinked_uvs_layout *layout = s->uvs;
   memset(layout, 0, sizeof(*layout));

   unsigned sizes[UVS_NUM_GROUP] = {
      [UVS_POSITION] = 4,
      [UVS_PSIZ] = s->points,
      [UVS_CLIP_DIST] = nir->info.clip_distance_array_size,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(s->components); ++i) {
      if (i != VARYING_SLOT_POS && i != VARYING_SLOT_PSIZ && s->components[i]) {
         layout->components[i] = s->components[i];
         layout->written |= BITFIELD64_BIT(i);
         sizes[UVS_VARYINGS] += s->components[i];
      }
   }

   unsigned offs = 0;
   for (enum uvs_group g = 0; g < UVS_NUM_GROUP; ++g) {
      layout->group_offs[g] = offs;
      offs += sizes[g];
   }

   layout->size = offs;
   layout->user_size = sizes[UVS_VARYINGS];
   s->single_vary = util_is_power_of_two_nonzero64(layout->written);

   agx_pack(&layout->osel, OUTPUT_SELECT, cfg) {
      cfg.point_size = sizes[UVS_PSIZ];
      cfg.clip_distance_plane_0 = sizes[UVS_CLIP_DIST] > 0;
      cfg.clip_distance_plane_1 = sizes[UVS_CLIP_DIST] > 1;
      cfg.clip_distance_plane_2 = sizes[UVS_CLIP_DIST] > 2;
      cfg.clip_distance_plane_3 = sizes[UVS_CLIP_DIST] > 3;
      cfg.clip_distance_plane_4 = sizes[UVS_CLIP_DIST] > 4;
      cfg.clip_distance_plane_5 = sizes[UVS_CLIP_DIST] > 5;
      cfg.clip_distance_plane_6 = sizes[UVS_CLIP_DIST] > 6;
      cfg.clip_distance_plane_7 = sizes[UVS_CLIP_DIST] > 7;
   }

   agx_pack(&layout->vdm, VDM_STATE_VERTEX_OUTPUTS, cfg) {
      cfg.output_count_1 = offs;
      cfg.output_count_2 = offs;
   }
}

/* A UVS slot's index (16-bit), as agx_nir_lower_uvs's for a vertex shader */
static nir_def *
uvs_index(nir_builder *b, struct mesh_hw_state *s, nir_io_semantics sem,
          unsigned component, nir_def *offset)
{
   enum uvs_group group = sem.location == VARYING_SLOT_POS    ? UVS_POSITION
                          : sem.location == VARYING_SLOT_PSIZ ? UVS_PSIZ
                                                              : UVS_VARYINGS;
   nir_def *base;

   if (group == UVS_VARYINGS && !s->single_vary)
      base = nir_load_uvs_index_agx(b, .io_semantics = sem);
   else
      base = nir_imm_intN_t(b, s->uvs->group_offs[group], 16);

   return nir_iadd(b, nir_iadd_imm(b, base, component),
                   nir_imul_imm(b, nir_u2u16(b, offset), 4));
}

/* The threadgroup address of a slot's dword for a vertex */
static nir_def *
vertex_address(nir_builder *b, struct mesh_hw_state *s, nir_def *index,
               nir_def *vertex)
{
   nir_def *el = nir_iadd(
      b, nir_imul_imm(b, nir_u2u32(b, index), s->vtx_stride),
      nir_u2u32(b, vertex));

   return nir_iadd_imm(b, nir_imul_imm(b, el, 4), s->vtx_base);
}

static unsigned
static_vertex_address(struct mesh_hw_state *s, unsigned slot, unsigned vertex)
{
   return s->vtx_base + 4 * (slot * s->vtx_stride + vertex);
}

static void
lower_vertex_output(nir_builder *b, nir_intrinsic_instr *intr,
                    struct mesh_hw_state *s)
{
   bool store = intr->intrinsic == nir_intrinsic_store_per_vertex_output;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_def *vertex = intr->src[store ? 1 : 0].ssa;
   nir_def *offset = nir_get_io_offset_src(intr)->ssa;
   unsigned component = nir_intrinsic_component(intr);

   /* Only points have a point size */
   if (sem.location == VARYING_SLOT_PSIZ && !s->points) {
      if (!store) {
         nir_def_rewrite_uses(&intr->def,
                              nir_imm_floatN_t(b, 1.0, intr->def.bit_size));
      }

      return;
   }

   if (!store) {
      nir_def *comps[4];

      for (unsigned i = 0; i < intr->def.num_components; ++i) {
         nir_def *index = uvs_index(b, s, sem, component + i, offset);
         comps[i] = nir_load_shared(b, 1, 32, vertex_address(b, s, index, vertex),
                                    .align_mul = 4);
      }

      nir_def_rewrite_uses(&intr->def,
                           nir_vec(b, comps, intr->def.num_components));
      return;
   }

   u_foreach_bit(i, nir_intrinsic_write_mask(intr)) {
      unsigned c = component + i;
      nir_def *value = nir_channel(b, intr->src[0].ssa, i);
      nir_def *index = uvs_index(b, s, sem, c, offset);

      nir_store_shared(b, value, vertex_address(b, s, index, vertex),
                       .align_mul = 4);

      /* The clip distances go to their varyings and to the clip group, as a
       * vertex shader's (agx_nir_lower_uvs) */
      if (sem.location == VARYING_SLOT_CLIP_DIST0 ||
          sem.location == VARYING_SLOT_CLIP_DIST1) {
         unsigned cd = 4 * (sem.location - VARYING_SLOT_CLIP_DIST0) + c;

         if (cd < b->shader->info.clip_distance_array_size) {
            nir_def *clip = nir_iadd_imm(
               b, nir_imul_imm(b, nir_u2u16(b, offset), 4),
               s->uvs->group_offs[UVS_CLIP_DIST] + cd);

            nir_store_shared(b, value, vertex_address(b, s, clip, vertex),
                             .align_mul = 4);
         }
      }
   }
}

/* The indices, a byte per corner, primitive after primitive */
static void
lower_indices(nir_builder *b, nir_intrinsic_instr *intr,
              struct mesh_hw_state *s)
{
   bool store = intr->intrinsic == nir_intrinsic_store_per_primitive_output;
   nir_def *primitive = intr->src[store ? 1 : 0].ssa;
   unsigned component = nir_intrinsic_component(intr);

   nir_def *base = nir_iadd_imm(b, nir_imul_imm(b, primitive, s->vpp),
                                s->idx_base + component);

   if (store) {
      u_foreach_bit(i, nir_intrinsic_write_mask(intr)) {
         nir_def *value = nir_channel(b, intr->src[0].ssa, i);
         nir_store_shared(b, nir_u2u8(b, value), nir_iadd_imm(b, base, i),
                          .align_mul = 1);
      }
   } else {
      nir_def *comps[4];

      for (unsigned i = 0; i < intr->def.num_components; ++i) {
         comps[i] = nir_u2uN(
            b,
            nir_load_shared(b, 1, 8, nir_iadd_imm(b, base, i), .align_mul = 1),
            intr->def.bit_size);
      }

      nir_def_rewrite_uses(&intr->def,
                           nir_vec(b, comps, intr->def.num_components));
   }
}

static bool
lower_mesh(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct mesh_hw_state *s = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *repl;

   switch (intr->intrinsic) {
   case nir_intrinsic_store_per_vertex_output:
   case nir_intrinsic_load_per_vertex_output:
      lower_vertex_output(b, intr, s);
      nir_instr_remove(&intr->instr);
      return true;

   case nir_intrinsic_store_per_primitive_output:
   case nir_intrinsic_load_per_primitive_output:
      lower_indices(b, intr, s);
      nir_instr_remove(&intr->instr);
      return true;

   case nir_intrinsic_set_vertex_and_primitive_count:
      nir_store_var(b, s->vertex_count, intr->src[0].ssa, 1);
      nir_store_var(b, s->primitive_count, intr->src[1].ssa, 1);
      nir_instr_remove(&intr->instr);
      return true;

   case nir_intrinsic_barrier: {
      /* The outputs live in threadgroup memory */
      nir_variable_mode modes = nir_intrinsic_memory_modes(intr);
      if (!(modes & nir_var_shader_out))
         return false;

      nir_intrinsic_set_memory_modes(
         intr, (modes & ~nir_var_shader_out) | nir_var_mem_shared);
      return true;
   }

   case nir_intrinsic_load_num_workgroups:
      repl = load_grid(b);
      break;

   case nir_intrinsic_load_draw_id:
      repl = LOAD_PARAM(b, draw_id, 1, 32);
      break;

   case nir_intrinsic_load_global_invocation_id: {
      const uint16_t *wg = b->shader->info.workgroup_size;

      repl = nir_iadd(b,
                      nir_imul(b, nir_load_workgroup_id(b),
                               nir_imm_ivec3(b, wg[0], wg[1], wg[2])),
                      nir_load_local_invocation_id(b));
      break;
   }

   default:
      return false;
   }

   nir_def_replace(&intr->def, nir_u2uN(b, repl, intr->def.bit_size));
   return true;
}

/* A padded threadgroup's IDs (pad_threadgroup): the shader's own, from the
 * index the hardware gives in the 32 x 1 x 1 it runs */
static bool
lower_padded_id(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const uint16_t *wg = data;
   nir_def *repl;

   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_local_invocation_id: {
      nir_def *i = nir_u2u32(b, nir_load_local_invocation_index(b));

      repl = nir_vec3(b, nir_umod_imm(b, i, wg[0]),
                      nir_umod_imm(b, nir_udiv_imm(b, i, wg[0]), wg[1]),
                      nir_udiv_imm(b, i, wg[0] * wg[1]));
      break;
   }
   case nir_intrinsic_load_workgroup_size:
      repl = nir_imm_ivec3(b, wg[0], wg[1], wg[2]);
      break;
   default:
      return false;
   }

   nir_def_replace(&intr->def, nir_u2uN(b, repl, intr->def.bit_size));
   return true;
}

/*
 * A threadgroup under a SIMD group's 32 threads runs as one of 32: its
 * epilogue then moves the vertices 32 a uvst, the form Apple's compiler
 * emits at 256 vertices (mesh.md), each uvst's threadgroup address 128 B
 * after the last. A uvst from an address not 16 B aligned draws nothing
 * (mesh-hw.md, builds 6 and 7 on the Mac: a line or a triangle stored a
 * vertex a uvst from one thread, or from slots 12 B apart, drew nothing; a
 * point, one uvst, drew). The shader's invocations keep their IDs and the
 * others skip its body, so they reach only the epilogue. Not with quad
 * derivatives, whose IDs the hardware lays out.
 */
static void
pad_threadgroup(nir_shader *nir, struct hk_mesh_hw_info *info)
{
   uint16_t wg[3] = {info->workgroup_size[0], info->workgroup_size[1],
                     info->workgroup_size[2]};
   unsigned threads = wg[0] * wg[1] * wg[2];

   if (threads >= 32 || nir->info.derivative_group != DERIVATIVE_GROUP_NONE)
      return;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_shader_intrinsics_pass(nir, lower_padded_id, nir_metadata_control_flow,
                              wg);

   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));

   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_push_if(&b, nir_ult_imm(&b, nir_load_local_invocation_index(&b), threads));
   nir_cf_reinsert(&body, b.cursor);
   nir_pop_if(&b, NULL);

   info->workgroup_size[0] = 32;
   info->workgroup_size[1] = 1;
   info->workgroup_size[2] = 1;
   nir->info.workgroup_size[0] = 32;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;
}

/* Apple's agc.mesh.write_uvb (mesh.md), on SIMD group 0 */
static void
emit_epilogue(nir_builder *b, struct mesh_hw_state *s)
{
   const struct hk_mesh_hw_info *info = s->info;
   const struct agx_unlinked_uvs_layout *uvs = s->uvs;

   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);

   nir_push_if(b, nir_ieq_imm(b, nir_load_subgroup_id(b), 0));
   {
      /* The counts are the first invocation's (OpSetMeshOutputsEXT), which
       * a padded threadgroup's other lanes never set */
      nir_def *zero = nir_imm_int(b, 0);
      nir_def *verts = nir_umin(
         b, nir_read_invocation(b, nir_load_var(b, s->vertex_count), zero),
         nir_imm_int(b, info->max_vertices));
      nir_def *prims = nir_umin(
         b, nir_read_invocation(b, nir_load_var(b, s->primitive_count), zero),
         nir_imm_int(b, info->max_primitives));
      prims = nir_bcsel(b, nir_ieq_imm(b, verts, 0), nir_imm_int(b, 0), prims);

      nir_def *nidx = nir_imul_imm(b, prims, s->vpp);
      nir_def *ndw = nir_ushr_imm(b, nir_iadd_imm(b, nidx, 3), 2);
      nir_def *lane = nir_load_subgroup_invocation(b);
      nir_def *last = nir_iadd_imm(b, verts, -1);

      /* SIMD group 0's lanes: a threadgroup under 32 threads is padded to
       * 32, unless it has quad derivatives; each uvst moves only the active
       * lanes' dwords (Apple's 8-thread, 4-vertex capture needs no more) */
      const uint16_t *wg = b->shader->info.workgroup_size;
      unsigned lanes = MIN2(32, wg[0] * wg[1] * wg[2]);

      unsigned idx_chunks =
         DIV_ROUND_UP(DIV_ROUND_UP(info->max_primitives * s->vpp, 4), lanes);
      unsigned vtx_chunks = DIV_ROUND_UP(info->max_vertices, lanes);

      /* Each lane's dword of indices: clamped to the vertices, the bytes past
       * the count zeroed, as Apple's epilogue */
      for (unsigned c = 0; c < idx_chunks; ++c) {
         nir_def *d = nir_iadd_imm(b, lane, lanes * c);

         nir_push_if(b, nir_ult(b, d, ndw));
         {
            nir_def *addr = nir_iadd_imm(b, nir_imul_imm(b, d, 4), s->idx_base);
            nir_def *w = nir_load_shared(b, 1, 32, addr, .align_mul = 4);
            nir_def *out = nir_imm_int(b, 0);

            for (unsigned k = 0; k < 4; ++k) {
               nir_def *index = nir_ubitfield_extract_imm(b, w, 8 * k, 8);
               nir_def *corner = nir_iadd_imm(b, nir_imul_imm(b, d, 4), k);

               index = nir_bcsel(b, nir_ult(b, corner, nidx),
                                 nir_umin(b, index, last), nir_imm_int(b, 0));
               out = nir_ior(b, out, nir_ishl_imm(b, index, 8 * k));
            }

            nir_store_shared(b, out, addr, .align_mul = 4);
         }
         nir_pop_if(b, NULL);
      }

      /* Each lane's vertex: the position through the guard band (hk_shader.c,
       * lower_guard_band) and the point size clamped (pointSizeRange), here
       * so that the shader read its own outputs back */
      unsigned psiz = uvs->group_offs[UVS_PSIZ];

      for (unsigned c = 0; c < vtx_chunks; ++c) {
         nir_def *v = nir_iadd_imm(b, lane, lanes * c);

         nir_push_if(b, nir_ult(b, v, verts));
         {
            for (unsigned k = 0; k < 2; ++k) {
               nir_def *addr = vertex_address(b, s, nir_imm_int(b, k), v);
               nir_def *x = nir_load_shared(b, 1, 32, addr, .align_mul = 4);

               nir_store_shared(b, nir_fmul_imm(b, x, 1.0 / HK_GUARD_BAND), addr,
                                .align_mul = 4);
            }

            if (s->points) {
               nir_def *addr = vertex_address(b, s, nir_imm_int(b, psiz), v);
               nir_def *size = nir_imm_float(b, 1.0);

               if (s->psiz_written) {
                  size = nir_fclamp(b, nir_load_shared(b, 1, 32, addr, .align_mul = 4),
                                    nir_imm_float(b, 1.0),
                                    nir_imm_float(b, 511.95));
               }

               nir_store_shared(b, size, addr, .align_mul = 4);
            }
         }
         nir_pop_if(b, NULL);
      }

      nir_allocate_uvb_agx(b, nir_u2u16(b, prims), nir_u2u16(b, verts));

      /* The indices, a dword a lane: the pair is {element of lane 0 << 16,
       * its address}, and the wait follows each */
      for (unsigned c = 0; c < idx_chunks; ++c) {
         nir_push_if(b, nir_ult(b, nir_iadd_imm(b, lane, lanes * c), ndw));
         {
            uint64_t pair = ((uint64_t)(s->idx_base + 4 * lanes * c) << 32) |
                            ((lanes * c) << 16);

            nir_store_uvb_agx(b, nir_imm_int64(b, pair),
                              .base = AGX_UVB_STORE_INDICES |
                                      AGX_UVB_STORE_WAIT);
         }
         nir_pop_if(b, NULL);
      }

      /* The vertices, a vertex a lane, slot by slot: {slot | first vertex <<
       * 16, its dword's address} */
      for (unsigned c = 0; c < vtx_chunks; ++c) {
         nir_push_if(b, nir_ult(b, nir_iadd_imm(b, lane, lanes * c), verts));
         {
            for (unsigned slot = 0; slot < uvs->size; ++slot) {
               uint64_t pair =
                  ((uint64_t)static_vertex_address(s, slot, lanes * c) << 32) |
                  ((lanes * c) << 16) | slot;

               bool wait = slot + 1 == uvs->size && c + 1 < vtx_chunks;

               nir_store_uvb_agx(b, nir_imm_int64(b, pair),
                                 .base = AGX_UVB_STORE_VERTEX |
                                         (wait ? AGX_UVB_STORE_WAIT : 0));
            }
         }
         nir_pop_if(b, NULL);
      }
   }
   nir_pop_if(b, NULL);
}

/*
 * The mesh program of a mesh shader without a task shader, or false when the
 * hardware path does not cover the shader (the draws then run as compute).
 * The shader is the mesh stage before poly_nir_lower_mesh, its I/O lowered.
 */
bool
hk_nir_lower_mesh_hw(nir_shader *nir, struct hk_mesh_hw_info *info,
                     struct agx_unlinked_uvs_layout *uvs)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   enum mesa_prim prim = nir->info.mesh.primitive_type;

   *info = (struct hk_mesh_hw_info){
      .max_vertices = nir->info.mesh.max_vertices_out,
      .max_primitives = nir->info.mesh.max_primitives_out,
      .prim = prim,
      .workgroup_size = {nir->info.workgroup_size[0],
                         nir->info.workgroup_size[1],
                         nir->info.workgroup_size[2]},
   };

   struct mesh_hw_state s = {
      .info = info,
      .uvs = uvs,
      .vpp = mesa_vertices_per_prim(prim),
      .points = prim == MESA_PRIM_POINTS,
   };

   if (!info->max_vertices || !info->max_primitives ||
       info->max_vertices > HK_MESH_HW_MAX_VERTICES ||
       info->max_primitives > HK_MESH_HW_MAX_PRIMITIVES ||
       nir->info.cull_distance_array_size || nir->info.workgroup_size_variable)
      return false;

   nir_shader_intrinsics_pass(nir, gather, nir_metadata_all, &s);
   if (s.unsupported)
      return false;

   layout_uvs(nir, &s);

   s.idx_base = align(nir->info.shared_size, 16);
   s.vtx_base = s.idx_base + align(info->max_primitives * s.vpp, 16);
   s.vtx_stride = align(info->max_vertices, 4);
   unsigned shared_B = s.vtx_base + uvs->size * s.vtx_stride * 4;

   if (uvs->size > 128 || shared_B > HK_MESH_HW_MAX_SHARED_B)
      return false;

   /* Every invocation reaches the epilogue's barrier */
   NIR_PASS(_, nir, nir_lower_returns);

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   s.vertex_count = nir_local_variable_create(impl, glsl_uint_type(), NULL);
   s.primitive_count = nir_local_variable_create(impl, glsl_uint_type(), NULL);
   nir_store_var(&b, s.vertex_count, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, s.primitive_count, nir_imm_int(&b, 0), 1);

   nir_shader_intrinsics_pass(nir, lower_mesh, nir_metadata_none, &s);
   pad_threadgroup(nir, info);

   b.cursor = nir_after_impl(impl);
   emit_epilogue(&b, &s);

   nir->info.shared_size = shared_B;

   /* A compute-form program */
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->info.outputs_written = 0;
   nir->info.per_primitive_outputs = 0;

   return nir_progress(true, impl, nir_metadata_none);
}

/* Metal's passthrough object program for a mesh pipeline without an object
 * function (mesh.md, the 2026-10-31 capture): three threads, thread t loads
 * the grid's component t and writes it (a uvsw with no slot). No payload, so
 * no barrier. */
nir_shader *
hk_build_mesh_hw_passthrough(const nir_shader_compiler_options *options)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, options,
                                                  "mesh passthrough object");
   nir_shader *s = b.shader;

   s->info.io_lowered = true;
   s->info.internal = true;
   s->info.workgroup_size[0] = 3;
   s->info.workgroup_size[1] = 1;
   s->info.workgroup_size[2] = 1;

   nir_def *t = nir_load_local_invocation_index(&b);
   nir_def *grid = LOAD_PARAM(&b, grid, 1, 64);
   nir_def *addr = nir_iadd(&b, grid, nir_u2u64(&b, nir_imul_imm(&b, t, 4)));

   nir_store_mesh_child_count_agx(
      &b, nir_load_global_constant(&b, 1, 32, addr, .align_mul = 4));

   return s;
}
