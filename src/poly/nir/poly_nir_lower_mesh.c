/*
 * Copyright 2026 hitori1403
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "compiler/nir/nir_builder.h"
#include "poly/cl/libpoly.h"
#include "poly/geometry.h"
#include "poly/nir/poly_nir.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "util/u_math.h"
#include "nir.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "shader_enums.h"

/*
 * Task and mesh shaders run as compute shaders. Each mesh workgroup writes its
 * outputs to a record in memory:
 *
 *    u32  vertex_count, primitive_count, pad[2]
 *    u8   index[max_primitives][vertices per primitive]   (padded to 16 bytes)
 *    vec4 vertex[vertex slots][max_vertices]
 *    vec4 primitive[primitive slots][max_primitives]
 *
 * The slots are the outputs written, compacted. A slot's values for all
 * vertices (primitives) are contiguous, so a workgroup's stores of one output
 * are too.
 *
 * A hardware vertex shader then rasterizes the records: a non-indexed draw of
 * max_primitives * vertices per primitive vertices, one instance per mesh
 * workgroup. Each vertex loads its corner and its primitive's per-primitive
 * outputs. No vertex is shared between primitives, so the per-primitive values
 * (layer and viewport too) reach the rasterizer on every corner.
 */
#define RECORD_INDICES 16

static unsigned
record_vertex_base(const struct poly_mesh_info *info)
{
   unsigned vpp = mesa_vertices_per_prim(info->prim);
   return RECORD_INDICES + align(info->max_primitives * vpp, 16);
}

static unsigned
record_primitive_base(const struct poly_mesh_info *info)
{
   return record_vertex_base(info) +
          util_bitcount64(info->vertex_outputs) * info->max_vertices * 16;
}

static nir_def *
load_param_offset(nir_builder *b, nir_def *p, uint32_t offset,
                  unsigned components, unsigned bit_size)
{
   return nir_load_global_constant(b, components, bit_size,
                                   nir_iadd_imm(b, p, offset), .align_mul = 4);
}

#define load_param(b, p, field, components, bit_size)                         \
   load_param_offset(b, p, offsetof(struct poly_mesh_params, field),           \
                     components, bit_size)

/* Linear index of a workgroup in a grid */
static nir_def *
linear_id(nir_builder *b, nir_def *id, nir_def *count)
{
   nir_def *yz = nir_iadd(b, nir_channel(b, id, 1),
                          nir_imul(b, nir_channel(b, id, 2),
                                   nir_channel(b, count, 1)));

   return nir_iadd(b, nir_channel(b, id, 0),
                   nir_imul(b, yz, nir_channel(b, count, 0)));
}

static nir_def *
payload_base(nir_builder *b, nir_def *p, nir_def *task)
{
   nir_def *stride = load_param(b, p, payload_stride, 1, 32);

   return nir_iadd(b, load_param(b, p, payloads, 1, 64),
                   nir_imul_2x32_64(b, task, stride));
}

struct gather_outputs {
   struct poly_mesh_info *info;
   uint8_t components[64];
};

static bool
gather_outputs(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct gather_outputs *g = data;
   bool store = false, primitive = false;

   switch (intr->intrinsic) {
   case nir_intrinsic_store_per_primitive_output:
      primitive = true;
      FALLTHROUGH;
   case nir_intrinsic_store_per_vertex_output:
      store = true;
      break;
   case nir_intrinsic_load_per_primitive_output:
      primitive = true;
      break;
   case nir_intrinsic_load_per_vertex_output:
      break;
   default:
      return false;
   }

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location == VARYING_SLOT_PRIMITIVE_INDICES)
      return false;

   unsigned components =
      nir_intrinsic_component(intr) +
      (store ? util_last_bit(nir_intrinsic_write_mask(intr))
             : intr->def.num_components);

   /* An output indexed indirectly takes all its slots, whole */
   nir_src *offset = nir_get_io_offset_src(intr);
   unsigned first = sem.location, count = sem.num_slots;

   if (nir_src_is_const(*offset)) {
      first += nir_src_as_uint(*offset);
      count = 1;
   } else {
      components = 4;
   }

   for (unsigned slot = first; slot < first + count; ++slot) {
      assert(slot < 64 && "no 16-bit outputs");

      if (primitive)
         g->info->primitive_outputs |= BITFIELD64_BIT(slot);
      else
         g->info->vertex_outputs |= BITFIELD64_BIT(slot);

      g->components[slot] = MAX2(g->components[slot], components);
   }

   return false;
}

/* Index of an output among the record's slots of its kind */
static nir_def *
slot_index(nir_builder *b, uint64_t mask, nir_intrinsic_instr *intr)
{
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_src *offset = nir_get_io_offset_src(intr);

   if (nir_src_is_const(*offset)) {
      unsigned slot = sem.location + nir_src_as_uint(*offset);
      return nir_imm_int(b, util_bitcount64(mask & BITFIELD64_MASK(slot)));
   }

   nir_def *slot = nir_iadd_imm(b, offset->ssa, sem.location);
   nir_def *below = nir_iadd_imm(b, nir_ishl(b, nir_imm_int64(b, 1), slot), -1);
   return nir_bit_count(b, nir_iand_imm(b, below, mask));
}

static nir_def *
output_offset(nir_builder *b, const struct poly_mesh_info *info,
              nir_intrinsic_instr *intr, nir_def *index, bool primitive)
{
   uint64_t mask = primitive ? info->primitive_outputs : info->vertex_outputs;
   unsigned count = primitive ? info->max_primitives : info->max_vertices;
   unsigned base =
      primitive ? record_primitive_base(info) : record_vertex_base(info);

   nir_def *el = nir_iadd(b, nir_imul_imm(b, slot_index(b, mask, intr), count),
                          index);

   return nir_iadd_imm(b, nir_imul_imm(b, el, 16),
                       base + (nir_intrinsic_component(intr) * 4));
}

/* Offset of a primitive's index of corner (component) c */
static nir_def *
index_offset(nir_builder *b, const struct poly_mesh_info *info,
             nir_intrinsic_instr *intr, nir_def *primitive, unsigned c)
{
   unsigned vpp = mesa_vertices_per_prim(info->prim);

   return nir_iadd_imm(b, nir_imul_imm(b, primitive, vpp),
                       RECORD_INDICES + nir_intrinsic_component(intr) + c);
}

struct lower_mesh_state {
   const struct poly_mesh_info *info;

   nir_def *record, *payload, *workgroup_id, *num_workgroups, *draw_id;
   nir_def *native_workgroup_id;
   nir_variable *vertex_count, *primitive_count;
   bool task;
};

static nir_def *
record_address(nir_builder *b, struct lower_mesh_state *s, nir_def *offset)
{
   return nir_iadd(b, s->record, nir_u2u64(b, offset));
}

static bool
lower_output(nir_builder *b, nir_intrinsic_instr *intr,
             struct lower_mesh_state *s)
{
   bool primitive = intr->intrinsic == nir_intrinsic_store_per_primitive_output ||
                    intr->intrinsic == nir_intrinsic_load_per_primitive_output;
   bool store = intr->intrinsic == nir_intrinsic_store_per_vertex_output ||
                intr->intrinsic == nir_intrinsic_store_per_primitive_output;

   nir_def *index = intr->src[store ? 1 : 0].ssa;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);

   /* Indices are bytes, corner by corner */
   if (sem.location == VARYING_SLOT_PRIMITIVE_INDICES) {
      if (store) {
         nir_def *value = intr->src[0].ssa;

         u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
            nir_def *addr =
               record_address(b, s, index_offset(b, s->info, intr, index, c));

            nir_store_global(b, nir_u2u8(b, nir_channel(b, value, c)), addr,
                             .align_mul = 1);
         }
      } else {
         nir_def *corners[4];

         for (unsigned c = 0; c < intr->def.num_components; ++c) {
            nir_def *addr =
               record_address(b, s, index_offset(b, s->info, intr, index, c));

            corners[c] = nir_u2uN(b, nir_load_global(b, 1, 8, addr, .align_mul = 1),
                                  intr->def.bit_size);
         }

         nir_def_rewrite_uses(&intr->def,
                              nir_vec(b, corners, intr->def.num_components));
      }

      return true;
   }

   nir_def *addr =
      record_address(b, s, output_offset(b, s->info, intr, index, primitive));

   if (store) {
      assert(nir_src_bit_size(intr->src[0]) == 32 && "no 16-bit outputs");
      nir_store_global(b, intr->src[0].ssa, addr,
                       .write_mask = nir_intrinsic_write_mask(intr),
                       .align_mul = 4);
   } else {
      nir_def_rewrite_uses(&intr->def,
                           nir_load_global(b, intr->def.num_components,
                                           intr->def.bit_size, addr,
                                           .align_mul = 4));
   }

   return true;
}

/* A barrier on outputs or the payload is one on the global memory backing them.
 * A mesh workgroup may span several subgroups: keep the scopes.
 */
static bool
lower_barrier(nir_intrinsic_instr *intr)
{
   nir_variable_mode modes = nir_intrinsic_memory_modes(intr);
   nir_variable_mode memory = nir_var_shader_out | nir_var_mem_task_payload;

   if (!(modes & memory))
      return false;

   nir_intrinsic_set_memory_modes(intr, (modes & ~memory) | nir_var_mem_global);
   return true;
}

static nir_def *
payload_address(nir_builder *b, nir_def *payload, nir_intrinsic_instr *intr)
{
   nir_def *offset = nir_iadd_imm(b, nir_get_io_offset_src(intr)->ssa,
                                  nir_intrinsic_base(intr));

   return nir_iadd(b, payload, nir_u2u64(b, offset));
}

static bool
lower_mesh(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower_mesh_state *s = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *repl;

   switch (intr->intrinsic) {
   case nir_intrinsic_store_per_vertex_output:
   case nir_intrinsic_store_per_primitive_output:
   case nir_intrinsic_load_per_vertex_output:
   case nir_intrinsic_load_per_primitive_output:
      lower_output(b, intr, s);
      nir_instr_remove(&intr->instr);
      return true;

   case nir_intrinsic_set_vertex_and_primitive_count:
      nir_store_var(b, s->vertex_count, intr->src[0].ssa, 1);
      nir_store_var(b, s->primitive_count, intr->src[1].ssa, 1);
      nir_instr_remove(&intr->instr);
      return true;

   case nir_intrinsic_barrier:
      return lower_barrier(intr);

   case nir_intrinsic_load_task_payload:
      repl = nir_load_global(b, intr->def.num_components, intr->def.bit_size,
                             payload_address(b, s->payload, intr),
                             .align_mul = nir_intrinsic_align_mul(intr),
                             .align_offset = nir_intrinsic_align_offset(intr));
      break;

   case nir_intrinsic_load_workgroup_id:
      if (intr == nir_def_as_intrinsic(s->native_workgroup_id))
         return false;

      repl = s->workgroup_id;
      break;

   case nir_intrinsic_load_num_workgroups:
      repl = s->num_workgroups;
      break;

   case nir_intrinsic_load_global_invocation_id:
      /* After a task shader, the workgroup ID is not the hardware's */
      if (!s->task)
         return false;

      repl = nir_iadd(b,
                      nir_imul(b, s->workgroup_id,
                               nir_imm_ivec3(b, b->shader->info.workgroup_size[0],
                                             b->shader->info.workgroup_size[1],
                                             b->shader->info.workgroup_size[2])),
                      nir_load_local_invocation_id(b));
      break;

   case nir_intrinsic_load_draw_id:
      repl = s->draw_id;
      break;

   default:
      return false;
   }

   nir_def_replace(&intr->def, nir_u2uN(b, repl, intr->def.bit_size));
   return true;
}

/*
 * The rasterization shader: a hardware vertex shader drawing the records. A
 * primitive past the count, or culled, is killed with a NaN position.
 */
static nir_shader *
build_rast_shader(const nir_shader *mesh, const struct poly_mesh_info *info,
                  const uint8_t *components, bool multiview)
{
   nir_builder b_ = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, mesh->options, "%s rast",
      mesh->info.name ? mesh->info.name : "mesh");
   nir_builder *b = &b_;
   nir_shader *s = b->shader;

   s->info.io_lowered = true;
   s->info.clip_distance_array_size = mesh->info.clip_distance_array_size;
   s->info.cull_distance_array_size = mesh->info.cull_distance_array_size;

   unsigned vpp = mesa_vertices_per_prim(info->prim);
   nir_def *p = nir_load_mesh_param_buffer_poly(b);
   nir_def *vertex = nir_load_vertex_id(b);
   nir_def *primitive = nir_udiv_imm(b, vertex, vpp);

   nir_def *record =
      nir_iadd(b, load_param(b, p, outputs, 1, 64),
               nir_imul_2x32_64(b, nir_load_instance_id(b),
                                nir_imm_int(b, info->record_stride)));

   nir_def *count =
      nir_load_global(b, 1, 32, nir_iadd_imm(b, record, 4), .align_mul = 4);
   nir_def *alive = nir_ult(b, primitive, count);

   /* The corner's index sits at the vertex ID. Out-of-range indices are
    * undefined: clamp them to stay within the record.
    */
   nir_def *index_addr =
      nir_iadd(b, record, nir_u2u64(b, nir_iadd_imm(b, vertex, RECORD_INDICES)));
   nir_def *index = nir_u2u32(b, nir_load_global(b, 1, 8, index_addr,
                                                 .align_mul = 1));
   nir_def *corner = nir_umin(b, index, nir_imm_int(b, info->max_vertices - 1));

   nir_def *position = NULL;
   uint64_t outputs = info->vertex_outputs | info->primitive_outputs;

   u_foreach_bit64(slot, outputs) {
      bool per_primitive = info->primitive_outputs & BITFIELD64_BIT(slot);
      uint64_t mask =
         per_primitive ? info->primitive_outputs : info->vertex_outputs;

      unsigned el_count =
         per_primitive ? info->max_primitives : info->max_vertices;
      unsigned base = per_primitive ? record_primitive_base(info)
                                    : record_vertex_base(info);
      unsigned slot_base =
         base + (util_bitcount64(mask & BITFIELD64_MASK(slot)) * el_count * 16);

      nir_def *el = per_primitive ? primitive : corner;
      nir_def *offset = nir_iadd_imm(b, nir_imul_imm(b, el, 16), slot_base);

      unsigned n = slot == VARYING_SLOT_POS ? 4 : components[slot];
      nir_def *value = nir_load_global(b, n, 32, nir_iadd(b, record,
                                                          nir_u2u64(b, offset)),
                                       .align_mul = 16);

      if (slot == VARYING_SLOT_CULL_PRIMITIVE) {
         alive = nir_iand(b, alive, nir_ieq_imm(b, nir_channel(b, value, 0), 0));
      } else if (slot == VARYING_SLOT_POS) {
         position = value;
      } else if (slot != VARYING_SLOT_LAYER || !multiview) {
         bool is_float = slot == VARYING_SLOT_PSIZ ||
                         slot == VARYING_SLOT_CLIP_DIST0 ||
                         slot == VARYING_SLOT_CLIP_DIST1 ||
                         slot == VARYING_SLOT_CULL_DIST0 ||
                         slot == VARYING_SLOT_CULL_DIST1;

         nir_store_output(b, value, nir_imm_int(b, 0),
                          .src_type = is_float ? nir_type_float32
                                               : nir_type_uint32,
                          .io_semantics.location = slot,
                          .io_semantics.num_slots = 1);
      }
   }

   if (position) {
      nir_def *nan = nir_imm_vec4(b, NAN, NAN, NAN, NAN);

      nir_store_output(b, nir_bcsel(b, alive, position, nan), nir_imm_int(b, 0),
                       .src_type = nir_type_float32,
                       .io_semantics.location = VARYING_SLOT_POS,
                       .io_semantics.num_slots = 1);
   }

   /* Multiview draws each view, the view index as the layer */
   if (multiview) {
      nir_store_output(b, nir_load_view_index(b), nir_imm_int(b, 0),
                       .src_type = nir_type_uint32,
                       .io_semantics.location = VARYING_SLOT_LAYER,
                       .io_semantics.num_slots = 1);
   }

   nir_shader_gather_info(s, b->impl);

   if (info->prim == MESA_PRIM_POINTS && position &&
       !(s->info.outputs_written & VARYING_BIT_PSIZ)) {
      nir_lower_default_point_size(s);
   }

   return s;
}

bool
poly_nir_lower_mesh(nir_shader *mesh, bool task, bool multiview,
                    nir_shader **rast, struct poly_mesh_info *info)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(mesh);

   *info = (struct poly_mesh_info){
      .max_vertices = mesh->info.mesh.max_vertices_out,
      .max_primitives = mesh->info.mesh.max_primitives_out,
      .prim = mesh->info.mesh.primitive_type,
   };

   struct gather_outputs g = {.info = info};
   nir_shader_intrinsics_pass(mesh, gather_outputs, nir_metadata_all, &g);

   info->record_stride = record_primitive_base(info) +
                         util_bitcount64(info->primitive_outputs) *
                            info->max_primitives * 16;

   *rast = build_rast_shader(mesh, info, g.components, multiview);

   /* The counts are stored at the end, so a shader that does not set them
    * outputs nothing
    */
   NIR_PASS(_, mesh, nir_lower_returns);

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *p = nir_load_mesh_param_buffer_poly(&b);

   struct lower_mesh_state s = {
      .info = info,
      .task = task,
      .native_workgroup_id = nir_load_workgroup_id(&b),
      .vertex_count = nir_local_variable_create(impl, glsl_uint_type(), NULL),
      .primitive_count = nir_local_variable_create(impl, glsl_uint_type(), NULL),
      .draw_id = load_param(&b, p, draw_id, 1, 32),
   };

   nir_store_var(&b, s.vertex_count, zero, 1);
   nir_store_var(&b, s.primitive_count, zero, 1);

   /* After a task shader, mesh workgroups are dispatched in one dimension:
    * find each one's task workgroup and its place in that one's grid.
    */
   nir_def *w;
   if (task) {
      w = nir_channel(&b, s.native_workgroup_id, 0);
      nir_def *t = poly_mesh_task_index(&b, p, w);

      nir_def *launch = nir_iadd(&b, load_param(&b, p, launches, 1, 64),
                                 nir_u2u64(&b, nir_imul_imm(&b, t, 16)));

      s.workgroup_id = poly_mesh_workgroup_id(&b, p, t, w);
      s.num_workgroups = nir_load_global(&b, 3, 32, launch, .align_mul = 16);
      s.payload = payload_base(&b, p, t);
   } else {
      s.workgroup_id = s.native_workgroup_id;
      s.num_workgroups = load_param(&b, p, group_count, 3, 32);
      s.payload = payload_base(&b, p, zero);
      w = linear_id(&b, s.workgroup_id, s.num_workgroups);
   }

   s.record = nir_iadd(&b, load_param(&b, p, outputs, 1, 64),
                       nir_imul_2x32_64(&b, w,
                                        nir_imm_int(&b, info->record_stride)));

   nir_shader_intrinsics_pass(mesh, lower_mesh, nir_metadata_none, &s);

   b.cursor = nir_after_impl(impl);
   nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
   {
      nir_def *counts = nir_vec2(&b, nir_load_var(&b, s.vertex_count),
                                 nir_load_var(&b, s.primitive_count));

      nir_store_global(&b, counts, s.record, .align_mul = 16);
   }
   nir_pop_if(&b, NULL);

   /* Run as a compute shader */
   mesh->info.stage = MESA_SHADER_COMPUTE;
   memset(&mesh->info.cs, 0, sizeof(mesh->info.cs));
   mesh->info.outputs_written = 0;
   mesh->info.per_primitive_outputs = 0;

   return nir_progress(true, impl, nir_metadata_none);
}

struct lower_task_state {
   nir_def *payload, *launch, *num_workgroups, *draw_id;
};

static bool
lower_task(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower_task_state *s = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *repl;

   switch (intr->intrinsic) {
   case nir_intrinsic_store_task_payload:
      nir_store_global(b, intr->src[0].ssa,
                       payload_address(b, s->payload, intr),
                       .write_mask = nir_intrinsic_write_mask(intr),
                       .align_mul = nir_intrinsic_align_mul(intr),
                       .align_offset = nir_intrinsic_align_offset(intr));
      nir_instr_remove(&intr->instr);
      return true;

   case nir_intrinsic_load_task_payload:
      repl = nir_load_global(b, intr->def.num_components, intr->def.bit_size,
                             payload_address(b, s->payload, intr),
                             .align_mul = nir_intrinsic_align_mul(intr),
                             .align_offset = nir_intrinsic_align_offset(intr));
      break;

   case nir_intrinsic_task_payload_atomic:
      repl = nir_global_atomic(b, intr->def.bit_size,
                               payload_address(b, s->payload, intr),
                               intr->src[1].ssa,
                               .atomic_op = nir_intrinsic_atomic_op(intr));
      break;

   case nir_intrinsic_task_payload_atomic_swap:
      repl = nir_global_atomic_swap(b, intr->def.bit_size,
                                    payload_address(b, s->payload, intr),
                                    intr->src[1].ssa, intr->src[2].ssa,
                                    .atomic_op = nir_intrinsic_atomic_op(intr));
      break;

   case nir_intrinsic_launch_mesh_workgroups: {
      /* Uniform: one invocation records the launch, with its size for the
       * prefix sum. A launch past the limits is undefined: launch nothing.
       */
      nir_def *grid = intr->src[0].ssa;
      nir_def *xy = nir_imul(b, nir_channel(b, grid, 0), nir_channel(b, grid, 1));
      nir_def *size64 = nir_imul_2x32_64(b, xy, nir_channel(b, grid, 2));

      nir_def *valid = nir_ball(b, nir_ule_imm(b, grid, 65535));
      valid = nir_iand(b, valid,
                       nir_ule_imm(b, size64, POLY_MAX_MESH_WORKGROUPS));

      grid = nir_bcsel(b, valid, grid, nir_imm_zero(b, 3, 32));
      nir_def *size = nir_bcsel(b, valid, nir_u2u32(b, size64), nir_imm_int(b, 0));

      nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
      {
         nir_store_global(b, nir_vec4(b, nir_channel(b, grid, 0),
                                      nir_channel(b, grid, 1),
                                      nir_channel(b, grid, 2), size),
                          s->launch, .align_mul = 16);
      }
      nir_pop_if(b, NULL);
      nir_instr_remove(&intr->instr);
      return true;
   }

   case nir_intrinsic_barrier:
      return lower_barrier(intr);

   case nir_intrinsic_load_num_workgroups:
      repl = s->num_workgroups;
      break;

   case nir_intrinsic_load_draw_id:
      repl = s->draw_id;
      break;

   default:
      return false;
   }

   nir_def_replace(&intr->def, repl);
   return true;
}

bool
poly_nir_lower_task(nir_shader *task, struct poly_mesh_info *info)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(task);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   *info = (struct poly_mesh_info){
      .payload_size = task->info.task_payload_size,
   };

   nir_def *p = nir_load_mesh_param_buffer_poly(&b);

   struct lower_task_state s = {
      .num_workgroups = load_param(&b, p, group_count, 3, 32),
      .draw_id = load_param(&b, p, draw_id, 1, 32),
   };

   nir_def *t = linear_id(&b, nir_load_workgroup_id(&b), s.num_workgroups);
   s.payload = payload_base(&b, p, t);
   s.launch = nir_iadd(&b, load_param(&b, p, launches, 1, 64),
                       nir_u2u64(&b, nir_imul_imm(&b, t, 16)));

   nir_shader_intrinsics_pass(task, lower_task, nir_metadata_none, &s);

   /* Run as a compute shader */
   task->info.stage = MESA_SHADER_COMPUTE;
   memset(&task->info.cs, 0, sizeof(task->info.cs));

   return nir_progress(true, impl, nir_metadata_none);
}
