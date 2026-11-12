/*
 * Copyright © 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

/* A ray-tracing pipeline as one compute kernel, after lavapipe's
 * (lvp_ray_tracing_pipeline.c): the AGX backend has no calls. Two tiers.
 *
 * Tier 1 inlines every stage where it is called, when that ends: only ray
 * generation shaders trace (or the pipeline's depth is 1) and callable
 * shaders neither call nor trace. Each inlined stage is a clone of its shader
 * with its own copies of the ray-tracing state, mapped onto its caller's when
 * it is inlined. Payloads live in scratch: [0, 32) holds the hit attributes,
 * then a frame for each stage that passes payloads, its callees' frames after
 * its own. Once inlined every offset is a constant, and
 * nir_lower_scratch_to_var turns the scratch back into variables.
 *
 * Tier 2, recursion, is a dispatcher loop (below hk_nir_build_rt_kernel).
 */

#include "bvh/hk_bvh_defines.h"
#include "hk_nir_ray_tracing.h"

#include "nir/nir_builder.h"
#include "spirv/spirv.h"
#include "util/hash_table.h"
#include "util/u_dynarray.h"

#define HIT_ATTRIBS_SIZE 32

/* The ray-tracing state, a variable a value */
union rt_vars {
   struct {
      nir_variable *bvh_base;
      nir_variable *flags;
      nir_variable *cull_mask;
      nir_variable *sbt_offset;
      nir_variable *sbt_stride;
      nir_variable *miss_index;
      nir_variable *origin;
      nir_variable *tmin;
      nir_variable *dir;
      nir_variable *tmax;

      nir_variable *instance_addr;
      nir_variable *primitive_addr;
      nir_variable *primitive_id;
      nir_variable *geometry_id_and_flags;
      nir_variable *hit_kind;
      nir_variable *sbt_index;

      nir_variable *shader_record_ptr;
      /* The scratch offset of the payload passed to the next callee */
      nir_variable *call_data;

      nir_variable *accept;
      nir_variable *terminate;
      nir_variable *opaque;

      /* Tier 2: the case to run next, the current frame's base on the
       * thread-private stack, and the callee's payload relative to its frame
       */
      nir_variable *next;
      nir_variable *stack_ptr;
      nir_variable *arg;
   };
   nir_variable *all[24];
};

static_assert(sizeof(union rt_vars) == 24 * sizeof(nir_variable *),
              "every variable in the array");

struct rt_traversal {
   nir_variable *origin;
   nir_variable *dir;
   nir_variable *inv_dir;
   nir_variable *bvh_base;
   nir_variable *current_node;
   nir_variable *stack_base;
   nir_variable *stack_ptr;
   nir_variable *stack;
   nir_variable *hit;
   nir_variable *instance_addr;
   nir_variable *sbt_offset_and_flags;
};

struct rt_builder {
   const struct hk_rt_kernel_info *info;

   /* Stage indices by type */
   struct util_dynarray by_stage[MESA_SHADER_CALLABLE + 1];

   /* The end of the deepest frame */
   unsigned scratch_end;

   /* Tier 2: each stage's first case, by stage index, and its frame size */
   uint32_t *first_case;
   uint32_t *frame;
};

/* A stage being inlined, or the kernel */
struct rt_clone {
   struct rt_builder *rb;
   nir_shader *shader;
   mesa_shader_stage stage;
   uint32_t index;

   union rt_vars vars;

   /* Where the frames of the stages this one calls start */
   unsigned frame_end;

   /* The walk's state, made at the first trace */
   struct rt_traversal trav;
   bool has_trav;
};

static void inline_stage(struct rt_clone *parent, nir_builder *b,
                         uint32_t index);

static void
rt_vars_create(nir_shader *s, union rt_vars *v)
{
   const struct glsl_type *u32 = glsl_uint_type();
   const struct glsl_type *u64 = glsl_uint64_t_type();
   const struct glsl_type *f32 = glsl_float_type();
   const struct glsl_type *vec3 = glsl_vec_type(3);
   const struct glsl_type *bool_type = glsl_bool_type();

#define VAR(name, type)                                                        \
   v->name = nir_variable_create(s, nir_var_shader_temp, type, "rt." #name)

   VAR(bvh_base, u64);
   VAR(flags, u32);
   VAR(cull_mask, u32);
   VAR(sbt_offset, u32);
   VAR(sbt_stride, u32);
   VAR(miss_index, u32);
   VAR(origin, vec3);
   VAR(tmin, f32);
   VAR(dir, vec3);
   VAR(tmax, f32);
   VAR(instance_addr, u64);
   VAR(primitive_addr, u64);
   VAR(primitive_id, u32);
   VAR(geometry_id_and_flags, u32);
   VAR(hit_kind, u32);
   VAR(sbt_index, u32);
   VAR(shader_record_ptr, u64);
   VAR(call_data, u32);
   VAR(accept, bool_type);
   VAR(terminate, bool_type);
   VAR(opaque, bool_type);
   VAR(next, u32);
   VAR(stack_ptr, u32);
   VAR(arg, u32);
#undef VAR
}

static struct rt_traversal *
rt_traversal(struct rt_clone *c, nir_function_impl *impl)
{
   struct rt_traversal *t = &c->trav;
   if (c->has_trav)
      return t;

   const struct glsl_type *u32 = glsl_uint_type();
   const struct glsl_type *vec3 = glsl_vec_type(3);

#define VAR(name, type)                                                        \
   t->name = nir_local_variable_create(impl, type, "traversal." #name)

   VAR(origin, vec3);
   VAR(dir, vec3);
   VAR(inv_dir, vec3);
   VAR(bvh_base, glsl_uint64_t_type());
   VAR(current_node, u32);
   VAR(stack_base, u32);
   VAR(stack_ptr, u32);
   VAR(stack, glsl_array_type(u32, HK_BVH_STACK_SIZE, 0));
   VAR(hit, glsl_bool_type());
   VAR(instance_addr, glsl_uint64_t_type());
   VAR(sbt_offset_and_flags, u32);
#undef VAR

   c->has_trav = true;
   return t;
}

static nir_def *
load_param(nir_builder *b, unsigned offset, unsigned num_components,
           unsigned bit_size)
{
   nir_def *addr = nir_iadd_imm(b, nir_load_rt_params_agx(b), offset);
   return nir_load_global_constant(b, num_components, bit_size, addr);
}

/* Points the shader record at a table's entry and returns the entry's shader
 * ID in slot, 0 when the table is null.
 */
static nir_def *
sbt_lookup(struct rt_clone *c, nir_builder *b, unsigned table, nir_def *index,
           unsigned slot)
{
   nir_def *base = load_param(b, table, 1, 64);
   nir_def *entry = base;

   if (index) {
      /* maxShaderGroupStride fits the stride's low 32 bits */
      nir_def *stride = load_param(b, table + 16, 1, 32);
      entry = nir_iadd(b, entry, nir_u2u64(b, nir_imul(b, index, stride)));
   }

   nir_store_var(b, c->vars.shader_record_ptr,
                 nir_iadd_imm(b, entry, HK_RT_HANDLE_SIZE), 0x1);

   nir_def *null_id = nir_imm_int64(b, 0);
   nir_push_if(b, nir_ine_imm(b, base, 0));
   nir_def *id =
      nir_load_global_constant(b, 1, 64, nir_iadd_imm(b, entry, slot * 8));
   nir_pop_if(b, NULL);

   return nir_if_phi(b, id, null_id);
}

/* Runs the stage among these whose ID is id */
static void
build_ladder(struct rt_clone *c, nir_builder *b, nir_def *id,
             const uint32_t *stages, unsigned count)
{
   const struct hk_rt_stage *all = c->rb->info->stages;

   for (unsigned i = 0; i < count; i++) {
      nir_push_if(b, nir_ieq_imm(b, id, all[stages[i]].id));
      inline_stage(c, b, stages[i]);
      nir_push_else(b, NULL);
   }

   for (unsigned i = 0; i < count; i++)
      nir_pop_if(b, NULL);
}

static void
build_stage_ladder(struct rt_clone *c, nir_builder *b, nir_def *id,
                   mesa_shader_stage stage)
{
   struct util_dynarray *list = &c->rb->by_stage[stage];
   build_ladder(c, b, id, util_dynarray_begin(list),
                util_dynarray_num_elements(list, uint32_t));
}

static nir_def *
sbt_index(struct rt_clone *c, nir_builder *b, nir_def *geometry_id_and_flags)
{
   nir_def *geometry_id = nir_iand_imm(b, geometry_id_and_flags, 0xfffffff);
   nir_def *instance_offset =
      nir_iand_imm(b, nir_load_var(b, c->trav.sbt_offset_and_flags), 0xffffff);

   return nir_iadd(
      b, nir_iadd(b, nir_load_var(b, c->vars.sbt_offset), instance_offset),
      nir_imul(b, nir_load_var(b, c->vars.sbt_stride), geometry_id));
}

static void
handle_triangle(nir_builder *b, struct hk_triangle_intersection *intersection,
                const struct hk_ray_traversal_args *args,
                const struct hk_ray_flags *ray_flags)
{
   struct rt_clone *c = args->data;
   union rt_vars *v = &c->vars;
   nir_def *attribs = nir_imm_int(b, 0);

   nir_store_var(b, v->accept, nir_imm_true(b), 0x1);
   nir_store_var(b, v->terminate, ray_flags->terminate_on_first_hit, 0x1);

   nir_def *prev_tmax = nir_load_var(b, v->tmax);
   nir_def *prev_instance_addr = nir_load_var(b, v->instance_addr);
   nir_def *prev_primitive_id = nir_load_var(b, v->primitive_id);
   nir_def *prev_geometry_id_and_flags =
      nir_load_var(b, v->geometry_id_and_flags);
   nir_def *prev_hit_kind = nir_load_var(b, v->hit_kind);
   nir_def *prev_primitive_addr = nir_load_var(b, v->primitive_addr);
   nir_def *prev_barycentrics =
      nir_load_scratch(b, 2, 32, attribs, .align_mul = 4);

   nir_store_var(b, v->tmax, intersection->t, 0x1);
   nir_store_var(b, v->instance_addr, nir_load_var(b, c->trav.instance_addr),
                 0x1);
   nir_store_var(b, v->primitive_id, intersection->base.primitive_id, 0x1);
   nir_store_var(b, v->geometry_id_and_flags,
                 intersection->base.geometry_id_and_flags, 0x1);
   nir_store_var(b, v->hit_kind,
                 nir_bcsel(b, intersection->frontface, nir_imm_int(b, 0xFE),
                           nir_imm_int(b, 0xFF)),
                 0x1);
   nir_store_var(b, v->primitive_addr, intersection->base.node_addr, 0x1);
   nir_store_scratch(b, intersection->barycentrics, attribs, .align_mul = 4);

   nir_def *index = sbt_index(c, b, intersection->base.geometry_id_and_flags);

   nir_push_if(b, nir_inot(b, intersection->base.opaque));
   {
      nir_def *id = sbt_lookup(
         c, b,
         offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableAddress),
         index, 1);
      build_stage_ladder(c, b, id, MESA_SHADER_ANY_HIT);
   }
   nir_pop_if(b, NULL);

   nir_push_if(b, nir_load_var(b, v->accept));
   {
      nir_store_var(b, v->sbt_index, index, 0x1);
      nir_store_var(b, c->trav.hit, nir_imm_true(b), 0x1);
      nir_break_if(b, nir_load_var(b, v->terminate));
   }
   nir_push_else(b, NULL);
   {
      nir_store_var(b, v->tmax, prev_tmax, 0x1);
      nir_store_var(b, v->instance_addr, prev_instance_addr, 0x1);
      nir_store_var(b, v->primitive_id, prev_primitive_id, 0x1);
      nir_store_var(b, v->geometry_id_and_flags, prev_geometry_id_and_flags,
                    0x1);
      nir_store_var(b, v->hit_kind, prev_hit_kind, 0x1);
      nir_store_var(b, v->primitive_addr, prev_primitive_addr, 0x1);
      nir_store_scratch(b, prev_barycentrics, attribs, .align_mul = 4);
   }
   nir_pop_if(b, NULL);
}

static void
handle_aabb(nir_builder *b, struct hk_leaf_intersection *intersection,
            const struct hk_ray_traversal_args *args,
            const struct hk_ray_flags *ray_flags)
{
   struct rt_clone *c = args->data;
   union rt_vars *v = &c->vars;

   if (!util_dynarray_num_elements(&c->rb->by_stage[MESA_SHADER_INTERSECTION],
                                   uint32_t))
      return;

   nir_store_var(b, v->accept, nir_imm_false(b), 0x1);
   nir_store_var(b, v->terminate, ray_flags->terminate_on_first_hit, 0x1);
   nir_store_var(b, v->opaque, intersection->opaque, 0x1);

   nir_def *prev_instance_addr = nir_load_var(b, v->instance_addr);
   nir_def *prev_primitive_id = nir_load_var(b, v->primitive_id);
   nir_def *prev_geometry_id_and_flags =
      nir_load_var(b, v->geometry_id_and_flags);

   nir_store_var(b, v->instance_addr, nir_load_var(b, c->trav.instance_addr),
                 0x1);
   nir_store_var(b, v->primitive_id, intersection->primitive_id, 0x1);
   nir_store_var(b, v->geometry_id_and_flags,
                 intersection->geometry_id_and_flags, 0x1);

   nir_def *index = sbt_index(c, b, intersection->geometry_id_and_flags);
   nir_def *id = sbt_lookup(
      c, b,
      offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableAddress),
      index, 2);

   nir_def *prev_attribs[HIT_ATTRIBS_SIZE / 4];
   for (unsigned i = 0; i < ARRAY_SIZE(prev_attribs); i++) {
      prev_attribs[i] =
         nir_load_scratch(b, 1, 32, nir_imm_int(b, i * 4), .align_mul = 4);
   }

   build_stage_ladder(c, b, id, MESA_SHADER_INTERSECTION);

   nir_push_if(b, nir_load_var(b, v->accept));
   {
      nir_store_var(b, v->sbt_index, index, 0x1);
      nir_store_var(b, c->trav.hit, nir_imm_true(b), 0x1);
      nir_break_if(b, nir_load_var(b, v->terminate));
   }
   nir_push_else(b, NULL);
   {
      for (unsigned i = 0; i < ARRAY_SIZE(prev_attribs); i++) {
         nir_store_scratch(b, prev_attribs[i], nir_imm_int(b, i * 4),
                           .align_mul = 4);
      }
      nir_store_var(b, v->instance_addr, prev_instance_addr, 0x1);
      nir_store_var(b, v->primitive_id, prev_primitive_id, 0x1);
      nir_store_var(b, v->geometry_id_and_flags, prev_geometry_id_and_flags,
                    0x1);
   }
   nir_pop_if(b, NULL);
}

/* The ray's state as a trace sets it: the arguments trace_ray and
 * rt_trace_ray share
 */
static void
store_ray(struct rt_clone *c, nir_builder *b, nir_intrinsic_instr *intr)
{
   union rt_vars *v = &c->vars;

   nir_store_var(b, v->bvh_base, intr->src[0].ssa, 0x1);
   nir_store_var(b, v->flags, intr->src[1].ssa, 0x1);
   nir_store_var(b, v->cull_mask, intr->src[2].ssa, 0x1);
   nir_store_var(b, v->sbt_offset, nir_iand_imm(b, intr->src[3].ssa, 0xF), 0x1);
   nir_store_var(b, v->sbt_stride, nir_iand_imm(b, intr->src[4].ssa, 0xF), 0x1);
   nir_store_var(b, v->miss_index, nir_iand_imm(b, intr->src[5].ssa, 0xFFFF),
                 0x1);
   nir_store_var(b, v->origin, intr->src[6].ssa, 0x7);
   nir_store_var(b, v->tmin, intr->src[7].ssa, 0x1);
   nir_store_var(b, v->dir, intr->src[8].ssa, 0x7);
   nir_store_var(b, v->tmax, intr->src[9].ssa, 0x1);
}

/* The walk of the ray in c->vars, the any-hit and intersection shaders
 * inlined: c->trav.hit tells whether it committed a hit
 */
static void
build_walk(struct rt_clone *c, nir_builder *b)
{
   union rt_vars *v = &c->vars;
   const struct hk_rt_kernel_info *info = c->rb->info;

   nir_def *accel_struct = nir_load_var(b, v->bvh_base);
   nir_def *flags = nir_load_var(b, v->flags);
   nir_def *cull_mask = nir_load_var(b, v->cull_mask);
   nir_def *origin = nir_load_var(b, v->origin);
   nir_def *tmin = nir_load_var(b, v->tmin);
   nir_def *dir = nir_load_var(b, v->dir);

   struct rt_traversal *t = rt_traversal(c, b->impl);
   nir_store_var(b, t->bvh_base, accel_struct, 0x1);
   nir_store_var(b, t->origin, origin, 0x7);
   nir_store_var(b, t->dir, dir, 0x7);
   nir_store_var(b, t->inv_dir, nir_frcp(b, dir), 0x7);
   nir_store_var(b, t->current_node, nir_imm_int(b, HK_BVH_ROOT_NODE), 0x1);
   nir_store_var(b, t->stack_base, nir_imm_int(b, -1), 0x1);
   nir_store_var(b, t->stack_ptr, nir_imm_int(b, 0), 0x1);
   nir_store_var(b, t->hit, nir_imm_false(b), 0x1);

   struct hk_ray_traversal_args args = {
      .root_bvh_base = accel_struct,
      .flags = flags,
      .cull_mask = nir_ishl_imm(b, cull_mask, 24),
      .origin = origin,
      .tmin = tmin,
      .dir = dir,
      .vars =
         {
            .tmax = nir_build_deref_var(b, v->tmax),
            .origin = nir_build_deref_var(b, t->origin),
            .dir = nir_build_deref_var(b, t->dir),
            .inv_dir = nir_build_deref_var(b, t->inv_dir),
            .bvh_base = nir_build_deref_var(b, t->bvh_base),
            .current_node = nir_build_deref_var(b, t->current_node),
            .stack_base = nir_build_deref_var(b, t->stack_base),
            .stack_ptr = nir_build_deref_var(b, t->stack_ptr),
            .stack = nir_build_deref_var(b, t->stack),
            .instance_addr = nir_build_deref_var(b, t->instance_addr),
            .sbt_offset_and_flags =
               nir_build_deref_var(b, t->sbt_offset_and_flags),
         },
      .stack_size = HK_BVH_STACK_SIZE,
      .aabb_cb = info->skip_aabbs ? NULL : handle_aabb,
      .triangle_cb = info->skip_triangles ? NULL : handle_triangle,
      .data = c,
   };

   nir_push_if(b, nir_ine_imm(b, accel_struct, 0));
   hk_build_ray_traversal(b, &args);
   nir_pop_if(b, NULL);
}

static void
lower_trace_ray(struct rt_clone *c, nir_builder *b, nir_intrinsic_instr *intr)
{
   /* Tier 1: a trace outside a ray generation shader is past this
    * pipeline's depth (hk_nir_rt_needs_dispatcher): undefined, and dropped.
    */
   if (c->stage != MESA_SHADER_RAYGEN)
      return;

   union rt_vars *v = &c->vars;

   nir_store_var(b, v->call_data, intr->src[10].ssa, 0x1);
   store_ray(c, b, intr);
   build_walk(c, b);

   nir_def *flags = nir_load_var(b, v->flags);
   nir_def *miss_index = nir_load_var(b, v->miss_index);
   struct rt_traversal *t = &c->trav;

   nir_push_if(b, nir_load_var(b, t->hit));
   {
      nir_def *skip_chit =
         nir_test_mask(b, flags, SpvRayFlagsSkipClosestHitShaderKHRMask);
      nir_push_if(b, nir_inot(b, skip_chit));
      {
         nir_def *id = sbt_lookup(c, b,
                                  offsetof(VkTraceRaysIndirectCommand2KHR,
                                           hitShaderBindingTableAddress),
                                  nir_load_var(b, v->sbt_index), 0);
         build_stage_ladder(c, b, id, MESA_SHADER_CLOSEST_HIT);
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      nir_def *id = sbt_lookup(c, b,
                               offsetof(VkTraceRaysIndirectCommand2KHR,
                                        missShaderBindingTableAddress),
                               miss_index, 0);
      build_stage_ladder(c, b, id, MESA_SHADER_MISS);
   }
   nir_pop_if(b, NULL);
}

static void
lower_execute_callable(struct rt_clone *c, nir_builder *b,
                       nir_intrinsic_instr *intr)
{
   nir_def *id = sbt_lookup(c, b,
                            offsetof(VkTraceRaysIndirectCommand2KHR,
                                     callableShaderBindingTableAddress),
                            intr->src[0].ssa, 0);

   nir_store_var(b, c->vars.call_data, intr->src[1].ssa, 0x1);
   build_stage_ladder(c, b, id, MESA_SHADER_CALLABLE);
}

/* An intersection shader's report runs the any-hit shader of the hit group,
 * one of those paired with this intersection shader.
 */
static nir_def *
lower_report_intersection(struct rt_clone *c, nir_builder *b,
                          nir_intrinsic_instr *intr)
{
   union rt_vars *v = &c->vars;
   const struct hk_rt_kernel_info *info = c->rb->info;

   uint32_t ahits[info->group_count];
   unsigned ahit_count = 0;
   for (uint32_t g = 0; g < info->group_count; g++) {
      const struct hk_rt_group *group = &info->groups[g];
      if (group->intersection != c->index || group->any_hit == HK_RT_UNUSED)
         continue;

      bool seen = false;
      for (unsigned i = 0; i < ahit_count; i++)
         seen |= ahits[i] == group->any_hit;
      if (!seen)
         ahits[ahit_count++] = group->any_hit;
   }

   nir_def *t = intr->src[0].ssa;
   nir_def *hit_kind = intr->src[1].ssa;

   nir_def *prev_accept = nir_load_var(b, v->accept);
   nir_def *prev_tmax = nir_load_var(b, v->tmax);
   nir_def *prev_hit_kind = nir_load_var(b, v->hit_kind);

   nir_variable *commit =
      nir_local_variable_create(b->impl, glsl_bool_type(), "commit");
   nir_store_var(b, commit, nir_imm_false(b), 0x1);

   nir_def *in_range = nir_iand(b, nir_fge(b, t, nir_load_var(b, v->tmin)),
                                nir_fge(b, nir_load_var(b, v->tmax), t));
   nir_def *terminated =
      nir_iand(b, nir_load_var(b, v->terminate), nir_load_var(b, v->accept));
   nir_push_if(b, nir_iand(b, in_range, nir_inot(b, terminated)));
   {
      nir_store_var(b, v->accept, nir_imm_true(b), 0x1);
      nir_store_var(b, v->tmax, t, 0x1);
      nir_store_var(b, v->hit_kind, hit_kind, 0x1);

      if (ahit_count) {
         nir_def *prev_terminate = nir_load_var(b, v->terminate);
         nir_store_var(b, v->terminate, nir_imm_false(b), 0x1);

         nir_push_if(b, nir_inot(b, nir_load_var(b, v->opaque)));
         {
            nir_def *entry = nir_iadd_imm(
               b, nir_load_var(b, v->shader_record_ptr), -HK_RT_HANDLE_SIZE);
            nir_def *id =
               nir_load_global_constant(b, 1, 64, nir_iadd_imm(b, entry, 8));
            build_ladder(c, b, id, ahits, ahit_count);
         }
         nir_pop_if(b, NULL);

         nir_def *terminate = nir_load_var(b, v->terminate);
         nir_store_var(b, v->terminate, nir_ior(b, terminate, prev_terminate),
                       0x1);

         nir_push_if(b, terminate);
         nir_jump(b, nir_jump_return);
         nir_pop_if(b, NULL);
      }

      nir_push_if(b, nir_load_var(b, v->accept));
      {
         nir_store_var(b, commit, nir_imm_true(b), 0x1);
      }
      nir_push_else(b, NULL);
      {
         nir_store_var(b, v->accept, prev_accept, 0x1);
         nir_store_var(b, v->tmax, prev_tmax, 0x1);
         nir_store_var(b, v->hit_kind, prev_hit_kind, 0x1);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   return nir_load_var(b, commit);
}

static void
return_now(nir_builder *b)
{
   /* Inside an if, so that what follows stays reachable for the builder */
   nir_push_if(b, nir_imm_true(b));
   nir_jump(b, nir_jump_return);
   nir_pop_if(b, NULL);
}

static nir_def *
lower_rt_intrinsic(struct rt_clone *c, nir_builder *b,
                   nir_intrinsic_instr *intr)
{
   union rt_vars *v = &c->vars;

   switch (intr->intrinsic) {
   case nir_intrinsic_trace_ray:
      lower_trace_ray(c, b, intr);
      return NULL;
   case nir_intrinsic_execute_callable:
      lower_execute_callable(c, b, intr);
      return NULL;
   case nir_intrinsic_report_ray_intersection:
      return lower_report_intersection(c, b, intr);
   case nir_intrinsic_ignore_ray_intersection:
      nir_store_var(b, v->accept, nir_imm_false(b), 0x1);
      return_now(b);
      return NULL;
   case nir_intrinsic_terminate_ray:
      nir_store_var(b, v->accept, nir_imm_true(b), 0x1);
      nir_store_var(b, v->terminate, nir_imm_true(b), 0x1);
      return_now(b);
      return NULL;

   case nir_intrinsic_load_ray_launch_id:
      return nir_load_global_invocation_id(b, 32);
   case nir_intrinsic_load_ray_launch_size:
      return load_param(b, offsetof(VkTraceRaysIndirectCommand2KHR, width), 3,
                        32);
   case nir_intrinsic_load_shader_record_ptr:
      return nir_load_var(b, v->shader_record_ptr);
   case nir_intrinsic_load_ray_t_min:
      return nir_load_var(b, v->tmin);
   case nir_intrinsic_load_ray_t_max:
      return nir_load_var(b, v->tmax);
   case nir_intrinsic_load_ray_world_origin:
      return nir_load_var(b, v->origin);
   case nir_intrinsic_load_ray_world_direction:
      return nir_load_var(b, v->dir);
   case nir_intrinsic_load_ray_flags:
      return nir_load_var(b, v->flags);
   case nir_intrinsic_load_ray_hit_kind:
      return nir_load_var(b, v->hit_kind);
   case nir_intrinsic_load_primitive_id:
      return nir_load_var(b, v->primitive_id);
   case nir_intrinsic_load_ray_geometry_index:
      return nir_iand_imm(b, nir_load_var(b, v->geometry_id_and_flags),
                          0xFFFFFFF);
   case nir_intrinsic_load_cull_mask:
      return nir_iand_imm(b, nir_load_var(b, v->cull_mask), 0xFF);

   case nir_intrinsic_load_ray_instance_custom_index: {
      nir_def *addr = nir_iadd_imm(
         b, nir_load_var(b, v->instance_addr),
         offsetof(struct hk_bvh_instance_node, custom_instance_and_mask));
      return nir_iand_imm(b, nir_load_global(b, 1, 32, addr), 0xFFFFFF);
   }
   case nir_intrinsic_load_instance_id: {
      nir_def *addr =
         nir_iadd_imm(b, nir_load_var(b, v->instance_addr),
                      offsetof(struct hk_bvh_instance_node, instance_id));
      return nir_load_global(b, 1, 32, addr);
   }
   case nir_intrinsic_load_ray_world_to_object: {
      unsigned col = nir_intrinsic_column(intr);
      nir_def *wto_matrix[3];
      hk_load_wto_matrix(b, nir_load_var(b, v->instance_addr), NULL,
                         wto_matrix);

      nir_def *vals[3];
      for (unsigned i = 0; i < 3; ++i)
         vals[i] = nir_channel(b, wto_matrix[i], col);

      return nir_vec(b, vals, 3);
   }
   case nir_intrinsic_load_ray_object_to_world: {
      unsigned col = nir_intrinsic_column(intr);
      nir_def *instance_addr = nir_load_var(b, v->instance_addr);
      nir_def *rows[3];
      for (unsigned r = 0; r < 3; ++r) {
         rows[r] = nir_load_global(
            b, 4, 32,
            nir_iadd_imm(
               b, instance_addr,
               offsetof(struct hk_bvh_instance_node, otw_matrix) + r * 16));
      }
      return nir_vec3(b, nir_channel(b, rows[0], col),
                      nir_channel(b, rows[1], col),
                      nir_channel(b, rows[2], col));
   }
   case nir_intrinsic_load_ray_object_origin: {
      nir_def *wto_matrix[3];
      hk_load_wto_matrix(b, nir_load_var(b, v->instance_addr), NULL,
                         wto_matrix);
      return hk_mul_vec3_mat(b, nir_load_var(b, v->origin), wto_matrix, true);
   }
   case nir_intrinsic_load_ray_object_direction: {
      nir_def *wto_matrix[3];
      hk_load_wto_matrix(b, nir_load_var(b, v->instance_addr), NULL,
                         wto_matrix);
      return hk_mul_vec3_mat(b, nir_load_var(b, v->dir), wto_matrix, false);
   }
   case nir_intrinsic_load_ray_triangle_vertex_positions:
      return hk_load_vertex_position(b, nir_load_var(b, v->primitive_addr),
                                     nir_intrinsic_column(intr));

   default:
      UNREACHABLE("not a ray-tracing intrinsic");
   }
}

static bool
is_rt_intrinsic(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_trace_ray:
   case nir_intrinsic_execute_callable:
   case nir_intrinsic_report_ray_intersection:
   case nir_intrinsic_ignore_ray_intersection:
   case nir_intrinsic_terminate_ray:
   case nir_intrinsic_load_ray_launch_id:
   case nir_intrinsic_load_ray_launch_size:
   case nir_intrinsic_load_shader_record_ptr:
   case nir_intrinsic_load_ray_t_min:
   case nir_intrinsic_load_ray_t_max:
   case nir_intrinsic_load_ray_world_origin:
   case nir_intrinsic_load_ray_world_direction:
   case nir_intrinsic_load_ray_flags:
   case nir_intrinsic_load_ray_hit_kind:
   case nir_intrinsic_load_primitive_id:
   case nir_intrinsic_load_ray_geometry_index:
   case nir_intrinsic_load_cull_mask:
   case nir_intrinsic_load_ray_instance_custom_index:
   case nir_intrinsic_load_instance_id:
   case nir_intrinsic_load_ray_world_to_object:
   case nir_intrinsic_load_ray_object_to_world:
   case nir_intrinsic_load_ray_object_origin:
   case nir_intrinsic_load_ray_object_direction:
   case nir_intrinsic_load_ray_triangle_vertex_positions:
      return true;
   default:
      return false;
   }
}

/* What a stage reads of its ray as it starts: what it calls changes the
 * state after.
 */
static bool
is_entry_value(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_load_shader_record_ptr:
   case nir_intrinsic_load_ray_flags:
   case nir_intrinsic_load_ray_object_origin:
   case nir_intrinsic_load_ray_world_origin:
   case nir_intrinsic_load_ray_t_min:
   case nir_intrinsic_load_ray_object_direction:
   case nir_intrinsic_load_ray_world_direction:
   case nir_intrinsic_load_ray_t_max:
      return true;
   default:
      return false;
   }
}

/* The payloads this stage passes go to scratch at frame; those it receives
 * are at call_data's offset, the hit attributes at 0.
 */
static nir_variable *
lower_payloads(struct rt_clone *c, unsigned frame)
{
   nir_shader *s = c->shader;
   nir_function_impl *impl = nir_shader_get_entrypoint(s);

   NIR_PASS(_, s, nir_lower_global_vars_to_local);

   bool moved = false;
   nir_foreach_variable_with_modes(var, s, nir_var_shader_call_data_out) {
      var->data.mode = nir_var_shader_temp;
      moved = true;
   }

   if (moved)
      nir_fixup_deref_modes(s);

   s->scratch_size = frame;
   NIR_PASS(_, s, nir_lower_vars_to_explicit_types,
            nir_var_shader_temp | nir_var_shader_call_data_in |
               nir_var_ray_hit_attrib,
            glsl_get_natural_size_align_bytes);

   nir_variable *call_data =
      nir_local_variable_create(impl, glsl_uint_type(), "call_data");
   nir_builder b = nir_builder_create(impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;

         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (!nir_deref_mode_is_one_of(
                deref, nir_var_shader_call_data_in | nir_var_ray_hit_attrib))
            continue;

         bool is_call_data =
            nir_deref_mode_is(deref, nir_var_shader_call_data_in);
         deref->modes = nir_var_shader_temp;

         if (deref->deref_type == nir_deref_type_var) {
            b.cursor = nir_before_instr(&deref->instr);
            nir_def *offset =
               is_call_data ? nir_load_var(&b, call_data) : nir_imm_int(&b, 0);
            nir_deref_instr *cast = nir_build_deref_cast(
               &b, offset, nir_var_shader_temp, deref->var->type, 0);
            nir_def_replace(&deref->def, &cast->def);
         }
      }
   }

   nir_progress(true, impl, nir_metadata_control_flow);

   NIR_PASS(_, s, nir_lower_explicit_io, nir_var_shader_temp,
            nir_address_format_32bit_offset);

   c->frame_end = ALIGN_POT(s->scratch_size, 16);
   c->rb->scratch_end = MAX2(c->rb->scratch_end, c->frame_end);
   return call_data;
}

static void
inline_stage(struct rt_clone *parent, nir_builder *pb, uint32_t index)
{
   struct rt_builder *rb = parent->rb;
   nir_shader *s = nir_shader_clone(NULL, rb->info->stages[index].nir);
   nir_function_impl *impl = nir_shader_get_entrypoint(s);

   struct rt_clone c = {
      .rb = rb,
      .shader = s,
      .stage = s->info.stage,
      .index = index,
   };

   nir_variable *call_data = lower_payloads(&c, parent->frame_end);
   rt_vars_create(s, &c.vars);

   struct util_dynarray intrs;
   util_dynarray_init(&intrs, NULL);

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_jump) {
            nir_jump_instr *jump = nir_instr_as_jump(instr);
            if (jump->type == nir_jump_halt)
               jump->type = nir_jump_return;
         } else if (instr->type == nir_instr_type_intrinsic &&
                    is_rt_intrinsic(nir_instr_as_intrinsic(instr)->intrinsic)) {
            util_dynarray_append(&intrs, nir_instr_as_intrinsic(instr));
         }
      }
   }

   util_dynarray_foreach(&intrs, nir_intrinsic_instr *, intr) {
      if (is_entry_value((*intr)->intrinsic))
         nir_instr_move(nir_before_impl(impl), &(*intr)->instr);
   }

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_store_var(&b, call_data, nir_load_var(&b, c.vars.call_data), 0x1);

   util_dynarray_foreach(&intrs, nir_intrinsic_instr *, intr) {
      b.cursor = nir_before_instr(&(*intr)->instr);
      nir_def *def = lower_rt_intrinsic(&c, &b, *intr);

      if (def)
         nir_def_replace(&(*intr)->def, def);
      else
         nir_instr_remove(&(*intr)->instr);
   }

   util_dynarray_fini(&intrs);
   nir_progress(true, impl, nir_metadata_none);

   NIR_PASS(_, s, nir_lower_continue_constructs);
   NIR_PASS(_, s, nir_lower_returns);

   struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
   for (unsigned i = 0; i < ARRAY_SIZE(c.vars.all); i++)
      _mesa_hash_table_insert(remap, c.vars.all[i], parent->vars.all[i]);

   nir_inline_function_impl(pb, impl, NULL, remap);

   /* The inlined defs keep the clone's indices */
   nir_index_ssa_defs(pb->impl);

   _mesa_hash_table_destroy(remap, NULL);
   ralloc_free(s);
}

/* Whether a stage traces rays and whether it calls callable shaders */
static void
stage_calls(const nir_shader *nir, bool *traces, bool *calls)
{
   *traces = false;
   *calls = false;

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            *traces |= op == nir_intrinsic_trace_ray;
            *calls |= op == nir_intrinsic_execute_callable;
         }
      }
   }
}

bool
hk_nir_rt_needs_dispatcher(const struct hk_rt_kernel_info *info)
{
   for (uint32_t i = 0; i < info->stage_count; i++) {
      const nir_shader *nir = info->stages[i].nir;
      bool traces, calls;
      stage_calls(nir, &traces, &calls);

      switch (nir->info.stage) {
      case MESA_SHADER_CALLABLE:
         /* Inlining would not end: it may call itself, or trace into a
          * closest-hit shader that calls it
          */
         if (traces || calls)
            return true;
         break;

      case MESA_SHADER_CLOSEST_HIT:
      case MESA_SHADER_MISS:
         /* At depth 1 the trace never runs (lower_trace_ray drops it) */
         if (traces && info->max_recursion_depth > 1)
            return true;
         break;

      default:
         break;
      }
   }

   return false;
}

nir_shader *
hk_nir_build_rt_kernel(const nir_shader_compiler_options *options,
                       const struct hk_rt_kernel_info *info)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, options,
                                                  "hk-ray-tracing-pipeline");
   nir_shader *s = b.shader;
   s->info.workgroup_size[0] = 8;
   s->info.workgroup_size[1] = 4;
   s->info.workgroup_size[2] = 1;

   struct rt_builder rb = {
      .info = info,
      .scratch_end = HIT_ATTRIBS_SIZE,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(rb.by_stage); i++)
      util_dynarray_init(&rb.by_stage[i], NULL);

   for (uint32_t i = 0; i < info->stage_count; i++) {
      mesa_shader_stage stage = info->stages[i].nir->info.stage;
      assert(mesa_shader_stage_is_rt(stage));
      util_dynarray_append(&rb.by_stage[stage], i);
   }

   struct rt_clone kernel = {
      .rb = &rb,
      .shader = s,
      .stage = MESA_SHADER_COMPUTE,
      .index = HK_RT_UNUSED,
      .frame_end = HIT_ATTRIBS_SIZE,
   };
   rt_vars_create(s, &kernel.vars);

   nir_def *launch_id = nir_load_global_invocation_id(&b, 32);
   nir_def *launch_size =
      load_param(&b, offsetof(VkTraceRaysIndirectCommand2KHR, width), 3, 32);

   nir_push_if(&b, nir_bany(&b, nir_uge(&b, launch_id, launch_size)));
   nir_jump(&b, nir_jump_return);
   nir_pop_if(&b, NULL);

   nir_def *id = sbt_lookup(
      &kernel, &b,
      offsetof(VkTraceRaysIndirectCommand2KHR, raygenShaderRecordAddress), NULL,
      0);
   build_stage_ladder(&kernel, &b, id, MESA_SHADER_RAYGEN);

   for (unsigned i = 0; i < ARRAY_SIZE(rb.by_stage); i++)
      util_dynarray_fini(&rb.by_stage[i]);

   NIR_PASS(_, s, nir_lower_continue_constructs);
   NIR_PASS(_, s, nir_lower_returns);

   s->scratch_size = rb.scratch_end;
   NIR_PASS(_, s, nir_lower_global_vars_to_local);
   NIR_PASS(_, s, nir_lower_vars_to_ssa);
   NIR_PASS(_, s, nir_lower_scratch_to_var);
   NIR_PASS(_, s, nir_remove_dead_variables,
            nir_var_shader_temp | nir_var_function_temp, NULL);

   return s;
}

/*
 * Tier 2: a dispatcher loop, as RADV's first ray-tracing pipelines were one
 * shader.
 *
 * Each ray generation, closest-hit, miss and callable shader is split at its
 * calls by nir_lower_shader_calls, and each part is a case of a loop:
 *
 *    next = the ray generation shader's case
 *    loop { if next == 0 break; if next == k { case k } ... }
 *
 * A call pushes the case that resumes the caller onto the thread-private
 * stack and sets next to the callee: the traversal, a case of its own with
 * the any-hit and intersection shaders inlined as in tier 1, or a callable
 * shader. A stage's end pops the case to resume. A stage's frame holds what
 * lives across its calls: the payloads it passes, its locals indexed
 * dynamically and the values nir_lower_shader_calls spills, then the return
 * slot; its callee's frame follows. The frames start past the kernel's own
 * stack, at a base the dispatch passes with the stack's end, which the
 * specTPR allocates from the pipeline's stack size.
 */

/* Cases: 0 ends the thread, 1 is the traversal, the stages' parts follow */
#define CASE_DONE      0
#define CASE_TRAVERSAL 1

/* A frame's last 16 B: the case that resumes it */
#define RETURN_SLOT 16

struct rt_split {
   /* The entry, then the part resuming after each call */
   nir_shader **parts;
   unsigned count;

   /* The frame, return slot included: hk_nir_rt_stage_stack_size */
   unsigned frame;
};

struct rt_part {
   nir_shader *nir;

   /* Its stage's index, and its own: 0 the entry */
   uint32_t stage;
   uint32_t index;
};

/* A closest-hit shader that calls reads its hit attributes from a copy taken
 * as it starts: a nested trace writes the kernel's.
 */
static void
copy_hit_attribs(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   bool progress = false;

   nir_foreach_variable_with_modes(var, s, nir_var_ray_hit_attrib) {
      nir_variable *copy =
         nir_local_variable_create(impl, var->type, "hit_attribs");

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;

            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->deref_type == nir_deref_type_var && deref->var == var)
               deref->var = copy;
         }
      }

      nir_builder b = nir_builder_at(nir_before_impl(impl));
      nir_copy_var(&b, copy, var);
      progress = true;
   }

   if (!progress)
      return;

   nir_fixup_deref_modes(s);
   nir_progress(true, impl, nir_metadata_control_flow);
   NIR_PASS(_, s, nir_lower_var_copies);
}

/* The incoming payload sits in the caller's frame, at an offset from this
 * stage's frame that the caller passes. Read once as the stage starts, the
 * offset is spilled across its calls as any value (RADV's CPS lowering).
 */
static bool
lower_call_data_in(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_create(impl);
   nir_def *arg = NULL;
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;

         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type != nir_deref_type_var ||
             !nir_deref_mode_is(deref, nir_var_shader_call_data_in))
            continue;

         if (arg == NULL) {
            b.cursor = nir_before_impl(impl);
            arg = nir_load_rt_arg_scratch_offset_amd(&b);
         }

         b.cursor = nir_before_instr(instr);
         nir_deref_instr *cast = nir_build_deref_cast(
            &b, arg, nir_var_shader_call_data_in, deref->var->type, 0);
         nir_def_replace(&deref->def, &cast->def);
         progress = true;
      }
   }

   return nir_progress(progress, impl, nir_metadata_control_flow);
}

/* The ray-tracing state a stage reads, which its calls change */
static bool
is_ray_state(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_trace_ray:
   case nir_intrinsic_execute_callable:
   case nir_intrinsic_report_ray_intersection:
   case nir_intrinsic_ignore_ray_intersection:
   case nir_intrinsic_terminate_ray:
   case nir_intrinsic_load_ray_launch_id:
   case nir_intrinsic_load_ray_launch_size:
      return false;
   default:
      return is_rt_intrinsic(op);
   }
}

/* The state is the caller's until this stage calls, so it is read as the
 * stage starts and spilled across the calls, like any value (RADV's
 * move_rt_instructions, for all of it). None takes a source.
 */
static void
hoist_ray_state(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   struct util_dynarray moved;
   util_dynarray_init(&moved, NULL);

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             is_ray_state(nir_instr_as_intrinsic(instr)->intrinsic))
            util_dynarray_append(&moved, instr);
      }
   }

   util_dynarray_foreach(&moved, nir_instr *, instr)
      nir_instr_move(nir_before_impl(impl), *instr);

   nir_progress(true, impl, nir_metadata_control_flow);
   util_dynarray_fini(&moved);
}

/* A ray generation, closest-hit, miss or callable shader in parts, its
 * frame-relative scratch at offsets from its frame's base
 */
static struct rt_split
split_stage(void *mem_ctx, const nir_shader *nir)
{
   nir_shader *s = nir_shader_clone(mem_ctx, nir);
   bool traces, calls;
   stage_calls(s, &traces, &calls);
   bool calling = traces || calls;

   /* Every way out reaches the end, where the stage returns */
   NIR_PASS(_, s, nir_lower_returns);

   if (calling && s->info.stage == MESA_SHADER_CLOSEST_HIT)
      copy_hit_attribs(s);

   /* The payloads it passes in its frame, the one it gets in its caller's */
   NIR_PASS(_, s, nir_remove_dead_variables, nir_var_shader_call_data_out,
            NULL);
   s->scratch_size = 0;
   NIR_PASS(_, s, nir_lower_vars_to_explicit_types,
            nir_var_shader_call_data_in | nir_var_shader_call_data_out,
            glsl_get_natural_size_align_bytes);
   NIR_PASS(_, s, lower_call_data_in);
   NIR_PASS(_, s, nir_lower_explicit_io,
            nir_var_shader_call_data_in | nir_var_shader_call_data_out,
            nir_address_format_32bit_offset);

   /* nir_lower_shader_calls keeps SSA values across calls: locals indexed
    * dynamically go to the frame. A leaf's stay variables, for the backend.
    * Ray queries live across a call are not kept (they stay variables).
    */
   if (calling) {
      NIR_PASS(_, s, nir_lower_vars_to_ssa);
      NIR_PASS(_, s, nir_lower_vars_to_scratch, 0,
               glsl_get_natural_size_align_bytes,
               glsl_get_natural_size_align_bytes);
   }

   hoist_ray_state(s);

   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(s)));
   nir_rt_return_amd(&b);

   struct rt_split split = {
      .count = 1,
      .frame = ALIGN_POT(s->scratch_size, 16),
   };

   nir_shader **resume = NULL;
   uint32_t resume_count = 0;

   if (calling) {
      const nir_lower_shader_calls_options options = {
         .address_format = nir_address_format_32bit_offset,
         .stack_alignment = 16,
         .localized_loads = true,
      };

      nir_lower_shader_calls(s, &options, &resume, &resume_count, mem_ctx);
      split.count += resume_count;
   }

   split.parts = ralloc_array(mem_ctx, nir_shader *, split.count);
   split.parts[0] = s;
   for (uint32_t i = 0; i < resume_count; i++)
      split.parts[1 + i] = resume[i];

   /* Each call's stack size covers what the frame holds at that call */
   for (uint32_t i = 0; i < split.count; i++) {
      nir_foreach_function_impl(impl, split.parts[i]) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;

               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_rt_trace_ray ||
                   intr->intrinsic == nir_intrinsic_rt_execute_callable) {
                  split.frame =
                     MAX2(split.frame,
                          ALIGN_POT(nir_intrinsic_stack_size(intr), 16) +
                             RETURN_SLOT);
               }
            }
         }
      }
   }

   return split;
}

unsigned
hk_nir_rt_stage_stack_size(const nir_shader *nir)
{
   switch (nir->info.stage) {
   case MESA_SHADER_RAYGEN:
   case MESA_SHADER_CLOSEST_HIT:
   case MESA_SHADER_MISS:
   case MESA_SHADER_CALLABLE:
      break;
   default:
      /* Inlined into the traversal, on the kernel's own stack */
      return 0;
   }

   void *mem_ctx = ralloc_context(NULL);
   unsigned frame = split_stage(mem_ctx, nir).frame;
   ralloc_free(mem_ctx);

   return frame;
}

/* The case of the stage among these whose ID is id, CASE_DONE for none */
static nir_def *
select_case(struct rt_clone *c, nir_builder *b, nir_def *id,
            mesa_shader_stage stage)
{
   const struct hk_rt_stage *all = c->rb->info->stages;
   nir_def *next = nir_imm_int(b, CASE_DONE);

   util_dynarray_foreach(&c->rb->by_stage[stage], uint32_t, i) {
      next = nir_bcsel(b, nir_ieq_imm(b, id, all[*i].id),
                       nir_imm_int(b, c->rb->first_case[*i]), next);
   }

   return next;
}

/* Back to the caller: the case on top of the stack */
static void
build_return(struct rt_clone *c, nir_builder *b)
{
   nir_def *sp =
      nir_iadd_imm(b, nir_load_var(b, c->vars.stack_ptr), -RETURN_SLOT);

   nir_store_var(b, c->vars.stack_ptr, sp, 0x1);
   nir_store_var(b, c->vars.next,
                 nir_load_scratch(b, 1, 32, sp, .align_mul = 16), 0x1);
}

/* To the callee's case; a null or unknown callee returns at once */
static void
goto_case(struct rt_clone *c, nir_builder *b, nir_def *next)
{
   nir_store_var(b, c->vars.next, next, 0x1);

   nir_push_if(b, nir_ieq_imm(b, next, CASE_DONE));
   build_return(c, b);
   nir_pop_if(b, NULL);
}

/* Pushes the case that resumes after this call and moves the frame past its
 * return slot, the payload relative to the callee's frame
 */
static void
push_call(struct rt_clone *c, nir_builder *b, nir_intrinsic_instr *intr,
          nir_def *payload)
{
   unsigned size = ALIGN_POT(nir_intrinsic_stack_size(intr), 16);
   uint32_t resume =
      c->rb->first_case[c->index] + nir_intrinsic_call_idx(intr) + 1;
   nir_def *sp = nir_load_var(b, c->vars.stack_ptr);

   nir_store_scratch(b, nir_imm_int(b, resume), nir_iadd_imm(b, sp, size),
                     .align_mul = 16);
   nir_store_var(b, c->vars.stack_ptr, nir_iadd_imm(b, sp, size + RETURN_SLOT),
                 0x1);
   nir_store_var(b, c->vars.arg,
                 nir_iadd_imm(b, payload, -(int64_t)(size + RETURN_SLOT)), 0x1);
}

static nir_def *
lower_part_intrinsic(struct rt_clone *c, nir_builder *b,
                     nir_intrinsic_instr *intr)
{
   union rt_vars *v = &c->vars;

   switch (intr->intrinsic) {
   case nir_intrinsic_rt_trace_ray:
      push_call(c, b, intr, intr->src[10].ssa);
      store_ray(c, b, intr);
      nir_store_var(b, v->next, nir_imm_int(b, CASE_TRAVERSAL), 0x1);
      return NULL;

   case nir_intrinsic_rt_execute_callable: {
      push_call(c, b, intr, intr->src[1].ssa);
      nir_def *id = sbt_lookup(c, b,
                               offsetof(VkTraceRaysIndirectCommand2KHR,
                                        callableShaderBindingTableAddress),
                               intr->src[0].ssa, 0);
      goto_case(c, b, select_case(c, b, id, MESA_SHADER_CALLABLE));
      return NULL;
   }

   case nir_intrinsic_rt_resume: {
      /* The callee popped the return slot */
      int size = ALIGN_POT(nir_intrinsic_stack_size(intr), 16);
      nir_store_var(b, v->stack_ptr,
                    nir_iadd_imm(b, nir_load_var(b, v->stack_ptr), -size), 0x1);
      return NULL;
   }

   case nir_intrinsic_rt_return_amd:
      if (c->stage == MESA_SHADER_RAYGEN)
         nir_store_var(b, v->next, nir_imm_int(b, CASE_DONE), 0x1);
      else
         build_return(c, b);
      return NULL;

   case nir_intrinsic_load_rt_arg_scratch_offset_amd:
      return nir_load_var(b, v->arg);

   default:
      return lower_rt_intrinsic(c, b, intr);
   }
}

static bool
is_part_intrinsic(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_rt_trace_ray:
   case nir_intrinsic_rt_execute_callable:
   case nir_intrinsic_rt_resume:
   case nir_intrinsic_rt_return_amd:
   case nir_intrinsic_load_rt_arg_scratch_offset_amd:
      return true;
   default:
      return is_rt_intrinsic(op);
   }
}

/* The hit attributes are the kernel's, at 0 (tier 1's lower_payloads) */
static void
lower_hit_attribs(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_create(impl);
   bool progress = false;

   NIR_PASS(_, s, nir_lower_vars_to_explicit_types, nir_var_ray_hit_attrib,
            glsl_get_natural_size_align_bytes);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;

         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (!nir_deref_mode_is(deref, nir_var_ray_hit_attrib))
            continue;

         deref->modes = nir_var_shader_temp;
         progress = true;

         if (deref->deref_type == nir_deref_type_var) {
            b.cursor = nir_before_instr(&deref->instr);
            nir_deref_instr *cast =
               nir_build_deref_cast(&b, nir_imm_int(&b, 0), nir_var_shader_temp,
                                    deref->var->type, 0);
            nir_def_replace(&deref->def, &cast->def);
         }
      }
   }

   if (!nir_progress(progress, impl, nir_metadata_control_flow))
      return;

   NIR_PASS(_, s, nir_lower_explicit_io, nir_var_shader_temp,
            nir_address_format_32bit_offset);
}

static void
inline_part(struct rt_clone *k, nir_builder *kb, const struct rt_part *p)
{
   nir_shader *s = p->nir;
   nir_function_impl *impl = nir_shader_get_entrypoint(s);

   struct rt_clone c = {
      .rb = k->rb,
      .shader = s,
      .stage = s->info.stage,
      .index = p->stage,
   };

   /* The frame's accesses, before anything adds others */
   struct util_dynarray frame_ops;
   util_dynarray_init(&frame_ops, NULL);

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_scratch ||
             intr->intrinsic == nir_intrinsic_store_scratch)
            util_dynarray_append(&frame_ops, intr);
      }
   }

   /* Before the state's variables, which are shader_temp too */
   lower_hit_attribs(s);
   rt_vars_create(s, &c.vars);

   nir_builder b = nir_builder_create(impl);

   util_dynarray_foreach(&frame_ops, nir_intrinsic_instr *, op) {
      nir_intrinsic_instr *intr = *op;
      nir_src *offset =
         &intr->src[intr->intrinsic == nir_intrinsic_store_scratch ? 1 : 0];

      b.cursor = nir_before_instr(&intr->instr);
      nir_src_rewrite(
         offset, nir_iadd(&b, nir_load_var(&b, c.vars.stack_ptr), offset->ssa));

      /* A frame is only 16 B aligned */
      unsigned align_mul = MIN2(nir_intrinsic_align_mul(intr), 16);
      nir_intrinsic_set_align(intr, align_mul,
                              nir_intrinsic_align_offset(intr) % align_mul);
   }

   util_dynarray_fini(&frame_ops);

   struct util_dynarray intrs;
   util_dynarray_init(&intrs, NULL);

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_jump) {
            /* The halts after calls end the case */
            nir_jump_instr *jump = nir_instr_as_jump(instr);
            if (jump->type == nir_jump_halt)
               jump->type = nir_jump_return;
         } else if (instr->type == nir_instr_type_intrinsic &&
                    is_part_intrinsic(
                       nir_instr_as_intrinsic(instr)->intrinsic)) {
            util_dynarray_append(&intrs, nir_instr_as_intrinsic(instr));
         }
      }
   }

   util_dynarray_foreach(&intrs, nir_intrinsic_instr *, intr) {
      b.cursor = nir_before_instr(&(*intr)->instr);
      nir_def *def = lower_part_intrinsic(&c, &b, *intr);

      if (def)
         nir_def_replace(&(*intr)->def, def);
      else
         nir_instr_remove(&(*intr)->instr);
   }

   util_dynarray_fini(&intrs);
   nir_progress(true, impl, nir_metadata_none);

   NIR_PASS(_, s, nir_lower_continue_constructs);
   NIR_PASS(_, s, nir_lower_returns);

   struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
   for (unsigned i = 0; i < ARRAY_SIZE(c.vars.all); i++)
      _mesa_hash_table_insert(remap, c.vars.all[i], k->vars.all[i]);

   nir_inline_function_impl(kb, impl, NULL, remap);
   nir_index_ssa_defs(kb->impl);

   _mesa_hash_table_destroy(remap, NULL);
}

static void
build_traversal_case(struct rt_clone *k, nir_builder *b)
{
   union rt_vars *v = &k->vars;

   /* The any-hit and intersection shaders, inlined as in tier 1, find the
    * payload at an absolute offset
    */
   nir_store_var(
      b, v->call_data,
      nir_iadd(b, nir_load_var(b, v->stack_ptr), nir_load_var(b, v->arg)), 0x1);

   build_walk(k, b);

   nir_push_if(b, nir_load_var(b, k->trav.hit));
   {
      nir_def *skip_chit = nir_test_mask(
         b, nir_load_var(b, v->flags), SpvRayFlagsSkipClosestHitShaderKHRMask);
      nir_push_if(b, skip_chit);
      {
         build_return(k, b);
      }
      nir_push_else(b, NULL);
      {
         nir_def *id = sbt_lookup(k, b,
                                  offsetof(VkTraceRaysIndirectCommand2KHR,
                                           hitShaderBindingTableAddress),
                                  nir_load_var(b, v->sbt_index), 0);
         goto_case(k, b, select_case(k, b, id, MESA_SHADER_CLOSEST_HIT));
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      nir_def *id = sbt_lookup(k, b,
                               offsetof(VkTraceRaysIndirectCommand2KHR,
                                        missShaderBindingTableAddress),
                               nir_load_var(b, v->miss_index), 0);
      goto_case(k, b, select_case(k, b, id, MESA_SHADER_MISS));
   }
   nir_pop_if(b, NULL);
}

/* The parts that are entries or not, of ray generation shaders or not, in
 * the loop's order
 */
static void
build_cases(struct rt_clone *k, nir_builder *b, struct util_dynarray *parts,
            bool resume, bool raygen, nir_def *stack_end)
{
   const struct rt_builder *rb = k->rb;

   util_dynarray_foreach(parts, struct rt_part, p) {
      if ((p->index > 0) != resume)
         continue;

      if (!resume && (p->nir->info.stage == MESA_SHADER_RAYGEN) != raygen)
         continue;

      uint32_t id = rb->first_case[p->stage] + p->index;
      nir_push_if(b, nir_ieq_imm(b, nir_load_var(b, k->vars.next), id));

      /* A stage whose frame would pass the stack's end stops its thread:
       * deeper than the stack size, which the spec leaves undefined, or than
       * the specTPR can give (hk_rt_stack_end). Its frame stays within the
       * thread's memory.
       */
      uint32_t frame = rb->frame[p->stage];
      if (p->index == 0 && frame > 0) {
         nir_def *end =
            nir_iadd_imm(b, nir_load_var(b, k->vars.stack_ptr), frame);
         nir_push_if(b, nir_ult(b, stack_end, end));
         nir_store_var(b, k->vars.next, nir_imm_int(b, CASE_DONE), 0x1);
         nir_push_else(b, NULL);
      }

      inline_part(k, b, p);

      if (p->index == 0 && frame > 0)
         nir_pop_if(b, NULL);

      nir_pop_if(b, NULL);
   }
}

nir_shader *
hk_nir_build_rt_dispatcher(const nir_shader_compiler_options *options,
                           const struct hk_rt_kernel_info *info)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, options,
                                                  "hk-ray-tracing-dispatcher");
   nir_shader *s = b.shader;
   s->info.workgroup_size[0] = 8;
   s->info.workgroup_size[1] = 4;
   s->info.workgroup_size[2] = 1;

   /* The application's code: its scratch stays scratch (an internal shader's
    * goes back to variables in agx_preprocess_nir, which cannot hold the
    * frames' moving offsets)
    */
   s->info.internal = false;

   void *mem_ctx = ralloc_context(NULL);
   uint32_t stage_count = MAX2(info->stage_count, 1);

   struct rt_builder rb = {
      .info = info,
      .scratch_end = HIT_ATTRIBS_SIZE,
      .first_case = rzalloc_array(mem_ctx, uint32_t, stage_count),
      .frame = rzalloc_array(mem_ctx, uint32_t, stage_count),
   };

   for (unsigned i = 0; i < ARRAY_SIZE(rb.by_stage); i++)
      util_dynarray_init(&rb.by_stage[i], mem_ctx);

   struct util_dynarray parts;
   util_dynarray_init(&parts, mem_ctx);
   uint32_t next_case = CASE_TRAVERSAL + 1;
   bool any_traces = false;

   for (uint32_t i = 0; i < info->stage_count; i++) {
      const nir_shader *nir = info->stages[i].nir;
      mesa_shader_stage stage = nir->info.stage;
      assert(mesa_shader_stage_is_rt(stage));
      util_dynarray_append(&rb.by_stage[stage], i);

      bool traces, calls;
      stage_calls(nir, &traces, &calls);
      any_traces |= traces;

      if (stage == MESA_SHADER_ANY_HIT || stage == MESA_SHADER_INTERSECTION)
         continue;

      struct rt_split split = split_stage(mem_ctx, nir);
      rb.first_case[i] = next_case;
      rb.frame[i] = split.frame;

      for (uint32_t j = 0; j < split.count; j++) {
         struct rt_part part = {split.parts[j], i, j};
         util_dynarray_append(&parts, part);
      }

      next_case += split.count;
   }

   struct rt_clone kernel = {
      .rb = &rb,
      .shader = s,
      .stage = MESA_SHADER_COMPUTE,
      .index = HK_RT_UNUSED,
      .frame_end = HIT_ATTRIBS_SIZE,
   };
   rt_vars_create(s, &kernel.vars);
   union rt_vars *v = &kernel.vars;

   nir_def *launch_id = nir_load_global_invocation_id(&b, 32);
   nir_def *launch_size =
      load_param(&b, offsetof(VkTraceRaysIndirectCommand2KHR, width), 3, 32);

   nir_push_if(&b, nir_bany(&b, nir_uge(&b, launch_id, launch_size)));
   nir_jump(&b, nir_jump_return);
   nir_pop_if(&b, NULL);

   nir_def *stack = nir_load_rt_stack_agx(&b);
   nir_store_var(&b, v->stack_ptr, nir_channel(&b, stack, 0), 0x1);

   nir_def *id = sbt_lookup(
      &kernel, &b,
      offsetof(VkTraceRaysIndirectCommand2KHR, raygenShaderRecordAddress), NULL,
      0);
   nir_store_var(&b, v->next, select_case(&kernel, &b, id, MESA_SHADER_RAYGEN),
                 0x1);

   nir_push_loop(&b);
   {
      nir_break_if(&b, nir_ieq_imm(&b, nir_load_var(&b, v->next), CASE_DONE));

      /* In the order a trace's cases follow each other, so that one pass
       * runs a trace with its callee and its return when the lanes agree
       */
      nir_def *stack_end = nir_channel(&b, stack, 1);
      build_cases(&kernel, &b, &parts, false, true, stack_end);

      if (any_traces) {
         nir_push_if(
            &b, nir_ieq_imm(&b, nir_load_var(&b, v->next), CASE_TRAVERSAL));
         build_traversal_case(&kernel, &b);
         nir_pop_if(&b, NULL);
      }

      build_cases(&kernel, &b, &parts, false, false, stack_end);
      build_cases(&kernel, &b, &parts, true, false, stack_end);
   }
   nir_pop_loop(&b, NULL);

   NIR_PASS(_, s, nir_lower_continue_constructs);
   NIR_PASS(_, s, nir_lower_returns);

   /* The hit attributes and the inlined any-hit and intersection shaders'
    * frames; the frames are past the kernel's stack
    */
   s->scratch_size = rb.scratch_end;
   NIR_PASS(_, s, nir_lower_global_vars_to_local);
   NIR_PASS(_, s, nir_lower_vars_to_ssa);
   NIR_PASS(_, s, nir_remove_dead_variables,
            nir_var_shader_temp | nir_var_function_temp, NULL);

   ralloc_free(mem_ctx);
   return s;
}
