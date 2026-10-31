/*
 * Copyright © 2021 Google
 * Copyright © 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef HK_NIR_RAY_TRACING_H
#define HK_NIR_RAY_TRACING_H

#include "nir/nir.h"
#include "nir/nir_builder.h"

#include <vulkan/vulkan_core.h>

nir_def *hk_mul_vec3_mat(nir_builder *b, nir_def *vec, nir_def *matrix[], bool translation);

void hk_load_wto_matrix(nir_builder *b, nir_def *instance_addr, nir_def **node_data, nir_def **out);

nir_def *hk_load_vertex_position(nir_builder *b, nir_def *primitive_addr, uint32_t index);

struct hk_ray_traversal_args;

struct hk_ray_flags {
   nir_def *force_opaque;
   nir_def *force_not_opaque;
   nir_def *terminate_on_first_hit;
   nir_def *no_cull_front;
   nir_def *no_cull_back;
   nir_def *no_cull_opaque;
   nir_def *no_cull_no_opaque;
   nir_def *no_skip_triangles;
   nir_def *no_skip_aabbs;
};

struct hk_leaf_intersection {
   nir_def *node_addr;
   nir_def *primitive_id;
   nir_def *geometry_id_and_flags;
   nir_def *opaque;
};

typedef void (*hk_aabb_intersection_cb)(nir_builder *b, struct hk_leaf_intersection *intersection,
                                         const struct hk_ray_traversal_args *args,
                                         const struct hk_ray_flags *ray_flags);

struct hk_triangle_intersection {
   struct hk_leaf_intersection base;

   nir_def *t;
   nir_def *frontface;
   nir_def *barycentrics;
};

typedef void (*hk_triangle_intersection_cb)(nir_builder *b,
                                             struct hk_triangle_intersection *intersection,
                                             const struct hk_ray_traversal_args *args,
                                             const struct hk_ray_flags *ray_flags);

struct hk_ray_traversal_vars {
   /* For each accepted hit, tmax will be set to the t value. This allows for automatic intersection
    * culling.
    */
   nir_deref_instr *tmax;

   /* Those variables change when entering and exiting BLASes. */
   nir_deref_instr *origin;
   nir_deref_instr *dir;
   nir_deref_instr *inv_dir;

   /* The base address of the current TLAS/BLAS. */
   nir_deref_instr *bvh_base;

   nir_deref_instr *current_node;

   nir_deref_instr *stack_base;
   nir_deref_instr *stack_ptr;
   nir_deref_instr *stack;

   /* Information about the current instance used for culling. */
   nir_deref_instr *instance_addr;
   nir_deref_instr *sbt_offset_and_flags;
};

struct hk_ray_traversal_args {
   nir_def *root_bvh_base;
   nir_def *flags;
   nir_def *cull_mask;
   nir_def *origin;
   nir_def *tmin;
   nir_def *dir;

   struct hk_ray_traversal_vars vars;

   /* vars.stack's entries */
   unsigned stack_size;

   hk_aabb_intersection_cb aabb_cb;
   hk_triangle_intersection_cb triangle_cb;

   void *data;
};

/* Builds the ray traversal loop and returns whether traversal is incomplete, similar to
 * rayQueryProceedEXT. Traversal will only be considered incomplete, if one of the specified
 * callbacks breaks out of the traversal loop.
 */
nir_def *hk_build_ray_traversal(nir_builder *b, const struct hk_ray_traversal_args *args);

bool hk_nir_lower_ray_queries(struct nir_shader *shader);

/* A ray-tracing pipeline's kernel (hk_nir_rt_pipeline.c) */

/* A group handle: the IDs of its general or closest-hit, any-hit and
 * intersection shaders, 0 for none, and a zero. The shader record follows.
 */
#define HK_RT_HANDLE_SIZE 32

#define HK_RT_UNUSED (~0u)

struct hk_rt_stage {
   nir_shader *nir;
   uint64_t id;
};

/* Indices into hk_rt_kernel_info::stages, or HK_RT_UNUSED */
struct hk_rt_group {
   uint32_t general;
   uint32_t closest_hit;
   uint32_t any_hit;
   uint32_t intersection;
};

struct hk_rt_kernel_info {
   /* One a shader ID */
   const struct hk_rt_stage *stages;
   uint32_t stage_count;

   const struct hk_rt_group *groups;
   uint32_t group_count;

   bool skip_triangles;
   bool skip_aabbs;
};

bool hk_nir_rt_kernel_supported(const struct hk_rt_kernel_info *info);

nir_shader *hk_nir_build_rt_kernel(const nir_shader_compiler_options *options,
                                   const struct hk_rt_kernel_info *info);

#endif
