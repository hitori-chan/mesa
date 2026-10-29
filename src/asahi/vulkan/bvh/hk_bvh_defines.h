/*
 * Copyright © 2021 Google
 * Copyright © 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 *
 * From lavapipe's lvp_acceleration_structure.h.
 */

/* Honeykrisp's acceleration structures, shared by the encode shader and the
 * driver. A binary tree in lavapipe's node layout, walked in software by
 * hk_nir_ray_tracing.c: the header, the box nodes (the root first), then the
 * leaves, padded to a box node's size, since the walk loads 56 bytes of every
 * node. A node reference is the node's byte offset from the header with its
 * type (hk_bvh_node_*) in the low two bits, so a structure is position
 * independent: a copy or compaction is a copy of its bytes.
 */

#ifndef HK_BVH_DEFINES_H
#define HK_BVH_DEFINES_H

#ifdef VULKAN
#define VK_UUID_SIZE 16
#else
#include <vulkan/vulkan.h>
#endif

#include "vk_bvh_defines.h"

#define HK_GEOMETRY_OPAQUE (1u << 31)

#define HK_INSTANCE_FORCE_OPAQUE                 (1u << 31)
#define HK_INSTANCE_NO_FORCE_NOT_OPAQUE          (1u << 30)
#define HK_INSTANCE_TRIANGLE_FACING_CULL_DISABLE (1u << 29)
#define HK_INSTANCE_TRIANGLE_FLIP_FACING         (1u << 28)

#define hk_bvh_node_triangle 0
#define hk_bvh_node_internal 1
#define hk_bvh_node_instance 2
#define hk_bvh_node_aabb     3

/* 48 bytes */
struct hk_bvh_triangle_node {
   float coords[3][3];
   uint32_t padding;
   uint32_t primitive_id;
   /* flags in the upper 4 bits */
   uint32_t geometry_id_and_flags;
};

/* 32 bytes */
struct hk_bvh_aabb_node {
   vk_aabb bounds;
   uint32_t primitive_id;
   /* flags in the upper 4 bits */
   uint32_t geometry_id_and_flags;
};

/* 120 bytes */
struct hk_bvh_instance_node {
   /* the bottom level's header; its root is at HK_BVH_ROOT_NODE */
   uint64_t bvh_ptr;
   /* lower 24 bits the custom instance index, upper 8 bits the visibility mask */
   uint32_t custom_instance_and_mask;
   /* lower 24 bits the SBT offset, upper 8 bits VkGeometryInstanceFlagsKHR */
   uint32_t sbt_offset_and_flags;
   mat3x4 wto_matrix;
   uint32_t padding;
   uint32_t instance_id;
   /* the object-to-world matrix, transposed from the instance's transform */
   mat3x4 otw_matrix;
};

/* 56 bytes; a child is HK_BVH_INVALID_NODE when absent */
struct hk_bvh_box_node {
   vk_aabb bounds[2];
   uint32_t children[2];
};

/* 64 bytes */
struct hk_bvh_header {
   vk_aabb bounds;
   uint64_t compacted_size;
   uint64_t serialization_size;
   uint64_t instance_count;
   uint32_t leaf_nodes_offset;
   uint32_t padding[3];
};

#ifdef VULKAN
TYPE(hk_bvh_header, 8);
TYPE(hk_bvh_box_node, 4);
TYPE(hk_bvh_triangle_node, 4);
TYPE(hk_bvh_aabb_node, 4);
TYPE(hk_bvh_instance_node, 8);
#endif

/* The root box node is the first node after the header; an empty structure's
 * has no child.
 */
#define HK_BVH_ROOT_NODE_OFFSET 64
#define HK_BVH_ROOT_NODE        (HK_BVH_ROOT_NODE_OFFSET | hk_bvh_node_internal)
#define HK_BVH_INVALID_NODE     0xFFFFFFFF

/* The walk loads this much of every node: a box node. */
#define HK_BVH_NODE_PREFETCH_SIZE 56

/* The walk's stack, for both levels. The builder is LBVH, whose tree over
 * 32-bit Morton codes, ties broken by index, is at most 32 + log2(n) deep, so
 * 61 a level for 2^29 leaves; no tree is flattened.
 */
#define HK_BVH_STACK_SIZE 128

struct hk_encode_args {
   VOID_REF intermediate_bvh;
   VOID_REF output_bvh;
   REF(vk_ir_header) header;
   /* the structure's size, hk_get_as_size's */
   uint64_t size;
   uint32_t leaf_node_count;
   uint32_t geometry_type;
};

#define HK_COPY_MODE_SERIALIZE   0
#define HK_COPY_MODE_DESERIALIZE 1

/* A serialized structure is the common header (the driver's and the format's
 * UUIDs, the sizes, the instance count), the instances' bottom-level
 * pointers, then the structure's bytes.
 */
struct hk_copy_args {
   VOID_REF src;
   VOID_REF dst;
   /* the serialized header's two UUIDs */
   uint32_t uuid[8];
   uint32_t mode;
   uint32_t padding;
};

#endif
