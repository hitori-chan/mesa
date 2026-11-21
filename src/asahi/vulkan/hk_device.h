/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_device.h"
#include "util/rwlock.h"
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"
#include "util/u_queue.h"
#include "agx_bg_eot.h"
#include "agx_pack.h"
#include "agx_scratch.h"
#include "decode.h"
#include "vk_cmd_queue.h"
#include "vk_dispatch_table.h"

#include "hk_private.h"

#include "hk_descriptor_table.h"
#include "hk_queue.h"
#include "vk_device.h"
#include "vk_meta.h"
#include "vk_queue.h"

struct hk_physical_device;
struct vk_pipeline_cache;

typedef void (*hk_internal_builder_t)(struct nir_builder *b, const void *key);

struct hk_internal_key {
   hk_internal_builder_t builder;
   size_t key_size;
   uint8_t key[];
};

struct hk_internal_shaders {
   simple_mtx_t lock;
   struct hash_table *ht;
};

struct hk_rc_sampler {
   struct agx_sampler_packed key;

   /* Reference count for this hardware sampler, protected by the heap mutex */
   uint16_t refcount;

   /* Index of this hardware sampler in the hardware sampler heap */
   uint16_t index;
};

struct hk_sampler_heap {
   simple_mtx_t lock;

   struct hk_descriptor_table table;

   /* Map of agx_sampler_packed to hk_rc_sampler */
   struct hash_table *ht;
};

/* Results by the hash of their input, up to a size, then cleared */
struct hk_result_cache {
   simple_mtx_t lock;
   struct hash_table *ht;
   size_t bytes;
};

struct hk_device {
   struct vk_device vk;
   struct agx_device dev;
   struct agxdecode_ctx *decode_ctx;

   struct hk_descriptor_table occlusion_queries;
   struct hk_sampler_heap samplers;

   struct vk_meta_device meta;
   struct agx_bg_eot_cache bg_eot;

   struct {
      struct agx_bo *bo;
      uint64_t heap;
   } rodata;

   /* HK_ATOMIC64_LOCKS 32-bit locks for 64-bit atomics, zeroed */
   struct agx_bo *atomic64_locks;

   struct hk_internal_shaders prolog_epilog;
   struct hk_internal_shaders kernels;
   struct hk_api_shader *null_fs;

   /* Indirected for common secondary emulation */
   struct vk_device_dispatch_table cmd_dispatch;

   /* Heap used for GPU-side memory allocation for geometry/tessellation.
    *
    * Control streams accessing the heap must be serialized. This is not
    * expected to be a legitimate problem. If it is, we can rework later.
    */
   struct agx_bo *heap;
   util_once_flag heap_init_once;

   struct {
      struct agx_scratch vs, fs, cs;
      simple_mtx_t lock;
   } scratch;

   uint32_t perftest;

   /* A pipeline's stages but its last, compiled beside the calling thread
    * (hk_compile_shaders): started at the first pipeline of several stages,
    * busy counting the stages it holds */
   struct {
      struct util_queue queue;
      util_once_flag once;
      bool ok;
      uint32_t busy;
   } compile;

   /* A graphics pipeline's stages, compiled on these threads after its
    * creation returns (hk_compile_shaders_later): started at the first such
    * pipeline. A full queue makes the creating thread wait for room. */
   struct {
      struct util_queue queue;
      util_once_flag once;
      bool ok;
   } later;

   struct {
      struct u_rwlock lock;
      struct util_dynarray list;
      struct util_dynarray counts;
   } external_bos;

   /* Results by the hash of their input (hk_shader.c): the backend's
    * (hk_compile_nir) and the stages lowered (hk_compile_shaders) */
   struct hk_result_cache backend_cache, lower_cache;
};

VK_DEFINE_HANDLE_CASTS(hk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

enum hk_perftest {
   HK_PERF_NOTESS = BITFIELD_BIT(0),
   HK_PERF_NOBORDER = BITFIELD_BIT(1),
   HK_PERF_NOBARRIER = BITFIELD_BIT(2),
   HK_PERF_BATCH = BITFIELD_BIT(3),
   HK_PERF_NOROBUST = BITFIELD_BIT(4),
   HK_PERF_NOBATCH = BITFIELD_BIT(5),
   HK_PERF_NOEOTRESOLVE = BITFIELD_BIT(6),
   HK_PERF_NOSLC = BITFIELD_BIT(7),
   HK_PERF_HOSTBARRIER = BITFIELD_BIT(8),
   HK_PERF_ALLEXT = BITFIELD_BIT(9),
   HK_PERF_NOOVERLAP = BITFIELD_BIT(10),
   HK_PERF_ZSSTORE = BITFIELD_BIT(11),
   HK_PERF_IMGCHAIN = BITFIELD_BIT(12),
   HK_PERF_SWTESS = BITFIELD_BIT(13),
   HK_PERF_HWTESS32 = BITFIELD_BIT(14),
   /* Not an option: on G15 unless swtess is set */
   HK_PERF_HWTESS = BITFIELD_BIT(15),
   HK_PERF_SERIALCOMPILE = BITFIELD_BIT(16),
   HK_PERF_NOBACKENDCACHE = BITFIELD_BIT(17),
   HK_PERF_NOLOWERCACHE = BITFIELD_BIT(18),
   HK_PERF_SYNCCOMPILE = BITFIELD_BIT(19),
};

#define HK_PERF(dev, flag) unlikely((dev)->perftest &HK_PERF_##flag)

/* The options that change the code compiled, part of the shader cache's key:
 * a shader compiled under one is never loaded under another */
#define HK_PERF_SHADER_CODE                                                    \
   (HK_PERF_NOBORDER | HK_PERF_NOROBUST | HK_PERF_HWTESS | HK_PERF_HWTESS32)

/* HK_PERFTEST, with the driconf options and the defaults that set its bits */
uint32_t hk_parse_perftest(const struct hk_physical_device *pdev);

static inline struct hk_physical_device *
hk_device_physical(struct hk_device *dev)
{
   return (struct hk_physical_device *)dev->vk.physical;
}

VkResult hk_device_init_meta(struct hk_device *dev);
void hk_device_init_accel_struct(struct hk_device *dev);
void hk_device_finish_meta(struct hk_device *dev);

VkResult hk_sampler_heap_add(struct hk_device *dev,
                             struct agx_sampler_packed desc,
                             struct hk_rc_sampler **out);

void hk_sampler_heap_remove(struct hk_device *dev, struct hk_rc_sampler *rc);

static inline struct agx_scratch *
hk_device_scratch_locked(struct hk_device *dev, mesa_shader_stage stage)
{
   simple_mtx_assert_locked(&dev->scratch.lock);

   switch (stage) {
   case MESA_SHADER_FRAGMENT:
      return &dev->scratch.fs;
   case MESA_SHADER_VERTEX:
      return &dev->scratch.vs;
   default:
      return &dev->scratch.cs;
   }
}

static inline void
hk_device_alloc_scratch(struct hk_device *dev, mesa_shader_stage stage,
                        unsigned size)
{
   simple_mtx_lock(&dev->scratch.lock);
   agx_scratch_alloc(hk_device_scratch_locked(dev, stage), size, 0);
   simple_mtx_unlock(&dev->scratch.lock);
}
