/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/compiler/agx_compile.h"
#include "util/simple_mtx.h"
#include "agx_tilebuffer.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "pool.h"

struct agx_precompiled_shader {
   struct agx_shader b;
   struct agx_bo *bo;
   uint64_t ptr;
};

struct agx_bg_eot_cache {
   struct agx_device *dev;
   struct agx_pool pool;
   simple_mtx_t lock;

   /* Map from agx_bg_eot_key to agx_bg_eot_shader */
   struct hash_table *ht;

   struct agx_precompiled_shader *precomp[LIBAGX_NUM_PROGRAMS];
};

/*
 * Get a precompiled shader, uploading if necessary. This is thread-safe.
 */
struct agx_precompiled_shader *
agx_get_precompiled(struct agx_bg_eot_cache *cache, unsigned program);

/*
 * Get the address of the cached helper program. This is thread-safe.
 */
uint64_t agx_helper_program(struct agx_bg_eot_cache *cache);

enum agx_bg_eot_op {
   AGX_BG_EOT_NONE,
   AGX_BG_CLEAR,
   AGX_BG_LOAD,
   AGX_EOT_STORE,
};

/* G15: a background program that loads render targets reads their texture
 * descriptors (2 per render target, 32 B each, the texture at 2 * rt) through
 * the 64-bit uniform at u0 and the sampler table's first entry through
 * another (a texture instruction names the pair of a table below u32): the
 * first one past the clear colours of the render targets it clears, at 4 + 8
 * * rt */
#define AGX_G15_MAX_BG_SAMPLER_UNIFORM 60

struct agx_bg_eot_key {
   struct agx_tilebuffer_layout tib;
   enum agx_bg_eot_op op[8];
   unsigned reserved_preamble;

   /* G15, loading programs: the uniform (in 16-bit halves) of the sampler
    * table's base */
   unsigned g15_sampler_uniform;
};

/* G15: places a loading program's sampler uniform (above) */
static inline void
agx_bg_eot_key_set_g15_sampler(struct agx_bg_eot_key *key)
{
   bool loads = false;
   unsigned first_free = 4;

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
      loads |= key->op[rt] == AGX_BG_LOAD;
      if (key->op[rt] == AGX_BG_CLEAR)
         first_free = 4 + 8 * (rt + 1);
   }

   if (loads) {
      key->g15_sampler_uniform = first_free;
      assert(first_free <= AGX_G15_MAX_BG_SAMPLER_UNIFORM);
   }
}

struct agx_bg_eot_shader {
   struct agx_bg_eot_key key;
   struct agx_shader_info info;
   struct agx_bo *bo;
   uint64_t ptr;
};

struct agx_bg_eot_shader *agx_get_bg_eot_shader(struct agx_bg_eot_cache *cache,
                                                struct agx_bg_eot_key *key);

void agx_bg_eot_init(struct agx_bg_eot_cache *cache, struct agx_device *dev);
void agx_bg_eot_cleanup(struct agx_bg_eot_cache *cache);
