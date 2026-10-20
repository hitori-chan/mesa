/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "agx_pack.h"

#pragma once

struct libagx_decompress_images {
   struct agx_texture_packed compressed;
   struct agx_pbe_packed uncompressed;
};
static_assert(sizeof(struct libagx_decompress_images) == 48);

/* G15 indexes tables of 32 B entries: the texture, its uncompressed PBE with
 * the software word a multisampled image's store reads (PBE G15 software
 * MSAA), and the txf sampler its loads fetch with */
struct libagx_decompress_images_g15 {
   struct agx_texture_packed compressed;
   uint32_t pad0[2];
   struct agx_pbe_packed uncompressed;
   uint32_t uncompressed_sw[2];
   struct agx_sampler_packed txf;
   uint32_t pad1[6];
};
static_assert(sizeof(struct libagx_decompress_images_g15) == 96);
