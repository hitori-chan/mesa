/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>

struct nir_shader;
struct nir_instr;
struct nir_builder;
struct nir_def;
struct nir_intrinsic_instr;
struct nir_tex_instr;

bool agx_nir_lower_address(struct nir_shader *shader);
bool agx_nir_lower_algebraic_late(struct nir_shader *shader);
bool agx_nir_cleanup_amul(struct nir_shader *shader);
bool agx_nir_fuse_lea(struct nir_shader *shader);
bool agx_nir_lower_lea(struct nir_shader *shader);
bool agx_nir_fuse_selects(struct nir_shader *shader);
bool agx_nir_fuse_algebraic_late(struct nir_shader *shader);
bool agx_nir_fence_images(struct nir_shader *shader);
bool agx_nir_lower_layer(struct nir_shader *s);
bool agx_nir_lower_clip_distance(struct nir_shader *s);
bool agx_nir_lower_subgroups(struct nir_shader *s, bool g15);
bool agx_nir_lower_fminmax(struct nir_shader *s);
bool agx_nir_lower_tile_raw(struct nir_shader *s, unsigned nr_samples,
                            bool mask_coverage);

bool agx_nir_lower_texture_early(struct nir_shader *s, bool support_lod_bias);
bool agx_nir_lower_texture(struct nir_shader *s);
bool agx_nir_lower_texture_gen(struct nir_shader *s, bool g15);
bool agx_nir_lower_cube_g15(struct nir_shader *s);
struct nir_tex_instr *
agx_nir_image_load_to_txf_g15(struct nir_builder *b,
                              struct nir_intrinsic_instr *intr,
                              struct nir_def *sampler);
bool agx_nir_lower_multisampled_image_store(struct nir_shader *s, bool g15);
bool agx_nir_needs_texture_crawl(struct nir_instr *instr);
