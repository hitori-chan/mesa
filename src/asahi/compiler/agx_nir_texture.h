/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

/* Texture backend flags */
#define AGX_TEXTURE_FLAG_NO_CLAMP (1 << 0)

/* Indicates that the sampler should be overriden to clamp to 0 instead of 1 */
#define AGX_TEXTURE_FLAG_CLAMP_TO_0 (1 << 1)

/* G15: the bias is q5.6 in a 16-bit uniform, for F28 (agx_nir_g15_bias_q56) */
#define AGX_TEXTURE_FLAG_BIAS_Q56 (1 << 2)

/* An array access whose layer is 0 by construction (a plain view read as an
 * array of one): no layer to clamp or bounds-check */
#define AGX_TEXTURE_FLAG_LAYER_ZERO (1 << 3)

/* Texel buffers lowered to (at most) 16384x16384 2D textures */
#define AGX_TEXTURE_BUFFER_WIDTH      16384
#define AGX_TEXTURE_BUFFER_MAX_HEIGHT 16384
#define AGX_TEXTURE_BUFFER_MAX_SIZE                                            \
   (AGX_TEXTURE_BUFFER_WIDTH * AGX_TEXTURE_BUFFER_MAX_HEIGHT)
