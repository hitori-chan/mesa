/*
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2024 Valve Corporation
 * Copyright 2022 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl.h"

#pragma once

struct libagx_xfb_counter_copy {
   DEVICE(uint32_t) dest[4];
   DEVICE(uint32_t) src[4];
};

/* An array of these crosses to the GPU: the pad keeps i386's 4-byte aligned
 * uint64_t to the GPU's 16-byte stride.
 */
struct libagx_imm_write {
   DEVICE(uint32_t) address;
   uint32_t value;
   uint32_t pad;
};
static_assert(sizeof(struct libagx_imm_write) == 16);

#define LIBAGX_QUERY_UNAVAILABLE (uint64_t)((int64_t)-1)
