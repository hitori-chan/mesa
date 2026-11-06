/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"

#include "vk_device_memory.h"

#include "util/list.h"

struct hk_device;
struct hk_image_plane;

struct hk_device_memory {
   struct vk_device_memory vk;

   struct agx_bo *bo;

   void *map;

   /* A swapchain's memory, external under virtio: the host syncs a submit on
    * it only while the application owns it (hk_wsi_set_memory_ownership).
    */
   bool wsi;
   bool wsi_owned;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)

void hk_wsi_set_memory_ownership(VkDevice device, VkDeviceMemory memory,
                                 VkBool32 ownership);

extern const VkExternalMemoryProperties hk_opaque_fd_mem_props;
extern const VkExternalMemoryProperties hk_dma_buf_mem_props;
