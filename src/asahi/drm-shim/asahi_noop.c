/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2018 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#include "drm-shim/drm_shim.h"
#include "drm-uapi/asahi_drm.h"
#include "util/macros.h"

static struct drm_asahi_params_global params = {
   .gpu_generation = 13,
   .gpu_variant = 'G',
   .gpu_revision = 0,
   .vm_start = 0x4000,
   .vm_end = 0x7fffff8000ull,
   .command_timestamp_frequency_hz = 1000000000,
};

struct asahi_bo {
   struct shim_bo base;
   uint32_t offset;
};

static struct asahi_bo *
asahi_bo(struct shim_bo *bo)
{
   return (struct asahi_bo *)bo;
}

/* ASAHI_SHIM_G15=1: report the M3 Max (G15C B1) as the g15-7.4 kernel does,
 * keep the VM binds, and with ASAHI_SHIM_DUMP=dir write each submit's
 * command buffer and every bound range to dir/submit-N/ for offline decoding.
 */
static const struct drm_asahi_params_global params_g15c = {
   .gpu_generation = 15,
   .gpu_variant = 'C',
   .gpu_revision = 0x11,
   .chip_id = 0x6034,
   .num_dies = 1,
   .num_clusters_total = 4,
   .num_cores_per_cluster = 10,
   .core_masks = {0x3ff, 0x3ff, 0x3ff, 0},
   .vm_start = 0x4000,
   .vm_end = 0x3ffffff8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 64,
   .max_attachments = 16,
   .max_frequency_khz = 1398000,
   .command_timestamp_frequency_hz = 1000000000,
};

struct asahi_bind {
   uint64_t addr, range, offset;
   struct shim_bo *bo;
};

static struct asahi_bind *binds;
static unsigned nr_binds, max_binds, nr_submits;

struct asahi_device {
   uint64_t next_offset;
};

static struct asahi_device asahi = {
   .next_offset = 0x1000,
};

static int
asahi_ioctl_noop(int fd, unsigned long request, void *arg)
{
   return 0;
}

/* A bound object (timestamps) gets a nonzero handle, as from the kernel */
static int
asahi_ioctl_gem_bind_object(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_gem_bind_object *bind = arg;
   static uint32_t next_object = 1;

   if (bind->op == DRM_ASAHI_BIND_OBJECT_OP_BIND)
      bind->object_handle = next_object++;

   return 0;
}

static int
asahi_ioctl_gem_create(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_asahi_gem_create *create = arg;
   struct asahi_bo *bo = calloc(1, sizeof(*bo));

   drm_shim_bo_init(&bo->base, create->size);

   assert(UINT64_MAX - asahi.next_offset > create->size);
   bo->offset = asahi.next_offset;
   asahi.next_offset += create->size;

   create->handle = drm_shim_bo_get_handle(shim_fd, &bo->base);

   drm_shim_bo_put(&bo->base);

   return 0;
}

static int
asahi_ioctl_gem_mmap_offset(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_asahi_gem_mmap_offset *map = arg;
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, map->handle);

   map->offset = drm_shim_bo_get_mmap_offset(shim_fd, bo);

   drm_shim_bo_put(bo);

   return 0;
}

static int
asahi_ioctl_vm_bind(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_asahi_vm_bind *vb = arg;

   for (unsigned i = 0; i < vb->num_binds; ++i) {
      struct drm_asahi_gem_bind_op *op =
         (void *)(uintptr_t)(vb->userptr + (uint64_t)i * vb->stride);
      uint64_t end = op->addr + op->range;

      /* Drop (or trim) what the new range covers */
      for (unsigned j = 0; j < nr_binds;) {
         struct asahi_bind *b = &binds[j];
         if (b->addr >= op->addr && b->addr + b->range <= end) {
            if (b->bo)
               drm_shim_bo_put(b->bo);
            *b = binds[--nr_binds];
         } else {
            ++j;
         }
      }

      if (op->flags & DRM_ASAHI_BIND_UNBIND)
         continue;

      if (nr_binds == max_binds) {
         max_binds = MAX2(max_binds * 2, 4096);
         binds = realloc(binds, max_binds * sizeof(*binds));
         assert(binds);
      }
      binds[nr_binds++] = (struct asahi_bind){
         .addr = op->addr,
         .range = op->range,
         .offset = op->offset,
         .bo = drm_shim_bo_lookup(shim_fd, op->handle),
      };
   }

   return 0;
}

static void
asahi_dump(const char *path, const void *data, size_t size)
{
   FILE *f = fopen(path, "wb");
   if (f) {
      fwrite(data, 1, size, f);
      fclose(f);
   }
}

static void
asahi_dump_bo(const char *path, struct shim_bo *bo, uint64_t offset,
              uint64_t size)
{
   void *data = malloc(size);
   if (data && pread(shim_device.mem_fd, data, size, bo->mem_addr + offset) ==
                  (ssize_t)size)
      asahi_dump(path, data, size);
   free(data);
}

static int
asahi_ioctl_submit(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_submit *sub = arg;
   const char *dir = getenv("ASAHI_SHIM_DUMP");
   char path[512];

   if (!dir)
      return 0;

   snprintf(path, sizeof(path), "%s/submit-%u", dir, nr_submits++);
   mkdir(dir, 0755);
   mkdir(path, 0755);

   char *p = path + strlen(path);
   strcpy(p, "/cmdbuf.bin");
   asahi_dump(path, (void *)(uintptr_t)sub->cmdbuf, sub->cmdbuf_size);

   for (unsigned i = 0; i < nr_binds; ++i) {
      struct asahi_bind *b = &binds[i];
      if (!b->bo || b->range > 64 << 20 || b->offset >= b->bo->size)
         continue;

      snprintf(p, path + sizeof(path) - p, "/%016" PRIx64 ".bin", b->addr);
      asahi_dump_bo(path, b->bo, b->offset,
                    MIN2(b->range, b->bo->size - b->offset));
   }

   return 0;
}

static int
asahi_ioctl_get_param(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_get_params *gp = arg;

   switch (gp->param_group) {
   case 0:
      assert(gp->size == sizeof(struct drm_asahi_params_global));
      memcpy((void *)gp->pointer, &params, gp->size);
      return 0;

   default:
      fprintf(stderr, "Unknown DRM_IOCTL_ASAHI_GET_PARAMS %d\n",
              gp->param_group);
      return -1;
   }
}

static ioctl_fn_t driver_ioctls[] = {
   [DRM_ASAHI_GET_PARAMS] = asahi_ioctl_get_param,
   [DRM_ASAHI_VM_CREATE] = asahi_ioctl_noop,
   [DRM_ASAHI_VM_DESTROY] = asahi_ioctl_noop,
   [DRM_ASAHI_VM_BIND] = asahi_ioctl_vm_bind,
   [DRM_ASAHI_GEM_CREATE] = asahi_ioctl_gem_create,
   [DRM_ASAHI_GEM_MMAP_OFFSET] = asahi_ioctl_gem_mmap_offset,
   [DRM_ASAHI_QUEUE_CREATE] = asahi_ioctl_noop,
   [DRM_ASAHI_QUEUE_DESTROY] = asahi_ioctl_noop,
   [DRM_ASAHI_GEM_BIND_OBJECT] = asahi_ioctl_gem_bind_object,
   [DRM_ASAHI_SUBMIT] = asahi_ioctl_submit,
};

void
drm_shim_driver_init(void)
{
   if (getenv("ASAHI_SHIM_G15"))
      params = params_g15c;

   shim_device.driver_ioctls = driver_ioctls;
   shim_device.driver_ioctl_count = ARRAY_SIZE(driver_ioctls);

   drm_shim_platform_device_setup("asahi", "/soc/agx", "apple,gpu-g13g");
}
