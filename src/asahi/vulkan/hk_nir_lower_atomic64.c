/*
 * Copyright 2026 hitori1403
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_shader.h"
#include "nir_builder.h"

/*
 * 64-bit global atomics under a lock. G15's device atomics are 32-bit but for
 * a 64-bit min and max without a result, and no 64-bit compare-exchange, so
 * every 64-bit operation takes one of the device's HK_ATOMIC64_LOCKS 32-bit
 * locks, hashed from its address: it reads and writes the value's halves
 * with 32-bit atomics, which the L2 serves, and releases the lock. A shader
 * whose 64-bit atomics are all min and max without a result keeps them on
 * G15's own form; with any other there, they take the lock too, or one
 * landing between another operation's read and write would be lost. Each lane retries in one loop and leaves it once it
 * has had the lock, so the lanes of a SIMD group never wait on each other.
 * No acquire fence: the lock's compare-exchange returns a result, which the
 * lane waits for before the reads. No release fence either: a G15 device
 * fence inside the lock made every submit of the CTS's u64 memory_model
 * cases time out. Instead the release's value, 0, is computed from the
 * halves' exchanges' results (held & (lo | hi), held being 0 there), so
 * they return and the lane waits for both before it releases; result-less
 * exchanges could land after the release (the CTS's R64 image end_result
 * cases lost updates). The branch tests held != 1, not held == 0, or
 * nir_opt_if would rewrite held to 0 inside it and fold the dependency
 * away. An exchange reads nothing first: its value does not depend on the
 * old one, which its own result returns. Image atomics reach it as global
 * atomics on their texel's address (agx_nir_lower_texture_gen), so it runs
 * after them. A fragment shader's helper lanes skip it: their
 * atomics do nothing and return undefined (nir_lower_helper_writes), so the
 * lock's compare-exchange would never succeed for them.
 */

static nir_def *
apply(nir_builder *b, nir_atomic_op op, nir_def *old, nir_def *data,
      nir_def *data2)
{
   switch (op) {
   case nir_atomic_op_iadd:
      return nir_iadd(b, old, data);
   case nir_atomic_op_imin:
      return nir_imin(b, old, data);
   case nir_atomic_op_umin:
      return nir_umin(b, old, data);
   case nir_atomic_op_imax:
      return nir_imax(b, old, data);
   case nir_atomic_op_umax:
      return nir_umax(b, old, data);
   case nir_atomic_op_iand:
      return nir_iand(b, old, data);
   case nir_atomic_op_ior:
      return nir_ior(b, old, data);
   case nir_atomic_op_ixor:
      return nir_ixor(b, old, data);
   case nir_atomic_op_xchg:
      return data;
   case nir_atomic_op_cmpxchg:
      return nir_bcsel(b, nir_ieq(b, old, data), data2, old);
   default:
      UNREACHABLE("not a 64-bit integer atomic");
   }
}

static nir_def *
atomic32(nir_builder *b, nir_def *addr, nir_def *data, nir_atomic_op op)
{
   return nir_global_atomic(b, 32, addr, data, .atomic_op = op);
}

static bool
is_native(nir_intrinsic_instr *intr)
{
   switch (nir_intrinsic_atomic_op(intr)) {
   case nir_atomic_op_imax:
   case nir_atomic_op_imin:
   case nir_atomic_op_umax:
   case nir_atomic_op_umin:
      return intr->intrinsic == nir_intrinsic_global_atomic &&
             nir_def_is_unused(&intr->def);
   default:
      return false;
   }
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data_)
{
   if ((intr->intrinsic != nir_intrinsic_global_atomic &&
        intr->intrinsic != nir_intrinsic_global_atomic_swap) ||
       intr->def.bit_size != 64)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_atomic_op op = nir_intrinsic_atomic_op(intr);
   nir_def *addr = intr->src[0].ssa;
   nir_def *data = intr->src[1].ssa;
   nir_def *data2 = op == nir_atomic_op_cmpxchg ? intr->src[2].ssa : NULL;

   nir_def *root = nir_load_root_agx(b);
   nir_def *table = nir_load_global_constant(
      b, 1, 64,
      nir_iadd_imm(b, root, hk_root_descriptor_offset(atomic64_locks)),
      .align_mul = 8, .access = ACCESS_CAN_SPECULATE);
   nir_def *index = nir_iand_imm(b, nir_u2u32(b, nir_ushr_imm(b, addr, 3)),
                                 HK_ATOMIC64_LOCKS - 1);
   nir_def *lock = nir_iadd(b, table, nir_u2u64(b, nir_imul_imm(b, index, 4)));
   nir_def *hi_addr = nir_iadd_imm(b, addr, 4);

   nir_variable *result =
      nir_local_variable_create(b->impl, glsl_uint64_t_type(), "atomic64");
   nir_store_var(b, result, nir_imm_int64(b, 0), 1);

   bool fs = b->shader->info.stage == MESA_SHADER_FRAGMENT;
   if (fs)
      nir_push_if(b, nir_inot(b, nir_is_helper_invocation(b, 1)));

   nir_push_loop(b);
   {
      nir_def *held = nir_global_atomic_swap(
         b, 32, lock, nir_imm_int(b, 0), nir_imm_int(b, 1),
         .atomic_op = nir_atomic_op_cmpxchg);
      nir_push_if(b, nir_ine_imm(b, held, 1));
      {
         nir_def *zero = nir_imm_int(b, 0);
         nir_def *old = NULL;
         if (op != nir_atomic_op_xchg) {
            old = nir_pack_64_2x32_split(
               b, atomic32(b, addr, zero, nir_atomic_op_iadd),
               atomic32(b, hi_addr, zero, nir_atomic_op_iadd));
         }
         nir_def *new = apply(b, op, old, data, data2);
         nir_def *lo = atomic32(b, addr, nir_unpack_64_2x32_split_x(b, new),
                                nir_atomic_op_xchg);
         nir_def *hi = atomic32(b, hi_addr,
                                nir_unpack_64_2x32_split_y(b, new),
                                nir_atomic_op_xchg);
         atomic32(b, lock, nir_iand(b, held, nir_ior(b, lo, hi)),
                  nir_atomic_op_xchg);

         nir_store_var(b, result, nir_pack_64_2x32_split(b, lo, hi), 1);
         nir_jump(b, nir_jump_break);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, NULL);

   if (fs)
      nir_pop_if(b, NULL);

   nir_def_replace(&intr->def, nir_load_var(b, result));
   return true;
}

bool
hk_nir_lower_atomic64(nir_shader *nir)
{
   /* G15's own 64-bit max and min, without a result, need no lock. A shader
    * whose 64-bit atomics are all such stays native (a visibility buffer's
    * max); with any other, they take the lock too, or one landing between a
    * locked operation's read and write would be lost. */
   bool other = false;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if ((intr->intrinsic == nir_intrinsic_global_atomic ||
                 intr->intrinsic == nir_intrinsic_global_atomic_swap) &&
                intr->def.bit_size == 64 && !is_native(intr))
               other = true;
         }
      }
   }

   if (!other)
      return false;

   bool progress =
      nir_shader_intrinsics_pass(nir, lower, nir_metadata_none, NULL);

   if (progress) {
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   }

   return progress;
}

/*
 * R64 image loads and stores go through the image's R32G32_UINT descriptors
 * (hk_image_view.c): a load returns (x, 0, 0, 1), and a sparse one its
 * residency code after, a store writes x's halves.
 */
static bool
lower_image_r64(nir_builder *b, nir_intrinsic_instr *intr, void *data_)
{
   bool sparse = intr->intrinsic == nir_intrinsic_bindless_image_sparse_load;

   if ((intr->intrinsic == nir_intrinsic_bindless_image_load || sparse) &&
       intr->def.bit_size == 64) {
      unsigned n = intr->def.num_components - sparse;
      b->cursor = nir_after_instr(&intr->instr);

      intr->num_components = 4 + sparse;
      intr->def.num_components = 4 + sparse;
      intr->def.bit_size = 32;
      nir_intrinsic_set_dest_type(intr, nir_type_uint32);
      nir_intrinsic_set_format(intr, PIPE_FORMAT_R32G32_UINT);

      nir_def *zero = nir_imm_int64(b, 0);
      nir_def *v[5] = {
         nir_pack_64_2x32(b, nir_trim_vector(b, &intr->def, 2)),
         zero,
         zero,
         nir_imm_int64(b, 1),
      };
      if (sparse)
         v[n] = nir_u2u64(b, nir_channel(b, &intr->def, 4));

      nir_def_rewrite_uses_after(&intr->def, nir_vec(b, v, n + sparse));
      return true;
   }

   if (intr->intrinsic == nir_intrinsic_bindless_image_store &&
       intr->src[3].ssa->bit_size == 64) {
      b->cursor = nir_before_instr(&intr->instr);

      nir_def *x = nir_unpack_64_2x32(b, nir_channel(b, intr->src[3].ssa, 0));
      nir_def *zero = nir_imm_int(b, 0);
      nir_src_rewrite(
         &intr->src[3],
         nir_vec4(b, nir_channel(b, x, 0), nir_channel(b, x, 1), zero, zero));

      intr->num_components = 4;
      nir_intrinsic_set_src_type(intr, nir_type_uint32);
      nir_intrinsic_set_format(intr, PIPE_FORMAT_R32G32_UINT);
      return true;
   }

   return false;
}

bool
hk_nir_lower_image_r64(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_image_r64,
                                     nir_metadata_control_flow, NULL);
}
