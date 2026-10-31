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
 * with 32-bit atomics, which the L2 serves, and releases the lock. Even min
 * and max take it, or one landing between another operation's read and
 * write would be lost. Each lane retries in one loop and leaves it once it
 * has had the lock, so the lanes of a SIMD group never wait on each other.
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
   bool progress =
      nir_shader_intrinsics_pass(nir, lower, nir_metadata_none, NULL);

   if (progress) {
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   }

   return progress;
}
