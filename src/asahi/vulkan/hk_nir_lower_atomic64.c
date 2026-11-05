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
 * landing between another operation's read and write would be lost. So do
 * the shader's 64-bit atomic stores (an exchange) and loads (the word read
 * whole, between the halves' writes no more), which without a locked
 * operation are single accesses. Each lane retries in one loop and leaves it
 * once it has had the lock, so the lanes of a SIMD group never wait on each
 * other.
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
 *
 * A lane that misses the lock backs off as Apple's compiler has a compute
 * kernel's compare-exchange loop do (AGXCompilerCore 26A434's
 * InsertYieldsPass, __yield_tame_backoff): after 8 misses, each miss yields
 * the SIMD group's issue slot for 1, 2, 3... up to 16384. Without it, a
 * SIMD group spinning on a lock its holder's group must issue to release
 * stalls them both.
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

/* A 64-bit atomic load or store: one access, unlocked unless the shader locks
 * its read-modify-writes */
static bool
is_atomic_access64(const nir_intrinsic_instr *intr, nir_intrinsic_op op)
{
   const nir_def *v = op == nir_intrinsic_store_global ? intr->src[0].ssa
                                                       : &intr->def;
   return intr->intrinsic == op && v->bit_size == 64 &&
          v->num_components == 1 &&
          (nir_intrinsic_access(intr) & ACCESS_ATOMIC);
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data_)
{
   bool ld = is_atomic_access64(intr, nir_intrinsic_load_global);
   bool st = is_atomic_access64(intr, nir_intrinsic_store_global);
   if (!ld && !st &&
       ((intr->intrinsic != nir_intrinsic_global_atomic &&
         intr->intrinsic != nir_intrinsic_global_atomic_swap) ||
        intr->def.bit_size != 64))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_atomic_op op =
      ld || st ? nir_atomic_op_xchg : nir_intrinsic_atomic_op(intr);
   nir_def *addr = intr->src[st ? 1 : 0].ssa;
   nir_def *data = ld ? NULL : intr->src[st ? 0 : 1].ssa;
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

   nir_variable *misses =
      nir_local_variable_create(b->impl, glsl_uint_type(), "atomic64_misses");
   nir_variable *amount = nir_local_variable_create(
      b->impl, glsl_uint16_t_type(), "atomic64_yield");
   nir_store_var(b, misses, nir_imm_int(b, 0), 1);
   nir_store_var(b, amount, nir_imm_intN_t(b, 1, 16), 1);

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
         nir_def *lo, *hi;
         if (ld) {
            /* A load reads the word whole, coherent (G15's loads see other
             * cores' writes), and writes nothing: a robust load's address
             * may be the zero page */
            nir_def *v = nir_load_global(
               b, 1, 64, addr, .align_mul = 8,
               .access = ACCESS_COHERENT | ACCESS_VOLATILE);
            lo = nir_unpack_64_2x32_split_x(b, v);
            hi = nir_unpack_64_2x32_split_y(b, v);
         } else {
            if (op != nir_atomic_op_xchg) {
               old = nir_pack_64_2x32_split(
                  b, atomic32(b, addr, zero, nir_atomic_op_iadd),
                  atomic32(b, hi_addr, zero, nir_atomic_op_iadd));
            }
            nir_def *new = apply(b, op, old, data, data2);
            lo = atomic32(b, addr, nir_unpack_64_2x32_split_x(b, new),
                          nir_atomic_op_xchg);
            hi = atomic32(b, hi_addr, nir_unpack_64_2x32_split_y(b, new),
                          nir_atomic_op_xchg);
         }
         atomic32(b, lock, nir_iand(b, held, nir_ior(b, lo, hi)),
                  nir_atomic_op_xchg);

         nir_store_var(b, result, nir_pack_64_2x32_split(b, lo, hi), 1);
         nir_jump(b, nir_jump_break);
      }
      nir_pop_if(b, NULL);

      nir_def *m = nir_load_var(b, misses);
      nir_store_var(b, misses, nir_iadd_imm(b, m, 1), 1);
      nir_push_if(b, nir_uge_imm(b, m, 8));
      {
         nir_def *a = nir_load_var(b, amount);
         nir_yield_agx(b, a);
         nir_store_var(b, amount, nir_umin_imm(b, nir_iadd_imm(b, a, 1), 16384),
                       1);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, NULL);

   if (fs)
      nir_pop_if(b, NULL);

   if (st)
      nir_instr_remove(&intr->instr);
   else
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
 * 64-bit shared atomics under locks in the workgroup's own memory. G15's
 * threadgroup atomic is 32-bit only (Apple's encoder has one atomictg form,
 * its data a 32-bit register), so each 64-bit word of the workgroup's memory
 * gets a lock bit, in lock words appended past it. A lane sets its word's bit
 * with an atomic or; if the bit was clear, it reads the word, writes the new
 * value with one 64-bit store and clears the bit with an atomic and. A
 * workgroup runs on one core, whose threadgroup memory takes a lane's
 * accesses in order (Mesa emits no fence for shared memory), so the lock
 * needs no fence, and its lanes never wait on another core. The word moves
 * whole, so a 64-bit atomic load needs no lock; an atomic store takes it as
 * an exchange, or it could land between a locked operation's read and write.
 * Word w's bit is bit w / n of lock word w % n (n a power of two), so
 * neighbouring words' locks sit in different lock words: a SIMD group's
 * lanes on neighbouring words do not queue on one lock word. The lock words
 * are zeroed at the shader's start, before a workgroup barrier. A workgroup
 * whose memory leaves too little room hashes its words onto the bits there
 * are; one that leaves none takes the device's locks, hashed from the word
 * and the workgroup, and releases after a load that returns its store.
 */
struct shared_locks {
   unsigned base;  /* the lock words' offset */
   unsigned shift; /* log2 of their count */
   bool device;    /* no room: the device's locks */
};

static bool
is_int_op(nir_atomic_op op)
{
   switch (op) {
   case nir_atomic_op_iadd:
   case nir_atomic_op_imin:
   case nir_atomic_op_umin:
   case nir_atomic_op_imax:
   case nir_atomic_op_umax:
   case nir_atomic_op_iand:
   case nir_atomic_op_ior:
   case nir_atomic_op_ixor:
   case nir_atomic_op_xchg:
   case nir_atomic_op_cmpxchg:
      return true;
   default:
      return false;
   }
}

static bool
is_shared_rmw64(const nir_intrinsic_instr *intr)
{
   return (intr->intrinsic == nir_intrinsic_shared_atomic ||
           intr->intrinsic == nir_intrinsic_shared_atomic_swap) &&
          intr->def.bit_size == 64 && is_int_op(nir_intrinsic_atomic_op(intr));
}

static bool
lower_shared(nir_builder *b, nir_intrinsic_instr *intr, void *data_)
{
   const struct shared_locks *locks = data_;
   bool store = intr->intrinsic == nir_intrinsic_store_shared &&
                intr->src[0].ssa->bit_size == 64 &&
                intr->src[0].ssa->num_components == 1 &&
                (nir_intrinsic_access(intr) & ACCESS_ATOMIC);
   if (!store && !is_shared_rmw64(intr))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_atomic_op op =
      store ? nir_atomic_op_xchg : nir_intrinsic_atomic_op(intr);
   nir_def *off = nir_iadd_imm(b, intr->src[store ? 1 : 0].ssa,
                               nir_intrinsic_base(intr));
   nir_def *data = intr->src[store ? 0 : 1].ssa;
   nir_def *data2 = op == nir_atomic_op_cmpxchg ? intr->src[2].ssa : NULL;
   bool unused = store || nir_def_is_unused(&intr->def);
   nir_def *word = nir_ushr_imm(b, off, 3);

   nir_def *lock, *bit = NULL;
   if (locks->device) {
      nir_def *root = nir_load_root_agx(b);
      nir_def *table = nir_load_global_constant(
         b, 1, 64,
         nir_iadd_imm(b, root, hk_root_descriptor_offset(atomic64_locks)),
         .align_mul = 8, .access = ACCESS_CAN_SPECULATE);
      nir_def *wg = nir_load_workgroup_id(b);
      nir_def *mix = nir_ixor(
         b, nir_imul_imm(b, nir_channel(b, wg, 0), 0x9e3779b1),
         nir_ixor(b, nir_imul_imm(b, nir_channel(b, wg, 1), 0x85ebca77),
                  nir_imul_imm(b, nir_channel(b, wg, 2), 0xc2b2ae3d)));
      nir_def *index = nir_iand_imm(
         b, nir_ixor(b, word, nir_ushr_imm(b, mix, 16)), HK_ATOMIC64_LOCKS - 1);
      lock = nir_iadd(b, table, nir_u2u64(b, nir_imul_imm(b, index, 4)));
   } else {
      nir_def *index = nir_iand_imm(b, word, (1u << locks->shift) - 1);
      lock = nir_iadd_imm(b, nir_imul_imm(b, index, 4), locks->base);
      /* ishl takes the count's low 5 bits: past 32 lock bits a word, the
       * words hash onto them */
      bit = nir_ishl(b, nir_imm_int(b, 1), nir_ushr_imm(b, word, locks->shift));
   }

   nir_variable *result =
      nir_local_variable_create(b->impl, glsl_uint64_t_type(), "atomic64");
   nir_push_loop(b);
   {
      nir_def *held, *free;
      if (bit) {
         held = nir_shared_atomic(b, 32, lock, bit,
                                  .atomic_op = nir_atomic_op_ior);
         free = nir_ieq_imm(b, nir_iand(b, held, bit), 0);
      } else {
         held = nir_global_atomic_swap(b, 32, lock, nir_imm_int(b, 0),
                                       nir_imm_int(b, 1),
                                       .atomic_op = nir_atomic_op_cmpxchg);
         free = nir_ine_imm(b, held, 1);
      }
      nir_push_if(b, free);
      {
         nir_def *old = NULL;
         if (op != nir_atomic_op_xchg || !unused)
            old = nir_load_shared(b, 1, 64, off, .align_mul = 8);
         nir_store_shared(b, apply(b, op, old, data, data2), off,
                          .align_mul = 8);
         if (bit) {
            nir_shared_atomic(b, 32, lock, nir_inot(b, bit),
                              .atomic_op = nir_atomic_op_iand);
         } else {
            /* The device's lock is released through the L2: its value, 0
             * (held), depends on a load that returns the store */
            nir_def *back = nir_load_shared(b, 1, 32, off, .align_mul = 8);
            nir_global_atomic(b, 32, lock, nir_iand(b, held, back),
                              .atomic_op = nir_atomic_op_xchg);
         }
         if (!unused)
            nir_store_var(b, result, old, 1);
         nir_jump(b, nir_jump_break);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, NULL);

   if (unused)
      nir_instr_remove(&intr->instr);
   else
      nir_def_replace(&intr->def, nir_load_var(b, result));
   return true;
}

bool
hk_nir_lower_shared_atomic64(nir_shader *nir)
{
   bool rmw = false;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                is_shared_rmw64(nir_instr_as_intrinsic(instr)))
               rmw = true;
         }
      }
   }

   /* Without a read-modify-write, a 64-bit atomic load or store is one
    * access of the word, atomic as it stands */
   if (!rmw)
      return false;

   assert(mesa_shader_stage_is_compute(nir->info.stage) &&
          !nir->info.workgroup_size_variable && nir->info.shared_size > 0);

   unsigned words = DIV_ROUND_UP(nir->info.shared_size, 8);
   unsigned base = align(nir->info.shared_size, 4);
   unsigned room = (HK_MAX_SHARED_SIZE - base) / 4;
   unsigned n = util_next_power_of_two(DIV_ROUND_UP(words, 32));
   while (n > room)
      n >>= 1;

   struct shared_locks locks = {
      .base = base,
      .shift = n ? util_logbase2(n) : 0,
      .device = n == 0,
   };

   if (!locks.device) {
      nir->info.shared_size = base + n * 4;

      /* A shader zeroing its memory zeroes the locks with it, after this */
      if (!nir->info.zero_initialize_shared_memory) {
         nir_function_impl *impl = nir_shader_get_entrypoint(nir);
         nir_builder b = nir_builder_at(nir_before_impl(impl));
         unsigned threads = nir->info.workgroup_size[0] *
                            nir->info.workgroup_size[1] *
                            nir->info.workgroup_size[2];
         nir_def *index = nir_load_local_invocation_index(&b);

         for (unsigned i = 0; i < n; i += threads) {
            nir_def *w = nir_iadd_imm(&b, index, i);
            bool all = i + threads <= n;
            if (!all)
               nir_push_if(&b, nir_ult_imm(&b, w, n));
            nir_store_shared(&b, nir_imm_int(&b, 0),
                             nir_iadd_imm(&b, nir_imul_imm(&b, w, 4), base),
                             .align_mul = 4);
            if (!all)
               nir_pop_if(&b, NULL);
         }

         nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                     .memory_scope = SCOPE_WORKGROUP,
                     .memory_semantics = NIR_MEMORY_ACQ_REL,
                     .memory_modes = nir_var_mem_shared);
      }
   }

   nir_shader_intrinsics_pass(nir, lower_shared, nir_metadata_none, &locks);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   return true;
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
