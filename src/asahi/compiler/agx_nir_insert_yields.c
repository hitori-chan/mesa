/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include <string.h>
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_dynarray.h"
#include "agx_nir.h"

/*
 * G15: a compute loop that spins on memory yields. A SIMD group spinning on
 * another's progress keeps issuing, and the one it waits on, on the same
 * core, may not get to: the BVH sort's single dispatch hung with each core's
 * oldest workgroup the only one going on, and a 64-bit atomic's lock with
 * its holder stalled until its partner gave up. Apple's compiler
 * (AGXCompilerCore 26A434, InsertYieldsPass) gives a compute kernel's loop a
 * backoff when an exit of the loop depends on the result of an atomic, an
 * atomic load among them, the case here: after 8 misses, each yields for 1,
 * 2, 3... up to 16384 (__yield_tame_backoff). It leaves out a loop whose exit
 * atomics all address one word that the loop writes once (a work loop's
 * atomicAdd, a lock-free compare-exchange's update), which progresses each
 * time it misses, and a loop that yields already. Its other policy, for a
 * loop that loads and stores device-coherent memory, is not taken: Vulkan's
 * coherent buffers are common in loops that do not spin.
 *
 * The yield goes at the top of each iteration but the first, which only a
 * miss reaches.
 */

#define SKIP    8
#define INITIAL 1
#define MAX     16384

/* An atomic that returns a value, or an atomic load: its address source */
static nir_src *
atomic_address(nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return NULL;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
   if (!info->has_dest || info->num_srcs == 0)
      return NULL;

   if (nir_intrinsic_has_atomic_op(intr) ||
       (nir_intrinsic_has_access(intr) &&
        (nir_intrinsic_access(intr) & ACCESS_ATOMIC)))
      return &intr->src[0];

   return NULL;
}

struct exits {
   nir_loop *loop;
   bool any;
   bool one_address;
   nir_def *address;
};

static void
add_exit_atomic(struct exits *e, nir_src *address)
{
   if (!e->any) {
      e->any = true;
      e->one_address = true;
      e->address = address->ssa;
   } else if (e->address != address->ssa) {
      e->one_address = false;
   }
}

/* The atomics a branch condition depends on, within a few ALU steps */
static void
find_exit_atomics(struct exits *e, nir_def *def, unsigned depth)
{
   nir_instr *instr = nir_def_instr(def);
   nir_src *address = atomic_address(instr);

   if (address) {
      add_exit_atomic(e, address);
   } else if (instr->type == nir_instr_type_alu && depth > 0) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++)
         find_exit_atomics(e, alu->src[i].src.ssa, depth - 1);
   }
}

static bool
breaks(struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_instr *last = nir_block_last_instr(nir_cf_node_as_block(node));
         if (last && last->type == nir_instr_type_jump &&
             nir_instr_as_jump(last)->type == nir_jump_break)
            return true;
         break;
      }
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         if (breaks(&nif->then_list) || breaks(&nif->else_list))
            return true;
         break;
      }
      default:
         /* An inner loop's breaks are its own */
         break;
      }
   }

   return false;
}

/* The conditions of the ifs that lead to the loop's breaks */
static void
find_exits(struct exits *e, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type != nir_cf_node_if)
         continue;

      nir_if *nif = nir_cf_node_as_if(node);
      if (breaks(&nif->then_list) || breaks(&nif->else_list))
         find_exit_atomics(e, nif->condition.ssa, 4);

      find_exits(e, &nif->then_list);
      find_exits(e, &nif->else_list);
   }
}

static bool
in_loop(nir_loop *loop, nir_instr *instr)
{
   for (nir_cf_node *n = &instr->block->cf_node; n; n = n->parent) {
      if (n == &loop->cf_node)
         return true;
   }

   return false;
}

/* Is a compare-exchange's compare value its own last result or a load of its
 * word in the loop (a lock-free update, which progresses when it misses)? */
static bool
updates(nir_loop *loop, nir_intrinsic_instr *intr)
{
   nir_def *cmp = intr->src[1].ssa;
   nir_instr *instr = nir_def_instr(cmp);

   if (instr->type == nir_instr_type_phi &&
       instr->block == nir_loop_first_block(loop)) {
      nir_foreach_phi_src(src, nir_instr_as_phi(instr)) {
         if (src->src.ssa == &intr->def)
            return true;
      }
   }

   if (instr->type == nir_instr_type_intrinsic && in_loop(loop, instr)) {
      nir_intrinsic_instr *load = nir_instr_as_intrinsic(instr);
      const nir_intrinsic_info *info = &nir_intrinsic_infos[load->intrinsic];
      return info->num_srcs > 0 && info->has_dest &&
             !nir_intrinsic_has_atomic_op(load) &&
             strstr(info->name, "load") &&
             load->src[0].ssa == intr->src[0].ssa;
   }

   return false;
}

/* The loop's writes to the word at address: atomic read-modify-writes but
 * the exchanges, atomic stores and the lock-free updates */
static unsigned
progress_writes(nir_loop *loop, nir_def *address)
{
   unsigned n = 0;

   nir_foreach_use(src, address) {
      nir_instr *instr = nir_src_use_instr(src);
      if (instr->type != nir_instr_type_intrinsic || !in_loop(loop, instr))
         continue;

      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];

      if (nir_intrinsic_has_atomic_op(intr)) {
         if (src != &intr->src[0])
            continue;

         nir_atomic_op op = nir_intrinsic_atomic_op(intr);
         if (op == nir_atomic_op_cmpxchg)
            n += updates(loop, intr);
         else if (op != nir_atomic_op_xchg)
            n++;
      } else if (!info->has_dest && info->num_srcs > 1 &&
                 src == &intr->src[1] && nir_intrinsic_has_access(intr) &&
                 (nir_intrinsic_access(intr) & ACCESS_ATOMIC)) {
         /* An atomic store: its value, then its address */
         n++;
      }
   }

   return n;
}

static bool
yields(nir_loop *loop)
{
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_yield_agx)
            return true;
      }
   }

   return false;
}

static bool
spins(nir_loop *loop)
{
   struct exits e = {.loop = loop};
   find_exits(&e, &loop->body);

   if (!e.any || yields(loop))
      return false;

   return !(e.one_address && progress_writes(loop, e.address) == 1);
}

static void
insert_backoff(nir_builder *b, nir_loop *loop)
{
   nir_variable *count =
      nir_local_variable_create(b->impl, glsl_uint_type(), "yield_count");
   nir_variable *amount =
      nir_local_variable_create(b->impl, glsl_uint16_t_type(), "yield_amount");

   b->cursor = nir_before_cf_node(&loop->cf_node);
   nir_store_var(b, count, nir_imm_int(b, 0), 1);
   nir_store_var(b, amount, nir_imm_intN_t(b, INITIAL, 16), 1);

   b->cursor = nir_before_cf_list(&loop->body);
   nir_def *c = nir_load_var(b, count);
   nir_store_var(b, count, nir_iadd_imm(b, c, 1), 1);
   nir_push_if(b, nir_uge_imm(b, c, SKIP + 1));
   {
      nir_def *a = nir_load_var(b, amount);
      nir_yield_agx(b, a);
      nir_store_var(b, amount, nir_umin_imm(b, nir_iadd_imm(b, a, 1), MAX), 1);
   }
   nir_pop_if(b, NULL);
}

static void
collect_loops(struct exec_list *list, struct util_dynarray *loops)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         collect_loops(&nif->then_list, loops);
         collect_loops(&nif->else_list, loops);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         collect_loops(&loop->body, loops);
         if (spins(loop))
            util_dynarray_append(loops, loop);
         break;
      }
      default:
         break;
      }
   }
}

bool
agx_nir_insert_yields(nir_shader *s)
{
   bool progress = false;

   nir_foreach_function_impl(impl, s) {
      struct util_dynarray loops = UTIL_DYNARRAY_INIT;
      collect_loops(&impl->body, &loops);

      nir_builder b = nir_builder_create(impl);
      util_dynarray_foreach(&loops, nir_loop *, loop) {
         insert_backoff(&b, *loop);
         progress = true;
      }

      util_dynarray_fini(&loops);
      nir_progress(progress, impl, nir_metadata_none);
   }

   return progress;
}
