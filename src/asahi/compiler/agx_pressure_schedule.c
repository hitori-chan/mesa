/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

/* Bottom-up local scheduler to reduce register pressure */

#include "util/bitset.h"
#include "util/dag.h"
#include "util/sparse_bitset.h"
#include "util/u_dynarray.h"
#include "util/u_debug.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "agx_opcodes.h"

struct sched_ctx {
   /* Dependency graph */
   struct dag *dag;

   /* Live set, dense: calculate_pressure_delta tests it for every source of
    * every candidate at every step */
   BITSET_WORD *live;
   unsigned alloc;

   /* G15: issue the textures early (agx_pressure_schedule), while the live
    * registers (pressure, in 16-bit halves) stay within budget */
   bool early_tex;
   signed pressure, budget;

   /* The block's readers of each value, so that a value becoming live marks
    * the deltas that change: first_reader[value] indexes readers, -1 none */
   int32_t *first_reader;
   struct util_dynarray readers;
};

struct sched_node {
   struct dag_node dag;

   /* Instruction this node represents */
   agx_instr *instr;

   /* calculate_pressure_delta of the instruction, unless dirty: only a source
    * becoming live changes it once the node is a candidate, since its readers
    * are scheduled by then */
   int32_t delta;
   bool dirty;
};

struct sched_reader {
   struct sched_node *node;
   int32_t next;
};

static void
add_dep(struct sched_node *a, struct sched_node *b)
{
   assert(a != b && "no self-dependencies");

   if (a && b)
      dag_add_edge(&a->dag, &b->dag, 0);
}

static void
serialize(struct sched_node *a, struct sched_node **b)
{
   add_dep(a, *b);
   *b = a;
}

/* last_write is all NULL on entry and on return */
static struct dag *
create_dag(agx_context *ctx, agx_block *block, void *memctx,
           struct sched_node **last_write, struct sched_ctx *s)
{
   struct dag *dag = dag_create(ctx);

   struct sched_node *coverage = NULL;
   struct sched_node *preload = NULL;

   /* Last memory load, to serialize stores against */
   struct sched_node *memory_load = NULL;

   /* Last memory store, to serialize loads and stores against */
   struct sched_node *memory_store = NULL;

   agx_foreach_instr_in_block(block, I) {
      /* Don't touch control flow */
      if (instr_after_logical_end(I))
         break;

      struct sched_node *node = rzalloc(memctx, struct sched_node);
      node->instr = I;
      node->dirty = true;
      dag_init_node(dag, &node->dag);

      /* Reads depend on writes, no other hazards in SSA */
      agx_foreach_ssa_src(I, src) {
         unsigned v = I->src[src].value;
         add_dep(node, last_write[v]);

         struct sched_reader r = {node, s->first_reader[v]};
         s->first_reader[v] = util_dynarray_num_elements(&s->readers,
                                                         struct sched_reader);
         util_dynarray_append(&s->readers, r);
      }

      agx_foreach_ssa_dest(I, d) {
         assert(I->dest[d].value < ctx->alloc);
         last_write[I->dest[d].value] = node;
      }

      /* Classify the instruction and add dependencies according to the class */
      enum agx_schedule_class dep = agx_opcodes_info[I->op].schedule_class;
      assert(dep != AGX_SCHEDULE_CLASS_INVALID && "invalid instruction seen");

      /* G15 samples after the pixel fence its fragment shaders wait on
       * (agx_compile.c): keep texture accesses below the barriers */
      if (ctx->key->dev.gpu_generation >= 15 &&
          (I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
           I->op == AGX_OPCODE_TEXTURE_LOAD || I->op == AGX_OPCODE_IMAGE_LOAD))
         dep = AGX_SCHEDULE_CLASS_LOAD;

      bool barrier = dep == AGX_SCHEDULE_CLASS_BARRIER;
      bool discards =
         I->op == AGX_OPCODE_SAMPLE_MASK || I->op == AGX_OPCODE_ZS_EMIT;

      if (dep == AGX_SCHEDULE_CLASS_STORE)
         add_dep(node, memory_load);
      else if (dep == AGX_SCHEDULE_CLASS_ATOMIC || barrier)
         serialize(node, &memory_load);

      if (dep == AGX_SCHEDULE_CLASS_LOAD || dep == AGX_SCHEDULE_CLASS_STORE ||
          dep == AGX_SCHEDULE_CLASS_ATOMIC || barrier)
         serialize(node, &memory_store);

      if (dep == AGX_SCHEDULE_CLASS_COVERAGE || barrier)
         serialize(node, &coverage);

      /* Make sure side effects happen before a discard */
      if (discards)
         add_dep(node, memory_store);

      if (dep == AGX_SCHEDULE_CLASS_PRELOAD)
         serialize(node, &preload);
      else
         add_dep(node, preload);
   }

   agx_foreach_instr_in_block(block, I) {
      if (instr_after_logical_end(I))
         break;

      agx_foreach_ssa_dest(I, d)
         last_write[I->dest[d].value] = NULL;
   }

   return dag;
}

/*
 * Calculate the change in register pressure from scheduling a given
 * instruction. Equivalently, calculate the difference in the number of live
 * registers before and after the instruction, given the live set after the
 * instruction. This calculation follows immediately from the dataflow
 * definition of liveness:
 *
 *      live_in = (live_out - KILL) + GEN
 */
static signed
calculate_pressure_delta(agx_instr *I, const BITSET_WORD *live)
{
   signed delta = 0;

   /* Destinations must be unique */
   agx_foreach_ssa_dest(I, d) {
      if (BITSET_TEST(live, I->dest[d].value))
         delta -= agx_index_size_16(I->dest[d]);
   }

   agx_foreach_ssa_src(I, src) {
      /* Filter duplicates */
      bool dupe = false;

      for (unsigned i = 0; i < src; ++i) {
         if (agx_is_equiv(I->src[i], I->src[src])) {
            dupe = true;
            break;
         }
      }

      if (!dupe && !BITSET_TEST(live, I->src[src].value))
         delta += agx_index_size_16(I->src[src]);
   }

   return delta;
}

static bool
is_texture(const agx_instr *I)
{
   return I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
          I->op == AGX_OPCODE_TEXTURE_LOAD || I->op == AGX_OPCODE_IMAGE_LOAD;
}

/*
 * Choose the next instruction, bottom-up. For now we use a simple greedy
 * heuristic: choose the instruction that has the best effect on liveness, while
 * hoisting sample_mask.
 */
static struct sched_node *
choose_instr(struct sched_ctx *s)
{
   int32_t min_delta = INT32_MAX, tex_delta = INT32_MAX;
   struct sched_node *best = NULL, *tex = NULL, *last = NULL;

   list_for_each_entry(struct sched_node, n, &s->dag->heads, dag.link) {
      /* Heuristic: hoist sample_mask/zs_emit. This allows depth/stencil tests
       * to run earlier, and potentially to discard the entire quad invocation
       * earlier, reducing how much redundant fragment shader we run.
       *
       * Since we schedule backwards, we make that happen by only choosing
       * sample_mask when all other instructions have been exhausted.
       */
      if (n->instr->op == AGX_OPCODE_SAMPLE_MASK ||
          n->instr->op == AGX_OPCODE_ZS_EMIT) {

         if (!last)
            last = n;

         continue;
      }

      /* Heuristic: sink wait_pix to increase parallelism. Since wait_pix does
       * not read or write registers, this has no effect on pressure.
       */
      if (n->instr->op == AGX_OPCODE_WAIT_PIX)
         return n;

      if (n->dirty) {
         n->delta = calculate_pressure_delta(n->instr, s->live);
         n->dirty = false;
      }

      int32_t delta = n->delta;

      /* G15 heuristic: hide a texture's latency by choosing it after the
       * instructions that do not need it, so that it issues before them, the
       * textures of a block together, and their readers wait late, as
       * Apple's B1 code samples back to back. Its result stays live meanwhile,
       * so only while the pressure allows.
       */
      if (s->early_tex && is_texture(n->instr) && s->pressure <= s->budget) {
         if (delta < tex_delta) {
            tex = n;
            tex_delta = delta;
         }

         continue;
      }

      if (delta < min_delta) {
         best = n;
         min_delta = delta;
      }
   }

   /* ...and not past the budget: a schedule over it is not kept */
   if (best && tex && s->pressure + min_delta > s->budget)
      return tex;

   return best ? best : tex ? tex : last;
}

/* The live set at a block's end */
static void
live_init(BITSET_WORD *live, unsigned alloc, struct u_sparse_bitset *live_out)
{
   memset(live, 0, BITSET_BYTES(alloc));

   U_SPARSE_BITSET_FOREACH_SET(live_out, v)
      BITSET_SET(live, v);
}

/* The live set before I from the one after it, as agx_liveness_ins_update */
static void
live_update(BITSET_WORD *live, agx_instr *I)
{
   agx_foreach_ssa_dest(I, d)
      BITSET_CLEAR(live, I->dest[d].value);

   agx_foreach_ssa_src(I, s)
      BITSET_SET(live, I->src[s].value);
}

/* The 16-bit halves live */
static signed
live_size(const BITSET_WORD *live, unsigned alloc, const uint8_t *size)
{
   signed n = 0;
   unsigned v;

   BITSET_FOREACH_SET(v, live, alloc)
      n += size[v];

   return n;
}

static void
pressure_schedule_block(agx_context *ctx, agx_block *block, struct sched_ctx *s,
                        const uint8_t *size)
{
   /* off by a constant, that's ok */
   signed pressure = 0;
   signed orig_max_pressure = 0;
   unsigned nr_ins = 0;
   bool textures = false;

   live_init(s->live, s->alloc, &block->live_out);

   agx_foreach_instr_in_block_rev(block, I) {
      pressure += calculate_pressure_delta(I, s->live);
      orig_max_pressure = MAX2(pressure, orig_max_pressure);
      live_update(s->live, I);
      textures |= is_texture(I);
      nr_ins++;
   }

   live_init(s->live, s->alloc, &block->live_out);
   signed base = s->early_tex ? live_size(s->live, s->alloc, size) : 0;
   s->pressure = base;

   /* off by a constant, that's ok */
   signed max_pressure = 0;
   pressure = 0;

   struct sched_node **schedule = calloc(nr_ins, sizeof(struct sched_node *));
   nr_ins = 0;

   while (!list_is_empty(&s->dag->heads)) {
      struct sched_node *node = choose_instr(s);
      signed delta = calculate_pressure_delta(node->instr, s->live);
      pressure += delta;
      s->pressure += delta;
      max_pressure = MAX2(pressure, max_pressure);
      dag_prune_head(s->dag, &node->dag);

      schedule[nr_ins++] = node;

      /* A source becoming live changes the deltas of its other readers */
      agx_foreach_ssa_src(node->instr, src) {
         unsigned v = node->instr->src[src].value;
         if (!BITSET_TEST(s->live, v)) {
            for (int32_t r = s->first_reader[v]; r >= 0;) {
               struct sched_reader *e = util_dynarray_element(
                  &s->readers, struct sched_reader, r);
               e->node->dirty = true;
               r = e->next;
            }
         }
      }

      live_update(s->live, node->instr);
   }

   /* Bail if it looks like it's worse, unless it issues textures early within
    * the budget */
   bool early =
      s->early_tex && textures &&
      base + max_pressure <= MAX2(s->budget, base + orig_max_pressure);

   if (max_pressure >= orig_max_pressure && !early) {
      free(schedule);
      return;
   }

   /* Apply the schedule */
   for (unsigned i = 0; i < nr_ins; ++i) {
      agx_remove_instruction(schedule[i]->instr);
      list_add(&schedule[i]->instr->link, &block->instructions);
   }

   free(schedule);
}

/*
 * G15: the budget for issuing textures early, in 16-bit halves, or -1 to not
 * issue early: the shader's pressure and AGX_G15_TEX_SLACK more (default 16),
 * within the occupancy that pressure already has (agx_ra rounds its registers
 * up to it) less a margin for the allocator. The slack is small because M3's
 * register file is allocated as the threads use it (Dynamic Caching), so
 * that the occupancy may fall before G13's table says (not measured).
 */
DEBUG_GET_ONCE_NUM_OPTION(agx_g15_tex_slack, "AGX_G15_TEX_SLACK", 16)

static signed
early_tex_budget(agx_context *ctx, const uint8_t *size, void *memctx)
{
   BITSET_WORD *live = ralloc_array(memctx, BITSET_WORD, BITSET_WORDS(ctx->alloc));
   signed max = 0;

   agx_foreach_block(ctx, block) {
      live_init(live, ctx->alloc, &block->live_out);
      signed pressure = live_size(live, ctx->alloc, size);
      max = MAX2(max, pressure);

      agx_foreach_instr_in_block_rev(block, I) {
         pressure += calculate_pressure_delta(I, live);
         max = MAX2(max, pressure);
         live_update(live, I);
      }
   }

   if (ctx->key->is_helper || max > AGX_G15_MAX_HALFREGS)
      return -1;

   signed regs =
      agx_occupancy_for_register_count(ALIGN_POT(MAX2(max, 1), 8)).max_registers;
   signed slack = debug_get_option_agx_g15_tex_slack();
   return MIN3(regs, AGX_G15_MAX_HALFREGS, max + slack + 16) - 16;
}

void
agx_pressure_schedule(agx_context *ctx)
{
   agx_compute_liveness(ctx);
   void *memctx = ralloc_context(ctx);

   /* G15 issues its textures early (choose_instr) */
   bool early_tex = ctx->key->dev.gpu_generation >= 15 &&
                    !(agx_compiler_debug & AGX_DBG_NOLATENCY);
   uint8_t *size = NULL;
   signed budget = -1;

   if (early_tex) {
      size = rzalloc_array(memctx, uint8_t, ctx->alloc);
      agx_foreach_instr_global(ctx, I) {
         agx_foreach_ssa_dest(I, d)
            size[I->dest[d].value] = agx_index_size_16(I->dest[d]);
      }

      budget = early_tex_budget(ctx, size, memctx);
      early_tex = budget >= 0;
   }

   BITSET_WORD *live = ralloc_array(memctx, BITSET_WORD, BITSET_WORDS(ctx->alloc));
   struct sched_node **last_write =
      rzalloc_array(memctx, struct sched_node *, ctx->alloc);
   int32_t *first_reader = ralloc_array(memctx, int32_t, ctx->alloc);
   memset(first_reader, 0xff, ctx->alloc * sizeof(int32_t));

   struct sched_ctx sctx = {
      .live = live,
      .alloc = ctx->alloc,
      .early_tex = early_tex,
      .budget = budget,
      .first_reader = first_reader,
   };
   util_dynarray_init(&sctx.readers, memctx);

   agx_foreach_block(ctx, block) {
      sctx.dag = create_dag(ctx, block, memctx, last_write, &sctx);
      pressure_schedule_block(ctx, block, &sctx, size);

      /* Back to no readers for the next block */
      util_dynarray_foreach(&sctx.readers, struct sched_reader, r) {
         agx_foreach_ssa_src(r->node->instr, src)
            first_reader[r->node->instr->src[src].value] = -1;
      }
      util_dynarray_clear(&sctx.readers);
   }

   /* Clean up after liveness analysis */
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s)
         I->src[s].kill = false;
   }

   ralloc_free(memctx);
}
