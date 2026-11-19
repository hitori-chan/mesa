/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include <stdlib.h>
#include "agx_builder.h"
#include "agx_compiler.h"

/*
 * A collect whose sources are read again after its vector's last use costs
 * moves: the vector and its sources are live together, so RA gives the vector
 * registers of its own and copies the sources in. Texture coordinates are the
 * common case: a parallax step reads its coordinates again for the next one,
 * so each sample waits on a dependent copy (Grim Dawn's G-buffer shaders, 30 a
 * pixel). The later uses read the vector instead, through a split after its
 * last use: the sources then die at the collect, which RA places where they
 * were, and the split, the vector's last use, splits it in place.
 *
 * Within one block: the vector's uses and every use of a source lie in the
 * collect's block, none between the collect and the vector's last use. Run
 * before RA, after scheduling, which would move the uses again.
 */
void
agx_opt_split_collect(agx_context *ctx)
{
   /* each value's last use, and the one block of its uses (MULTI past one) */
   unsigned n = ctx->alloc;
   agx_instr **last = calloc(n, sizeof(*last));
   agx_block **where = calloc(n, sizeof(*where));
   agx_block *const MULTI = (agx_block *)(uintptr_t)1;

   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         agx_foreach_ssa_src(I, s) {
            unsigned v = I->src[s].value;

            /* a phi reads its source in a predecessor */
            if (I->op == AGX_OPCODE_PHI || (where[v] && where[v] != block))
               where[v] = MULTI;
            else
               where[v] = block;

            last[v] = I;
         }
      }
   }

   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block_safe(block, C) {
         if (C->op != AGX_OPCODE_COLLECT || C->nr_srcs > 4 ||
             C->dest[0].type != AGX_INDEX_NORMAL)
            continue;

         agx_index vec = C->dest[0];
         agx_instr *L = last[vec.value];

         /* the split goes after L, which must not end the block */
         if (!L || where[vec.value] != block ||
             L == list_last_entry(&block->instructions, agx_instr, link))
            continue;

         bool ok = true, later = false;

         agx_foreach_src(C, s) {
            agx_index src = C->src[s];

            /* a split's part, made here, is not counted: n */
            ok &= src.type == AGX_INDEX_NORMAL && src.size == vec.size &&
                  src.value < n && where[src.value] == block;

            for (unsigned t = 0; t < s; ++t)
               ok &= C->src[t].value != src.value;

            later |= ok && last[src.value] != C;
         }

         if (!ok || !later)
            continue;

         /* no source read between the collect and L, L included */
         for (agx_instr *I = list_entry(C->link.next, agx_instr, link);;) {
            agx_foreach_ssa_src(I, s) {
               agx_foreach_src(C, c)
                  ok &= I->src[s].value != C->src[c].value;
            }

            if (I == L || !ok)
               break;

            I = list_entry(I->link.next, agx_instr, link);
         }

         if (!ok)
            continue;

         /* the sources read after L, which the split gives a part each: RA
          * takes a destination never read as live to the end of the block, so
          * the others are null, as agx_dce leaves them
          */
         unsigned left = 0;
         agx_foreach_src(C, c)
            left |= (last[C->src[c].value] != C) << c;

         agx_builder b = agx_init_builder(ctx, agx_after_instr(L));
         agx_instr *split = agx_split(&b, C->nr_srcs, vec);
         agx_index parts[4];

         agx_foreach_dest(split, d) {
            parts[d] = (left & BITFIELD_BIT(d)) ? agx_temp(ctx, vec.size)
                                                : agx_null();
            split->dest[d] = parts[d];
         }

         /* the uses after L read the split's parts, up to the sources' last
          * uses (last[] keeps them: they come after the split)
          */

         for (agx_instr *I = list_entry(split->link.next, agx_instr, link);
              left; I = list_entry(I->link.next, agx_instr, link)) {
            agx_foreach_src(C, c) {
               if (!(left & BITFIELD_BIT(c)))
                  continue;

               agx_foreach_ssa_src(I, s) {
                  if (I->src[s].value == C->src[c].value)
                     I->src[s].value = parts[c].value;
               }

               if (I == last[C->src[c].value])
                  left &= ~BITFIELD_BIT(c);
            }
         }
      }
   }

   free(last);
   free(where);
}
