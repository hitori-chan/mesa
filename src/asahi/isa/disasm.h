/*
 * Copyright 2025 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include "util/bitset.h"

signed agx2_disassemble_instr(BITSET_WORD *code, FILE *fp, unsigned offset,
                              bool verbose);

/* G15 (M3) */
signed agx3_disassemble_instr(BITSET_WORD *code, FILE *fp, unsigned offset,
                              bool verbose);

static inline bool
agx2_disassemble(uint8_t *code, size_t max_len, FILE *fp)
{
   int i = 0;
   bool errors = false;

   while (i < max_len) {
      /* Break on trap */
      if (code[i + 0] == 0x08 && code[i + 1] == 0x00)
         break;

      signed ret =
         agx2_disassemble_instr((BITSET_WORD *)(code + i), fp, i, true);
      if (ret < 0)
         fprintf(fp, "XXX error here\n");

      errors |= ret < 0;
      i += abs(ret);
   }

   return errors;
}

static inline bool
agx3_disassemble(uint8_t *code, size_t max_len, FILE *fp)
{
   int i = 0;
   bool errors = false;

   while (i < max_len) {
      signed ret =
         agx3_disassemble_instr((BITSET_WORD *)(code + i), fp, i, true);
      if (ret < 0)
         fprintf(fp, "XXX error here\n");

      errors |= ret < 0;

      /* Break after stop */
      if (code[i + 0] == 0x0e && code[i + 1] == 0x00 && code[i + 2] == 0x00 &&
          code[i + 3] == 0x00)
         break;

      i += abs(ret);
   }

   return errors;
}
