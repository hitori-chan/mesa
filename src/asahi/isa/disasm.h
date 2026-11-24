/*
 * Copyright 2025 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/bitset.h"
#include "util/macros.h"

/* The bytes the disassemblers read an instruction from, whatever its length
 * (DISASM_MAX_WORDS) */
#define AGX_DISASM_WINDOW_B 24

/* The instruction at code + i, aligned and zero-padded past max_len: the
 * disassemblers read a whole window, and a shader's last instruction ends
 * where its buffer may */
static inline void
agx_disasm_window(BITSET_WORD *window, const uint8_t *code, size_t max_len,
                  size_t i)
{
   memset(window, 0, AGX_DISASM_WINDOW_B);
   memcpy(window, code + i, MIN2(max_len - i, AGX_DISASM_WINDOW_B));
}

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
      BITSET_WORD window[AGX_DISASM_WINDOW_B / sizeof(BITSET_WORD)];
      agx_disasm_window(window, code, max_len, i);
      const uint8_t *b = (const uint8_t *)window;

      /* Break on trap */
      if (b[0] == 0x08 && b[1] == 0x00)
         break;

      signed ret = agx2_disassemble_instr(window, fp, i, true);
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
      BITSET_WORD window[AGX_DISASM_WINDOW_B / sizeof(BITSET_WORD)];
      agx_disasm_window(window, code, max_len, i);
      const uint8_t *b = (const uint8_t *)window;

      signed ret = agx3_disassemble_instr(window, fp, i, true);
      if (ret < 0)
         fprintf(fp, "XXX error here\n");

      errors |= ret < 0;

      /* Break after stop */
      if (b[0] == 0x0e && b[1] == 0x00 && b[2] == 0x00 && b[3] == 0x00)
         break;

      i += abs(ret);
   }

   return errors;
}
