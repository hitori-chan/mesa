/*
 * Copyright 2026 hitori1403
 * SPDX-License-Identifier: GPL-2.0-only
 */

/*
 * The G15 packer against Apple's bytes: each case builds one instruction and
 * compares what agx3_pack_instr emits with the encoding of Metal's ray-query
 * kernel (macOS 27.0.1), changed only where the IR differs on purpose (ldscs's
 * load group, a wait Metal folded in, stscs's last-use flag on its data).
 */

#include "agx_test.h"

#include <gtest/gtest.h>

static std::string
hex(const struct util_dynarray *d)
{
   std::string s;
   char buf[3];

   for (unsigned i = 0; i < d->size; i++) {
      snprintf(buf, sizeof(buf), "%02x", ((const uint8_t *)d->data)[i]);
      s += buf;
   }

   return s;
}

class Pack3 : public testing::Test {
 protected:
   Pack3()
   {
      mem_ctx = ralloc_context(NULL);
      b = agx_test_builder(mem_ctx);
   }

   ~Pack3()
   {
      ralloc_free(mem_ctx);
   }

   std::string pack(agx_instr *I)
   {
      struct util_dynarray emission, fixups;
      util_dynarray_init(&emission, mem_ctx);
      util_dynarray_init(&fixups, mem_ctx);
      I->g15_nosync = true;
      agx3_pack_instr(&emission, &fixups, I, 0);
      return hex(&emission);
   }

   void *mem_ctx;
   agx_builder *b;
};

/* 32-bit registers and uniforms by their 32-bit number */
static agx_index
r32(unsigned n)
{
   return agx_register(2 * n, AGX_SIZE_32);
}

static agx_index
u32(unsigned n)
{
   return agx_uniform(2 * n, AGX_SIZE_32);
}

TEST_F(Pack3, LoadScs)
{
   /* ldscs 2, i32, x, r17, r0, r11, lsl 3, 0 (Metal's group 5) */
   EXPECT_EQ(pack(agx_scs_load_to(b, r32(17), r32(0), r32(11), AGX_FORMAT_I32,
                                  1, 3, 0)),
             "5f025422002c2000400500006800");

   /* ldscs 2, i8, x, r2l, r0, r11, lsl 3, 263 (Metal's waits for group 5) */
   EXPECT_EQ(pack(agx_scs_load_to(b, agx_register(4, AGX_SIZE_16), r32(0),
                                  r32(11), AGX_FORMAT_I8, 1, 3, 263)),
             "5f025404002c2000808583006800");

   /* ldscs 2, i32, x, r0, 0, 0, 0x600c: an absolute address */
   EXPECT_EQ(pack(agx_scs_load_to(b, r32(0), agx_zero(), agx_zero(),
                                  AGX_FORMAT_I32, 1, 0, 0x600c)),
             "5f02540000000000400406300200");
}

TEST_F(Pack3, StoreScs)
{
   agx_index v4 = agx_register(10, AGX_SIZE_32);

   /* stscs i32, xyzw, r5_r6_r7_r8, u60, r1, 0 */
   EXPECT_EQ(
      pack(agx_scs_store(b, v4, u32(60), r32(1), AGX_FORMAT_I32, 0xf, 0, 0)),
      "df02540a780420005c0500400800");

   /* stscs i32, xyzw, r5_r6_r7_r8, u59, r38, lsl 2, 256 */
   EXPECT_EQ(
      pack(agx_scs_store(b, v4, u32(59), r32(38), AGX_FORMAT_I32, 0xf, 2, 256)),
      "df02540a769820005c0520c00800");

   /* stscs i32, xy, r7_r8, r0, r11, lsl 3, 0 */
   EXPECT_EQ(pack(agx_scs_store(b, r32(7), r32(0), r32(11), AGX_FORMAT_I32, 0x3,
                                3, 0)),
             "df02540e002c2000640100000d00");

   /* stscs i32, xyzw, r0_r1_r2_r3, 0, 0, 0x6000 (Metal's waits for group 5
    * and marks r0 a last use) */
   EXPECT_EQ(pack(agx_scs_store(b, r32(0), agx_zero(), agx_zero(),
                                AGX_FORMAT_I32, 0xf, 0, 0x6000)),
             "df025400000000005c00004c0000");
}

TEST_F(Pack3, ReleaseScs)
{
   /* scs_release 15, i32, r0, r4l, lsl 4, 0 */
   EXPECT_EQ(
      pack(agx_scs_release(b, r32(0), agx_register(8, AGX_SIZE_16), 0xf, 4, 0)),
      "5f02540f00100000402d00006100");
}

TEST_F(Pack3, IntersectRay)
{
   /* mov cr4096, r37, group 0; mov cr4097, r8, group 0;
    * intersect_ray du16, u46, u54, u62, 0, wait 0 */
   EXPECT_EQ(pack(agx_intersect_ray(b, agx_uniform(32, AGX_SIZE_64), u32(46),
                                    u32(54), u32(62), r32(37), r32(8), 0)),
             "0b4b2700420000001000"
             "1b112700420000001000"
             "5f10540000c125c0864f4c00");
}

TEST_F(Pack3, ContinueRay)
{
   std::string s = pack(agx_continue_ray(b, r32(3), 4));

   /* continue_ray 4, wait 0, after the slot's mov to cr2 */
   ASSERT_GE(s.size(), 12u);
   EXPECT_EQ(s.substr(s.size() - 12), "5f1154040000");
}
