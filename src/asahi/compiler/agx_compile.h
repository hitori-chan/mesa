/*
 * Copyright 2018-2021 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "compiler/nir/nir.h"
#include "util/shader_stats.h"
#include "util/u_dynarray.h"
#include "util/u_tristate.h"
#include "shader_enums.h"

struct agx_cf_binding {
   /* Base coefficient register */
   uint8_t cf_base;

   /* Slot being bound */
   gl_varying_slot slot : 8;

   /* First component bound.
    *
    * Must be 2 (Z) or 3 (W) if slot == VARYING_SLOT_POS.
    */
   unsigned offset : 2;

   /* Number of components bound */
   unsigned count : 3;

   /* Is smooth shading enabled? If false, flat shading is used */
   bool smooth : 1;

   /* Perspective correct interpolation */
   bool perspective : 1;

   uint8_t pad;
};

/* Conservative bound, * 4 due to offsets (TODO: maybe worth eliminating
 * coefficient register aliasing?)
 */
#define AGX_MAX_CF_BINDINGS (VARYING_SLOT_MAX * 4)

struct agx_varyings_fs {
   /* Number of coefficient registers used */
   unsigned nr_cf;

   /* Number of coefficient register bindings */
   unsigned nr_bindings;

   /* Whether gl_FragCoord.z is read */
   bool reads_z;

   /* Coefficient register bindings */
   struct agx_cf_binding bindings[AGX_MAX_CF_BINDINGS];
};

union agx_varyings {
   struct agx_varyings_fs fs;
};

struct agx_interp_info {
   /* Bit masks indexed by I/O location of flat and linear varyings */
   uint64_t flat;
   uint64_t linear;
};
static_assert(sizeof(struct agx_interp_info) == 16, "packed");

struct agx_rodata {
   /* Offset in the binary */
   uint32_t offset;

   /* Base uniform to map constants */
   uint16_t base_uniform;

   /* Number of 16-bit constants to map contiguously there */
   uint16_t size_16;
};

struct agx_shader_info {
   mesa_shader_stage stage;
   uint32_t binary_size;

   union agx_varyings varyings;

   /* Number of uniforms */
   uint16_t push_count;

   /* Local memory allocation in bytes */
   uint16_t local_size;

   /* Local imageblock allocation in bytes per thread */
   uint16_t imageblock_stride;

   /* Scratch memory allocation in bytes for main/preamble respectively */
   unsigned scratch_size, preamble_scratch_size;

   /* G15: the stack registers in bytes per thread (specTPR) */
   unsigned stack_size;

   /* G15: the coordinate registers (cr) the shader uses, the highest written
    * plus one: the specTPR backs each with 8 bytes per thread (Apple's
    * kernels and render programs, 2026-10-16) */
   unsigned cr_count;

   /* Size in bytes of the main sahder */
   unsigned main_size;

   /* G15: the pixel fences (wait_pix masks) the shader waits on. A fragment
    * program signals them at its end, and its entry program releases all the
    * others (demo15's fragment shader waits 0x20C and signals it, its entry
    * program releases 0x2F3).
    */
   uint16_t pix_wait_mask;

   /* G15: the fences of pix_wait_mask the program releases itself before its
    * end (fence 1 right after zs_emit, as Apple's depth-output shader)
    */
   uint16_t pix_signal_mask;

   /* Does the shader have a preamble? If so, it is at offset preamble_offset.
    * The main shader is at offset main_offset. The preamble is executed first.
    */
   bool has_preamble;
   unsigned preamble_offset, main_offset;

   /* G15: the preamble's bytes before its stop, which the entry program
    * copies (lib/agx_esl.c), 0 to find them there
    */
   unsigned preamble_size;

   /* G15: the preamble did not fit in the registers, so the shader is
    * compiled again without one (agx_compile_shader_nir)
    */
   bool preamble_too_big;

   /* Does the shader read the tilebuffer? */
   bool reads_tib;

   /* Does the shader require early fragment tests? */
   bool early_fragment_tests;

   /* Does the shader potentially draw to a nonzero viewport? */
   bool nonzero_viewport;

   /* Does the shader write layer and/or viewport index? Written together */
   bool writes_layer_viewport;

   /* Does the shader control the sample mask? */
   bool writes_sample_mask;

   /* Does the shader write depth or stencil (zs_emit)? */
   bool writes_zs;

   /* Depth layout, never equal to NONE */
   enum gl_frag_depth_layout depth_layout;

   /* Based only the compiled shader, should tag writes be disabled? This is set
    * based on what is outputted. Note if rasterizer discard is used, that needs
    * to disable tag writes regardless of this flag.
    */
   bool tag_write_disable;

   /* Shader is incompatible with triangle merging */
   bool disable_tri_merging;

   /* Reads draw ID system value */
   bool uses_draw_id;

   /* Reads base vertex/instance */
   bool uses_base_param;

   /* Uses txf and hence needs a txf sampler mapped */
   bool uses_txf;

   /* Potentially uses the sampler heap (conservative) */
   bool uses_sampler_heap;

   /* Number of texture/sampler state registers pushed by the preamble. */
   uint8_t texture_state_count, sampler_state_count;

   /* Number of 16-bit registers used by the main shader and preamble
    * respectively.
    */
   uint16_t nr_gprs, nr_preamble_gprs;

   /* Output mask set during driver lowering */
   uint64_t outputs;

   /* Workgroup size */
   uint16_t workgroup_size[3];

   /* There may be constants in the binary. The driver must map these to uniform
    * registers as specified hre.
    */
   struct agx_rodata rodata;

   struct agx2_stats stats;
};

struct agx_precompiled_kernel_info {
   uint32_t preamble_offset, preamble_size, main_offset;
   uint32_t main_size, binary_size;
   struct agx_rodata rodata;
   uint16_t nr_gprs, nr_preamble_gprs;
   uint16_t push_count;
   uint16_t workgroup_size[3];
   uint16_t local_size;
   uint16_t imageblock_stride;
   bool uses_txf;

   /* G15's entry program (lib/agx_esl.c) */
   uint32_t stack_size;
   uint32_t cr_count;
};

static inline struct agx_precompiled_kernel_info
agx_compact_kernel_info(struct agx_shader_info *info)
{
   assert(info->has_preamble == (info->nr_preamble_gprs > 0));
   assert(info->texture_state_count <= 8 && "static maximum, no need to plumb");

   return (struct agx_precompiled_kernel_info){
      .preamble_offset = info->preamble_offset,
      .preamble_size = info->preamble_size,
      .main_offset = info->main_offset,
      .main_size = info->main_size,
      .binary_size = info->binary_size,
      .rodata = info->rodata,
      .nr_gprs = info->nr_gprs,
      .nr_preamble_gprs = info->nr_preamble_gprs,
      .push_count = info->push_count,
      .workgroup_size = {info->workgroup_size[0], info->workgroup_size[1],
                         info->workgroup_size[2]},
      .local_size = info->local_size,
      .imageblock_stride = info->imageblock_stride,
      .uses_txf = info->uses_txf,
      .stack_size = info->stack_size,
      .cr_count = info->cr_count,
   };
}

/* G15: a preamble reads pushed uniform word N from register N +
 * AGX_G15_PUSH_REG, where the entry program loads it (lib/agx_esl.c) */
#define AGX_G15_PUSH_REG 2

/* Only the words below this go through those registers: a fragment program
 * whose entry program wrote r95 hung the GPU, one that stopped at r94 ran
 * (boots #391-#393: shaders with 40 promoted constants, whose loads reach
 * word 93; synthetic ones with 80 to 120), so the entry program stays
 * below r90 */
#define AGX_G15_PUSH_WORDS 88

/* The registers a G15 program may use, in 16-bit halves: r0-r93. A main
 * program that used r97 hung the GPU and one that stopped at r81 ran; an entry
 * program that wrote r95 hung and one that stopped at r94 ran
 * (compute-program.md, Registers). Apple's compiler gives G15C 96 GPRs
 * (AGCLLVMG15TargetLowerer +0x42C = 0x60), and no shader of the QuartzCore
 * corpus uses a register above r93. The allocator spills the rest to the
 * thread-private stack. Its register file is a multiple of 8 halves (16 with a
 * 6-word gradient vector), so it ends at r91 (r87) in practice, which still
 * holds the pushed words a preamble reads in place (r2 to r89).
 * AGX_G15_MAX_REG=N makes rN the highest register instead, for an experiment
 * (in the same steps; the disk cache ignores it). */
#define AGX_G15_MAX_HALFREGS 188

/* G15 (AGX3) building blocks for the entry program, from the compiler's
 * G15 packer (agx_pack3.c). Registers and uniforms are 32-bit indices. */
struct util_dynarray;
void agx3_emit_mov_imm(struct util_dynarray *out, unsigned reg, uint32_t imm);
void agx3_emit_load(struct util_dynarray *out, unsigned slot, unsigned dst,
                    unsigned base, unsigned index16, unsigned count);
void agx3_emit_wait(struct util_dynarray *out, unsigned slot);
void agx3_emit_uniform_store(struct util_dynarray *out, unsigned uniform,
                             unsigned src);
void agx3_emit_stop(struct util_dynarray *out);
void agx3_emit_signal_pix(struct util_dynarray *out, unsigned mask);
void agx3_emit_speclm(struct util_dynarray *out, unsigned samples,
                      unsigned stride, unsigned shape, bool imageblock,
                      unsigned threadgroup_64B);
void agx3_emit_ldshdr(struct util_dynarray *out, uint64_t va,
                      unsigned size_class);
void agx3_emit_spectpr(struct util_dynarray *out, unsigned crs,
                       unsigned stack_words);
void agx3_emit_sample_loop_header(struct util_dynarray *out);
void agx3_emit_sample_loop_footer(struct util_dynarray *out,
                                  unsigned nr_samples, unsigned footer_offset,
                                  unsigned loop_begin);
unsigned agx3_instr_length(const uint8_t *code, size_t size);
int agx3_stop_offset(const uint8_t *code, size_t size);

struct agx_shader_part {
   struct agx_shader_info info;
   void *binary;
};

static inline bool
agx_is_shader_empty(struct agx_shader_part *s)
{
   /* Last instruction is a stop, so if there's one instruction, there is
    * nothing but a stop. The shader is thus empty.
    */
   return (s->info.stats.instrs == 1);
}

#define AGX_MAX_RTS (8)

enum agx_format {
   AGX_FORMAT_I8 = 0,
   AGX_FORMAT_I16 = 1,
   AGX_FORMAT_I32 = 2,
   AGX_FORMAT_F16 = 3,
   AGX_FORMAT_U8NORM = 4,
   AGX_FORMAT_S8NORM = 5,
   AGX_FORMAT_U16NORM = 6,
   AGX_FORMAT_S16NORM = 7,
   AGX_FORMAT_RGB10A2 = 8,
   AGX_FORMAT_SRGBA8 = 10,
   AGX_FORMAT_RG11B10F = 12,
   AGX_FORMAT_RGB9E5 = 13,

   /* Keep last */
   AGX_NUM_FORMATS,
};

struct agx_fs_shader_key {
   /* Normally, access to the tilebuffer must be guarded by appropriate fencing
    * instructions to ensure correct results in the presence of out-of-order
    * hardware optimizations. However, specially dispatched clear shaders are
    * not subject to these conditions and can omit the wait instructions.
    *
    * Must (only) be set for special clear shaders.
    *
    * Must not be used with sample mask writes (including discards) or
    * tilebuffer loads (including blending).
    */
   bool ignore_tib_dependencies;

   /* When dynamic sample shading is used, the fragment shader is wrapped in a
    * loop external to the API shader. This bit indicates that we are compiling
    * inside the sample loop, meaning the execution nesting counter is already
    * zero and must be preserved.
    */
   bool inside_sample_loop;

   /* Base coefficient register. 0 for API shaders but nonzero for FS prolog */
   uint8_t cf_base;

   /* The tilebuffer's sample count, or 0 if the shader does not know it (a
    * main fragment shader is compiled before the render pass is). G15 takes
    * the sample mask of a tile access literally: all samples is
    * (1 << nr_samples) - 1, not G13's 0xFF.
    */
   uint8_t nr_samples;
};

struct agx_device_key {
   /* Does the target GPU need explicit cluster coherency for atomics?
    * Only used on G13X.
    */
   enum u_tristate needs_g13x_coherency;

   /* Is soft fault enabled? This is technically system-wide policy set by the
    * kernel, but that's functionally a hardware feature.
    */
   bool soft_fault;

   /* GPU generation (13, 14 or 15). 0 is G13/G14 machine code, which is what
    * shaders built without a device (e.g. precompiled libagx) get. G15 (M3)
    * has its own instruction encoding, AGX3.
    */
   uint8_t gpu_generation;
};

struct agx_shader_key {
   /* Device info */
   struct agx_device_key dev;

   /* Number of reserved preamble slots at the start */
   unsigned reserved_preamble;

   /* Whether scratch memory is available in the given shader stage */
   bool has_scratch;

   /* Whether we're compiling the helper program used for scratch allocation.
    * This has special register allocation requirements.
    */
   bool is_helper;

   /* Whether the driver supports uploading constants for this shader. If
    * false, constants will not be promoted to uniforms.
    */
   bool promote_constants;

   /* Similarly whether the driver supports promoting bindless
    * textures/samplers.  Currently this works only if non-bindless
    * textures/samplers are not used, but none of our drivers mix bindless /
    * non-bindless usage.
    */
   bool promote_textures;

   /* Set if this is a non-monolithic shader that must be linked with additional
    * shader parts before the program can be used. This suppresses omission of
    * `stop` instructions, which the linker must insert instead.
    */
   bool no_stop;

   /* Set if this is a secondary shader part (prolog or epilog). This prevents
    * the compiler from allocating uniform registers. For example, this turns
    * off preambles.
    */
   bool secondary;

   /* Set for a vertex shader linked after a vertex prolog, which exports the
    * vertex and instance IDs. G15 reads them from special registers otherwise.
    */
   bool vs_prolog;

   union {
      struct agx_fs_shader_key fs;
   };
};

uint64_t agx_gather_texcoords(nir_shader *nir);

void agx_preprocess_nir(nir_shader *nir);
bool agx_nir_lower_discard_zs_emit(nir_shader *s);
bool agx_nir_lower_stencil_only_g15(nir_shader *s);
bool agx_nir_lower_sample_mask(nir_shader *s);
bool agx_nir_lower_interpolation(nir_shader *s, bool g15);

bool agx_nir_lower_cull_distance_vs(struct nir_shader *s);
bool agx_nir_lower_cull_distance_fs(struct nir_shader *s,
                                    unsigned nr_distances);
bool agx_mem_vectorize_cb(unsigned align_mul, unsigned align_offset,
                          unsigned bit_size, unsigned num_components,
                          int64_t hole_size, nir_intrinsic_instr *low,
                          nir_intrinsic_instr *high, void *data);

void agx_compile_shader_nir(nir_shader *nir, struct agx_shader_key *key,
                            struct agx_shader_part *out);

struct agx_occupancy {
   unsigned max_registers;
   unsigned max_threads;
};

struct agx_occupancy agx_occupancy_for_register_count(unsigned halfregs);
unsigned agx_max_registers_for_occupancy(unsigned occupancy);

/* G15: whether G13's occupancy table bounds a compute program's registers by
 * its workgroup size, and its threads by its registers (AGX_G15_WG_CAP) */
bool agx_g15_wg_cap(void);

static inline unsigned
agx_round_registers(unsigned halfregs)
{
   return agx_occupancy_for_register_count(halfregs).max_registers;
}

static const nir_shader_compiler_options agx_nir_options = {
   .lower_fdiv = true,
   .float_mul_add16 = nir_float_muladd_support_has_ffma | nir_float_muladd_support_fuse,
   .float_mul_add32 = nir_float_muladd_support_has_ffma | nir_float_muladd_support_fuse,
   .lower_flrp16 = true,
   .lower_flrp32 = true,
   .lower_flrp64 = true,
   .lower_fpow = true,
   .lower_fmod = true,
   .lower_bitfield_extract8 = true,
   .lower_bitfield_extract16 = true,
   .lower_bitfield_insert = true,
   .lower_ifind_msb = true,
   .lower_find_lsb = true,
   .lower_fisnormal = true,
   .lower_scmp = true,
   .lower_isign = true,
   .lower_fsign = true,
   .lower_iabs = true,
   .lower_fminmax_signed_zero = true,
   .lower_fdph = true,
   .lower_ffract = true,
   .lower_pack_half_2x16 = true,
   .lower_pack_unorm_2x16 = true,
   .lower_pack_snorm_2x16 = true,
   .lower_pack_unorm_4x8 = true,
   .lower_pack_snorm_4x8 = true,
   .lower_pack_64_2x32 = true,
   .lower_unpack_half_2x16 = true,
   .lower_unpack_unorm_2x16 = true,
   .lower_unpack_snorm_2x16 = true,
   .lower_unpack_unorm_4x8 = true,
   .lower_unpack_snorm_4x8 = true,
   .lower_extract_byte = true,
   .lower_insert_byte = true,
   .lower_insert_word = true,
   .has_cs_global_id = true,
   .lower_device_index_to_zero = true,
   .lower_hadd = true,
   .has_amul = true,
   .has_isub = true,
   .has_load_global_bounded = true,
   .support_16bit_alu = true,
   .max_unroll_iterations = 32,
   .lower_uniforms_to_ubo = true,
   .late_lower_int64 = true,
   .lower_int64_options =
      (nir_lower_int64_options) ~(nir_lower_iadd64 | nir_lower_imul_2x32_64),
   .lower_doubles_options = (nir_lower_doubles_options)(~0),
   .support_indirect_inputs = BITFIELD_BIT(MESA_SHADER_TESS_CTRL) |
                              BITFIELD_BIT(MESA_SHADER_TESS_EVAL) |
                              BITFIELD_BIT(MESA_SHADER_FRAGMENT),
   .support_indirect_outputs = (uint8_t)BITFIELD_MASK(MESA_SHADER_STAGES),
   .max_samples = 4,
   .lower_fquantize2f16 = true,
   .compact_arrays = true,
   .discard_is_demote = true,
   .scalarize_ddx = true,
   .io_options =
      (nir_io_options)(nir_io_always_interpolate_convergent_fs_inputs |
                       nir_io_linear_if_pos_w_is_constant),
   /* A uniform expression moved into the next stage is computed once by its
    * preamble, while a varying costs per-pixel interpolation (on G15 about
    * eight instructions a channel).
    */
   .max_varying_expression_cost = 8,
};
