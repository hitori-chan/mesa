/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

/* Hit objects (VK_EXT_ray_tracing_invocation_reorder) in a ray-tracing stage,
 * before it is inlined (tier 1) or split at its calls (tier 2).
 *
 * Each hit object variable becomes a struct of what its shaders would read,
 * which nir_lower_vars_to_ssa keeps in registers and nir_lower_shader_calls
 * spills across a call as any value. A trace into a hit object is
 * hit_object_trace_ray_agx, the kernel's hit state it leaves read back at once
 * (load_hit_state_agx); a run of the recorded shader stores that state
 * (store_hit_state_agx) and is hit_object_execute_agx. hk_nir_rt_pipeline.c
 * lowers both. Nothing is reordered: the hints are dropped, as
 * rayTracingInvocationReorderReorderingHint (NONE) tells the application.
 */

#include "bvh/hk_bvh_defines.h"
#include "hk_nir_ray_tracing.h"

#include "nir/nir_builder.h"
#include "util/set.h"
#include "util/u_dynarray.h"

enum {
   HO_EMPTY,
   HO_HIT,
   HO_MISS,
};

/* The struct's members */
enum {
   HO_KIND,
   HO_FLAGS,
   /* The hit group's record index, or the miss shader's */
   HO_INDEX,
   HO_ORIGIN,
   HO_TMIN,
   HO_DIR,
   HO_TMAX,
   HO_INSTANCE_ADDR,
   HO_PRIMITIVE_ADDR,
   HO_PRIMITIVE_ID,
   HO_GEOMETRY_ID_AND_FLAGS,
   HO_HIT_KIND,
   HO_ATTRIBS,
   HO_COUNT,
};

static const struct glsl_type *
hit_object_type(void)
{
   const struct glsl_type *u32 = glsl_uint_type();
   const struct glsl_type *u64 = glsl_uint64_t_type();
   const struct glsl_type *f32 = glsl_float_type();
   const struct glsl_type *vec3 = glsl_vec_type(3);

   struct glsl_struct_field fields[HO_COUNT] = {
      [HO_KIND] = {.type = u32, .name = "kind"},
      [HO_FLAGS] = {.type = u32, .name = "flags"},
      [HO_INDEX] = {.type = u32, .name = "index"},
      [HO_ORIGIN] = {.type = vec3, .name = "origin"},
      [HO_TMIN] = {.type = f32, .name = "tmin"},
      [HO_DIR] = {.type = vec3, .name = "dir"},
      [HO_TMAX] = {.type = f32, .name = "tmax"},
      [HO_INSTANCE_ADDR] = {.type = u64, .name = "instance_addr"},
      [HO_PRIMITIVE_ADDR] = {.type = u64, .name = "primitive_addr"},
      [HO_PRIMITIVE_ID] = {.type = u32, .name = "primitive_id"},
      [HO_GEOMETRY_ID_AND_FLAGS] = {.type = u32,
                                    .name = "geometry_id_and_flags"},
      [HO_HIT_KIND] = {.type = u32, .name = "hit_kind"},
      [HO_ATTRIBS] = {.type = glsl_array_type(u32, HK_HIT_ATTRIB_WORDS, 0),
                      .name = "attribs"},
   };

   return glsl_struct_type(fields, HO_COUNT, "hk_hit_object", false);
}

static bool
is_hit_object_intrinsic(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_hit_object_trace_ray:
   case nir_intrinsic_hit_object_record_miss:
   case nir_intrinsic_hit_object_record_empty:
   case nir_intrinsic_hit_object_record_from_query:
   case nir_intrinsic_hit_object_execute_shader:
   case nir_intrinsic_hit_object_get_attributes:
   case nir_intrinsic_hit_object_set_sbt_record_index:
   case nir_intrinsic_hit_object_reorder_thread:
   case nir_intrinsic_hit_object_get_is_empty:
   case nir_intrinsic_hit_object_get_is_hit:
   case nir_intrinsic_hit_object_get_is_miss:
   case nir_intrinsic_hit_object_get_ray_flags:
   case nir_intrinsic_hit_object_get_ray_t_min:
   case nir_intrinsic_hit_object_get_ray_t_max:
   case nir_intrinsic_hit_object_get_world_ray_origin:
   case nir_intrinsic_hit_object_get_world_ray_direction:
   case nir_intrinsic_hit_object_get_object_ray_origin:
   case nir_intrinsic_hit_object_get_object_ray_direction:
   case nir_intrinsic_hit_object_get_object_to_world:
   case nir_intrinsic_hit_object_get_world_to_object:
   case nir_intrinsic_hit_object_get_instance_custom_index:
   case nir_intrinsic_hit_object_get_instance_id:
   case nir_intrinsic_hit_object_get_primitive_index:
   case nir_intrinsic_hit_object_get_geometry_index:
   case nir_intrinsic_hit_object_get_hit_kind:
   case nir_intrinsic_hit_object_get_sbt_record_index:
   case nir_intrinsic_hit_object_get_shader_record_buffer_handle:
   case nir_intrinsic_hit_object_get_triangle_vertex_positions:
      return true;
   default:
      return false;
   }
}

static nir_deref_instr *
member(nir_builder *b, nir_deref_instr *hit_object, unsigned field)
{
   return nir_build_deref_struct(b, hit_object, field);
}

static nir_def *
load(nir_builder *b, nir_deref_instr *hit_object, unsigned field)
{
   return nir_load_deref(b, member(b, hit_object, field));
}

static void
store(nir_builder *b, nir_deref_instr *hit_object, unsigned field,
      nir_def *value)
{
   nir_store_deref(b, member(b, hit_object, field), value,
                   nir_component_mask(value->num_components));
}

static nir_deref_instr *
attrib(nir_builder *b, nir_deref_instr *hit_object, unsigned word)
{
   return nir_build_deref_array_imm(b, member(b, hit_object, HO_ATTRIBS),
                                    word);
}

/* The hit attributes as words in the layout every stage gives them (natural
 * size and alignment: hk_nir_rt_pipeline.c's lower_payloads), copied between
 * words[] and a variable's scalar leaves; past HK_HIT_ATTRIB_WORDS nothing.
 */
static void
copy_attribs(nir_builder *b, nir_deref_instr *deref,
             const struct glsl_type *explicit, unsigned offset, nir_def **words,
             bool to_var)
{
   const struct glsl_type *type = deref->type;

   if (glsl_type_is_vector_or_scalar(type)) {
      bool is_bool = glsl_type_is_boolean(type);
      unsigned bits = is_bool ? 32 : glsl_get_bit_size(type);
      unsigned comps = glsl_get_vector_elements(type);
      nir_def *value = to_var ? NULL : nir_load_deref(b, deref);
      nir_def *chans[NIR_MAX_VEC_COMPONENTS];

      for (unsigned c = 0; c < comps; c++) {
         unsigned byte = offset + c * (bits / 8);
         unsigned w = byte / 4, shift = (byte % 4) * 8;
         bool in = byte + bits / 8 <= HK_HIT_ATTRIB_WORDS * 4;

         if (to_var) {
            if (!in) {
               chans[c] = nir_imm_zero(b, 1, is_bool ? 1 : bits);
            } else if (bits == 64) {
               chans[c] = nir_pack_64_2x32_split(b, words[w], words[w + 1]);
            } else if (bits == 32) {
               chans[c] = is_bool ? nir_ine_imm(b, words[w], 0) : words[w];
            } else {
               chans[c] = nir_u2uN(b, nir_ushr_imm(b, words[w], shift), bits);
            }
         } else if (in) {
            nir_def *chan = nir_channel(b, value, c);
            if (bits == 64) {
               words[w] = nir_unpack_64_2x32_split_x(b, chan);
               words[w + 1] = nir_unpack_64_2x32_split_y(b, chan);
            } else if (bits == 32) {
               words[w] = is_bool ? nir_b2i32(b, chan) : chan;
            } else {
               words[w] = nir_bitfield_insert(b, words[w], nir_u2u32(b, chan),
                                              nir_imm_int(b, shift),
                                              nir_imm_int(b, bits));
            }
         }
      }

      if (to_var)
         nir_store_deref(b, deref, nir_vec(b, chans, comps),
                         nir_component_mask(comps));
   } else if (glsl_type_is_struct_or_ifc(type)) {
      for (unsigned i = 0; i < glsl_get_length(type); i++) {
         copy_attribs(b, nir_build_deref_struct(b, deref, i),
                      glsl_get_struct_field(explicit, i),
                      offset + glsl_get_struct_field_offset(explicit, i), words,
                      to_var);
      }
   } else {
      /* An array or a matrix's columns */
      unsigned stride = glsl_get_explicit_stride(explicit);
      const struct glsl_type *elem = glsl_type_is_matrix(explicit)
                                        ? glsl_get_column_type(explicit)
                                        : glsl_get_array_element(explicit);

      for (unsigned i = 0; i < glsl_get_length(type); i++) {
         if (offset + i * stride >= HK_HIT_ATTRIB_WORDS * 4)
            break;

         copy_attribs(b, nir_build_deref_array_imm(b, deref, i), elem,
                      offset + i * stride, words, to_var);
      }
   }
}

static const struct glsl_type *
attribs_layout(const struct glsl_type *type)
{
   unsigned size, align;
   return glsl_get_explicit_type_for_size_align(
      type, glsl_get_natural_size_align_bytes, &size, &align);
}

static void
lower_trace(nir_builder *b, nir_deref_instr *ho, nir_intrinsic_instr *intr)
{
   nir_def *origin_tmin = intr->src[7].ssa;
   nir_def *dir_tmax = intr->src[8].ssa;
   nir_def *origin = nir_trim_vector(b, origin_tmin, 3);
   nir_def *tmin = nir_channel(b, origin_tmin, 3);
   nir_def *dir = nir_trim_vector(b, dir_tmax, 3);
   nir_def *tmax = nir_channel(b, dir_tmax, 3);
   nir_def *flags = intr->src[2].ssa;
   nir_def *miss_index = intr->src[6].ssa;

   nir_hit_object_trace_ray_agx(b, intr->src[1].ssa, flags, intr->src[3].ssa,
                                intr->src[4].ssa, intr->src[5].ssa, miss_index,
                                origin, tmin, dir, tmax, intr->src[9].ssa);

   nir_def *hit = nir_load_hit_state_agx(b, 1, 1, .base = HK_HIT_STATE_HIT);

   store(b, ho, HO_KIND,
         nir_bcsel(b, hit, nir_imm_int(b, HO_HIT), nir_imm_int(b, HO_MISS)));
   store(b, ho, HO_FLAGS, flags);
   store(b, ho, HO_INDEX,
         nir_bcsel(b, hit,
                   nir_load_hit_state_agx(b, 1, 32,
                                          .base = HK_HIT_STATE_SBT_INDEX),
                   nir_iand_imm(b, miss_index, 0xffff)));
   store(b, ho, HO_ORIGIN, origin);
   store(b, ho, HO_TMIN, tmin);
   store(b, ho, HO_DIR, dir);
   store(b, ho, HO_TMAX,
         nir_bcsel(b, hit,
                   nir_load_hit_state_agx(b, 1, 32, .base = HK_HIT_STATE_TMAX),
                   tmax));
   store(b, ho, HO_INSTANCE_ADDR,
         nir_load_hit_state_agx(b, 1, 64, .base = HK_HIT_STATE_INSTANCE_ADDR));
   store(
      b, ho, HO_PRIMITIVE_ADDR,
      nir_load_hit_state_agx(b, 1, 64, .base = HK_HIT_STATE_PRIMITIVE_ADDR));
   store(b, ho, HO_PRIMITIVE_ID,
         nir_load_hit_state_agx(b, 1, 32, .base = HK_HIT_STATE_PRIMITIVE_ID));
   store(b, ho, HO_GEOMETRY_ID_AND_FLAGS,
         nir_load_hit_state_agx(b, 1, 32,
                                .base = HK_HIT_STATE_GEOMETRY_ID_AND_FLAGS));
   store(b, ho, HO_HIT_KIND,
         nir_load_hit_state_agx(b, 1, 32, .base = HK_HIT_STATE_HIT_KIND));

   for (unsigned i = 0; i < HK_HIT_ATTRIB_WORDS; i++) {
      nir_store_deref(
         b, attrib(b, ho, i),
         nir_load_hit_state_agx(b, 1, 32, .base = HK_HIT_STATE_ATTRIB + i),
         0x1);
   }
}

static void
lower_execute(nir_builder *b, nir_deref_instr *ho, nir_def *payload)
{
   nir_def *kind = load(b, ho, HO_KIND);

   nir_push_if(b, nir_ine_imm(b, kind, HO_EMPTY));
   {
      nir_def *index = load(b, ho, HO_INDEX);

      nir_store_hit_state_agx(b, nir_ieq_imm(b, kind, HO_HIT),
                              .base = HK_HIT_STATE_HIT);
      nir_store_hit_state_agx(b, load(b, ho, HO_INSTANCE_ADDR),
                              .base = HK_HIT_STATE_INSTANCE_ADDR);
      nir_store_hit_state_agx(b, load(b, ho, HO_PRIMITIVE_ADDR),
                              .base = HK_HIT_STATE_PRIMITIVE_ADDR);
      nir_store_hit_state_agx(b, load(b, ho, HO_PRIMITIVE_ID),
                              .base = HK_HIT_STATE_PRIMITIVE_ID);
      nir_store_hit_state_agx(b, load(b, ho, HO_GEOMETRY_ID_AND_FLAGS),
                              .base = HK_HIT_STATE_GEOMETRY_ID_AND_FLAGS);
      nir_store_hit_state_agx(b, load(b, ho, HO_HIT_KIND),
                              .base = HK_HIT_STATE_HIT_KIND);
      nir_store_hit_state_agx(b, index, .base = HK_HIT_STATE_SBT_INDEX);

      for (unsigned i = 0; i < HK_HIT_ATTRIB_WORDS; i++) {
         nir_store_hit_state_agx(b, nir_load_deref(b, attrib(b, ho, i)),
                                 .base = HK_HIT_STATE_ATTRIB + i);
      }

      nir_hit_object_execute_agx(
         b, nir_imm_int64(b, 0), load(b, ho, HO_FLAGS), nir_imm_int(b, 0),
         nir_imm_int(b, 0), nir_imm_int(b, 0), index, load(b, ho, HO_ORIGIN),
         load(b, ho, HO_TMIN), load(b, ho, HO_DIR), load(b, ho, HO_TMAX),
         payload);
   }
   nir_pop_if(b, NULL);
}

static void
lower_record_miss(nir_builder *b, nir_deref_instr *ho,
                  nir_intrinsic_instr *intr)
{
   nir_def *origin_tmin = intr->src[3].ssa;
   nir_def *dir_tmax = intr->src[4].ssa;

   store(b, ho, HO_KIND, nir_imm_int(b, HO_MISS));
   store(b, ho, HO_FLAGS, intr->src[1].ssa);
   store(b, ho, HO_INDEX, intr->src[2].ssa);
   store(b, ho, HO_ORIGIN, nir_trim_vector(b, origin_tmin, 3));
   store(b, ho, HO_TMIN, nir_channel(b, origin_tmin, 3));
   store(b, ho, HO_DIR, nir_trim_vector(b, dir_tmax, 3));
   store(b, ho, HO_TMAX, nir_channel(b, dir_tmax, 3));
}

static nir_def *
rq_committed(nir_builder *b, nir_def *rq, nir_ray_query_value value,
             unsigned comps, unsigned bits)
{
   return nir_rq_load(b, comps, bits, rq, .ray_query_value = value,
                      .committed = true);
}

/* The query's committed intersection: none (an empty hit object), a
 * triangle (its hit kind and barycentrics the query's) or an AABB (the
 * sources')
 */
static void
lower_record_from_query(nir_builder *b, nir_deref_instr *ho,
                        nir_intrinsic_instr *intr)
{
   nir_def *rq = intr->src[1].ssa;
   nir_deref_instr *attribs = nir_src_as_deref(intr->src[4]);

   nir_def *type = rq_committed(
      b, rq, nir_ray_query_value_intersection_type, 1, 32);
   nir_def *triangle = nir_ieq_imm(b, type, 1);

   store(b, ho, HO_KIND,
         nir_bcsel(b, nir_ieq_imm(b, type, 0), nir_imm_int(b, HO_EMPTY),
                   nir_imm_int(b, HO_HIT)));
   store(b, ho, HO_FLAGS,
         rq_committed(b, rq, nir_ray_query_value_flags, 1, 32));
   store(b, ho, HO_INDEX, intr->src[2].ssa);
   store(b, ho, HO_ORIGIN,
         rq_committed(b, rq, nir_ray_query_value_world_ray_origin, 3, 32));
   store(b, ho, HO_TMIN, rq_committed(b, rq, nir_ray_query_value_tmin, 1, 32));
   store(b, ho, HO_DIR,
         rq_committed(b, rq, nir_ray_query_value_world_ray_direction, 3, 32));
   store(b, ho, HO_TMAX,
         rq_committed(b, rq, nir_ray_query_value_intersection_t, 1, 32));
   store(b, ho, HO_INSTANCE_ADDR,
         rq_committed(b, rq, nir_ray_query_value_intersection_instance_node_agx,
                      1, 64));
   store(
      b, ho, HO_PRIMITIVE_ADDR,
      rq_committed(b, rq, nir_ray_query_value_intersection_primitive_node_agx,
                   1, 64));
   store(b, ho, HO_PRIMITIVE_ID,
         rq_committed(b, rq, nir_ray_query_value_intersection_primitive_index,
                      1, 32));
   store(b, ho, HO_GEOMETRY_ID_AND_FLAGS,
         rq_committed(b, rq, nir_ray_query_value_intersection_geometry_index,
                      1, 32));

   nir_def *front_face =
      rq_committed(b, rq, nir_ray_query_value_intersection_front_face, 1, 1);
   store(b, ho, HO_HIT_KIND,
         nir_bcsel(b, triangle,
                   nir_bcsel(b, front_face, nir_imm_int(b, 0xFE),
                             nir_imm_int(b, 0xFF)),
                   intr->src[3].ssa));

   nir_def *aabb_words[HK_HIT_ATTRIB_WORDS];
   for (unsigned i = 0; i < HK_HIT_ATTRIB_WORDS; i++)
      aabb_words[i] = nir_imm_int(b, 0);

   copy_attribs(b, attribs, attribs_layout(attribs->type), 0, aabb_words,
                false);

   nir_def *barycentrics = rq_committed(
      b, rq, nir_ray_query_value_intersection_barycentrics, 2, 32);

   for (unsigned i = 0; i < HK_HIT_ATTRIB_WORDS; i++) {
      nir_def *tri = i < 2 ? nir_channel(b, barycentrics, i) : nir_imm_int(b, 0);
      nir_store_deref(b, attrib(b, ho, i),
                      nir_bcsel(b, triangle, tri, aabb_words[i]), 0x1);
   }
}

static nir_def *
instance_load(nir_builder *b, nir_deref_instr *ho, unsigned offset)
{
   return nir_load_global(
      b, 1, 32, nir_iadd_imm(b, load(b, ho, HO_INSTANCE_ADDR), offset));
}

static nir_def *lower_get_node(nir_builder *b, nir_deref_instr *ho,
                               nir_intrinsic_instr *intr);

static nir_def *
lower_get(nir_builder *b, nir_deref_instr *ho, nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_hit_object_get_is_empty:
      return nir_ieq_imm(b, load(b, ho, HO_KIND), HO_EMPTY);
   case nir_intrinsic_hit_object_get_is_hit:
      return nir_ieq_imm(b, load(b, ho, HO_KIND), HO_HIT);
   case nir_intrinsic_hit_object_get_is_miss:
      return nir_ieq_imm(b, load(b, ho, HO_KIND), HO_MISS);
   case nir_intrinsic_hit_object_get_ray_flags:
      return load(b, ho, HO_FLAGS);
   case nir_intrinsic_hit_object_get_ray_t_min:
      return load(b, ho, HO_TMIN);
   case nir_intrinsic_hit_object_get_ray_t_max:
      return load(b, ho, HO_TMAX);
   case nir_intrinsic_hit_object_get_world_ray_origin:
      return load(b, ho, HO_ORIGIN);
   case nir_intrinsic_hit_object_get_world_ray_direction:
      return load(b, ho, HO_DIR);
   case nir_intrinsic_hit_object_get_primitive_index:
      return load(b, ho, HO_PRIMITIVE_ID);
   case nir_intrinsic_hit_object_get_geometry_index:
      return nir_iand_imm(b, load(b, ho, HO_GEOMETRY_ID_AND_FLAGS), 0xFFFFFFF);
   case nir_intrinsic_hit_object_get_hit_kind:
      return load(b, ho, HO_HIT_KIND);
   case nir_intrinsic_hit_object_get_sbt_record_index:
      return load(b, ho, HO_INDEX);

   case nir_intrinsic_hit_object_get_shader_record_buffer_handle: {
      /* The record after the handle in the hit or miss table's entry */
      nir_def *hit = nir_ieq_imm(b, load(b, ho, HO_KIND), HO_HIT);
      unsigned hit_table =
         offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableAddress);
      unsigned miss_table =
         offsetof(VkTraceRaysIndirectCommand2KHR, missShaderBindingTableAddress);
      nir_def *table = nir_iadd(b, nir_load_rt_params_agx(b),
                                nir_bcsel(b, hit, nir_imm_int64(b, hit_table),
                                          nir_imm_int64(b, miss_table)));
      nir_def *base = nir_load_global_constant(b, 1, 64, table);
      nir_def *stride =
         nir_load_global_constant(b, 1, 32, nir_iadd_imm(b, table, 16));
      nir_def *entry =
         nir_iadd(b, base, nir_umul_2x32_64(b, load(b, ho, HO_INDEX), stride));
      return nir_unpack_64_2x32(b, nir_iadd_imm(b, entry, HK_RT_HANDLE_SIZE));
   }

   default:
      break;
   }

   /* The rest read the instance's or the primitive's node, which only a hit
    * sets (a miss leaves them undefined): read on a hit, else the world's
    * ray for the object's, an identity matrix and zeros
    */
   nir_def *hit = nir_ieq_imm(b, load(b, ho, HO_KIND), HO_HIT);
   nir_push_if(b, hit);
   nir_def *value = lower_get_node(b, ho, intr);
   nir_push_else(b, NULL);
   nir_def *none;
   switch (intr->intrinsic) {
   case nir_intrinsic_hit_object_get_object_ray_origin:
      none = load(b, ho, HO_ORIGIN);
      break;
   case nir_intrinsic_hit_object_get_object_ray_direction:
      none = load(b, ho, HO_DIR);
      break;
   case nir_intrinsic_hit_object_get_object_to_world:
   case nir_intrinsic_hit_object_get_world_to_object: {
      unsigned col = nir_intrinsic_column(intr);
      nir_def *chans[3];
      for (unsigned r = 0; r < 3; r++)
         chans[r] = nir_imm_float(b, r == col ? 1.0 : 0.0);
      none = nir_vec(b, chans, 3);
      break;
   }
   default:
      none = nir_imm_zero(b, value->num_components, value->bit_size);
      break;
   }
   nir_pop_if(b, NULL);
   return nir_if_phi(b, value, none);
}

static nir_def *
lower_get_node(nir_builder *b, nir_deref_instr *ho, nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_hit_object_get_instance_custom_index:
      return nir_iand_imm(
         b,
         instance_load(
            b, ho, offsetof(struct hk_bvh_instance_node, custom_instance_and_mask)),
         0xFFFFFF);
   case nir_intrinsic_hit_object_get_instance_id:
      return instance_load(b, ho,
                           offsetof(struct hk_bvh_instance_node, instance_id));

   case nir_intrinsic_hit_object_get_object_ray_origin:
   case nir_intrinsic_hit_object_get_object_ray_direction:
   case nir_intrinsic_hit_object_get_world_to_object: {
      nir_def *wto[3];
      hk_load_wto_matrix(b, load(b, ho, HO_INSTANCE_ADDR), NULL, wto);

      if (intr->intrinsic == nir_intrinsic_hit_object_get_object_ray_origin)
         return hk_mul_vec3_mat(b, load(b, ho, HO_ORIGIN), wto, true);
      if (intr->intrinsic == nir_intrinsic_hit_object_get_object_ray_direction)
         return hk_mul_vec3_mat(b, load(b, ho, HO_DIR), wto, false);

      unsigned col = nir_intrinsic_column(intr);
      return nir_vec3(b, nir_channel(b, wto[0], col),
                      nir_channel(b, wto[1], col), nir_channel(b, wto[2], col));
   }

   case nir_intrinsic_hit_object_get_object_to_world: {
      unsigned col = nir_intrinsic_column(intr);
      nir_def *instance_addr = load(b, ho, HO_INSTANCE_ADDR);
      nir_def *rows[3];
      for (unsigned r = 0; r < 3; ++r) {
         rows[r] = nir_load_global(
            b, 4, 32,
            nir_iadd_imm(
               b, instance_addr,
               offsetof(struct hk_bvh_instance_node, otw_matrix) + r * 16));
      }
      return nir_vec3(b, nir_channel(b, rows[0], col),
                      nir_channel(b, rows[1], col),
                      nir_channel(b, rows[2], col));
   }

   case nir_intrinsic_hit_object_get_triangle_vertex_positions:
      return hk_load_vertex_position(b, load(b, ho, HO_PRIMITIVE_ADDR),
                                     nir_intrinsic_column(intr));

   default:
      UNREACHABLE("not a hit object query");
   }
}

static void
lower_intrinsic(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_deref_instr *ho = nir_src_as_deref(intr->src[0]);

   switch (intr->intrinsic) {
   case nir_intrinsic_hit_object_trace_ray:
      lower_trace(b, ho, intr);
      break;
   case nir_intrinsic_hit_object_execute_shader:
      lower_execute(b, ho, intr->src[1].ssa);
      break;
   case nir_intrinsic_hit_object_record_empty:
      store(b, ho, HO_KIND, nir_imm_int(b, HO_EMPTY));
      break;
   case nir_intrinsic_hit_object_record_miss:
      lower_record_miss(b, ho, intr);
      break;
   case nir_intrinsic_hit_object_record_from_query:
      lower_record_from_query(b, ho, intr);
      break;
   case nir_intrinsic_hit_object_set_sbt_record_index:
      store(b, ho, HO_INDEX, intr->src[1].ssa);
      break;
   case nir_intrinsic_hit_object_get_attributes: {
      nir_deref_instr *attribs = nir_src_as_deref(intr->src[1]);
      nir_def *words[HK_HIT_ATTRIB_WORDS];
      for (unsigned i = 0; i < HK_HIT_ATTRIB_WORDS; i++)
         words[i] = nir_load_deref(b, attrib(b, ho, i));

      copy_attribs(b, attribs, attribs_layout(attribs->type), 0, words, true);
      break;
   }
   case nir_intrinsic_hit_object_reorder_thread:
      break;
   default:
      nir_def_rewrite_uses(&intr->def, lower_get(b, ho, intr));
      break;
   }

   nir_instr_remove(&intr->instr);
}

bool
hk_nir_lower_hit_objects(nir_shader *nir)
{
   struct util_dynarray intrs;
   util_dynarray_init(&intrs, NULL);
   struct set *vars = _mesa_pointer_set_create(NULL);
   bool progress = false;

   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;

      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

            /* No reordering: the hint's computation dies with it */
            if (intr->intrinsic == nir_intrinsic_reorder_thread) {
               nir_instr_remove(instr);
               impl_progress = true;
               continue;
            }

            if (!is_hit_object_intrinsic(intr->intrinsic))
               continue;

            nir_variable *var =
               nir_deref_instr_get_variable(nir_src_as_deref(intr->src[0]));
            assert(var && "a hit object is a variable");
            _mesa_set_add(vars, var);
            util_dynarray_append(&intrs, intr);
         }
      }

      if (util_dynarray_num_elements(&intrs, nir_intrinsic_instr *)) {
         /* The placeholder type becomes the struct, arrays of it kept */
         const struct glsl_type *type = hit_object_type();
         set_foreach(vars, entry) {
            nir_variable *var = (nir_variable *)entry->key;
            if (glsl_without_array(var->type) != type)
               var->type = glsl_type_wrap_in_arrays(type, var->type);
         }
         nir_fixup_deref_types(nir);

         util_dynarray_foreach(&intrs, nir_intrinsic_instr *, intr)
            lower_intrinsic(&b, *intr);

         util_dynarray_clear(&intrs);
         _mesa_set_clear(vars, NULL);
         impl_progress = true;
      }

      progress |= nir_progress(impl_progress, impl, nir_metadata_none);
   }

   _mesa_set_destroy(vars, NULL);
   util_dynarray_fini(&intrs);
   return progress;
}
