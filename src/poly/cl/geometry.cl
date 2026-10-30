/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2023 Valve Corporation
 * Copyright 2025 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "compiler/libcl/libcl_vk.h"
#include "poly/geometry.h"
#include "poly/prim.h"
#include "poly/tessellator.h"
#include "util/macros.h"
#include "util/u_math.h"

uint64_t nir_ro_to_rw_poly(uint64_t address);

static inline uint
xfb_prim(uint id, uint n, uint copy)
{
   return sub_sat(id, n - 1u) + copy;
}

/*
 * Determine whether an output vertex has an n'th copy in the transform feedback
 * buffer. This is written weirdly to let constant folding remove unnecessary
 * stores when length is known statically.
 */
bool
poly_xfb_vertex_copy_in_strip(uint n, uint id, uint length, uint copy)
{
   uint prim = xfb_prim(id, n, copy);

   int num_prims = length - (n - 1);
   return copy == 0 || (prim < num_prims && id >= copy && copy < num_prims);
}

uint
poly_xfb_vertex_offset(uint n, uint invocation_base_prim, uint strip_base_prim,
                       uint id_in_strip, uint copy, bool flatshade_first)
{
   uint prim = xfb_prim(id_in_strip, n, copy);
   uint vert_0 = min(id_in_strip, n - 1);
   uint vert = vert_0 - copy;

   if (n == 3) {
      vert = poly_map_vertex_in_tri_strip(prim, vert, flatshade_first);
   }

   /* Tally up in the whole buffer */
   uint base_prim = invocation_base_prim + strip_base_prim;
   uint base_vertex = base_prim * n;
   return base_vertex + (prim * n) + vert;
}

uint64_t
poly_xfb_vertex_address(constant struct poly_geometry_params *p, uint index,
                        uint buffer, uint stride, uint output_offset)
{
   uint xfb_offset = (index * stride) + output_offset;

   return (uintptr_t)(p->xfb_base[buffer]) + xfb_offset;
}

uint
poly_vertex_id_for_line_class(enum mesa_prim mode, uint prim, uint vert,
                              uint num_prims)
{
   /* Line list, line strip, or line loop */
   if (mode == MESA_PRIM_LINE_LOOP && prim == (num_prims - 1) && vert == 1)
      return 0;

   if (mode == MESA_PRIM_LINES)
      prim *= 2;

   return prim + vert;
}

uint
poly_vertex_id_for_tri_class(enum mesa_prim mode, uint prim, uint vert,
                             bool flatshade_first)
{
   if (flatshade_first && mode == MESA_PRIM_TRIANGLE_FAN) {
      vert = vert + 1;
      vert = (vert == 3) ? 0 : vert;
   }

   if (mode == MESA_PRIM_TRIANGLE_FAN && vert == 0)
      return 0;

   if (mode == MESA_PRIM_TRIANGLES)
      prim *= 3;

   /* Triangle list, triangle strip, or triangle fan */
   if (mode == MESA_PRIM_TRIANGLE_STRIP) {
      unsigned pv = flatshade_first ? 0 : 2;

      bool even = (prim & 1) == 0;
      bool provoking = vert == pv;

      vert = ((provoking || even) ? vert : ((3 - pv) - vert));
   }

   return prim + vert;
}

uint
poly_vertex_id_for_line_adj_class(enum mesa_prim mode, uint prim, uint vert)
{
   /* Line list adj or line strip adj */
   if (mode == MESA_PRIM_LINES_ADJACENCY)
      prim *= 4;

   return prim + vert;
}

uint
poly_vertex_id_for_tri_adj_class(enum mesa_prim mode, uint prim, uint vert,
                                 uint nr, bool flatshade_first)
{
   /* Tri adj list or tri adj strip */
   if (mode == MESA_PRIM_TRIANGLE_STRIP_ADJACENCY) {
      return poly_vertex_id_for_tri_strip_adj(prim, vert, nr, flatshade_first);
   } else {
      return (6 * prim) + vert;
   }
}

uint
poly_map_to_line_adj(uint id)
{
   /* Sequence (1, 2), (5, 6), (9, 10), ... */
   return ((id & ~1) * 2) + (id & 1) + 1;
}

uint
poly_map_to_line_strip_adj(uint id)
{
   /* Sequence (1, 2), (2, 3), (4, 5), .. */
   uint prim = id / 2;
   uint vert = id & 1;
   return prim + vert + 1;
}

uint
poly_map_to_tri_strip_adj(uint id)
{
   /* Sequence (0, 2, 4), (2, 6, 4), (4, 6, 8), (6, 10, 8)
    *
    * Although tri strips with adjacency have 6 cases in general, after
    * disregarding the vertices only available in a geometry shader, there are
    * only even/odd cases. In other words, it's just a triangle strip subject to
    * extra padding.
    *
    * Dividing through by two, the sequence is:
    *
    *   (0, 1, 2), (1, 3, 2), (2, 3, 4), (3, 5, 4)
    */
   uint prim = id / 3;
   uint vtx = id % 3;

   /* Flip the winding order of odd triangles */
   if ((prim % 2) == 1) {
      if (vtx == 1)
         vtx = 2;
      else if (vtx == 2)
         vtx = 1;
   }

   return 2 * (prim + vtx);
}

uint
poly_load_index_buffer(constant struct poly_vertex_params *p, uint id,
                       uint index_size)
{
   return poly_load_index(p->index_buffer, p->index_buffer_range_el, id,
                          index_size);
}

static uint
setup_xfb_buffer(global struct poly_geometry_params *p, uint i, uint stride,
                 uint max_output_end, uint vertices_per_prim)
{
   uint xfb_offset = *(p->xfb_offs_ptrs[i]);
   p->xfb_base[i] = p->xfb_base_original[i] + xfb_offset;

   /* Let output_end = output_offset + output_size.
    *
    * Primitive P will write up to (but not including) offset:
    *
    *    xfb_offset + ((P - 1) * (verts_per_prim * stride))
    *               + ((verts_per_prim - 1) * stride)
    *               + output_end
    *
    * To fit all outputs for P, that value must be less than the XFB
    * buffer size for the output with maximal output_end, as everything
    * else is constant here across outputs within a buffer/primitive:
    *
    *    floor(P) <= (stride + size - xfb_offset - output_end)
    *                 // (stride * verts_per_prim)
    */
   int numer_s = p->xfb_size[i] + (stride - max_output_end) - xfb_offset;
   uint numer = max(numer_s, 0);
   return numer / (stride * vertices_per_prim);
}

void
poly_write_strip(GLOBAL uint32_t *index_buffer, uint32_t inv_index_offset,
                 uint32_t prim_index_offset, uint32_t vertex_offset,
                 uint32_t verts_in_prim, uint3 info)
{
   _poly_write_strip(index_buffer, inv_index_offset + prim_index_offset,
                     vertex_offset, verts_in_prim, info.x, info.y, info.z);
}

void
poly_pad_index_gs(global int *index_buffer, uint inv_index_offset,
                  uint nr_indices, uint alloc)
{
   for (uint i = nr_indices; i < alloc; ++i) {
      index_buffer[inv_index_offset + i] = -1;
   }
}

uintptr_t
poly_vertex_output_buffer(constant struct poly_vertex_params *p)
{
   return p->output_buffer;
}

uint64_t
poly_vertex_outputs(constant struct poly_vertex_params *p)
{
   return p->outputs;
}

uintptr_t
poly_vertex_output_address(constant struct poly_vertex_params *p,
                           uint64_t mask, uint vtx, gl_varying_slot location)
{
   /* Written like this to let address arithmetic work */
   return p->output_buffer +
      ((uintptr_t)poly_tcs_in_offs_el(vtx, location, mask)) * 16;
}

unsigned
poly_index_size(constant struct poly_vertex_params *p)
{
   return p->index_size_B;
}

unsigned
poly_input_vertices(constant struct poly_vertex_params *p)
{
   return p->verts_per_instance;
}

global uint *
poly_load_xfb_count_address(constant struct poly_geometry_params *p, int index,
                            int count_words, uint unrolled_id)
{
   return &p->count_buffer[(unrolled_id * count_words) + index];
}

uint
poly_previous_xfb_primitives(global struct poly_geometry_params *p,
                             int static_count, int count_index, int count_words,
                             bool prefix_sum, uint unrolled_id)
{
   if (static_count >= 0) {
      /* If the number of outputted vertices per invocation is known statically,
       * we can calculate the base.
       */
      return unrolled_id * static_count;
   } else {
      /* Otherwise, load from the count buffer buffer. Note that the sums are
       * inclusive, so index 0 is nonzero. This requires a little fixup here. We
       * use a saturating unsigned subtraction so we don't read out-of-bounds.
       *
       * If we didn't prefix sum, there's only one element.
       */
      uint prim_minus_1 = prefix_sum ? sub_sat(unrolled_id, 1u) : 0;
      uint count = p->count_buffer[(prim_minus_1 * count_words) + count_index];

      return unrolled_id == 0 ? 0 : count;
   }
}

/* Like u_foreach_bit, specialized for XFB to enable loop unrolling */
#define poly_foreach_xfb(word, index)                                          \
   for (uint i = 0; i < 4; ++i)                                                \
      if (word & BITFIELD_BIT(i))

void
poly_pre_gs(global struct poly_geometry_params *p, uint streams,
            uint buffers_written, uint4 buffer_to_stream, int4 count_index,
            uint4 stride, uint4 output_end, int4 static_count, uint invocations,
            uint vertices_per_prim, global uint *gs_invocations,
            global uint *gs_primitives, global uint *c_primitives,
            global uint *c_invocations)
{
   unsigned count_words = !!(count_index[0] >= 0) + !!(count_index[1] >= 0) +
                          !!(count_index[2] >= 0) + !!(count_index[3] >= 0);
   bool prefix_sum = count_words && buffers_written;
   uint unrolled_in_prims = p->input_primitives;

   /* Determine the number of primitives generated in each stream */
   uint4 in_prims = 0;
   poly_foreach_xfb(streams, i) {
      in_prims[i] = poly_previous_xfb_primitives(p, static_count[i],
                                                 count_index[i], count_words,
                                                 prefix_sum, unrolled_in_prims);

      *(p->prims_generated_counter[i]) += in_prims[i];
   }

   uint4 prims = in_prims;
   uint emitted_prims = prims[0] + prims[1] + prims[2] + prims[3];

   if (buffers_written) {
      poly_foreach_xfb(buffers_written, i) {
         uint max_prims =
            setup_xfb_buffer(p, i, stride[i], output_end[i], vertices_per_prim);

         unsigned stream = buffer_to_stream[i];
         prims[stream] = min(prims[stream], max_prims);
      }

      int4 overflow = prims < in_prims;

      poly_foreach_xfb(streams, i) {
         p->xfb_verts[i] = prims[i] * vertices_per_prim;

         *(p->xfb_overflow[i]) += (bool)overflow[i];
         *(p->xfb_prims_generated_counter[i]) += prims[i];
      }

      *(p->xfb_any_overflow) += any(overflow);

      /* Update XFB counters */
      poly_foreach_xfb(buffers_written, i) {
         uint32_t prim_stride_B = stride[i] * vertices_per_prim;
         unsigned stream = buffer_to_stream[i];

         global uint *ptr = p->xfb_offs_ptrs[i];

         ptr = (global uint *)nir_ro_to_rw_poly((uint64_t)ptr);
         *ptr += prims[stream] * prim_stride_B;
      }
   }

   /* The geometry shader is invoked once per primitive (after unrolling
    * primitive restart). From the spec:
    *
    *    In case of instanced geometry shaders (see section 11.3.4.2) the
    *    geometry shader invocations count is incremented for each separate
    *    instanced invocation.
    */
   *gs_invocations += unrolled_in_prims * invocations;
   *gs_primitives += emitted_prims;

   /* Clipper queries are not well-defined, so we can emulate them in lots of
    * silly ways. We need the hardware counters to implement them properly. For
    * now, just consider all primitives emitted as passing through the clipper.
    * This satisfies spec text:
    *
    *    The number of primitives that reach the primitive clipping stage.
    *
    * and
    *
    *    If at least one vertex of the primitive lies inside the clipping
    *    volume, the counter is incremented by one or more. Otherwise, the
    *    counter is incremented by zero or more.
    */
   *c_primitives += emitted_prims;
   *c_invocations += emitted_prims;
}

/*
 * Conservative rasterization (overestimation) of a triangle, from clip space.
 * Each corner is moved out along its two edges until every edge lies
 * conservative_r pixels further out (in units of the pixel square's half-width
 * along the edge's normal), and the outputs are extrapolated on the triangle's
 * plane. The fragment shader then tests the pixel square against the triangle
 * exactly: its bounding box and its three edges.
 */
static float
cr_cross(float2 a, float2 b)
{
   return a.x * b.y - a.y * b.x;
}

static float2
cr_screen(float4 P, float4 vp)
{
   return (P.xy / P.w) * vp.xy + vp.zw;
}

/* Snap like the rasterizer, to 8 subpixel bits */
static float2
cr_snap(float2 s)
{
   return rint(s * 256.0f) / 256.0f;
}

/* Inward normal of the edge from a to b, scaled by its length */
static float2
cr_normal(float2 a, float2 b, float sigma)
{
   float2 d = b - a;
   return (float2)(-d.y, d.x) * sigma;
}

static float
cr_l1(float2 n)
{
   return fabs(n.x) + fabs(n.y);
}

/* Corner at q, moved out until both edges through it are r out */
static float2
cr_corner(float2 q, float2 n1, float2 n2, float r)
{
   float r1 = -r * cr_l1(n1), r2 = -r * cr_l1(n2);
   float det = cr_cross(n1, n2);

   return q + (float2)(r1 * n2.y - r2 * n1.y, n1.x * r2 - n2.x * r1) / det;
}

/* Screen-space barycentrics of v */
static float3
cr_bary(float2 s0, float2 s1, float2 s2, float area, float2 v)
{
   float3 l = (float3)(cr_cross(s2 - s1, v - s1), cr_cross(s0 - s2, v - s2),
                       cr_cross(s1 - s0, v - s0));
   return l / area;
}

/* Edge value for the fragment test: the pixel square touches the edge's
 * half-plane iff the value at its centre is >= -0.5.
 */
static float
cr_edge(float2 n, float2 q, float2 v, float extra)
{
   float l1 = cr_l1(n);
   return (dot(n, v - q) + extra * sqrt(dot(n, n))) / l1 + (1.0f / 512.0f);
}

/*
 * For output corner `corner`, returns:
 *
 *    what = 0: the perspective weights of the three vertices, and the status
 *    what = 1: the screen-space weights, and the status
 *    what = 2: the bounding box (x0, y0, x1, y1) in pixels
 *    what = 3: the three edge values, for the edges opposite each vertex
 *
 * Status 0: widened. 1: a vertex has w <= 0 or the widened triangle reaches
 * the plane's horizon, so the triangle is not widened and the fragment test
 * passes. 2: the snapped triangle is degenerate, cull it.
 */
float4
poly_conservative_corner(constant struct poly_geometry_params *p, float4 P0,
                         float4 P1, float4 P2, uint corner, uint viewport,
                         uint what)
{
   uint vp = min(viewport, p->conservative_viewport_count - 1);
   float4 xf = vload4(vp, p->conservative_viewports);
   float r = p->conservative_r, extra = p->conservative_extra;

   float2 s0 = cr_screen(P0, xf), s1 = cr_screen(P1, xf),
          s2 = cr_screen(P2, xf);
   float2 q0 = cr_snap(s0), q1 = cr_snap(s1), q2 = cr_snap(s2);

   /* Twice the signed areas, snapped and exact */
   float area = cr_cross(q1 - q0, q2 - q0);
   float area_s = cr_cross(s1 - s0, s2 - s0);
   float sigma = area < 0.0f ? -1.0f : 1.0f;

   /* Edge k is opposite vertex k */
   float2 n0 = cr_normal(q1, q2, sigma);
   float2 n1 = cr_normal(q2, q0, sigma);
   float2 n2 = cr_normal(q0, q1, sigma);

   float2 v0 = cr_corner(q0, n1, n2, r);
   float2 v1 = cr_corner(q1, n2, n0, r);
   float2 v2 = cr_corner(q2, n0, n1, r);

   float3 rw = (float3)(1.0f / P0.w, 1.0f / P1.w, 1.0f / P2.w);
   float3 mu0 = cr_bary(s0, s1, s2, area_s, v0);
   float3 mu1 = cr_bary(s0, s1, s2, area_s, v1);
   float3 mu2 = cr_bary(s0, s1, s2, area_s, v2);

   /* 1/w along the plane, at each corner */
   float h0 = dot(mu0, rw), h1 = dot(mu1, rw), h2 = dot(mu2, rw);
   float eps = fmax(rw.x, fmax(rw.y, rw.z)) * (1.0f / 1024.0f);

   uint status = 0;
   if (!(P0.w > 0.0f && P1.w > 0.0f && P2.w > 0.0f))
      status = 1;
   else if (area == 0.0f)
      status = 2;
   else if (!(area_s != 0.0f && fmin(h0, fmin(h1, h2)) > eps))
      status = 1;

   float2 v = corner == 0 ? v0 : corner == 1 ? v1 : v2;
   float3 mu = corner == 0 ? mu0 : corner == 1 ? mu1 : mu2;
   float h = corner == 0 ? h0 : corner == 1 ? h1 : h2;

   if (what == 0 || what == 1) {
      float3 w = what == 0 ? (mu * rw) / h : mu;

      if (status != 0)
         w = (float3)(corner == 0, corner == 1, corner == 2);

      return (float4)(w, (float)status);
   } else if (what == 2) {
      if (status != 0)
         return (float4)(-1.0e30f, -1.0e30f, 1.0e30f, 1.0e30f);

      float2 lo = fmin(q0, fmin(q1, q2)) - extra;
      float2 hi = fmax(q0, fmax(q1, q2)) + extra;
      return (float4)(lo, hi);
   } else {
      if (status != 0)
         return (float4)(1.0e30f, 1.0e30f, 1.0e30f, 0.0f);

      return (float4)(cr_edge(n0, q1, v, extra), cr_edge(n1, q2, v, extra),
                      cr_edge(n2, q0, v, extra), 0.0f);
   }
}
