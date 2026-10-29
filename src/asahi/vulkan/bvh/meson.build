# SPDX-License-Identifier: GPL-2.0-only

hk_bvh_include_dir = dir_source_root + '/src/asahi/vulkan/bvh'

hk_bvh_spv = []
foreach s : ['copy', 'encode']
  hk_bvh_spv += custom_target(
    s + '.spv.h',
    input : s + '.comp',
    output : s + '.spv.h',
    command : [
      prog_glslang, '-V', '-I' + vk_bvh_include_dir, '-I' + hk_bvh_include_dir,
      '--target-env', 'spirv1.5', '-x', '-o', '@OUTPUT@', '@INPUT@',
      glslang_quiet, vk_glsl_shader_preamble,
    ],
    depend_files : [vk_bvh_includes, files('hk_bvh_defines.h')],
  )
endforeach
