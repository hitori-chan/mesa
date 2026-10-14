# Copyright 2025 Valve Corporation
# Copyright 2020-2021 Collabora
# SPDX-License-Identifier: MIT

# AGX2 is the G13/G14 (M1/M2) ISA, AGX3 the G15 (M3) ISA. Each XML generates
# its own disassembler, agx2_disassemble_instr and agx3_disassemble_instr.
agx2_disasm_c = custom_target(
  input : ['gen-disasm.py', 'AGX2.xml'],
  output : 'agx2_disasm.c',
  command : [prog_python, '@INPUT@', 'agx2'],
  capture : true,
  depend_files : files('isa.py'),
)

libagx2_disasm = static_library(
  'agx2_disasm',
  agx2_disasm_c,
  include_directories : [inc_include, inc_src],
  c_args : [no_override_init_args],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

agx3_disasm_c = custom_target(
  input : ['gen-disasm.py', 'AGX3.xml'],
  output : 'agx3_disasm.c',
  command : [prog_python, '@INPUT@', 'agx3'],
  capture : true,
  depend_files : files('isa.py'),
)

libagx3_disasm = static_library(
  'agx3_disasm',
  agx3_disasm_c,
  include_directories : [inc_include, inc_src],
  c_args : [no_override_init_args],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

# The AGX3 encoder tables, for the compiler's G15 packer
agx3_pack_h = custom_target(
  input : ['gen-pack.py', 'AGX3.xml'],
  output : 'agx3_pack.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : files('isa.py'),
)

idep_agx3_pack_h = declare_dependency(
  sources : [agx3_pack_h],
  include_directories : include_directories('.'),
)

if with_tests
  agx_disasm_test = executable(
    'agx_disasm_test',
    files('test/test-disassembler.c'),
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    include_directories : [inc_include, inc_src],
    link_with : [libagx2_disasm, libagx3_disasm],
  )

  test(
    'agx2_disasm',
    agx_disasm_test,
    suite : ['asahi'],
    args : [files('test/cases.txt'), 'agx2'],
  )

  test(
    'agx2_python_disasm',
    prog_python,
    args : files('test-disasm.py', 'test/cases.txt'),
    suite : ['asahi']
  )

  test(
    'agx3_disasm',
    agx_disasm_test,
    suite : ['asahi'],
    args : [files('test/cases-agx3.txt'), 'agx3'],
  )

  test(
    'agx3_python_disasm',
    prog_python,
    args : [files('test-disasm.py', 'test/cases-agx3.txt'),
            '--xml', files('AGX3.xml')],
    suite : ['asahi']
  )
endif
