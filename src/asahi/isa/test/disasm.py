# Copyright 2025 Valve Corporation
# SPDX-License-Identifier: MIT

import isa as isa_module
import math
import sys
import binascii

# The ISA being disassembled: AGX2 unless select() picks another XML
isa = isa_module.isa

def select(xmlfile):
    global isa
    isa = isa_module.load(xmlfile)

def match_instr(raw):
    for key in isa.instrs:
        instr = isa.instrs[key]
        length = instr.length[instr.length_spec.bits.extract(raw) if instr.length_spec is not None else 0]
        r = raw & ((1 << (length * 8)) - 1)
        if (r & instr.exact.mask) == instr.exact.value:
            return length, instr, r

    return 2, None, raw

def map_value(v, kind, length):
    if kind in isa.enums:
        val = isa.enums[kind].values.get(v)
        return f'unk{v}' if val is None else val.display
    elif kind == 'signed':
        return str(-((1 << length) - v) if (v & (1 << (length - 1))) else v)
    else:
        return str(v)

def map_mod(v, mod):
    if mod.kind in isa.enums:
        return map_value(v, mod.kind, 0)
    else:
        assert(mod.bits.size() == 1)
        return mod.display if v != 0 else None

def decode_float_immediate(n):
	sign = -1.0 if n & 0x80 else 1.0
	e = (n & 0x70) >> 4
	f = n & 0xF
	if e == 0:
		return sign * f / 64.0
	else:
		return sign * float(0x10 | f) * (2.0 ** (e - 7))

def operand(desc, length):
    count = desc.get('count', 1)
    if count == 0:
        count = 4
    if 'mask' in desc:
        count = desc['mask'].bit_count()
    if 'zx' in desc:
        desc['sx'] = desc['zx'] ^ 1

    size = 64 if desc.get('size64', 0) != 0 else 32 if desc.get('size32', 0) != 0 else 16
    base = desc.get('base')
    kind = desc.get('kind')
    cache = desc.get('cache', False)
    discard = desc.get('discard', False) or desc.get('lu', False)

    hint = desc.get('hint')
    if hint is not None:
        if hint == 3:
            discard = True
        elif hint == 2:
            cache = True
        else:
            assert(hint == 1)

    hint = ('^' if discard else '') + ('$' if cache else '')
    suffixes = ''.join(['.' + x for x in ['bfloat', 'abs', 'neg', 'sx', 'sat'] if desc.get(x, 0) != 0])

    if kind == None:
        return None
    elif kind in ['cf', 'ts', 'ss', 'cr']:
        assert(count == 1)
        return hint + f'{kind}{base}' + suffixes
    elif kind == 'fimm':
        return str(decode_float_immediate(base)) + suffixes
    elif kind in ['imm', 'signed']:
        if desc.get('hex', 0):
            return hex(base) + suffixes
        return map_value(base, kind, length) + suffixes

    hl = 'hl' if desc.get('hl', 0) else ''

    def m(offs):
        assert(size == 16 or size == 32 or size == 64)
        reg = base + (offs * (size // 16))
        whole = reg >> 1
        part = reg & 1
        prefix = {'reg': 'r', 'uniform': 'u', 'stack': 'st'}[kind]

        if size == 16:
            part_str = 'h' if part != 0 else 'l'
            return f'{prefix}{whole}{part_str}'
        else:
            assert(part == 0)
            return f'{"d" if size == 64 else ""}{prefix}{whole}'

    return hint + '_'.join([m(x) for x in range(count)]) + hl + suffixes

def map_op(v, kind, raw, ins, length):
    for c in isa.encodings[kind].cases:
        if (v & c.exact.mask) == c.exact.value:
            desc = {}
            if c.kind is not None:
                length = c.kind.bits.size() + c.kind.shift
                desc['kind'] = c.kind.name
                desc['base'] = c.kind.bits.extract(v) << c.kind.shift
                if c.kind.kind == 'signed':
                    desc['kind'] = 'signed'

            for mod in c.modifiers:
                desc[mod.name] = mod.bits.extract(v)
                if mod.table is not None:
                    desc[mod.name] = mod.table[desc[mod.name]]

            for prop in c.properties:
                desc[prop] = int(c.properties[prop])

            return operand(desc, length)

    # No case matched: an invalid encoding
    return f'XXX{v:#x}'

def disasm(raw_in):
    length, instr, raw = match_instr(raw_in)
    if instr is None:
        return length, "<unknown instr>"

    def show(category, x):
        if category == 'imm':
            return map_value(x.bits.extract(raw), x.kind, x.bits.size())
        elif category == 'mod':
            return map_mod(x.bits.extract(raw), x)

        op = map_op(x.bits.extract(raw), x.kind, raw, instr, x.bits.size())
        if category == 'dest' and op is None:
            return '_'
        return op

    if isa.ordered:
        ops = instr.ordered
    else:
        ops = [('dest', x) for x in instr.dests] + \
              [('src', x) for x in instr.sources] + \
              [('imm', x) for x in instr.immediates] + \
              [('mod', x) for x in instr.modifiers if x.name != 'length']

    ops = [show(c, x) for c, x in ops]
    end = ', '.join([str(x) for x in ops if x is not None])
    if end != "":
        end = " " + end

    asm = f'{instr.display:<8}{end}'.strip()

    unexpected = raw & ~instr.mask()
    unexpected = [i for i in range(length * 8) if unexpected & (1 << i)]
    if len(unexpected) != 0:
        print(f"Unexpected set bits: {unexpected}")
        length = -length

    return length, asm

if __name__ == '__main__':
    if len(sys.argv) > 2:
        select(sys.argv[2])

    prog = open(sys.argv[1], 'rb').read()
    print("")
    i = 0
    while i < len(prog):
        lst = prog[i:i + 16]
        raw = sum([int(x) << (8 * i) for i, x in enumerate(lst)])
        length, asm = disasm(raw)
        lst = lst[0:length]
        print(f'{hex(i)[2:]:>4}: {str(binascii.hexlify(lst).decode()):<20} {asm}')
        if asm.strip() == 'trap':
            break
        i += length
    print("")
