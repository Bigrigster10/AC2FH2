"""Minimal DXBC (SM4/5) parser + disassembler for the .pc.vso/.pc.pso shaders inside FH2 car zips.

    python tools/dxbc.py <car zip> Shaders/<material>/<name>.pc.vso

Prints resource bindings, constant buffers (with variable names), signatures and the instruction stream with
cb[slot][reg] references resolved to variable names. Used to find how the game applies wheel morph targets
(see CLAUDE.md, "Wheels"). Reads only the user's own files; nothing from the game is stored here.
"""
import struct, sys

OPS = '''add and break breakc call callc case continue continuec cut default deriv_rtx deriv_rty discard div dp2 dp3 dp4
else emit emitthencut endif endloop endswitch eq exp frc ftoi ftou ge iadd if ieq ige ilt imad imax imin imul ine ineg ishl
ishr itof label ld ld_ms log loop lt mad min max customdata mov movc mul ne nop not or resinfo ret retc round_ne round_ni
round_pi round_z rsq sample sample_c sample_c_lz sample_l sample_d sample_b sqrt switch sincos udiv ult uge umul umad umax
umin ushr utof xor dcl_resource dcl_constantbuffer dcl_sampler dcl_index_range dcl_gs_output_topology dcl_gs_input_primitive
dcl_maxout dcl_input dcl_input_sgv dcl_input_siv dcl_input_ps dcl_input_ps_sgv dcl_input_ps_siv dcl_output dcl_output_sgv
dcl_output_siv dcl_temps dcl_indexable_temp dcl_global_flags reserved0 lod gather4 sample_pos sample_info reserved1
hs_decls hs_control_point_phase hs_fork_phase hs_join_phase emit_stream cut_stream emitthencut_stream interface_call bufinfo
deriv_rtx_coarse deriv_rtx_fine deriv_rty_coarse deriv_rty_fine gather4_c gather4_po gather4_po_c rcp f32tof16 f16tof32
uaddc usubb countbits firstbit_hi firstbit_lo firstbit_shi ubfe ibfe bfi bfrev swapc dcl_stream dcl_function_body
dcl_function_table dcl_interface dcl_input_control_point_count dcl_output_control_point_count dcl_tess_domain
dcl_tess_partitioning dcl_tess_output_primitive dcl_hs_max_tessfactor dcl_hs_fork_phase_instance_count
dcl_hs_join_phase_instance_count dcl_thread_group dcl_uav_typed dcl_uav_raw dcl_uav_structured dcl_tgsm_raw
dcl_tgsm_structured dcl_resource_raw dcl_resource_structured ld_uav_typed store_uav_typed ld_raw store_raw ld_structured
store_structured atomic_and atomic_or atomic_xor atomic_cmp_store atomic_iadd atomic_imax atomic_imin atomic_umax
atomic_umin imm_atomic_alloc imm_atomic_consume imm_atomic_iadd imm_atomic_and imm_atomic_or imm_atomic_xor
imm_atomic_exch imm_atomic_cmp_exch imm_atomic_imax imm_atomic_imin imm_atomic_umax imm_atomic_umin sync dadd dmax dmin
dmul deq dge dlt dne dmov dmovc dtof ftod eval_snapped eval_sample_index eval_centroid dcl_gs_instance_count abort
debug_break'''.split()
OPT = {0: 'r', 1: 'v', 2: 'o', 3: 'x', 4: 'l', 5: 'd', 6: 's', 7: 't', 8: 'cb', 9: 'icb', 10: 'label', 11: 'vPrim',
       12: 'oDepth', 13: 'null', 30: 'u', 31: 'g', 32: 'vThreadID', 35: 'vCoverage'}
INT_OPS = {'iadd', 'and', 'or', 'xor', 'ishl', 'ishr', 'ushr', 'imad', 'umad', 'imul', 'umul', 'ieq', 'ine', 'ige',
           'ilt', 'ult', 'uge', 'imax', 'imin', 'umax', 'umin', 'not', 'ineg', 'ubfe', 'ibfe', 'bfi', 'udiv', 'ftoi',
           'ftou', 'movc', 'switch', 'case', 'ld', 'ld_structured', 'ld_raw', 'countbits', 'firstbit_lo', 'f16tof32',
           'f32tof16'}


def chunks(d):
    assert d[:4] == b'DXBC'
    n = struct.unpack_from('<I', d, 28)[0]
    offs = struct.unpack_from(f'<{n}I', d, 32)
    return {d[o:o + 4].decode(): d[o + 8:o + 8 + struct.unpack_from('<I', d, o + 4)[0]] for o in offs}


def cstr(d, o):
    e = d.index(b'\0', o)
    return d[o:e].decode('latin1')


def rdef(d):
    cbn, cbo, bn, bo = struct.unpack_from('<4I', d, 0)
    binds = []
    for i in range(bn):
        no, typ, ret, dim, ns, bp, bc, fl = struct.unpack_from('<8I', d, bo + 32 * i)
        binds.append(dict(name=cstr(d, no), type=typ, ret=ret, dim=dim, bind=bp, count=bc))
    cbs = []
    for i in range(cbn):
        no, vc, vo, size, fl, typ = struct.unpack_from('<6I', d, cbo + 24 * i)
        vars_ = []
        for j in range(vc):
            vno, start, vsize, vfl, vto, vdef = struct.unpack_from('<6I', d, vo + 40 * j)
            vars_.append((start, vsize, cstr(d, vno), vfl))
        cbs.append(dict(name=cstr(d, no), size=size, vars=vars_, type=typ))
    return binds, cbs


def sig(d):
    n = struct.unpack_from('<I', d, 0)[0]
    out = []
    for i in range(n):
        no, si, sv, ct, reg, mask = struct.unpack_from('<6I', d, 8 + 24 * i)
        out.append((cstr(d, no), si, sv, ct, reg, mask & 0xff, (mask >> 8) & 0xff))
    return out


class Dis:
    def __init__(self, shex, binds, cbs):
        self.d = shex; self.binds = binds; self.cbs = cbs
        self.cbmap = {}
        for b in binds:
            if b['type'] == 0:
                cb = next((c for c in cbs if c['name'] == b['name']), None)
                if cb: self.cbmap[b['bind']] = cb
        self.tmap = {b['bind']: b['name'] for b in binds if b['type'] in (1, 2, 5, 7)}
        self.smap = {b['bind']: b['name'] for b in binds if b['type'] == 3}

    def cbname(self, slot, reg, comps):
        cb = self.cbmap.get(slot)
        if not cb: return None
        names = []
        for c in comps:
            off = reg * 16 + c * 4
            v = next((v for v in cb['vars'] if v[0] <= off < v[0] + v[1]), None)
            if v:
                rel = off - v[0]
                names.append(f'{v[2]}+{rel}' if rel else v[2])
            else:
                names.append(f'?{off}')
        out = []
        for n in names:
            if not out or out[-1] != n: out.append(n)
        return ','.join(out)

    def operand(self, toks, p, intop=False):
        t = toks[p]; p += 1
        ncomp = t & 3; sel = (t >> 2) & 3; typ = (t >> 12) & 0xff; idim = (t >> 20) & 3
        reps = [(t >> 22) & 7, (t >> 25) & 7, (t >> 28) & 7]
        mod = ''
        if t >> 31:
            ext = toks[p]; p += 1
            if ext & 0x3f == 1:
                m = (ext >> 6) & 0xff
                mod = {1: '-', 2: '|', 3: '-|'}.get(m, '')
        comps = []
        sw = ''
        if ncomp == 2:
            if sel == 0:
                mask = (t >> 4) & 15
                comps = [i for i in range(4) if mask >> i & 1]
                sw = '.' + ''.join('xyzw'[i] for i in comps) if mask != 15 else ''
            elif sel == 1:
                s = [(t >> (4 + 2 * i)) & 3 for i in range(4)]
                comps = s; sw = '.' + ''.join('xyzw'[i] for i in s)
            else:
                c = (t >> 4) & 3; comps = [c]; sw = '.' + 'xyzw'[c]
        elif ncomp == 1:
            comps = [0]
        name = OPT.get(typ, f'T{typ}')
        if typ == 4:   # immediate
            n = 1 if ncomp == 1 else 4
            vals = toks[p:p + n]; p += n
            def fmt(v):
                f = struct.unpack('<f', struct.pack('<I', v))[0]
                if intop or (abs(f) < 1e-30 and v != 0) or abs(f) > 1e9:
                    return str(v if v < 2**31 else v - 2**32)
                return f'{f:.6g}'
            return f'l({", ".join(fmt(v) for v in vals)})', p
        idx = []
        for k in range(idim):
            r = reps[k]
            if r == 0:
                idx.append(str(toks[p])); p += 1
            elif r == 1:
                idx.append(str(toks[p] | toks[p + 1] << 32)); p += 2
            elif r == 2:
                s, p = self.operand(toks, p, True); idx.append(s)
            elif r == 3:
                base = toks[p]; p += 1
                s, p = self.operand(toks, p, True); idx.append(f'{s} + {base}')
            else:
                raise ValueError('idx rep %d' % r)
        s = name + ''.join(f'[{i}]' for i in idx) if typ in (8, 9, 3) else name + (idx[0] if idx else '') + ''.join(f'[{i}]' for i in idx[1:])
        if typ == 8 and len(idx) == 2 and idx[0].isdigit() and idx[1].isdigit():
            nm = self.cbname(int(idx[0]), int(idx[1]), comps if comps else [0, 1, 2, 3])
            if nm: s += '{' + nm + '}'
        if typ == 7 and idx and idx[0].isdigit(): s += '{' + self.tmap.get(int(idx[0]), '?') + '}'
        if typ == 6 and idx and idx[0].isdigit(): s += '{' + self.smap.get(int(idx[0]), '?') + '}'
        s = s + sw
        if mod == '-': s = '-' + s
        elif mod == '|': s = '|' + s + '|'
        elif mod == '-|': s = '-|' + s + '|'
        return s, p

    def run(self):
        toks = struct.unpack(f'<{len(self.d) // 4}I', self.d)
        ver = toks[0]; n = toks[1]
        p = 2; out = []; indent = 0
        while p < n:
            t = toks[p]
            op = t & 0x7ff
            if op == 53:   # customdata
                ln = toks[p + 1]
                out.append(f'customdata class {(t >> 11) & 0x1fffff} len {ln}')
                if (t >> 11) == 3 or True:
                    vals = toks[p + 2:p + ln]
                    fl = [struct.unpack('<f', struct.pack('<I', v))[0] for v in vals]
                    out.append('   icb: ' + ' '.join(f'{x:.6g}' for x in fl[:64]) + (' ...' if len(fl) > 64 else ''))
                p += ln; continue
            ln = (t >> 24) & 0x7f
            name = OPS[op] if op < len(OPS) else f'op{op}'
            q = p + 1
            ext = t >> 31
            while ext:
                e = toks[q]; q += 1; ext = e >> 31
            sat = '_sat' if (t >> 13) & 1 and not name.startswith('dcl') else ''
            if name in ('if', 'breakc', 'retc', 'discard', 'continuec', 'movc') and name != 'movc':
                name += '_nz' if (t >> 18) & 1 else '_z'
            ops = []
            try:
                if name.startswith('dcl_'):
                    if name in ('dcl_temps',):
                        ops = [str(toks[q])]
                    elif name == 'dcl_global_flags':
                        ops = [hex((t >> 11) & 0x1fff)]
                    elif name == 'dcl_indexable_temp':
                        ops = [str(x) for x in toks[q:q + 3]]
                    else:
                        s, q2 = self.operand(toks, q); ops = [s] + [str(x) for x in toks[q2:p + ln]]
                        if name == 'dcl_resource': ops.append(f'dim{(t >> 11) & 0x1f}')
                else:
                    intop = name in INT_OPS
                    while q < p + ln:
                        s, q = self.operand(toks, q, intop); ops.append(s)
            except Exception as e:
                ops = [f'<{e}> raw ' + ' '.join(hex(x) for x in toks[p:p + ln])]
            if name in ('endif', 'endloop', 'else', 'endswitch'): indent -= 1
            out.append('  ' * max(indent, 0) + f'{name}{sat} ' + ', '.join(ops))
            if name.startswith('if') or name in ('loop', 'else', 'switch'): indent += 1
            p += max(ln, 1)
        return out


def disasm(data):
    ch = chunks(data)
    binds, cbs = rdef(ch['RDEF'])
    res = {'binds': binds, 'cbs': cbs, 'isgn': sig(ch['ISGN']) if 'ISGN' in ch else [],
           'osgn': sig(ch['OSGN']) if 'OSGN' in ch else []}
    shex = ch.get('SHEX') or ch.get('SHDR')
    res['code'] = Dis(shex, binds, cbs).run()
    return res


if __name__ == '__main__':
    import zipfile
    z, n = sys.argv[1], sys.argv[2]
    data = zipfile.ZipFile(z).read(n)
    r = disasm(data)
    print('== bindings'); [print('  ', b) for b in r['binds']]
    print('== cbuffers')
    for c in r['cbs']:
        print('  ', c['name'], c['size'])
        for v in c['vars']: print('      ', v)
    print('== ISGN', r['isgn']); print('== OSGN', r['osgn'])
    print('== code'); print('\n'.join(r['code']))
