"""Working dashboard needles: AC ARROW_SPEED / ARROW_RPM meshes on the donor's boneSpeed / boneTach, driven by the
donor's Animations/speedometer.gr2 / tachometer.gr2 rewritten for the AC dial.

How FH2 drives a needle (verified on 5 stock cars): the game plays the gauge animation at frame = value
(speedometer: 1 frame per km/h, tachometer: 1 frame per 10 rpm, 30 frames per second), so a gauge animation is just
'needle orientation as a function of the reading'. Each animation has one track for the bone: a constant position
(D3Constant32f, relative to the Granny root = skeleton position minus a fixed per-car offset) and an orientation curve
(D4nK16uC15u: knots + 3 x 15-bit quaternion components, the 4th rebuilt) in the same axes as the skeleton.

Granny file (v7, uncompressed sections, 64-bit pointers): CRC32 of everything from byte 104 at byte 40; section 0 holds
the objects, its relocation table gives every pointer. We only overwrite values in place (same sizes), so all
pointers stay valid.
"""
import re
import struct
import zlib

import numpy as np

SCALE = [1.4142135, 0.70710677, 0.35355338, 0.35355338, 0.35355338, 0.17677669, 0.17677669, 0.17677669,
         -1.4142135, -0.70710677, -0.35355338, -0.35355338, -0.35355338, -0.17677669, -0.17677669, -0.17677669]
OFFSET = [-0.70710677, -0.35355338, -0.53033006, -0.17677669, 0.17677669, -0.17677669, -0.088388346, 0.0,
          0.70710677, 0.35355338, 0.53033006, 0.17677669, -0.17677669, 0.17677669, 0.088388346, -0.0]
FPS = 30.0
BONES = {'speed': ('boneSpeed', 'Animations/speedometer.gr2', 1.0),     # frames per unit: km/h
         'rpm': ('boneTach', 'Animations/tachometer.gr2', 0.1)}         # rpm / 10


# ------------------------------------------------------------------------------------------- Granny (in place)
class Track:
    """the one bone track of a stock gauge animation, located in the file (absolute offsets)"""

    def __init__(self, data, bone):
        d = self.data = bytearray(data)
        base = 0x20
        soff, scnt = struct.unpack_from('<2I', d, base + 12)
        sec = [struct.unpack_from('<11I', d, base + soff + i * 44) for i in range(scnt)]
        assert all(s[0] == 0 for s in sec), 'compressed Granny section'
        s0 = sec[0][1]; self.s0 = s0
        rel = [struct.unpack_from('<3I', d, sec[0][7] + k * 12) for k in range(sec[0][8])]
        ptr = {r[0]: (r[1], r[2]) for r in rel}
        name_at = bytes(d[s0:s0 + sec[0][2]]).find(bone.encode() + b'\0')
        assert name_at >= 0, f'{bone} not in the animation'
        tr = next(o for o, (sc, t) in ptr.items() if sc == 0 and t == name_at)
        self.ori = ptr[tr + 20][1]; self.pos = ptr[tr + 36][1]
        fmt, deg, sel, okn, cnt = struct.unpack_from('<BBHfI', d, s0 + self.ori)
        assert fmt == 8, f'orientation curve format {fmt} (expected D4nK16uC15u)'
        self.n = cnt // 4; self.arr = ptr[self.ori + 12][1]
        self.okn = okn
        # Animation.Duration: the float equal to the last knot time, right before TimeStep 1/60 and Oversampling
        last = struct.unpack_from('<H', d, s0 + self.arr + 2 * (self.n - 1))[0] / okn
        f = np.frombuffer(bytes(d[:len(d) // 4 * 4]), '<f4')
        hits = [i * 4 for i in np.nonzero(np.abs(f - last) < 2e-3 * max(1, last))[0]
                if abs(struct.unpack_from('<f', d, i * 4 + 4)[0] - 1 / 60) < 1e-4]
        assert hits, 'animation duration not found'
        self.duration_at = hits[0]

    def position(self):
        return np.array(struct.unpack_from('<3f', self.data, self.s0 + self.pos + 4))

    def quats(self):
        d, s0, n = self.data, self.s0, self.n
        _, _, sel, okn, _ = struct.unpack_from('<BBHfI', d, s0 + self.ori)
        kc = struct.unpack_from(f'<{4 * n}H', d, s0 + self.arr)
        sc = [SCALE[(sel >> 4 * k) & 15] / 32767 for k in range(4)]
        of = [OFFSET[(sel >> 4 * k) & 15] for k in range(4)]
        out = []
        for i in range(n):
            a, b, c = kc[n + 3 * i:n + 3 * i + 3]
            s1 = ((b & 0x8000) >> 14) | (c >> 15)
            q = [0.0] * 4
            for v, s in ((a, (s1 + 1) & 3), (b, (s1 + 2) & 3), (c, (s1 + 3) & 3)):
                q[s] = (v & 0x7fff) * sc[s] + of[s]
            w = np.sqrt(max(0.0, 1 - sum(x * x for x in q)))
            q[s1] = -w if a & 0x8000 else w
            out.append(q)
        return np.array([k / okn for k in kc[:n]]), np.array(out)

    def write(self, pos, times, quats):
        """constant position + linear (degree 1) orientation curve with exactly n knots; duration = last knot"""
        d, s0, n = self.data, self.s0, self.n
        assert len(times) == n == len(quats)
        struct.pack_into('<3f', d, s0 + self.pos + 4, *map(float, pos))
        dur = float(times[-1])
        okn = 65535.0 / dur
        struct.pack_into('<BBHf', d, s0 + self.ori, 8, 1, 0x0000, okn)
        sc, of = SCALE[0] / 32767, OFFSET[0]          # table entry 0 for all: range +-0.7071
        kc = [min(65535, int(round(t * okn))) for t in times]
        prev = None
        for q in quats:
            q = np.asarray(q, float) / np.linalg.norm(q)
            if prev is not None and np.dot(q, prev) < 0:
                q = -q                                   # same hemisphere as the previous key (shortest path)
            prev = q
            s1 = int(np.argmax(np.abs(q)))
            comps = [(int(round((q[s] - of) / sc)) & 0x7fff) for s in ((s1 + 1) & 3, (s1 + 2) & 3, (s1 + 3) & 3)]
            a = comps[0] | (0x8000 if q[s1] < 0 else 0)
            b = comps[1] | ((s1 >> 1) << 15)
            c = comps[2] | ((s1 & 1) << 15)
            kc += [a, b, c]
        struct.pack_into(f'<{4 * n}H', d, s0 + self.arr, *kc)
        struct.pack_into('<f', d, self.duration_at, dur)
        struct.pack_into('<I', d, 40, zlib.crc32(bytes(d[104:])))
        return bytes(d)


# ------------------------------------------------------------------------------------------------ AC side
def _ini(text):
    from .lights import parse_ini
    return parse_ini(text)


def needles(car, speed_scale=1.0):
    """-> {'speed'|'rpm': (node name, angle_deg(value) function, max value)} from AC analog_instruments.ini.
    speed_scale: multiplies the speedo sweep (for AC mods whose STEP does not match their own dial face)"""
    raw = car.data.get('analog_instruments.ini')
    text = raw.decode('latin1', 'replace') if isinstance(raw, bytes) else raw
    out = {}
    for sec in _ini(text):
        s = sec['section']
        kind = 'rpm' if s.startswith('RPM') else 'speed' if s.startswith('SPEED') else None
        obj = sec.get('OBJECT_NAME', '').strip()
        if not kind or not obj or obj not in car.nodes:
            continue
        lut = sec.get('LUT', '').strip('() ')
        if lut:
            pts = sorted((float(a), float(b)) for a, b in (p.split('=') for p in lut.split('|') if '=' in p))
            xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            fn = (lambda xs, ys: lambda v: np.interp(v, xs, ys))(xs, ys)
            vmax = float(xs[-1])
        else:
            step, zero = float(sec.get('STEP', 1)), float(sec.get('ZERO', 0))
            if kind == 'speed':
                step *= speed_scale
            fn = (lambda st, z: lambda v: z + st * np.asarray(v, float))(step, zero)
            vmax = min(500.0, 300.0 / max(abs(step), 1e-6)) if kind == 'speed' else 10000.0
        out[kind] = (obj, fn, vmax)
    return out


def pivot_axis(node_matrix, scale, offset):
    """needle pivot (FH2 world) and rotation axis = the node axis pointing most toward the driver (-z)"""
    M = np.asarray(node_matrix, float)
    piv = M[3, :3] * scale + offset
    ax = [M[i, :3] / (np.linalg.norm(M[i, :3]) + 1e-12) for i in range(3)]
    a = max(ax, key=lambda v: abs(v[2]))
    return piv, (a if a[2] < 0 else -a)


def _quat(axis, ang):
    h = np.radians(ang) / 2
    return np.r_[np.asarray(axis) * np.sin(h), np.cos(h)]


def _rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def clockwise_sign(axis):
    """sign s such that rotating by s * angle about axis turns 'up' toward the driver's right (clockwise as seen from
    the seat: the driver looks along +z, +x = right)"""
    up = np.array([0.0, 1.0, 0.0]) - axis * axis[1]
    r = _rot(_quat(axis, 10)) @ up
    return 1.0 if r[0] > up[0] else -1.0


def build(kind, gr2_data, donor_rest_pos, pivot, axis, angle_fn, vmax):
    """-> rewritten gauge animation bytes (bone rest rotation = identity, so the AC mesh is stored pivot-relative)"""
    bone, _, per_unit = BONES[kind]
    tr = Track(gr2_data, bone)
    root_offset = np.asarray(donor_rest_pos) - tr.position()       # skeleton position - Granny position
    n = tr.n
    vals = np.linspace(0, vmax, n)
    s = clockwise_sign(axis)
    quats = [_quat(axis, s * float(angle_fn(v))) for v in vals]      # AC: the mesh is modelled at angle 0
    times = vals * per_unit / FPS
    return tr.write(np.asarray(pivot) - root_offset, times, quats), s


# ------------------------------------------------------------------------------------------- night glow
DIAL = re.compile(r'gauge|needle|dial|cluster|instrument|speedo|tacho|rev_?counter', re.I)
LITE_TEXTURE = 0x4e0d5e89            # int_gauges: glow layer (shown with the lights on)
DIFFUSE_TEXTURE = 0x6dd98cd9


def dial_meshes(car):
    """AC mesh names lit as instruments: interior [LIGHT_n] entries of lights.ini that name gauges / needles"""
    from .lights import parse_ini
    raw = car.data.get('lights.ini')
    text = raw.decode('latin1', 'replace') if isinstance(raw, bytes) else raw
    return {sec['NAME'] for sec in parse_ini(text)
            if sec['section'].startswith('LIGHT') and sec.get('NAME') and DIAL.search(sec['NAME'])}


def gauge_template(donor_zip):
    """the donor's own instrument material (per-car int_gauges.materialbin: diffuse + 'lite' glow texture)"""
    from .modelbin import Model
    from .material import MaterialInstance
    for n in donor_zip.namelist():
        if not n.lower().endswith('.modelbin') or 'interior' not in n.lower():
            continue
        try:
            md = Model.load(donor_zip.read(n))
        except Exception:
            continue
        for blob in md.materials.values():
            if blob.data and MaterialInstance.from_blob(blob).library_path.lower().endswith('int_gauges.materialbin'):
                return blob
    return None


def dial_material(template_blob, name, tex, bank, old_media, new_media):
    """AC dial texture as both the daytime colour and the night glow layer"""
    from .material import MaterialInstance, T_TEX
    from .convert import _retarget_bytes
    mi = MaterialInstance.from_blob(template_blob)
    mi.name = name
    mi.library_path = _retarget_bytes(mi.library_path.encode('latin1'), old_media, new_media).decode('latin1')
    d = bank.diffuse(tex) if tex else bank.flat('black', (0, 0, 0))
    used = set()
    for p in mi.params:
        if p.type != T_TEX:
            continue
        if p.hash in (DIFFUSE_TEXTURE, LITE_TEXTURE):
            p.value = d
        elif '\\media\\cars\\' in p.value[0].lower() and '\\_library\\' not in p.value[0].lower():
            p.value = bank.flat('black', (0, 0, 0))
        used.add(p.value[1])
    return mi, used
