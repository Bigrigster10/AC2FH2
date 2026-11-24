"""Working lights: AC lights.ini meshes -> FH2 light materials (glow when the game switches that light function on).

FH2 light glow is purely material driven: light materials (library lbk_* reflectors / gls_lightbulb) carry
LightFunctionCarLightParam = index of a light preset in the car's LightPresets.bin (which we keep from the donor), an
emissive colour (0xc308c931) and strength (0x6f45c5f1); the game feeds the on/off level per function. So each AC light
mesh gets a copy of the donor's own material for that function (headlight / tail / brake), with its glow mask made
white (the stock masks are UV-mapped to the donor's lamps).
AC lights.ini: [BRAKE_n] NAME=mesh (glows when braking), [LIGHT_n] NAME=mesh COLOR=r,g,b (glows with the lights on).
"""
import re

import copy

import numpy as np

from .material import MaterialInstance, Param, name_hash, T_TEX, T_INT, T_COLOR

LIGHT_FUNCTION = name_hash('LightFunctionCarLightParam')
GLOW_COLOUR, GLOW_STRENGTH, GLOW_MASK = 0xc308c931, 0x6f45c5f1, 0x35c82561
INTERIOR = re.compile(r'cockpit|interior|gauge|needle|dash|display|lcd', re.I)


def parse_ini(text):
    out, sec = [], None
    for line in (text or '').splitlines():
        line = line.split(';')[0].strip()
        if line.startswith('['):
            sec = {'section': line.strip('[]').upper()}; out.append(sec)
        elif '=' in line and sec is not None:
            k, v = line.split('=', 1); sec[k.strip().upper()] = v.strip()
    return out


def roles(car):
    """{mesh name: (role, rgb 0..1)} for exterior lights: 'brake', 'head', 'amber', 'tail'"""
    raw = car.data.get('lights.ini')
    text = raw.decode('latin1', 'replace') if isinstance(raw, bytes) else raw
    out = {}
    pos = {}
    for m in car.meshes:
        pos.setdefault(m.path.split('/')[-1], m)
    for sec in parse_ini(text):
        name = sec.get('NAME')
        if not name or INTERIOR.search(name):
            continue
        try:
            rgb = [float(x) for x in sec.get('COLOR', '0,0,0').split(',')[:3]]
        except ValueError:
            rgb = [0, 0, 0]
        if max(rgb) <= 0 and sec['section'].startswith('LIGHT'):
            continue
        mx = max(max(rgb), 1e-6); col = tuple(c / mx for c in rgb)
        ms = [m for m in car.meshes if name in m.path.split('/')]
        if not ms:
            continue
        front = float(sum(m.pos[:, 2].mean() for m in ms) / len(ms)) > 0
        if sec['section'].startswith('BRAKE'):
            role = 'brake'
        elif front:
            role = 'head' if col[2] > 0.5 else 'amber'
        else:
            role = 'tail'
        out[name] = (role, col)
    for m in car.meshes:                     # reversing lamps: AC drives them by name, not lights.ini
        seg = m.path.split('/')[-1]
        if seg not in out and re.search(r'revers', m.material.name + seg, re.I) and not INTERIOR.search(m.path):
            out[seg] = ('reverse_' + ('L' if float(m.pos[:, 0].mean()) < 0 else 'R'), (1.0, 1.0, 1.0))
    return out


def mesh_role(m, table):
    for seg in m.path.split('/'):
        if seg in table:
            return table[seg]
    return None


def templates(donor_zip):
    """donor light materials (MatI blobs) for each role, chosen by name + light function"""
    from .modelbin import Model
    found = {}
    for n in donor_zip.namelist():
        low = n.lower()
        if not (low.endswith('.modelbin') and 'lights' in low and 'glass' not in low.split('/')[-1]):
            continue
        try:
            md = Model.load(donor_zip.read(n))
        except Exception:
            continue
        for blob in md.materials.values():
            if not blob.data:
                continue
            mi = MaterialInstance.from_blob(blob)
            if not any(p.hash == LIGHT_FUNCTION for p in mi.params):
                continue
            lib = mi.library_path.lower(); nm = mi.name.lower()
            score = 2 if 'lbk_custom' in lib else 1                  # reflector with a glow mask: best fit
            side = 'L' if 'lights/' in low and low.split('/')[-1][len(low.split('/')[-1].split('light')[0]) + 5:][:1] == 'l' else 'R'
            role = ('brake' if 'brake' in nm else 'reverse_' + side if 'reverse' in nm else
                    'head' if 'headlight' in nm or ('headlight' in low and 'drl' not in nm and 'default' not in nm) else
                    'tail' if 'taillight' in low and ('custom_01' in nm or 'tail' in nm) else None)
            if role and found.get(role, (0,))[0] < score:
                found[role] = (score, blob)
    out = {k: v[1] for k, v in found.items()}
    if 'head' in out:
        out['amber'] = out['head']
    for side, other in (('L', 'R'), ('R', 'L')):          # a donor with one reverse lamp serves both sides
        if f'reverse_{side}' not in out and f'reverse_{other}' in out:
            out[f'reverse_{side}'] = out[f'reverse_{other}']
    return out


def colour_param(templates_):
    """a glow-colour parameter to give templates that have none (same library parameter, any donor material)"""
    for blob in templates_.values():
        for p in MaterialInstance.from_blob(blob).params:
            if p.hash == GLOW_COLOUR:
                return p
    return None


def make_material(template_blob, name, role, rgb, bank, colour_param=None):
    """copy of the donor light material for this AC mesh group; glow mask -> white, amber lamps glow amber"""
    mi = MaterialInstance.from_blob(template_blob)
    mi.name = name
    if role == 'amber' and not any(p.hash == GLOW_COLOUR for p in mi.params) and colour_param is not None:
        mi.params.append(copy.deepcopy(colour_param))
    used = set()
    for p in mi.params:
        if p.type == T_TEX:
            if p.hash == GLOW_MASK or '\\media\\cars\\' in p.value[0].lower() and '\\_library\\' not in p.value[0].lower():
                p.value = bank.flat('white', (255, 255, 255))
            used.add(p.value[1])
        elif p.hash == GLOW_COLOUR and role == 'amber':
            p.value = (1.0, 0.33, 0.02, 1.0)
    return mi, used


def split_sides(m):
    """a mesh holding both sides' lamps -> (left part, right part) by triangle centre (FH2 -x = left)"""
    cx = m.pos[m.tris].mean(1)[:, 0]
    parts = []
    for sel in (cx < 0, cx >= 0):
        if not sel.any():
            parts.append(None); continue
        t = m.tris[sel]
        used, inv = np.unique(t, return_inverse=True)
        q = copy.copy(m)
        q.pos, q.nrm, q.uv, q.tan = m.pos[used], m.nrm[used], m.uv[used], m.tan[used]
        q.tris = inv.reshape(-1, 3)
        parts.append(q)
    return parts


# ---- Lights.bin: the car's light SOURCES (beams, glow points), each pinned to one of the car's lamp model files ----
# u32 0xdeadbeef, u32 1, u32 n, n x (u32 len, path, u16 1), u32 count, count x 76-byte records, u32 0.
# Record: u32 type, u32 function (low 16 bits; brake 69, headlight 17, DRL 13, indicators 20/21 ...), u32 model index,
# 2 x (position xyz0, direction xyz0). A source whose model is not in the carbin does not exist in game.
import struct as _st

LIGHT_REC = 76


def lights_bin(data):
    """-> (model paths, offset of the first record, record count)"""
    assert _st.unpack_from('<I', data, 0)[0] == 0xdeadbeef
    n = _st.unpack_from('<I', data, 8)[0]; p = 12; paths = []
    for _ in range(n):
        ln = _st.unpack_from('<I', data, p)[0]; paths.append(data[p + 4:p + 4 + ln].decode('latin1')); p += 6 + ln
    cnt = _st.unpack_from('<I', data, p)[0]
    assert p + 4 + cnt * LIGHT_REC + 4 == len(data), 'unexpected Lights.bin layout'
    return paths, p + 4, cnt


def light_sources(data):
    paths, p0, cnt = lights_bin(data)
    out = []
    for i in range(cnt):
        t, fn, mdl = _st.unpack_from('<3I', data, p0 + i * LIGHT_REC)
        pos = _st.unpack_from('<3f', data, p0 + i * LIGHT_REC + 12)
        out.append((t, fn & 0xffff, mdl, pos))
    return out


def shift_sources(data, deltas):
    """deltas: {model index: (dx, dy, dz)} -> Lights.bin with those models' source positions moved"""
    paths, p0, cnt = lights_bin(data)
    b = bytearray(data)
    for i in range(cnt):
        base = p0 + i * LIGHT_REC
        mdl = _st.unpack_from('<I', b, base + 8)[0]
        d = deltas.get(mdl)
        if d is None:
            continue
        for off in (12, 44):                       # position of each half (direction vectors untouched)
            v = _st.unpack_from('<3f', b, base + off)
            _st.pack_into('<3f', b, base + off, *(float(a) + float(c) for a, c in zip(v, d)))
    return bytes(b)


def lamp_slots(mesh, role):
    """[(slot, MeshIn)]: the stock lamp slot(s) an AC lamp mesh belongs to (front_L/front_R/rear_L/rear_R/centre;
    FH2 -x = left, +z = front). A narrow brake lamp on the centre line = centre (third brake light); a mesh holding
    both sides' lamps is split by triangle centre."""
    pos = mesh.pos
    front = (pos[:, 2].min() + pos[:, 2].max()) / 2 > 0
    if not front and role == 'brake' and np.abs(pos[:, 0]).max() < 0.3:
        return [('centre', mesh)]
    out = []
    cx = pos[mesh.tris].mean(1)[:, 0]
    for sel, side in ((cx < 0, 'L'), (cx >= 0, 'R')):
        if not sel.any():
            continue
        t = mesh.tris[sel]
        used, inv = np.unique(t, return_inverse=True)
        q = copy.copy(mesh)
        q.pos, q.nrm, q.uv, q.tan = mesh.pos[used], mesh.nrm[used], mesh.uv[used], mesh.tan[used]
        if mesh.uv1 is not None:
            q.uv1 = mesh.uv1[used]
        q.tris = inv.reshape(-1, 3)
        out.append((('front_' if front else 'rear_') + side, q))
    return out
