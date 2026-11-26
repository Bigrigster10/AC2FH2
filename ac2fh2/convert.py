"""Assetto Corsa car folder -> Forza Horizon 2 add-on car (Extra Options mod folder).

Pipeline
  1. load the AC car (meshes in FH2 axes, materials, textures, chosen skin)
  2. fit the body to the model donor's wheel positions
  3. convert materials + textures, write the body .modelbin
  4. rebuild the donor's car zip under the new media name (carbin, kept models re-targeted)
  5. database.json / strings.json / mod.json / thumbnails
"""
import io
import json
import os
import re
import uuid
import zipfile
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import accar, database, physics, swatchbin
from .bundle import Bundle, Blob, Meta
from .carbin import AOMap, CarScene
from .library import KINDS, DIFFUSE_PARAMS, ALPHA_PARAMS, TemplateLibrary, classify, car_relative
from .material import MaterialInstance, T_TEX, name_hash
from .modelbin import Model, ModelStats, skel_bones, skel_set
from .writer import ALL_LODS, ZERO_MORPH, pattern_uv, MeshIn, build_model

WHEELS = ('LF', 'RF', 'LR', 'RR')


@dataclass
class Settings:
    ac_folder: str
    fh2_cars: list                  # paths of FH2 car zips available (donors + templates)
    gamedb: str
    out_dir: str
    car_id: int = 3100
    name: str = None
    short: str = None
    year: int = None
    skin: str = None
    model_donor: str = None         # media name; auto when None
    physics_donor: int = None       # car id; auto when None
    max_texture: int = 2048
    price: int = None               # override the predicted price
    gauge_speed_scale: float = 1.0  # speedo needle sweep multiplier (AC mods whose STEP misses their own dial)
    measured_stats: dict = None     # in-game My Cars values {'PI': 779, 'Speed': 6.2, ...}: replace the prediction
    material_overrides: dict = field(default_factory=dict)   # AC material -> kind
    log: object = print


# ----------------------------------------------------------------- donors
class ModelDonor:
    def __init__(self, zip_path):
        self.path = zip_path
        self.zip = zipfile.ZipFile(zip_path)
        cb = [n for n in self.zip.namelist() if n.lower().endswith('.carbin') and '/' not in n]
        self.carbin_name = cb[0]
        self.scene = CarScene.parse(self.zip.read(cb[0]))
        self.media = os.path.splitext(cb[0])[0]
        self.car_id = self.scene.ordinal
        wheel = [m for p in self.scene.parts if p.part_type == 43 for m in p.models]
        self.wheels = {m.bone_name[-2:]: np.array(m.matrix[12:15]) for m in wheel}
        names = {n.lower(): n for n in self.zip.namelist()}
        self.locators_xml = self.zip.read(names['locators.xml']).decode('utf-8', 'replace') if 'locators.xml' in names else ''
        m = re.search(r'carLocator_steeringWheel"/>\s*<SceneTransform[^/]*value\._41="([-0-9.]+)"', self.locators_xml)
        self.steer_x = float(m.group(1)) if m else 0.0

    def wheel_layout(self):
        w = self.wheels
        return dict(wb=w['LF'][2] - w['LR'][2], tf=w['RF'][0] - w['LF'][0], tr=w['RR'][0] - w['LR'][0],
                    zf=w['LF'][2], zr=w['LR'][2], y=np.mean([v[1] for v in w.values()]))


def ac_wheel_centres(car):
    c = {}
    for k in WHEELS:
        if f'WHEEL_{k}' in car.nodes:
            c[k] = car.nodes[f'WHEEL_{k}'][3, :3].copy(); continue
        ms = [m for m in car.meshes if f'/WHEEL_{k}' in m.path]
        if ms:
            p = np.concatenate([m.pos for m in ms]); c[k] = (p.min(0) + p.max(0)) / 2
    if len(c) < 4:
        raise ValueError('the car needs WHEEL_LF/RF/LR/RR nodes')
    return c


def ac_steer_x(car):
    m = car.nodes.get('STEER_HR')
    return float(m[3, 0]) if m is not None else 0.0


def choose_model_donor(car, donors):
    """closest wheel layout; strongly prefer the same steering side (the driver animations come with the donor)"""
    ac = ac_wheel_centres(car)
    wb, tf, tr = ac['LF'][2] - ac['LR'][2], ac['RF'][0] - ac['LF'][0], ac['RR'][0] - ac['LR'][0]
    side = np.sign(ac_steer_x(car))
    def cost(d):   # wheels are moved sideways to fit, so track matters less than wheelbase
        l = d.wheel_layout()
        c = abs(np.log(l['wb'] / wb)) + 0.15 * (abs(np.log(l['tf'] / tf)) + abs(np.log(l['tr'] / tr)))
        return c + (0.12 if side and np.sign(d.steer_x) != side else 0)
    return min(donors, key=cost)


def body_fit(car, donor):
    """length scale + offset that puts the AC axles on the donor's (width is kept: the wheels move instead)"""
    ac, l = ac_wheel_centres(car), donor.wheel_layout()
    acwb = ac['LF'][2] - ac['LR'][2]
    sz = l['wb'] / acwb
    scale = np.array([1.0, 1.0, sz], np.float32)
    acy = np.mean([v[1] for v in ac.values()])
    offset = np.array([0.0, l['y'] - acy, l['zf'] - ac['LF'][2] * sz], np.float32)
    return scale, offset


def _straighten(W):
    """rotation (about the wheel centre) that takes the AC wheel axis to +x: removes baked-in camber/toe"""
    c = W[3, :3]
    a = W[0, :3] / np.linalg.norm(W[0, :3]); a = a if a[0] > 0 else -a
    v, cs = np.cross(a, [1, 0, 0]), float(a[0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Rot = (np.eye(3) + K + K @ K / (1 + cs)).astype(np.float32)          # column-vector rotation a -> +x
    return Rot, cs, (lambda p: ((p - c) @ Rot.T + c).astype(np.float32))


def _rim_meshes(car, corner):
    return [m for m in car.meshes if f'/WHEEL_{corner}' in m.path and '/RIM_BLUR' not in m.path.upper()
            and m.material.shader not in ('ksTyres', 'ksBrakeDisc') and classify(m.material).kind != 'skip']


def wheel_shifts(car, donor, ph):
    """carbin wheel transforms mark the wheel's outer face: put it on the AC rim's outer face (AC wheel nodes sit
    at the hub, often several cm inboard of the tyre's centre, so node +- half width puts the wheel too deep)"""
    ac = ac_wheel_centres(car)
    out = {}
    for k in WHEELS:
        w = (ph.tyres['F'] if k[1] == 'F' else ph.tyres['R'])[0] / 1000
        cx = ac[k][0]
        target = cx - w / 2 if cx < 0 else cx + w / 2
        W, ms = car.nodes.get(f'WHEEL_{k}'), _rim_meshes(car, k)
        if W is not None and ms:
            x = _straighten(W)[2](np.concatenate([m.pos for m in ms]))[:, 0]
            target = float(x.min()) if cx < 0 else float(x.max())
            c = W[3, :3]                    # camber is removed: keep the face 1 cm inside the arch (v0.13 poked out)
            body = [m.pos for m in car.meshes if '/WHEEL_' not in m.path and classify(m.material).kind == 'paint']
            if body:
                B = np.concatenate(body); B = B[(abs(B[:, 2] - c[2]) < 0.45) & (B[:, 1] > c[1])]
                if len(B):
                    target = max(target, float(B[:, 0].min()) + 0.01) if cx < 0 else min(target, float(B[:, 0].max()) - 0.01)
        out[k] = target - donor.wheels[k][0]
    return out


def stock_tyres(ph, log):
    """database tyre size = the AC tyre spec. The rim is sized by the game like a stock rim (unit convention below), so
    there is nothing to fudge visually: only keep the size inside what stock cars use (rim 10-24", aspect 25-95,
    width 125-450 mm), keeping the overall tyre diameter (= the physics)."""
    ph.tyres_orig = dict(ph.tyres)
    for axle in ('F', 'R'):
        w, a, rim = ph.tyres[axle]
        outer = rim * 0.0127 + w / 1000 * a / 100
        w2, rim2 = int(min(max(w, 125), 450)), int(min(max(rim, 10), 24))
        asp = lambda r: (outer - r * 0.0127) / (w2 / 1000) * 100
        while asp(rim2) < 25 and rim2 > 10:
            rim2 -= 1
        while asp(rim2) > 95 and rim2 < 24:
            rim2 += 1
        a2 = int(round(min(max(asp(rim2), 25), 95)))
        if (w2, a2, rim2) != (w, a, rim):
            log(f'  tyres ({axle}): AC {w}/{a} R{rim} -> {w2}/{a2} R{rim2} (inside the stock range, same diameter)')
        ph.tyres[axle] = (w2, a2, rim2)


def _rim_signature(car, corner, overrides):
    g = ac_rim(car, corner, overrides)
    return sorted((k, sum(len(m.pos) for m in v[2])) for k, v in g.items())


# ---- stock FH2 wheel convention (identical on every stock wheel examined; the game's morphing vertex shaders apply it):
# a wheel model is a UNIT rim: lip radius 0.14, face geometry at real depth (x 0 = outer face, +x inward), barrel back
# ring at x = 1.0. Two morph targets per vertex, each row's .w = which g_MorphWeights entry drives it:
#   radius (weight 0): delta = 1.318 * (0, y, z)            -> the game scales the rim to the database rim size
#   width  (weight 1): back ring delta x = -0.9             -> the game moves the barrel back to the tyre width
# The tyre the game draws uses the same weights, so a rim built this way always meets the tyre exactly, at any rim
# size (incl. rim-size / tyre-width upgrades). The at-speed blur texture is rendered from the UNIT geometry.
UNIT_LIP = 0.14
RADIAL = 1.318
BACK_X, BACK_DX = 1.0, -0.9
W_RADIUS, W_WIDTH = 0, 1


def rim_scale(inches):
    """(low, high) estimate of the game's radial scale for a rim size: lip at the nominal radius .. ~2 cm flange.
    Only used for clearance checks; the fit with the tyre does not depend on it."""
    lo = inches * 0.0127 / UNIT_LIP
    hi = 1 + RADIAL * (inches - 10) / 14
    return min(lo, hi), max(lo, hi)


def unit_morph(pos_u, nrm_u, dx_width):
    """morph rows [width pos (w1), radius pos (w0), width normal (w1), radius normal (w0)] like stock rims"""
    n = len(pos_u)
    m = np.zeros((n, 4, 4), np.float32)
    m[:, 0, 0] = dx_width; m[:, 0, 3] = W_WIDTH
    m[:, 1, 1:3] = RADIAL * pos_u[:, 1:3]; m[:, 1, 3] = W_RADIUS
    m[:, 2, 3] = W_WIDTH
    s = 1 + RADIAL
    nn = np.c_[nrm_u[:, 0], nrm_u[:, 1] / s, nrm_u[:, 2] / s]
    nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-9
    m[:, 3, :3] = nn - nrm_u; m[:, 3, 3] = W_RADIUS
    return m


def ac_rim(car, corner, overrides):
    """the AC rim meshes of one corner, grouped by material: {name: (kind, material, [meshes])}"""
    groups = {}
    for m in car.meshes:
        if f'/WHEEL_{corner}' not in m.path:
            continue
        if m.material.shader in ('ksTyres', 'ksBrakeDisc') or '/RIM_BLUR' in m.path.upper():
            continue
        kind = overrides.get(m.material.name) or classify(m.material).kind
        if kind != 'skip':
            groups.setdefault(m.material.name, (kind, m.material, []))[2].append(m)
    return groups


def build_wheel(car, ph, lib, bank, skel, overrides, log, axle='F', donor_wheel=None):
    """AC rim -> FH2 wheel model in the stock UNIT convention (see above), so the game sizes it exactly like a stock
    rim. The AC rim is straightened (camber/toe removed), its lip mapped to radius 0.14, its face kept at real depth
    and its barrel's back part given the width morph. The at-speed blur meshes are copied from the donor unchanged."""
    corner = 'LF' if axle == 'F' else 'LR'
    W = car.nodes.get(f'WHEEL_{corner}')
    groups = ac_rim(car, corner, overrides)
    if W is None or not groups:
        return None
    c = W[3, :3]
    Rot, cs, straight = _straighten(W)
    allp = straight(np.concatenate([m.pos for g in groups.values() for m in g[2]]))
    x_out, x_back = float(allp[:, 0].min()), float(allp[:, 0].max())    # left wheel: outer face = min x
    r_all = np.hypot(allp[:, 1] - c[1], allp[:, 2] - c[2])
    r_lip = float(r_all.max())
    ku = UNIT_LIP / r_lip
    width_m = ph.tyres[axle][0] / 1000
    # AC rims are often modelled wider than their tyre spec: fit the whole depth profile into the tyre width (only
    # ever compressed), so the face keeps its place in the wheel like in AC
    fz = min(1.0, width_m / max(x_back - x_out, 1e-3))
    depth_all = (allp[:, 0] - x_out) * fz
    D = (x_back - x_out) * fz                                            # rim depth (back edge of the barrel)
    face = depth_all[r_all < 0.8 * r_lip]
    F = float(face.max()) if len(face) else 0.0                          # deepest point of spokes / hub
    # barrel zone: behind the face, follows the game's width; never beyond what the narrowest tyre leaves
    split = min(max(F + 0.01, 0.4 * D), D - 0.01, 0.75 * min(width_m, D))
    split = max(split, 0.0)
    stretch = (BACK_X - split) / max(D - split, 1e-3)
    mats, idx, meshes_in, tref = [], [], [], set()
    for i, (name, (kind, acmat, gms)) in enumerate(sorted(groups.items(), key=lambda kv: kv[0].lower())):
        mi = lib.instance(kind); mi.name = name
        tref |= _fill_textures(mi, kind, acmat, bank)
        mats.append(mi.to_blob()); idx.append((name, 0))
        for m in gms:
            p = straight(m.pos)
            depth = (p[:, 0] - x_out) * fz
            back = depth > split
            t = np.clip((depth - split) / max(D - split, 1e-3), 0, 1)
            pos = np.zeros_like(p)
            pos[:, 0] = np.where(back, split + t * (BACK_X - split), depth)
            pos[:, 1] = (p[:, 1] - c[1]) * ku
            pos[:, 2] = (p[:, 2] - c[2]) * ku
            dx = np.where(back, BACK_DX * t, 0.0)
            sx = (fz * np.where(back, stretch, 1.0))[:, None]           # AC -> unit scale per vertex (x)
            nrm = m.nrm @ Rot.T
            nrm = np.c_[nrm[:, :1] / sx, nrm[:, 1:3] / ku]               # normals: inverse-transpose of the scale
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
            tan = m.tan.copy(); tan[:, :3] = tan[:, :3] @ Rot.T
            tan[:, :3] = np.c_[tan[:, :1] * sx, tan[:, 1:3] * ku]
            tan[:, :3] /= np.linalg.norm(tan[:, :3], axis=1, keepdims=True) + 1e-9
            meshes_in.append(MeshIn('wheel_LODS0', i, pos.astype(np.float32), nrm.astype(np.float32), m.uv,
                                    tan.astype(np.float32), m.tris, transparent=kind in ('glass', 'textured_alpha'),
                                    bone=0, morph=unit_morph(pos, nrm, dx)))
    nblur = 0
    if donor_wheel is not None:       # the donor's at-speed blur meshes (LOD0), already in the unit convention
        from .material import MaterialInstance
        dm = Model.load(donor_wheel)
        tr = dm.bundle.first('Modl').meta('TRef')
        if tr:
            n = int.from_bytes(tr[:4], 'little')
            tref |= {int.from_bytes(tr[4 + 4 * j:8 + 4 * j], 'little') for j in range(n)}
        used = {}
        for blob, me in dm.meshes:
            nm = bytes(blob.meta('Name') or b'').decode('latin1')
            if not nm.startswith('blur') or not (me.lod_flags & 2) or me.bucket_flags not in (17, 28):
                continue
            g = dm.geometry((blob, me))
            ch = min(i for (sem, i) in g['attrs'] if sem == 'TEXCOORD')     # blur discs use TEXCOORD2 only
            raw = g['attrs'][('TEXCOORD', ch)]
            uvt = np.frombuffer(me.uv_transforms, '<f4').reshape(5, 4)[ch]
            uv = np.c_[raw[:, 0] * uvt[1] + uvt[0], raw[:, 1] * uvt[3] + uvt[2]].astype(np.float32)
            t = next((v for (sem, i), v in sorted(g['attrs'].items()) if sem == 'TANGENT'), None)
            t = t if t is not None else np.tile([0, 1, 0, 1], (len(uv), 1)).astype(np.float32)
            if me.material not in used:
                mb = dm.materials[me.material]
                used[me.material] = len(mats)
                mats.append(mb); idx.append((MaterialInstance.from_blob(mb).name, 0))
            pos = np.asarray(g['pos'], np.float32); nrm = np.asarray(g['nrm'][:, :3], np.float32)
            rows = dm.morph_rows((blob, me))
            morph = np.tile(ZERO_MORPH, (len(pos), 1, 1))
            if rows is not None:      # stock blur: n targets (radius); keep every row with its own weight index
                nt = me.morph_targets
                for k in range(min(nt, 2)):
                    slot = 1 if int(rows[0, k, 3]) == W_RADIUS else 0
                    morph[:, slot] = rows[:, k]; morph[:, slot + 2] = rows[:, nt + k]
            else:
                morph = unit_morph(pos, nrm, np.zeros(len(pos)))
            meshes_in.append(MeshIn(nm, used[me.material], pos, nrm, uv, np.asarray(t, np.float32), g['tris'],
                                    transparent=me.bucket_flags == 28, bone=0, lod_flags=ALL_LODS & ~1,
                                    bucket=me.bucket_flags, cast_shadow=False, morph=morph))
            nblur += 1
    log(f'  rims ({axle}): {len(meshes_in)} meshes ({nblur} stock at-speed blur), stock unit convention: AC lip '
        f'{2 * r_lip / 0.0254:.1f}" -> 0.14, camber {np.degrees(np.arccos(min(cs, 1.0))):.1f} deg removed, face depth '
        f'{F * 100:.1f} cm (AC depth x{fz:.2f} to fit {width_m * 1000:.0f} mm), barrel {split * 100:.1f}..'
        f'{D * 100:.1f} cm follows the tyre width')
    rimv = [mi for mi in meshes_in if mi.lod_flags is None]
    prof = (np.concatenate([np.hypot(mi.pos[:, 1], mi.pos[:, 2]) for mi in rimv]),        # unit radius
            np.concatenate([mi.pos[:, 0] for mi in rimv]),                                 # unit x
            np.concatenate([mi.morph[:, 0, 0] for mi in rimv]))                            # width delta x
    return build_model(meshes_in, mats, skel, tref, morphs=True), idx, prof


def rim_profile(prof, inches, width_m, scale):
    """(radius m, depth m) of every rim vertex as the game draws it at a rim size / tyre width"""
    r_u, x_u, dx = prof
    w1 = min(max((BACK_X - width_m) / -BACK_DX, 0.0), 1.0)      # back ring at the tyre width
    return r_u * scale, x_u + w1 * dx


def _brake_clearance(prof, P, margin=0.005):
    """metres the brake part P (wheel space: x = depth inward, y/z radial) must move inward to clear the rim"""
    r_rim, d_rim = prof
    rb = float(np.hypot(P[:, 1], P[:, 2]).max())
    inner = d_rim[r_rim <= rb]                         # rim surface within the brake's radius (hub, spokes)
    return max(0.0, float(inner.max()) + margin - float(P[:, 0].min())) if len(inner) else 0.0


def _fill_textures(mi, kind, acmat, bank):
    diff = bank.diffuse(acmat.textures.get('txDiffuse')) if kind.startswith('textured') else None
    alpha = bank.alpha(acmat.textures.get('txDiffuse')) if kind == 'textured_alpha' else None
    used = set()
    for p in mi.params:
        if p.type != T_TEX:
            continue
        if p.hash in DIFFUSE_PARAMS and diff:
            p.value = diff
        elif p.hash in ALPHA_PARAMS and alpha:
            p.value = alpha
        elif car_relative(p.value[0]):
            p.value = bank.flat('gloss_neutral', (90, 90, 90)) if 'glos' in p.value[0].lower() else bank.flat('black', (0, 0, 0))
        used.add(p.value[1])
    return used


# --------------------------------------------------------------- physics
def ac_ini(text):
    out, sec = {}, None
    for line in text.splitlines():
        line = line.split(';')[0].strip()
        if line.startswith('['): sec = line.strip('[]'); out[sec] = {}
        elif '=' in line and sec: k, v = line.split('=', 1); out[sec][k.strip()] = v.strip()
    return out


def choose_identity_donor(db_path, brand, year):
    """same brand, closest year: where make, Autoshow group and factory colours come from"""
    import sqlite3
    db = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    prefix = (brand or '')[:3].upper()
    rows = [r for r in db.execute('select Id, MediaName, Year from Data_Car where IsDrivable=1')
            if r[1].upper().startswith(prefix + '_')]
    return min(rows, key=lambda r: abs((r[2] or year) - year)) if rows else None


def choose_physics_donor(db_path, brand, year, drive, mass_kg, kw=None, turbo=None, placement=1):
    """closest real car by drive type, engine position, aspiration, power and weight (any make)"""
    import sqlite3
    db = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    rows = db.execute('''select c.Id, c.MediaName, c.Year, c.DriveTypeID, c.CurbWeight, c.SimPeakPower, e.AspirationID_Stock
        from Data_Car c join List_UpgradeEngine u on u.Ordinal=c.Id and u.IsStock=1 join Data_Engine e on e.EngineID=u.EngineID
        where c.IsDrivable=1 and c.EnginePlacementID=?''', (placement,)).fetchall()
    prefix = (brand or '')[:3].upper()
    def cost(r):
        c = 0 if r[1].upper().startswith(prefix + '_') else 0.3
        c += abs((r[2] or year) - year) / 25 + (0 if r[3] == drive else 3)
        c += abs(r[4] * 100 - mass_kg) / 500 if mass_kg else 0
        if kw: c += abs(np.log(max(r[5], 1) / 10 / kw)) * 2
        if turbo is not None: c += 0 if (r[6] > 1) == turbo else 2
        return c
    return min(rows, key=cost)


def media_name_for(brand_prefix, model, year, length):
    core = re.sub(r'[^A-Za-z0-9]', '', model.title()) or 'Car'
    tail = f'_AC_{year % 100:02d}'
    room = length - len(brand_prefix) - 1 - len(tail)
    if room < 1:
        tail = f'_{year % 100:02d}'; room = length - len(brand_prefix) - 1 - len(tail)
    core = (core + 'X' * room)[:room]
    return f'{brand_prefix}_{core}{tail}'


# --------------------------------------------------------------- textures
class TextureBank:
    def __init__(self, car, skin, media, max_size, log):
        self.car, self.media, self.max, self.log = car, media.lower(), max_size, log
        self.skin_files = car.skins.get(skin, {}) if skin else {}
        self.done = {}             # (name, kind) -> (game path, hash)
        self.files = {}            # zip entry -> bytes

    def image(self, name):
        f = self.skin_files.get(name.lower())
        data = open(f, 'rb').read() if f else self.car.textures.get(name)
        if data is None:
            low = {k.lower(): v for k, v in self.car.textures.items()}
            data = low.get(name.lower())
        if data is None:
            return None
        try:
            return Image.open(io.BytesIO(data)).convert('RGBA')
        except Exception as e:          # unsupported DDS variants
            self.log(f'  ! could not read texture {name}: {e}')
            return None

    def add(self, key, img, srgb=True, kind='DIFF'):
        if key in self.done:
            return self.done[key]
        g = uuid.uuid4()
        stem = re.sub(r'[^A-Za-z0-9]+', '_', os.path.splitext(key[0])[0]).strip('_')[:40] or 'tex'
        fname = swatchbin.guid_name(stem, g, kind)
        entry = f'Textures/AC/{fname}'
        self.files[entry] = swatchbin.encode(img, srgb=srgb, guid=g, max_size=self.max)
        path = f'game:\\media\\cars\\{self.media}\\textures\\ac\\{fname.lower()}'
        self.done[key] = (path, name_hash(path.lower()))
        return self.done[key]

    def diffuse(self, name):
        img = self.image(name) if name else None
        if img is None:
            return None
        # AC's diffuse alpha is a reflection/specular mask (opacity has its own OPAC map here); FH2's materials read
        # diffuse alpha as coverage, so an AC alpha of 0 (common on rims) turned the texture black -> export opaque
        img = img.copy(); img.putalpha(255)
        return self.add((name, 'DIFF'), img)

    def alpha(self, name):
        img = self.image(name) if name else None
        if img is None:
            return None
        a = img.getchannel('A')
        return self.add((name, 'OPAC'), Image.merge('RGBA', (a, a, a, Image.new('L', a.size, 255))), srgb=False, kind='OPAC')

    def flat(self, label, rgb):
        return self.add((label, 'FLAT'), Image.new('RGBA', (8, 8), rgb + (255,)), srgb=False, kind='DIFF')


# ---------------------------------------------------------------- helpers
def _retarget_bytes(blob, old, new):
    """same-length, case-aware replacement of the donor media name (ASCII + UTF-16)"""
    assert len(old) == len(new)
    for enc in ('ascii', 'utf-16-le'):
        pat = re.compile(re.escape(old.encode(enc)), re.I)
        def sub(m, enc=enc):
            s = m.group().decode(enc)
            return (new.lower() if s.islower() else new.upper() if s.isupper() else new).encode(enc)
        blob = pat.sub(sub, blob)
    return blob


def _retarget_modelbin(data, old, new):
    """rename texture paths inside material instances and keep their hashes / TRef consistent"""
    b = Bundle.parse(data)
    hashmap = {}
    for i, blob in enumerate(b.blobs):
        if blob.tag != 'MatI' or not blob.data:
            continue
        mi = MaterialInstance.from_blob(blob)
        changed = False
        for p in mi.params:
            if p.type == T_TEX and old.lower() in p.value[0].lower():
                np_ = _retarget_bytes(p.value[0].encode('latin1'), old, new).decode('latin1')
                nh = name_hash(np_.lower())
                hashmap[p.value[1]] = nh
                p.value = (np_, nh); changed = True
        if changed:
            b.blobs[i] = mi.to_blob()
    if hashmap:
        modl = b.first('Modl'); tr = modl.meta('TRef')
        if tr:
            n = int.from_bytes(tr[:4], 'little')
            hs = [hashmap.get(int.from_bytes(tr[4 + 4 * k:8 + 4 * k], 'little'),
                              int.from_bytes(tr[4 + 4 * k:8 + 4 * k], 'little')) for k in range(n)]
            modl.set_meta('TRef', n.to_bytes(4, 'little') + b''.join(h.to_bytes(4, 'little') for h in hs))
    return b.build()


def _rel(path, media):
    """game:\\media\\cars\\<media>\\scene\\x.modelbin -> scene/x.modelbin (zip entry, lower case)"""
    p = path.replace('\\', '/')
    i = p.lower().find(f'/cars/{media.lower()}/')
    return p[i + len(media) + 7:].lower() if i >= 0 else None


def _thumb(img, w, h):
    img = img.convert('RGBA')
    r = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * r)), max(h, round(img.height * r))), Image.LANCZOS)
    l, t = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((l, t, l + w, t + h))


def _thumb_swatch(img, w, h):
    from . import xbtile
    from .swatchbin import _bc7
    import struct
    data = xbtile.tile_chain([_bc7(img)], w, h)
    # UI images: the format working add-on cars use (2e 01 00 b8, NOT sRGB-tagged). Tagged sRGB (2e 01 12 b0) the
    # Autoshow decoded our gamma-encoded pixels a second time: near-black car, orange rims (v0.13 report)
    txch = uuid.uuid4().bytes_le + struct.pack('<HHHHHH', w, h, 1, w, h, 1) + bytes([1, 1, 0]) + bytes([0x2e, 0x01, 0x00, 0xb8])
    return Bundle(1, 0, [Blob('TXCB', 2, 0, data, [Meta('TXCH', txch)])]).build()


# ---------------------------------------------------------------- convert
def convert(s: Settings):
    log = s.log
    log(f'Loading Assetto Corsa car: {s.ac_folder}')
    car = accar.load(s.ac_folder)
    ui = {}
    uip = os.path.join(s.ac_folder, 'ui', 'ui_car.json')
    if os.path.exists(uip):
        try:
            ui = json.loads(open(uip, 'rb').read().decode('utf-8-sig', 'replace'), strict=False)
        except ValueError:
            ui = {}
    brand = ui.get('brand', '') or 'AC'
    year = s.year or int(ui.get('year') or 2000)
    ph = physics.read(car)
    stock_tyres(ph, log)
    model_name = s.name or re.sub(rf'^{re.escape(brand)}\s*', '', ui.get('name', car.car_id)).strip() or car.car_id
    short = s.short or f"{brand} {model_name.split()[0]} '{year % 100:02d}"
    skin = s.skin or (sorted(car.skins)[0] if car.skins else None)
    from . import colors
    try:
        paints = colors.from_skins(car.skins, skin, colors.harvest(s.fh2_cars), None)
    except Exception as e:
        log(f'  ! paint list: {e}'); paints = []
    if paints:
        log(f'  factory paints: {", ".join(f"{p[0]} -> {p[1].split(chr(92))[-1][:-12]}" for p in paints)}')
    log(f'  {len(car.meshes)} meshes, {len(car.materials)} materials, skin: {skin}')

    # donors
    donors = [ModelDonor(p) for p in s.fh2_cars]
    donor = next((d for d in donors if d.media.lower() == (s.model_donor or '').lower()), None) \
        or choose_model_donor(car, donors)
    if s.physics_donor:
        phys_id = s.physics_donor
    else:
        tags = ' '.join(ui.get('tags', [])).lower()
        placement = 2 if 'mid' in tags else 3 if 'rear-engine' in tags or 'rear engine' in tags else 1
        r = choose_physics_donor(s.gamedb, brand, year, ph.drive, ph.mass, ph.peak_kw, ph.boost > 0, placement)
        phys_id = r[0]
        log(f'  physics donor: {r[1]} (car {r[0]})')
    ident = choose_identity_donor(s.gamedb, brand, year)
    ident_id = ident[0] if ident else None
    if ident: log(f'  brand / Autoshow group from: {ident[1]} (car {ident[0]})')
    else: log(f'  ! no {brand} cars in the game: make comes from the physics donor')
    log(f'  model donor:   {donor.media} (car {donor.car_id})')
    media = media_name_for(brand[:3].upper(), model_name, year, len(donor.media))
    log(f'  new car: id {s.car_id}, media {media}')

    # geometry
    scale, offset = body_fit(car, donor)
    log(f'  body fit: scale {np.round(scale, 3).tolist()} offset {np.round(offset, 3).tolist()}')
    lib = TemplateLibrary()
    missing = lib.harvest(s.fh2_cars)
    from .library import OPTIONAL_KINDS
    opt = {KINDS[k][0].lower() for k in OPTIONAL_KINDS}
    missing = [m_ for m_ in missing if m_ not in opt]
    if missing:
        raise RuntimeError(f'material templates not found in the FH2 cars given: {missing}')
    bank = TextureBank(car, skin, media, s.max_texture, log)

    keep = [m for m in car.meshes if not re.search(r'/(WHEEL|SUSP|HUB)_(LF|RF|LR|RR)\b', m.path)]
    # steering wheel: AC STEER_HR pivot -> the donor's spindle bone (so it turns with the driver's hands)
    spindle = None
    if 'STEER_HR' in car.nodes:
        A = car.nodes['STEER_HR']
        col = A[2, :3] * scale; col /= np.linalg.norm(col)
        if col[2] < 0: col = -col
        spindle = (A[3, :3] * scale + offset, col)
    from . import lights
    light_roles = lights.roles(car)
    light_tpl = lights.templates(donor.zip) if light_roles else {}
    from . import gauges
    dial_names = gauges.dial_meshes(car)
    dial_tpl = gauges.gauge_template(donor.zip) if dial_names else None
    # AC mirrors.ini meshes (AC paints a placeholder texture that it replaces at runtime) -> FH2 real-time mirrors
    raw = car.data.get('mirrors.ini')
    mirror_names = {sec['NAME'] for sec in lights.parse_ini(raw.decode('latin1', 'replace') if isinstance(raw, bytes)
                                                           else raw) if sec.get('NAME')}
    groups = {}
    for m in keep:
        kind = s.material_overrides.get(m.material.name) or classify(m.material).kind
        if kind == 'skip':
            continue
        if mirror_names and any(seg in mirror_names for seg in m.path.split('/')):
            cx = float(m.pos[:, 0].mean())
            mk = 'mirror_left' if cx < -0.3 else 'mirror_right' if cx > 0.3 else 'mirror_center'
            if KINDS[mk][0].lower() not in lib.by_lib:
                mk = 'chrome'
            groups.setdefault(f'{m.material.name}_{mk}', (mk, m.material, []))[2].append(m)
            continue
        if dial_names and any(seg in dial_names for seg in m.path.split('/')) and dial_tpl is not None:
            groups.setdefault(f'{m.material.name}_gauge', ('gauge', m.material, []))[2].append(m)
            continue
        lr = lights.mesh_role(m, light_roles)
        if lr and lr[0].startswith('reverse_'):          # per-side function: split a mesh holding both lamps
            for part, side in zip(lights.split_sides(m), 'LR'):
                role = f'reverse_{side}'
                if part is not None and role in light_tpl:
                    groups.setdefault(f'{m.material.name}_{role}light', (('light', role, lr[1]), m.material, []))[2].append(part)
            continue
        if lr and lr[0] in light_tpl:      # AC lights.ini lamp -> the donor's light material for that function
            key = f'{m.material.name}_{lr[0]}light'
            groups.setdefault(key, (('light',) + lr, m.material, []))[2].append(m)
            continue
        groups.setdefault(m.material.name, (kind, m.material, []))[2].append(m)
    nl = {}
    for k, v in groups.items():
        if isinstance(v[0], tuple):
            nl[v[0][1]] = nl.get(v[0][1], 0) + len(v[2])
    if nl:
        log(f'  lights: {", ".join(f"{n} {r}" for r, n in sorted(nl.items()))} lamp parts glow with the game\'s light functions')

    mats, indices, meshes_in, tref = [], [], [], set()
    entries = {n.lower(): n for n in donor.zip.namelist()}
    skel_entry = _rel(donor.scene.skeleton, donor.media)
    skel_data = Bundle.parse(donor.zip.read(entries[skel_entry])).first('Skel').data
    bones, _ = skel_bones(skel_data)
    bone_names = [b[0] for b in bones]
    spindle_index = bone_names.index('boneSteeringWheelSpindle') if 'boneSteeringWheelSpindle' in bone_names else 1
    if spindle is not None and 'boneSteeringWheelSpindle' in bone_names:
        D = bones[spindle_index][4].copy()
        a, b = D[0, :3] / np.linalg.norm(D[0, :3]), spindle[1]
        v, c = np.cross(a, b), float(np.dot(a, b))
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rm = np.eye(3) + K + K @ K / (1 + c)          # column-vector rotation taking a to b
        D[:3, :3] = D[:3, :3] @ Rm.T                  # rotate each bone axis (rows)
        D[3, :3] = spindle[0]
        skel_data = skel_set(skel_data, 'boneSteeringWheelSpindle', D)
        log(f'  steering wheel: pivot {np.round(spindle[0], 3).tolist()}, turns with the driver')
    # dashboard needles: AC ARROW_SPEED / ARROW_RPM -> the donor's boneSpeed / boneTach (rest = AC pivot, no rotation)
    # + the donor's gauge animations rewritten for the AC dial (gauges.py)
    needle_of, gauge_files = {}, {}
    for gk, (obj, fn, vmax) in gauges.needles(car, s.gauge_speed_scale).items():
        bname, gpath, _ = gauges.BONES[gk]
        gentry = next((n for n in donor.zip.namelist() if n.lower() == gpath.lower()), None)
        if bname not in bone_names or gentry is None:
            continue
        piv, ax = gauges.pivot_axis(car.nodes[obj], scale, offset)
        bi = bone_names.index(bname)
        try:
            gauge_files[gentry.lower()], _ = gauges.build(gk, donor.zip.read(gentry), bones[bi][4][3, :3], piv, ax, fn, vmax)
        except Exception as e:
            log(f'  ! {gk} needle: could not rewrite {gentry} ({e}); needle stays still'); continue
        D = np.eye(4, dtype=np.float32); D[3, :3] = piv
        skel_data = skel_set(skel_data, bname, D)
        needle_of[obj] = (bi, piv)
        log(f'  {gk} needle: {obj} on {bname}, 0..{vmax:.0f} {"km/h" if gk == "speed" else "rpm"} '
            f'-> {float(fn(vmax)):.0f} deg')
    lamp_meshes = {}                         # stock lamp slot (front_L ...) -> [MeshIn] (go into the donor's lamp files)
    for i, (mname, (kind, acmat, ms)) in enumerate(sorted(groups.items(), key=lambda kv: kv[0].lower())):
        role = kind[1] if isinstance(kind, tuple) else None
        if kind == 'gauge':                  # instrument faces + needles: glow at night like stock dials
            mi, used = gauges.dial_material(dial_tpl, mname, acmat.textures.get('txDiffuse'), bank,
                                            donor.media, media)
            tref |= used
            mats.append(mi.to_blob())
            indices.append((mname, 0))
            kind = 'textured'
        elif isinstance(kind, tuple):        # ('light', role, rgb)
            mi, used = lights.make_material(light_tpl[kind[1]], mname, kind[1], kind[2], bank, lights.colour_param(light_tpl))
            tref |= used
            mats.append(mi.to_blob())
            indices.append((mname, 0))
            kind = 'light'
        else:
            mi = lib.instance(kind)
            mi.name = mname
            tref |= _fill_textures(mi, kind, acmat, bank)
            mats.append(mi.to_blob())
            indices.append((mname, KINDS[kind][1]))
        for m in ms:
            pos, nrm, tan = m.pos * scale + offset, m.nrm, m.tan
            bone = 1
            if spindle is not None and '/STEER_HR' in m.path:
                # meshes on an animated bone are stored in that bone's space (the game applies the bone matrix;
                # the donor's own steering wheel sits within a few cm of 0,0,0). Row vectors: world = local @ D
                bone = spindle_index
                Dinv = np.linalg.inv(skel_bones(skel_data)[0][spindle_index][4].astype(np.float64))
                pos = (np.c_[pos, np.ones(len(pos))] @ Dinv)[:, :3].astype(np.float32)
                nrm = (nrm @ Dinv[:3, :3]).astype(np.float32)
                tan = np.c_[tan[:, :3] @ Dinv[:3, :3], tan[:, 3:]].astype(np.float32)
            for obj, (bi, piv) in needle_of.items():
                if f'/{obj}/' in m.path or m.path.endswith('/' + obj):
                    bone, pos = bi, (pos - piv).astype(np.float32)     # bone space (rest rotation = identity)
            uv, tris, uv1 = m.uv, m.tris, None
            if kind.startswith('mirror_'):     # stock mirror UVs: 0..1 over the glass, u along +x, v downwards
                uv = (uv - uv.min(0)) / np.maximum(uv.max(0) - uv.min(0), 1e-6)
                if np.corrcoef(pos[:, 0], uv[:, 0])[0, 1] < 0: uv[:, 0] = 1 - uv[:, 0]
                if np.corrcoef(pos[:, 1], uv[:, 1])[0, 1] > 0: uv[:, 1] = 1 - uv[:, 1]
                uv = uv.astype(np.float32)
            if kind == 'paint':       # special paints (camo, wood...) tile over TEXCOORD1 at an even density
                pos, nrm, uv, tan, tris, uv1 = pattern_uv(pos, nrm, uv, tan, tris)
            mesh = MeshIn(f'{m.name}_LODS0', i, pos, nrm, uv, tan, tris,
                          transparent=kind in ('glass', 'textured_alpha'), bone=bone, uv1=uv1)
            if role:
                for slot, part in lights.lamp_slots(mesh, role):
                    lamp_meshes.setdefault(slot, []).append(part)
            else:
                meshes_in.append(mesh)
    log(f'  {len(meshes_in)} meshes in {len(mats)} materials, {len(bank.files)} textures')

    # car scene: keep body record (re-targeted), brakes + wheels; upgrade parts -> empty model
    sc = donor.scene
    body_rec = next((m for p in sc.parts if p.part_type == 2 for m in p.models
                     if m.path.lower().endswith('\\platform\\body_a.modelbin')), None) \
        or max((m for p in sc.parts if p.part_type == 2 for m in p.models), key=lambda m: dict(m.material_indices).get('carpaint', 0))
    body_entry = _rel(body_rec.path, donor.media)
    skel = Bundle.parse(donor.zip.read(entries[body_entry])).first('Skel')
    if spindle is not None and 'boneSteeringWheelSpindle' in [b[0] for b in skel_bones(skel.data)[0]]:
        skel.data = skel_set(skel.data, 'boneSteeringWheelSpindle', skel_bones(skel_data)[0][spindle_index][4])
    for bname in ('boneSpeed', 'boneTach'):          # the body model's own Skel must match skeleton.modelbin
        if bname in bone_names and bname in [b[0] for b in skel_bones(skel.data)[0]]:
            skel.data = skel_set(skel.data, bname, skel_bones(skel_data)[0][bone_names.index(bname)][4])
    empty_mi = lib.instance('black_plastic'); empty_mi.name = 'empty'
    empty = build_model([MeshIn('empty_LODS0', 0, np.zeros((3, 3), np.float32), np.tile([0, 1, 0], (3, 1)).astype(np.float32),
                                np.zeros((3, 2), np.float32), np.tile([1, 0, 0, 1], (3, 1)).astype(np.float32),
                                np.array([[0, 1, 2]]))], [empty_mi.to_blob()], skel, shadows=False)
    old, new = donor.media, media
    gpath = lambda rel: f'game:\\media\\cars\\{new.lower()}\\' + rel.replace('/', '\\')
    body_rec.path = gpath(body_entry)
    body_rec.material_indices = indices
    ao_path = bank.add(('ao_white', 'AO'), Image.new('RGBA', (256, 256), (255, 255, 255, 255)), srgb=False, kind='AO')[0]
    ao_data = body_rec.ao_maps[0].data if body_rec.ao_maps else AOMap.none().data
    body_rec.ao_maps = [AOMap(3, ao_path, ao_data)]
    # lamps: the game's light sources (Lights.bin: beams, brake/tail glow) are pinned to the donor's lamp model files and
    # only exist if those files are in the carbin -> keep those records, fill them with the AC lamps of the same slot
    # (front/rear, side, centre brake) and move the sources by how far the AC lamps sit from the donor's
    lamp_files, lights_data = {}, None
    lb_name = next((n for n in donor.zip.namelist() if n.lower() == 'lights.bin'), None)
    part2 = [m for p in sc.parts if p.part_type == 2 for m in p.models]
    if lb_name and lamp_meshes:
        lights_data = donor.zip.read(lb_name)
        lpaths = lights.lights_bin(lights_data)[0]
        srcs = lights.light_sources(lights_data)
        slots = {}
        for k, lp in enumerate(lpaths):
            rec = next((m for m in part2 if m.path.lower() == lp.lower()), None)
            sp = [x[3] for x in srcs if x[2] == k]
            if rec is None or not sp:
                continue
            c = np.mean(sp, 0)
            slot = ('front_' if c[2] > 0 else 'centre' if abs(c[0]) < 0.25 else 'rear_') + \
                   ('' if (c[2] <= 0 and abs(c[0]) < 0.25) else 'L' if c[0] < 0 else 'R')
            slots.setdefault(slot, []).append((k, rec))
        if 'centre' not in slots and 'centre' in lamp_meshes:     # no centre lamp file: centre brake joins a side
            for mi_ in lamp_meshes.pop('centre'):
                lamp_meshes.setdefault('rear_' + ('L' if mi_.pos[:, 0].mean() < 0 else 'R'), []).append(mi_)
        deltas, placed = {}, set()
        for slot, recs in slots.items():
            ours = lamp_meshes.get(slot)
            donor_pts = []
            for k, rec in recs:
                md = Model.load(donor.zip.read(entries[_rel(rec.path, donor.media)]))
                donor_pts += [g for g in (md.geometry(bm)['pos'] for bm in md.meshes) if g is not None and len(g)]
            recs.sort(key=lambda kr: 'bulb' not in kr[1].path.lower())   # the bulbs file gets the lamps
            for j, (k, rec) in enumerate(recs):
                rel = _rel(rec.path, donor.media)
                if ours and j == 0:
                    used_m = sorted({mi_.material for mi_ in ours})
                    remap = {o: n for n, o in enumerate(used_m)}
                    sub = [MeshIn(**{**mi_.__dict__, 'material': remap[mi_.material]}) for mi_ in ours]
                    sk = Bundle.parse(donor.zip.read(entries[rel])).first('Skel')
                    lamp_files[rel] = build_model(sub, [mats[o] for o in used_m], sk, tref)
                    rec.material_indices = [indices[o] for o in used_m]
                    placed.add(slot)
                else:
                    lamp_files[rel] = empty
                    rec.material_indices = [('empty', 0)]
                rec.ao_maps = [AOMap(3, ao_path, ao_data)]
                if ours and donor_pts:
                    P = np.concatenate(donor_pts); Q = np.concatenate([mi_.pos for mi_ in ours])
                    dl = (Q.min(0) + Q.max(0)) / 2 - (P.min(0) + P.max(0)) / 2
                    # lengthwise: keep each source's distance from the lamp's OUTER face (beams start ahead of the glass)
                    dl[2] = Q[:, 2].max() - P[:, 2].max() if slot.startswith('front') else Q[:, 2].min() - P[:, 2].min()
                    deltas[k] = tuple(dl)
        lights_data = lights.shift_sources(lights_data, deltas)
        left = [m_ for sl, ms_ in lamp_meshes.items() if sl not in placed for m_ in ms_]
        if left:          # a slot the donor has no lamp file for: keep those lamps in the body (glow only)
            log(f'  ! {len(left)} lamp meshes have no matching donor lamp file; kept in the body')
        log(f'  lamps: {", ".join(sorted(placed))} in the donor lamp files; '
            f'{len(deltas)} light-source groups moved onto the AC lamps')
        keep_recs = [rec for recs in slots.values() for _, rec in recs]
    else:
        left, keep_recs = [m_ for ms_ in lamp_meshes.values() for m_ in ms_], []
    for p in sc.parts:
        if p.part_type == 2:
            p.models = [body_rec] + keep_recs
    body_data = build_model(meshes_in + left, mats, skel, tref)
    # upgrade bodykits: AC cars have none -> keep only the stock entry of each group (an empty model)
    for g in sc.groups:
        g.upgrades = [u for u in g.upgrades if u.is_stock]
        stock_ids = {u.id for u in g.upgrades}
        g.models = [m for m in g.models if set(m.upgrade_ids) & stock_ids][:1]
        for m in g.models:
            m.path = gpath('scene/exterior/ac_empty.modelbin'); m.ao_maps = [AOMap.none()]
            m.material_indices = [('empty', 0)]; m.upgrade_ids = sorted(stock_ids)
    sc.groups = [g for g in sc.groups if g.upgrades]
    # wheels + brakes: move sideways onto the AC wheel positions; the AC rim replaces the donor's
    shifts = wheel_shifts(car, donor, ph)
    wheel_files = {}
    for p in sc.parts:
        if p.part_type not in (4, 43):
            continue
        for m in p.models:
            corner = m.bone_name[-2:].upper()
            if corner in shifts:
                M = np.frombuffer(m.transform, '<f4').copy(); M[12] += shifts[corner]; m.transform = M.tobytes()
            if p.part_type == 43:     # all four corners usually share one model file: keep every record
                wheel_files.setdefault(_rel(m.path, donor.media), []).append(m)
    log(f'  wheels moved sideways by {", ".join(f"{k} {v * 100:+.1f} cm" for k, v in shifts.items())}')
    rim = rear = None
    if wheel_files:
        wentry = next(iter(wheel_files))
        wskel = Bundle.parse(donor.zip.read(entries[wentry])).first('Skel')
        wdata = donor.zip.read(entries[wentry])
        rim = build_wheel(car, ph, lib, bank, wskel, s.material_overrides, log, 'F', wdata)
        if rim:
            rear = None
            # one unit-convention model serves every size (the game morphs it per wheel); a separate rear model only
            # when the AC car really has a different rear rim design
            if _rim_signature(car, 'LR', s.material_overrides) != _rim_signature(car, 'LF', s.material_overrides):
                rear = build_wheel(car, ph, lib, bank, wskel, s.material_overrides, log, 'R', wdata)
            for wf, recs in wheel_files.items():
                for m in recs:
                    use = rear if (rear and m.bone_name[-1:].upper() == 'R') else rim   # LF/RF/LR/RR: axle last
                    m.material_indices = use[1]; m.ao_maps = [AOMap(3, ao_path, ao_data)]
                    if use is rear:
                        m.path = m.path[:m.path.lower().rfind('.modelbin')] + '_rear.modelbin'
                        rear_entry = wf[:-len('.modelbin')] + '_rear.modelbin'
    if rim:   # deep-dish AC rims: push each corner's brakes inward until the rotor/caliper clear the rim's hub
        wheel_T = {m.bone_name[-2:].upper(): np.frombuffer(m.transform, '<f4').reshape(4, 4)
                   for recs in wheel_files.values() for m in recs}
        moved = {}
        for p in sc.parts:
            if p.part_type != 4:
                continue
            by_corner = {}
            for m in p.models:
                by_corner.setdefault(m.bone_name[-2:].upper(), []).append(m)
            for corner, recs in by_corner.items():
                Tw = wheel_T.get(corner)
                if Tw is None:
                    continue
                prof = (rear if (rear and corner[1] == 'R') else rim)[2]
                tw, _, inches = ph.tyres[corner[1]]
                # the rim as the game draws it: stock size and the usual +1/+2" upgrades, both scale estimates
                profs = [rim_profile(prof, d, tw / 1000, sc_) for d in (inches, inches + 1, inches + 2)
                         for sc_ in rim_scale(d)]
                need = 0.0
                for m in recs:
                    M = np.frombuffer(m.transform, '<f4').reshape(4, 4)
                    md = Model.load(donor.zip.read(entries[_rel(m.path, donor.media)]))
                    geo = [md.geometry(bm)['pos'] for bm in md.meshes if bm[1].bucket_flags == 17]
                    P = np.concatenate([g for g in geo if g is not None])
                    L = np.c_[P, np.ones(len(P))] @ M @ np.linalg.inv(Tw)    # brake -> car -> wheel space
                    need = max([need] + [_brake_clearance(pr, L[:, :3]) for pr in profs])
                if need > 0:
                    side = np.sign(Tw[0, 0]) or 1.0                         # left wheels: inward = +x
                    for m in recs:
                        M = np.frombuffer(m.transform, '<f4').copy(); M[12] += side * need; m.transform = M.tobytes()
                    moved[corner] = need
        if moved:
            log(f'  brakes moved inward to clear the rim hub: {", ".join(f"{k} {v * 100:.1f} cm" for k, v in moved.items())}')
    sc.ordinal, sc.media_name = s.car_id, new.lower()
    sc.skeleton = _retarget_bytes(sc.skeleton.encode('latin1'), old, new).decode('latin1')
    for p in sc.parts:
        for m in p.models:
            m.path = _retarget_bytes(m.path.encode('latin1'), old, new).decode('latin1')
            for a in m.ao_maps:
                a.path = _retarget_bytes(a.path.encode('latin1'), old, new).decode('latin1')
    for g in sc.groups:
        for u in g.upgrades:
            if u.id // 1000 == donor.car_id: u.id = s.car_id * 1000 + u.id % 1000
            if u.car_body_id // 1000 == donor.car_id: u.car_body_id = s.car_id * 1000 + u.car_body_id % 1000
        for m in g.models:
            m.upgrade_ids = [s.car_id * 1000 + x % 1000 if x // 1000 == donor.car_id else x for x in m.upgrade_ids]

    # zip contents
    used = {_rel(m.path, new) for m in sc.all_models()} | {_rel(sc.skeleton, new)}
    files = {}
    for n in donor.zip.namelist():
        if n.endswith('/'):
            continue
        low = n.lower()
        if low.endswith('.modelbin') and low.startswith('scene/') and '/drivers/' not in low:
            if _retarget_bytes(low.encode(), old, new).decode() not in used and low != body_entry:
                continue                                   # replaced parts of the donor
        nn = '/'.join(_retarget_bytes(x.encode('latin1'), old, new).decode('latin1') for x in n.split('/'))
        data = donor.zip.read(n)
        if low == body_entry:
            data = body_data
        elif low in gauge_files:
            data = gauge_files[low]
        elif low in lamp_files:
            data = _retarget_modelbin(lamp_files[low], old, new)
        elif low == 'lights.bin' and lights_data is not None:
            data = _retarget_bytes(lights_data, old, new)
        elif rim and low in wheel_files:
            data = rim[0]
        elif low == donor.carbin_name.lower():
            data = sc.build()
        elif low == 'manufacturercolors.bin' and paints:
            data = colors.build([[(p[2], p[3], p[1])] for p in paints])
        elif low == skel_entry:
            sk = Bundle.parse(data); sk.first('Skel').data = skel_data; data = _retarget_modelbin(sk.build(), old, new)
        elif low == 'locators.xml' and spindle is not None:
            txt = data.decode('utf-8', 'replace')
            def put(mm):
                t = mm.group(0)
                for k, v in zip(('41', '42', '43'), spindle[0]):
                    t = re.sub(rf'value\._{k}="[-0-9.e]+"', f'value._{k}="{v:.6f}"', t)
                return t
            txt = re.sub(r'carLocator_steeringWheel"/>\s*<SceneTransform[^/]*/>', put, txt)
            data = _retarget_bytes(txt.encode('utf-8'), old, new)
        elif low.endswith('.modelbin'):
            data = _retarget_modelbin(data, old, new)
        elif not low.endswith('.swatchbin'):
            data = _retarget_bytes(data, old, new)
        files[nn] = data
    files['scene/exterior/ac_empty.modelbin'] = empty
    if rim and rear:
        files['/'.join(_retarget_bytes(x.encode('latin1'), old, new).decode('latin1') for x in rear_entry.split('/'))] = rear[0]
    files.update(bank.files)

    # output folder
    mod_id = re.sub(r'[^a-z0-9]+', '-', f'ac-{brand}-{model_name}-{year}'.lower()).strip('-')
    root = os.path.join(s.out_dir, mod_id)
    os.makedirs(os.path.join(root, 'media', 'cars'), exist_ok=True)
    zpath = os.path.join(root, 'media', 'cars', f'{new}.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n in sorted(files, key=str.upper):
            z.writestr(n, files[n])
    log(f'  wrote {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB, {len(files)} files)')

    nc = database.NewCar(s.car_id, new, model_name, short, year, donor.car_id, phys_id, ident_id)
    doc, strings = database.build(s.gamedb, nc)
    if paints:   # Autoshow / default paint rows follow the manufacturer list
        doc['tables']['Combo_Colors'] = {'key': ['ID'], 'insert': colors.combo_rows(s.gamedb, s.car_id, paints)}
    physics.apply(doc, s.gamedb, s.car_id, ph, log)
    physics.apply_identity(doc, ui, ph, log)
    bodykit = ('List_UpgradeCarBodyFrontBumper', 'List_UpgradeCarBodyRearBumper', 'List_UpgradeCarBodySideSkirt',
               'List_UpgradeCarBodyHood', 'List_UpgradeRearWing')
    for t in bodykit:
        if t in doc['tables']:
            doc['tables'][t]['insert'] = [r for r in doc['tables'][t]['insert'] if r.get('IsStock')]
    for r in doc['tables'].get('Data_CarBody', {}).get('insert', []):   # physics track = where the wheels now are
        wl = {k: donor.wheels[k][0] + shifts[k] for k in WHEELS}
        r['ModelFrontTrackOuter'] = round(float(wl['RF'] - wl['LF']), 4)
        r['ModelRearTrackOuter'] = round(float(wl['RR'] - wl['LR']), 4)
    # rim size is cosmetic here (same overall tyre size): rate/price the car on the AC tyre spec
    dc = doc['tables']['Data_Car']['insert'][0]
    cosmetic = {}
    for side, ax in (('Front', 'F'), ('Rear', 'R')):
        if getattr(ph, 'tyres_orig', None):
            cosmetic.update({f'{side}TireAspect': dc[f'{side}TireAspect'], f'{side}WheelDiameterIN': dc[f'{side}WheelDiameterIN']})
            dc[f'{side}TireAspect'], dc[f'{side}WheelDiameterIN'] = ph.tyres_orig[ax][1], ph.tyres_orig[ax][2]
    stats = physics.predict_stats(doc, s.gamedb, log, s.measured_stats)
    dc.update(cosmetic)
    if s.price: doc['tables']['Data_Car']['insert'][0]['BaseCost'] = int(s.price)
    n, bad = database.verify(s.gamedb, doc)
    log(f'  database: {n} rows, checked against your gamedb ({"OK" if not bad else bad})')
    json.dump(doc, open(os.path.join(root, 'database.json'), 'w'), indent=2)
    json.dump(strings, open(os.path.join(root, 'strings.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump({'format_version': 1, 'id': mod_id, 'name': f'{year} {brand} {model_name}',
               'version': '0.1.0', 'author': f'ac2fh2 (original: {ui.get("author", "unknown")})',
               'description': f'Converted from Assetto Corsa ({car.car_id}).',
               'database': 'database.json', 'media': 'media', 'dependencies': []},
              open(os.path.join(root, 'mod.json'), 'w'), indent=2)
    tdir = os.path.join(root, 'media', 'stripped', 'RC0'); os.makedirs(tdir, exist_ok=True)
    from . import thumbnail
    paint_rgb = paints[0][3] if paints else (0.5, 0.5, 0.52)
    for suf, (w, h) in (('Big', (816, 459)), ('Small', (192, 108))):
        try:
            img = thumbnail.render(car, paint_rgb, bank.image, size=(w, h), shadow=(suf == 'Big'))
        except Exception as e:                 # fall back to the AC preview photo
            log(f'  ! thumbnail render failed ({e}); using the AC preview')
            prev = car.skins.get(skin, {}).get('preview.jpg') if skin else None
            img = _thumb(Image.open(prev) if prev else Image.new('RGB', (w, h), (240, 240, 240)), w, h)
        open(os.path.join(tdir, f'Thumbnail_{s.car_id}_{suf}.swatchbin'), 'wb').write(_thumb_swatch(img.convert('RGBA'), w, h))
    log(f'Done: {root}')
    return root
