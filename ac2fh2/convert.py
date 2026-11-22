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
from .writer import pattern_uv, MeshIn, build_model

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


def wheel_shifts(car, donor, ph):
    """carbin wheel transforms mark the wheel's outer face: move each to the AC wheel position"""
    ac = ac_wheel_centres(car)
    out = {}
    for k in WHEELS:
        w = (ph.tyres['F'] if k[1] == 'F' else ph.tyres['R'])[0] / 1000
        cx = ac[k][0]
        target = cx - w / 2 if cx < 0 else cx + w / 2
        out[k] = target - donor.wheels[k][0]
    return out


def fit_rim_size(car, ph, log, min_aspect=29.5):
    """AC rims are often drawn bigger than the tyres.ini rim (thin tyre around them). FH2 draws its own tyre from the
    database size, so use the biggest rim (<= the drawn size) that keeps the overall wheel diameter -- i.e. the
    physics -- the same while the profile stays >= FH2's thinnest (30%)."""
    for axle, corner in (('F', 'LF'), ('R', 'LR')):
        W = car.nodes.get(f'WHEEL_{corner}')
        ms = [m for m in car.meshes if f'/WHEEL_{corner}' in m.path and '/RIM_BLUR' not in m.path.upper()
              and m.material.shader not in ('ksTyres', 'ksBrakeDisc') and classify(m.material).kind != 'skip']
        if W is None or not ms:
            continue
        c = W[3, :3]; P = np.concatenate([m.pos for m in ms])
        drawn = 2 * float(np.hypot(P[:, 1] - c[1], P[:, 2] - c[2]).max()) / 0.0254
        w_mm, aspect, rim = ph.tyres[axle]
        outer = rim * 0.0127 + w_mm / 1000 * aspect / 100            # tyre radius, m
        for inch in range(int(drawn), int(rim), -1):
            a = (outer - inch * 0.0127) / (w_mm / 1000) * 100
            if a >= min_aspect:
                ph.tyres[axle] = (w_mm, int(round(a)), inch)
                log(f'  rims ({axle}): AC draws {drawn:.1f}" rims -> {w_mm}/{round(a)} R{inch} (same overall size as '
                    f'{w_mm}/{aspect} R{rim})')
                break


def build_wheel(car, ph, lib, bank, skel, overrides, log, axle='F'):
    """AC rim -> FH2 wheel model at REAL size (metres): x 0 = outer face .. tyre width inward, radius = the
    database rim radius. Stock FH2 wheels are a unit rim (r 0.14, x 0..1) plus morph targets the game blends to the
    real rim size / width; there is no scaling in the car file, so a rim without morphs must already be real size.
    The AC rim is fitted by its OWN extents (AC rim meshes are often drawn bigger than the rim size in tyres.ini)."""
    corner = 'LF' if axle == 'F' else 'LR'
    W = car.nodes.get(f'WHEEL_{corner}')
    ms = [m for m in car.meshes if f'/WHEEL_{corner}' in m.path]
    if W is None or not ms:
        return None
    c = W[3, :3]
    groups = {}
    for m in ms:
        kind = overrides.get(m.material.name) or classify(m.material).kind
        if kind == 'skip' or m.material.shader in ('ksTyres', 'ksBrakeDisc') or '/RIM_BLUR' in m.path.upper():
            continue
        groups.setdefault(m.material.name, (kind, m.material, []))[2].append(m)
    if not groups:
        return None
    allp = np.concatenate([m.pos for g in groups.values() for m in g[2]])
    x_out, x_in = float(allp[:, 0].min()), float(allp[:, 0].max())     # left wheels: outer face is -x
    r_vis = float(np.hypot(allp[:, 1] - c[1], allp[:, 2] - c[2]).max())
    width_m, _, rim_in = ph.tyres[axle]
    R = rim_in * 0.0254 / 2
    k = R / r_vis
    sc = np.array([(width_m / 1000) / (x_in - x_out), k, k], np.float32)
    mats, idx, meshes_in = [], [], []
    for i, (name, (kind, acmat, gms)) in enumerate(sorted(groups.items(), key=lambda kv: kv[0].lower())):
        mi = lib.instance(kind); mi.name = name
        _fill_textures(mi, kind, acmat, bank)
        mats.append(mi.to_blob()); idx.append((name, 0))
        for m in gms:
            pos = (m.pos - np.array([x_out, c[1], c[2]], np.float32)) * sc
            nrm = m.nrm / sc; nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
            tan = m.tan.copy(); tan[:, :3] *= sc; tan[:, :3] /= np.linalg.norm(tan[:, :3], axis=1, keepdims=True) + 1e-9
            meshes_in.append(MeshIn(f'{m.name}_LODS0', i, pos.astype(np.float32), nrm.astype(np.float32), m.uv,
                                    tan.astype(np.float32), m.tris, transparent=kind in ('glass', 'textured_alpha'), bone=0))
    log(f'  rims ({axle}): {len(meshes_in)} meshes from the AC wheel (drawn {2 * r_vis / 0.0254:.1f}" across), '
        f'built at {rim_in}" x {width_m} mm, real size')
    # how deep the rim is at each radius (metres from the outer face): brakes must sit behind it
    allw = np.concatenate([mi.pos for mi in meshes_in])
    prof = (np.hypot(allw[:, 1], allw[:, 2]), allw[:, 0])
    return build_model(meshes_in, mats, skel), idx, prof


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
    from .swatchbin import _bc7, FMT_BC7_SRGB
    import struct
    data = xbtile.tile_chain([_bc7(img)], w, h)
    txch = uuid.uuid4().bytes_le + struct.pack('<HHHHHH', w, h, 1, w, h, 1) + bytes([1, 1, 0]) + FMT_BC7_SRGB
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
    fit_rim_size(car, ph, log)
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
    groups = {}
    for m in keep:
        kind = s.material_overrides.get(m.material.name) or classify(m.material).kind
        if kind == 'skip':
            continue
        groups.setdefault(m.material.name, (kind, m.material, []))[2].append(m)

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
    for i, (mname, (kind, acmat, ms)) in enumerate(sorted(groups.items(), key=lambda kv: kv[0].lower())):
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
            uv, tris, uv1 = m.uv, m.tris, None
            if kind == 'paint':       # special paints (camo, wood...) tile over TEXCOORD1 at an even density
                pos, nrm, uv, tan, tris, uv1 = pattern_uv(pos, nrm, uv, tan, tris)
            meshes_in.append(MeshIn(f'{m.name}_LODS0', i, pos, nrm, uv, tan, tris,
                                    transparent=kind in ('glass', 'textured_alpha'), bone=bone, uv1=uv1))
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
    body_data = build_model(meshes_in, mats, skel, tref)
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
    for p in sc.parts:
        if p.part_type == 2:
            p.models = [body_rec]
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
        rim = build_wheel(car, ph, lib, bank, wskel, s.material_overrides, log, 'F')
        if rim:
            rear = None
            if ph.tyres['R'] != ph.tyres['F']:      # different rear size: its own real-size model
                rear = build_wheel(car, ph, lib, bank, wskel, s.material_overrides, log, 'R')
            for wf, recs in wheel_files.items():
                for m in recs:
                    use = rear if (rear and m.bone_name[-2:-1].upper() == 'R') else rim
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
                prof = (rear if (rear and corner[0] == 'R') else rim)[2]
                need = 0.0
                for m in recs:
                    M = np.frombuffer(m.transform, '<f4').reshape(4, 4)
                    md = Model.load(donor.zip.read(entries[_rel(m.path, donor.media)]))
                    geo = [md.geometry(bm)['pos'] for bm in md.meshes if bm[1].bucket_flags == 17]
                    P = np.concatenate([g for g in geo if g is not None])
                    L = np.c_[P, np.ones(len(P))] @ M @ np.linalg.inv(Tw)    # brake -> car -> wheel space
                    need = max(need, _brake_clearance(prof, L[:, :3]))
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
    bodykit = ('List_UpgradeCarBodyFrontBumper', 'List_UpgradeCarBodyRearBumper', 'List_UpgradeCarBodySideSkirt',
               'List_UpgradeCarBodyHood', 'List_UpgradeRearWing')
    for t in bodykit:
        if t in doc['tables']:
            doc['tables'][t]['insert'] = [r for r in doc['tables'][t]['insert'] if r.get('IsStock')]
    for r in doc['tables'].get('Data_CarBody', {}).get('insert', []):   # physics track = where the wheels now are
        wl = {k: donor.wheels[k][0] + shifts[k] for k in WHEELS}
        r['ModelFrontTrackOuter'] = round(float(wl['RF'] - wl['LF']), 4)
        r['ModelRearTrackOuter'] = round(float(wl['RR'] - wl['LR']), 4)
    stats = physics.predict_stats(doc, s.gamedb, log)
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
