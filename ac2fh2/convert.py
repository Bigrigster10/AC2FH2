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

from . import accar, database, swatchbin
from .bundle import Bundle, Blob, Meta
from .carbin import AOMap, CarScene
from .library import KINDS, DIFFUSE_PARAMS, ALPHA_PARAMS, TemplateLibrary, classify, car_relative
from .material import MaterialInstance, T_TEX, name_hash
from .modelbin import ModelStats
from .writer import MeshIn, build_model

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

    def wheel_layout(self):
        w = self.wheels
        return dict(wb=w['LF'][2] - w['LR'][2], tf=w['RF'][0] - w['LF'][0], tr=w['RR'][0] - w['LR'][0],
                    zf=w['LF'][2], zr=w['LR'][2], y=np.mean([v[1] for v in w.values()]))


def ac_wheel_centres(car):
    c = {}
    for k in WHEELS:
        ms = [m for m in car.meshes if f'/WHEEL_{k}' in m.path]
        if ms:
            p = np.concatenate([m.pos for m in ms]); c[k] = (p.min(0) + p.max(0)) / 2
    if len(c) < 4:
        raise ValueError('the car needs WHEEL_LF/RF/LR/RR nodes')
    return c


def choose_model_donor(car, donors):
    ac = ac_wheel_centres(car)
    wb, tf, tr = ac['LF'][2] - ac['LR'][2], ac['RF'][0] - ac['LF'][0], ac['RR'][0] - ac['LR'][0]
    def cost(d):
        l = d.wheel_layout()
        return abs(np.log(l['wb'] / wb)) + abs(np.log(l['tf'] / tf)) + abs(np.log(l['tr'] / tr))
    return min(donors, key=cost)


def body_fit(car, donor):
    """per-axis scale + offset that puts the AC wheel centres on the donor's"""
    ac, l = ac_wheel_centres(car), donor.wheel_layout()
    acwb = ac['LF'][2] - ac['LR'][2]
    sz = l['wb'] / acwb
    sx = (l['tf'] + l['tr']) / ((ac['RF'][0] - ac['LF'][0]) + (ac['RR'][0] - ac['LR'][0]))
    scale = np.array([sx, 1.0, sz], np.float32)
    acy = np.mean([v[1] for v in ac.values()])
    offset = np.array([0.0, l['y'] - acy, l['zf'] - ac['LF'][2] * sz], np.float32)
    return scale, offset


# --------------------------------------------------------------- physics
def ac_ini(text):
    out, sec = {}, None
    for line in text.splitlines():
        line = line.split(';')[0].strip()
        if line.startswith('['): sec = line.strip('[]'); out[sec] = {}
        elif '=' in line and sec: k, v = line.split('=', 1); out[sec][k.strip()] = v.strip()
    return out


def choose_physics_donor(db_path, brand, year, drive, mass_kg):
    import sqlite3
    db = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    rows = db.execute('select Id, MediaName, Year, DriveTypeID, CurbWeight from Data_Car where IsDrivable=1').fetchall()
    prefix = (brand or '')[:3].upper()
    def cost(r):
        c = 0 if r[1].upper().startswith(prefix + '_') else 5
        c += abs((r[2] or year) - year) / 10 + (0 if r[3] == drive else 3)
        c += abs(r[4] * 100 - mass_kg) / 500 if mass_kg else 0
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
    ini = ac_ini(car.data.get('car.ini', ''))
    drv = ac_ini(car.data.get('drivetrain.ini', '')).get('TRACTION', {}).get('TYPE', 'RWD').upper()
    drive = {'FWD': 1, 'RWD': 2, 'AWD': 3, 'AWD2': 3}.get(drv, 2)
    mass = float(ini.get('BASIC', {}).get('TOTALMASS', '0') or 0) - 75
    model_name = s.name or re.sub(rf'^{re.escape(brand)}\s*', '', ui.get('name', car.car_id)).strip() or car.car_id
    short = s.short or f"{brand} {model_name.split()[0]} '{year % 100:02d}"
    skin = s.skin or (sorted(car.skins)[0] if car.skins else None)
    log(f'  {len(car.meshes)} meshes, {len(car.materials)} materials, skin: {skin}')

    # donors
    donors = [ModelDonor(p) for p in s.fh2_cars]
    donor = next((d for d in donors if d.media.lower() == (s.model_donor or '').lower()), None) \
        or choose_model_donor(car, donors)
    if s.physics_donor:
        phys_id = s.physics_donor
    else:
        r = choose_physics_donor(s.gamedb, brand, year, drive, mass)
        phys_id = r[0]
        log(f'  physics donor: {r[1]} (car {r[0]})')
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
    gloss = bank.flat('gloss_neutral', (90, 90, 90))
    black = bank.flat('black', (0, 0, 0))

    keep = [m for m in car.meshes if not re.search(r'/(WHEEL|SUSP|HUB)_(LF|RF|LR|RR)\b', m.path)]
    groups = {}
    for m in keep:
        kind = s.material_overrides.get(m.material.name) or classify(m.material).kind
        if kind == 'skip':
            continue
        groups.setdefault(m.material.name, (kind, m.material, []))[2].append(m)

    mats, indices, meshes_in, tref = [], [], [], set()
    for i, (mname, (kind, acmat, ms)) in enumerate(sorted(groups.items(), key=lambda kv: kv[0].lower())):
        mi = lib.instance(kind)
        mi.name = mname
        diff = bank.diffuse(acmat.textures.get('txDiffuse')) if kind.startswith('textured') else None
        alpha = bank.alpha(acmat.textures.get('txDiffuse')) if kind == 'textured_alpha' else None
        for p in mi.params:
            if p.type != T_TEX:
                continue
            if p.hash in DIFFUSE_PARAMS and diff:
                p.value = diff
            elif p.hash in ALPHA_PARAMS and alpha:
                p.value = alpha
            elif car_relative(p.value[0]):
                p.value = gloss if 'glos' in p.value[0].lower() else black
            tref.add(p.value[1])
        mats.append(mi.to_blob())
        indices.append((mname, KINDS[kind][1]))
        for m in ms:
            meshes_in.append(MeshIn(f'{m.name}_LODS0', i, m.pos * scale + offset, m.nrm, m.uv, m.tan, m.tris,
                                    transparent=kind in ('glass', 'textured_alpha')))
    log(f'  {len(meshes_in)} meshes in {len(mats)} materials, {len(bank.files)} textures')

    # car scene: keep body record (re-targeted), brakes + wheels; upgrade parts -> empty model
    sc = donor.scene
    body_rec = next((m for p in sc.parts if p.part_type == 2 for m in p.models
                     if m.path.lower().endswith('\\platform\\body_a.modelbin')), None) \
        or max((m for p in sc.parts if p.part_type == 2 for m in p.models), key=lambda m: dict(m.material_indices).get('carpaint', 0))
    body_entry = _rel(body_rec.path, donor.media)
    entries = {n.lower(): n for n in donor.zip.namelist()}
    skel = Bundle.parse(donor.zip.read(entries[body_entry])).first('Skel')
    body_data = build_model(meshes_in, mats, skel, tref)
    empty_mi = lib.instance('black_plastic'); empty_mi.name = 'empty'
    empty = build_model([MeshIn('empty_LODS0', 0, np.zeros((3, 3), np.float32), np.tile([0, 1, 0], (3, 1)).astype(np.float32),
                                np.zeros((3, 2), np.float32), np.tile([1, 0, 0, 1], (3, 1)).astype(np.float32),
                                np.array([[0, 1, 2]]))], [empty_mi.to_blob()], skel, shadows=False)
    old, new = donor.media, media
    gpath = lambda rel: f'game:\\media\\cars\\{new.lower()}\\' + rel.replace('/', '\\')
    body_rec.path = gpath(body_entry)
    body_rec.material_indices = indices
    body_rec.ao_maps = [AOMap.none()]
    for p in sc.parts:
        if p.part_type == 2:
            p.models = [body_rec]
    for g in sc.groups:
        for m in g.models:
            m.path = gpath('scene/exterior/ac_empty.modelbin'); m.ao_maps = [AOMap.none()]
            m.material_indices = [('empty', 0)]
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
        elif low == donor.carbin_name.lower():
            data = sc.build()
        elif low.endswith('.modelbin'):
            data = _retarget_modelbin(data, old, new)
        elif not low.endswith('.swatchbin'):
            data = _retarget_bytes(data, old, new)
        files[nn] = data
    files['scene/exterior/ac_empty.modelbin'] = empty
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

    nc = database.NewCar(s.car_id, new, model_name, short, year, donor.car_id, phys_id)
    doc, strings = database.build(s.gamedb, nc)
    n, bad = database.verify(s.gamedb, doc)
    log(f'  database: {n} rows, checked against your gamedb ({"OK" if not bad else bad})')
    json.dump(doc, open(os.path.join(root, 'database.json'), 'w'), indent=2)
    json.dump(strings, open(os.path.join(root, 'strings.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump({'format_version': 1, 'id': mod_id, 'name': f'{year} {brand} {model_name}',
               'version': '0.1.0', 'author': f'ac2fh2 (original: {ui.get("author", "unknown")})',
               'description': f'Converted from Assetto Corsa ({car.car_id}).',
               'database': 'database.json', 'media': 'media', 'dependencies': []},
              open(os.path.join(root, 'mod.json'), 'w'), indent=2)
    prev = car.skins.get(skin, {}).get('preview.jpg') if skin else None
    src = Image.open(prev) if prev else Image.new('RGB', (816, 459), (40, 40, 40))
    tdir = os.path.join(root, 'media', 'stripped', 'RC0'); os.makedirs(tdir, exist_ok=True)
    for suf, (w, h) in (('Big', (816, 459)), ('Small', (192, 108))):
        open(os.path.join(tdir, f'Thumbnail_{s.car_id}_{suf}.swatchbin'), 'wb').write(_thumb_swatch(_thumb(src, w, h), w, h))
    log(f'Done: {root}')
    return root
