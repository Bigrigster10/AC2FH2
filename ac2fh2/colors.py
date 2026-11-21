"""Factory paint list: ManufacturerColors.bin (the in-game "Manufacturer colors" tab) + matching Combo_Colors rows.

ManufacturerColors.bin = Grub bundle with one 'MNCL' 1.0 blob:
    u8 entry count; per entry: u8 colour count (1, or 2 for two-tone); per colour:
    u16 flags (29 = paint, 2 = second tone), f32 r, g, b (swatch), 7-bit-length string: library .materialbin path.
Verified to parse exactly on every stock car given. The material paths must exist in the game, so each AC skin is
mapped to the closest paint from the game's shared colour library (harvested from the user's car zips).
"""
import re
import sqlite3
import struct
import zipfile

import numpy as np
from PIL import Image

from .bundle import Blob, Bundle

# colour words in skin names win over the measured texture (AC tints some skins in the shader)
WORDS = {'black': (0.02, 0.02, 0.02), 'graphite': (0.13, 0.13, 0.14), 'grey': (0.42, 0.43, 0.45),
         'gray': (0.42, 0.43, 0.45), 'silver': (0.72, 0.73, 0.74), 'white': (0.93, 0.93, 0.92),
         'red': (0.62, 0.03, 0.04), 'blue': (0.10, 0.25, 0.65), 'yellow': (0.95, 0.80, 0.10),
         'green': (0.10, 0.40, 0.15), 'orange': (0.95, 0.40, 0.05), 'gold': (0.75, 0.60, 0.25),
         'purple': (0.35, 0.12, 0.45), 'brown': (0.35, 0.22, 0.12)}


def _str7(d, p):
    n = sh = 0
    while True:
        b = d[p]; p += 1; n |= (b & 0x7F) << sh; sh += 7
        if b < 0x80: break
    return d[p:p + n].decode('latin1'), p + n


def _w7(s):
    b, n = s.encode('latin1'), len(s)
    out = bytearray()
    while True:
        out.append((n & 0x7F) | (0x80 if n > 0x7F else 0)); n >>= 7
        if not n: break
    return bytes(out) + b


def parse(data):
    d = bytes(Bundle.parse(data).first('MNCL').data)
    p = 1; out = []
    for _ in range(d[0]):
        k = d[p]; p += 1; cols = []
        for _ in range(k):
            fl, r, g, b = struct.unpack_from('<H3f', d, p); p += 14
            path, p = _str7(d, p)
            cols.append((fl, (r, g, b), path))
        out.append(cols)
    assert p == len(d), 'ManufacturerColors: trailing bytes'
    return out


def build(entries):
    d = bytearray([len(entries)])
    for cols in entries:
        d.append(len(cols))
        for fl, (r, g, b), path in cols:
            d += struct.pack('<H3f', fl, r, g, b) + _w7(path)
    return Bundle(1, 0, [Blob("MNCL", 1, 0, bytes(d), [])]).build()


def harvest(zips):
    """shared-library paints from the user's cars: path -> (flags, rgb)"""
    pal = {}
    for z in zips:
        try:
            zf = zipfile.ZipFile(z)
            n = next((n for n in zf.namelist() if n.lower() == 'manufacturercolors.bin'), None)
            if not n: continue
            for cols in parse(zf.read(n)):
                fl, rgb, path = cols[0]
                if '\\_library\\' in path.lower():
                    pal.setdefault(path, (fl, rgb))
        except Exception:
            continue
    return pal


def skin_colour(name, files):
    low = name.lower()
    for w, rgb in WORDS.items():
        if re.search(rf'\b{w}', low):
            return np.array(rgb)
    f = files.get('skinbase.dds')
    if f:
        try:
            a = np.asarray(Image.open(f).convert('RGB').resize((64, 64))).reshape(-1, 3)
            return np.median(a, 0) / 255
        except Exception:
            pass
    return None


def from_skins(skins, first, palette, fallback):
    """AC skins -> list of (skin, library path, flags, rgb), nearest library paint per skin, no duplicates"""
    names = sorted(skins, key=lambda s: (s != first, s.lower()))
    paths = list(palette)
    P = np.array([palette[p][1] for p in paths]) if paths else np.zeros((0, 3))
    out, used = [], set()
    for s in names:
        c = skin_colour(s, skins[s])
        if c is None or not len(P):
            continue
        i = int(np.argmin(((P - c) ** 2).sum(1)))
        if paths[i] in used:
            continue
        used.add(paths[i])
        out.append((s, paths[i], palette[paths[i]][0], palette[paths[i]][1]))
    if not out and fallback:
        out = [(None, fallback[2], fallback[0], fallback[1])]
    return out[:20]


def combo_rows(db_path, car_id, picks):
    """Combo_Colors rows (Autoshow / default paint) matching the manufacturer list"""
    db = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    ref = db.execute('select RGB, ColorClassId from Combo_Colors').fetchall()
    db.close()
    R = np.array([[(v >> 16) & 255, (v >> 8) & 255, v & 255] for v, _ in ref], float) / 255
    rows = []
    for i, (_, path, fl, rgb) in enumerate(picks, 1):
        c = np.clip(np.array(rgb), 0, 1)
        cls = ref[int(np.argmin(((R - c) ** 2).sum(1)))][1]
        v = int(round(c[0] * 255)) << 16 | int(round(c[1] * 255)) << 8 | int(round(c[2] * 255))
        metallic = int(bool(re.search(r'met|pearl|flake|mica', path.lower())))
        rows.append({'ID': car_id * 1000 + i, 'Ordinal': car_id, 'Sequence': i, 'Metallic': metallic, 'RGB': v,
                     'BaseTexture': 'nodamage.tga', 'DamageTexture': 'damage.tga', 'ColorClassId': cls})
    return rows
