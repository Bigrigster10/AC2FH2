"""Re-open a finished mod folder and check it the way the game will read it."""
import json
import os
import struct
import zipfile

from .bundle import Bundle
from .carbin import CarScene
from .material import MaterialInstance, T_TEX, name_hash
from .modelbin import Model


def validate(mod_root, gamedb=None, log=print):
    problems, notes = [], []
    mj = json.load(open(os.path.join(mod_root, 'mod.json')))
    db = json.load(open(os.path.join(mod_root, 'database.json')))
    car = db['tables']['Data_Car']['insert'][0]
    media = car['MediaName']
    zp = os.path.join(mod_root, 'media', 'cars', f'{media}.zip')
    if not os.path.exists(zp):
        return [f'car zip missing: {zp}'], notes
    z = zipfile.ZipFile(zp)
    names = {n.lower(): n for n in z.namelist()}
    prefix = f'game:\\media\\cars\\{media.lower()}\\'

    def entry(path):
        p = path.lower()
        return names.get(p[len(prefix):].replace('\\', '/')) if p.startswith(prefix) else None

    cb = [n for n in names if n.endswith('.carbin')]
    sc = CarScene.parse(z.read(names[cb[0]]))
    if sc.ordinal != car['Id']: problems.append(f'carbin ordinal {sc.ordinal} != car id {car["Id"]}')
    if sc.media_name != media.lower(): problems.append(f'carbin media {sc.media_name} != {media.lower()}')
    up_db = {r['Id'] for t, s in db['tables'].items() if t.startswith('List_Upgrade') for r in s['insert']}
    for g in sc.groups:
        for u in g.upgrades:
            if u.id not in up_db and u.id // 1000 == car['Id']:
                problems.append(f'carbin upgrade {u.id} (part {g.part_type}) has no database row')
    tex_ok = tex_missing = models = meshes = 0
    for path in {m.path for m in sc.all_models()} | {sc.skeleton}:
        e = entry(path)
        if not e:
            if path.lower().startswith(prefix):
                problems.append(f'model missing from zip: {path}')
            continue
        m = Model.load(z.read(e)); models += 1
        for bm in m.meshes:
            g = m.geometry(bm); meshes += 1
            if g['pos'] is not None and len(g['tris']) and g['tris'].max() >= len(g['pos']):
                problems.append(f'{e}: bad indices in {g["name"]}')
        for blob in m.materials.values():
            if not blob.data: continue
            for p in MaterialInstance.from_blob(blob).params:
                if p.type != T_TEX: continue
                if p.value[1] != name_hash(p.value[0].lower()):
                    problems.append(f'{e}: texture hash mismatch {p.value[0]}')
                if p.value[0].lower().startswith(prefix):
                    if entry(p.value[0]): tex_ok += 1
                    else: tex_missing += 1; problems.append(f'{e}: texture not in zip {p.value[0]}')
    for n in names:
        if n.endswith('.swatchbin'):
            b = Bundle.parse(z.read(names[n])).first('TXCB')
            if b is None or b.meta('TXCH') is None: problems.append(f'bad texture {n}')
    th = os.path.join(mod_root, 'media', 'stripped', 'RC0', f'Thumbnail_{car["Id"]}_Big.swatchbin')
    if not os.path.exists(th): problems.append('thumbnail missing')
    if gamedb:
        from .database import verify
        n, bad = verify(gamedb, db)
        if bad: problems.append(f'database foreign key problems: {bad[:5]}')
        notes.append(f'database: {n} rows insert cleanly into the game database')
    notes.append(f'{models} models / {meshes} meshes load, {tex_ok} car textures resolve, '
                 f'{len([n for n in names if n.endswith(".swatchbin")])} textures, {len(names)} files')
    for x in notes: log('  ok: ' + x)
    for x in problems: log('  PROBLEM: ' + x)
    return problems, notes
