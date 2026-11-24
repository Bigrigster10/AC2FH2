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
    # each car-file record lists its model's materials by name: they must match the model we wrote
    from .material import MaterialInstance
    for rec in sc.all_models():
        e = entry(rec.path)
        if not e or not rec.material_indices:
            continue
        mm = Model.load(z.read(e))
        have = sorted(MaterialInstance.from_blob(b).name.lower() for b in mm.materials.values() if b.data)
        want = sorted(n.lower() for n, _ in rec.material_indices)
        if have != want:
            problems.append(f'{e}: car-file material list {want[:4]}... does not match the model {have[:4]}...')
    tex_ok = tex_missing = models = meshes = 0
    for path in {m.path for m in sc.all_models()} | {sc.skeleton}:
        e = entry(path)
        if not e:
            if path.lower().startswith(prefix):
                problems.append(f'model missing from zip: {path}')
            continue
        m = Model.load(z.read(e)); models += 1
        from .writer import layout_flags
        for li, lay in m.layouts.items():
            if lay.flags != layout_flags(lay.names, lay.elems):
                problems.append(f'{e}: vertex layout {li} flags {lay.flags} (should be {layout_flags(lay.names, lay.elems)})')
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
    # wheels: the game sizes rims through morph targets; ours must follow the stock unit convention exactly
    wheel_paths = {m.path for p in sc.parts if p.part_type == 43 for m in p.models}
    for path in wheel_paths:
        e = entry(path)
        if not e:
            continue
        problems += [f'{e}: {x}' for x in check_wheel(Model.load(z.read(e)))]
    if wheel_paths:
        notes.append(f'{len(wheel_paths)} wheel model(s) follow the stock rim convention (unit lip 0.14, '
                     f'radius morph x1.318 on weight 0, width morph on weight 1, rows in range)')
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


def check_wheel(m, lip=0.14, radial=1.318):
    """stock wheel convention (see convert.py): unit rim, radius morph on weight 0, width morph on weight 1, and every
    mesh's morph rows inside the MBuf (row = index + base_vertex + attribute stream start)."""
    import numpy as np
    from .modelbin import Buffer
    out = []
    mbufs = {b.id: Buffer.parse(b.data) for b in m.bundle.blobs if b.tag == 'MBuf'}
    if not mbufs:
        return ['no morph data (the game cannot size this rim)']
    rmax = 0.0
    for bm in m.meshes:
        blob, me = bm
        name = bytes(blob.meta('Name') or b'').decode('latin1')
        if me.morph_targets <= 0:
            out.append(f'{name}: no morph targets'); continue
        mb = mbufs.get(me.morph_buf)
        if mb is None:
            out.append(f'{name}: morph buffer {me.morph_buf} missing'); continue
        ib = m.ibufs[me.ib_index]
        idx = np.frombuffer(ib.data, '<u4' if ib.stride == 4 else '<u2')[me.ib_draw_offset:me.ib_draw_offset + me.index_count]
        attr = next((v for v in me.vbs if v[1] == 1), None)
        rows = idx.astype(np.int64) + me.base_vertex + (attr[3] // attr[2] if attr else 0)
        if rows.min() < 0 or rows.max() >= mb.count:
            out.append(f'{name}: morph rows {rows.min()}..{rows.max()} outside the buffer (0..{mb.count - 1})'); continue
        if me.bucket_flags == 9:
            continue
        g = m.geometry(bm); r = m.morph_rows(bm); nt = me.morph_targets
        if r is None or r.shape[1] != 2 * nt:
            out.append(f'{name}: morph rows have the wrong size'); continue
        pos = g['pos']; rad = np.hypot(pos[:, 1], pos[:, 2]); rmax = max(rmax, float(rad.max()))
        wi = r[:, :, 3]
        if not np.isin(wi, (0, 1)).all():
            out.append(f'{name}: morph weight index outside 0/1'); continue
        for k in range(nt):
            d = r[:, k, :3]; w = int(wi[0, k])
            if w == 0:          # radius: 1.318 * (0, y, z)
                err = np.abs(d - np.c_[np.zeros(len(pos)), radial * pos[:, 1:3]]).max()
                tol = 0.016 if any(f'LOD{k}' in name for k in range(2, 6)) else 2e-3   # stock decimated LODs
                if err > tol:
                    out.append(f'{name}: radius morph off by {err:.4f} (expected 1.318 * (0, y, z))')
            else:               # width: x only, never outward
                if np.abs(d[:, 1:]).max() > 1e-4 or d[:, 0].max() > 1e-4:
                    out.append(f'{name}: width morph must only move vertices inward in x')
        if not (wi[:, :nt] == 0).any(axis=1).all():
            out.append(f'{name}: some vertices have no radius morph (they would not scale with the rim)')
    if abs(rmax - lip) > 0.003:
        out.append(f'lip radius {rmax:.4f} (stock rims: {lip})')
    return out
