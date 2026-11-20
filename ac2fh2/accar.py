"""Load an Assetto Corsa car folder: world-space meshes, materials, textures, physics ini/lut."""
import os
from dataclasses import dataclass, field

import numpy as np

from . import kn5, acd


@dataclass
class ACMesh:
    name: str
    path: str                 # node path, e.g. 'sub_wrxsti_04_lod0_/WHEEL_LF/rim'
    material: kn5.Material
    pos: np.ndarray
    nrm: np.ndarray
    uv: np.ndarray
    tan: np.ndarray
    tris: np.ndarray
    lod_out: float = 0.0


@dataclass
class ACCar:
    folder: str
    car_id: str
    meshes: list
    textures: dict            # name -> DDS/PNG bytes
    materials: list
    data: dict                # physics files: name -> text
    skins: dict = field(default_factory=dict)   # skin -> {file: path}


def _flatten(node, parent=np.eye(4, dtype=np.float32), path=''):
    p = f'{path}/{node.name}' if path else node.name
    if node.kind == 1:
        m = np.array(node.matrix, np.float32).reshape(4, 4)   # row-vector convention (v * M)
        parent = m @ parent
    if node.kind in (2, 3) and node.verts is not None and len(node.indices):
        yield node, parent, p
    for c in node.children:
        yield from _flatten(c, parent, p)


def load(folder, kn5_name=None):
    car_id = os.path.basename(os.path.normpath(folder))
    kn5_path = os.path.join(folder, kn5_name or f'{car_id}.kn5')
    if not os.path.exists(kn5_path):
        cands = [f for f in os.listdir(folder) if f.lower().endswith('.kn5') and 'collider' not in f.lower()]
        kn5_path = os.path.join(folder, max(cands, key=lambda f: os.path.getsize(os.path.join(folder, f))))
    _ver, textures, mats, root = kn5.read(kn5_path)
    meshes = []
    for node, M, p in _flatten(root):
        v = node.verts
        pos = np.c_[v[:, 0:3], np.ones(len(v))] @ M
        R = M[:3, :3]
        nrm = v[:, 3:6] @ R; nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        tan3 = v[:, 8:11] @ R; tan3 /= np.linalg.norm(tan3, axis=1, keepdims=True) + 1e-12
        tris = node.indices.reshape(-1, 3).astype(np.int64)
        if np.linalg.det(R) < 0:
            tris = tris[:, ::-1]
        meshes.append(ACMesh(node.name, p, mats[node.material], pos[:, :3].astype(np.float32),
                             nrm.astype(np.float32), v[:, 6:8].copy(), np.c_[tan3, np.ones(len(v))].astype(np.float32),
                             tris))
    data = {}
    dfolder = os.path.join(folder, 'data')
    if os.path.isdir(dfolder):
        for f in os.listdir(dfolder):
            data[f] = open(os.path.join(dfolder, f), 'rb').read()
    elif os.path.exists(os.path.join(folder, 'data.acd')):
        data = acd.unpack(os.path.join(folder, 'data.acd'), car_id)
    data = {k: v.decode('utf-8', 'replace') for k, v in data.items()}
    skins = {}
    sdir = os.path.join(folder, 'skins')
    if os.path.isdir(sdir):
        for s in sorted(os.listdir(sdir)):
            sp = os.path.join(sdir, s)
            if os.path.isdir(sp):
                skins[s] = {f.lower(): os.path.join(sp, f) for f in os.listdir(sp)}
    return ACCar(folder, car_id, meshes, textures, mats, data, skins)
