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


def _handedness(pos, nrm, uv, tan, tris):
    """tangent w (+1/-1) from UV winding, accumulated per vertex"""
    p0, p1, p2 = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    u0, u1, u2 = uv[tris[:, 0]], uv[tris[:, 1]], uv[tris[:, 2]]
    e1, e2, d1, d2 = p1 - p0, p2 - p0, u1 - u0, u2 - u0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-12, 1e-12, r)
    bit = (e2 * d1[:, :1] - e1 * d2[:, :1]) / r[:, None]
    acc = np.zeros_like(pos)
    for k in range(3):
        np.add.at(acc, tris[:, k], bit)
    w = np.sign((np.cross(nrm, tan) * acc).sum(1))
    return np.where(w == 0, 1, w).astype(np.float32)


def _mirror_x(m):
    """Assetto Corsa: +X = car's left.  FH2: -X = car's left.  Mirror and keep faces/tangents valid."""
    for a in (m.pos, m.nrm, m.tan):
        a[:, 0] *= -1
    m.tan[:, 3] *= -1
    m.tris = m.tris[:, ::-1].copy()


def load(folder, kn5_name=None, fh2_axes=True):
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
        pos3, uv = pos[:, :3].astype(np.float32), v[:, 6:8].copy()
        nrm, tan3 = nrm.astype(np.float32), tan3.astype(np.float32)
        w = _handedness(pos3, nrm, uv, tan3, tris)
        m = ACMesh(node.name, p, mats[node.material], pos3, nrm, uv, np.c_[tan3, w].astype(np.float32), tris)
        if fh2_axes:
            _mirror_x(m)
        meshes.append(m)
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
