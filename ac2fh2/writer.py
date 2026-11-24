"""Build FH2 .modelbin files from plain mesh data.

Output layout follows the game's own single-LOD car models:
  Skel (copied from the donor part), MatI xN, Mesh xM (+ shadow copies), IndB (id 0, u32),
  VLay 0 (full: POSITION / NORMAL / TEXCOORD0-3 / TANGENT0-1), VLay -1 (POSITION only, shadows),
  VerB -1 (shared R32G32B32 positions), VerB 0 (attributes, 40 B/vertex), Modl.
"""
import struct
from dataclasses import dataclass, field

import numpy as np

from .bundle import Blob, Bundle, Meta
from .modelbin import Buffer, Layout, Mesh, ModelStats

# DXGI
R32G32B32_FLOAT, R16G16B16A16_FLOAT, R16G16_UNORM, R32_UINT = 6, 10, 35, 42
# 'packed' storage formats the game stores alongside (observed in FH2 layouts)
PACKED = {R32G32B32_FLOAT: 3, R16G16B16A16_FLOAT: 9, R16G16_UNORM: 46}

BUCKET_OPAQUE, BUCKET_DECAL, BUCKET_TRANSPARENT, BUCKET_SHADOW, BUCKET_NOT_SHADOW, BUCKET_A2C = 1, 2, 4, 8, 16, 32


@dataclass
class MeshIn:
    name: str
    material: int                      # index into the materials list
    pos: np.ndarray                    # (N,3) float32, metres, FH2 axes (x right, y up, z forward)
    nrm: np.ndarray                    # (N,3)
    uv: np.ndarray                     # (N,2) any range (tiling allowed)
    tan: np.ndarray                    # (N,4) xyz + handedness
    tris: np.ndarray                   # (T,3) int, local vertex indices
    transparent: bool = False
    cast_shadow: bool = True
    bone: int = 0
    uv1: np.ndarray = None             # TEXCOORD1 = pattern channel (special paints / flakes); default = uv
    lod_flags: int = None              # default: every detail level (ALL_LODS)
    bucket: int = None                 # default: opaque 17 / transparent 60
    morph: np.ndarray = None           # (N, 4, 4) morph rows [pos A, pos B, normal A, normal B], .w = weight index


# stock car paint: TEXCOORD1 is an even-density unwrap at ~2.2 repeats per metre (measured on stock bodies)
PATTERN_DENSITY = 2.2


def pattern_uv(pos, nrm, uv, tan, tris, density=PATTERN_DENSITY):
    """box projection for TEXCOORD1: each triangle uses the axis its face points along; vertices are split
    where neighbouring triangles use different axes. -> pos, nrm, uv, tan, tris, uv1"""
    P = pos[tris]
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    axis = np.argmax(np.abs(fn), 1)                              # 0 side, 1 top/bottom, 2 front/back
    key = tris * 3 + axis[:, None]
    uniq, inv = np.unique(key.ravel(), return_inverse=True)
    src, ax = uniq // 3, uniq % 3
    p = pos[src]
    # patterns run along the car: sides (z, y), top (z, x), ends (x, y)
    u = np.where(ax == 2, p[:, 0], p[:, 2])
    v = np.where(ax == 1, p[:, 0], p[:, 1])
    uv1 = (np.stack([u, -v], 1) * density).astype(np.float32)
    return pos[src], nrm[src], uv[src], tan[src], inv.reshape(-1, 3), uv1


def _id(i):
    return Meta('Id  ', struct.pack('<i', i))


def _bbox(p):
    return struct.pack('<6f', *p.min(0), *p.max(0))


def _uv_transform(uv):
    """quantise UVs to unorm16 with a per-mesh (offset, scale) per axis"""
    lo, hi = uv.min(0), uv.max(0)
    scale = np.where(hi - lo > 1e-8, hi - lo, 1.0)
    q = np.clip(np.round((uv - lo) / scale * 65535), 0, 65535).astype('<u2')
    return q, (float(lo[0]), float(scale[0]), float(lo[1]), float(scale[1]))


def _acmr(tris, cache=32):
    """average cache miss ratio (FIFO), like the game stores per mesh"""
    fifo, miss = [], 0
    for v in tris.ravel():
        if v not in fifo:
            miss += 1; fifo.append(v)
            if len(fifo) > cache: fifo.pop(0)
    return miss / max(len(tris), 1)


# Stock models carry 6 detail levels (LODS0/LOD0 = bits 0-1, LOD1..5 = bits 2..6; Modl lod_min 5, flags 127) and the
# game picks one by distance/camera -- wheels use LOD1+ while driving. We have one mesh set, so it is valid at all of them.
ALL_LODS = 0x7F


def layout_flags(names, elems):
    """VLay flags = which optional streams exist (verified on every stock model):
    TEXCOORDn -> 1 << n, TANGENTn -> 1 << (n + 5). With 0 the shaders treat the mesh as having no UVs/tangents."""
    f = 0
    for e in elems:
        if names[e[0]] == 'TEXCOORD': f |= 1 << e[1]
        elif names[e[0]] == 'TANGENT': f |= 1 << (e[1] + 5)
    return f


def build_layouts():
    names = ['POSITION', 'NORMAL', 'TEXCOORD', 'TANGENT']
    el = [(0, 0, 0, 0, R32G32B32_FLOAT, -1, 0),
          (1, 0, 1, 0, R16G16B16A16_FLOAT, -1, 0)]
    el += [(2, i, 1, 0, R16G16_UNORM, -1, 0) for i in range(4)]
    el += [(3, i, 1, 0, R16G16B16A16_FLOAT, -1, 0) for i in range(2)]
    full = Layout(names, el, [PACKED[e[4]] for e in el], layout_flags(names, el))
    pos_only = Layout(['POSITION'], [el[0]], [PACKED[R32G32B32_FLOAT]], 0)
    return full, pos_only


ATTR_STRIDE = 8 + 4 * 4 + 8 * 2      # normal + 4 uv + 2 tangents = 40


# morph rows of a vertex without deltas: stock rim pattern (A = width, weight index 1; B = radius, weight index 0)
ZERO_MORPH = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], np.float32)


def build_model(meshes, materials, skel_blob: Blob, tref_hashes=(), shadows=True, morphs=False) -> bytes:
    """meshes: [MeshIn]; materials: [Blob (MatI)]; skel_blob copied from the donor part.
    morphs: every mesh gets 2 morph targets like stock wheels ('Mrph' weight[0]/weight[1] + an MBuf with 4 half4 rows
    per vertex: MeshIn.morph, or zero deltas). The morphing vertex shaders find a vertex's rows from SV_VertexID plus
    a per-draw offset the game derives from the mesh, so morph models use the simplest layout that every variant
    agrees on: base_vertex 0, stream offsets 0, absolute indices -> MBuf row = index = vertex number."""
    positions, attrs, indices = [], [], []
    mesh_blobs = []
    vstart = istart = 0
    full, pos_only = build_layouts()
    for mi in meshes:
        n = len(mi.pos)
        tris = np.asarray(mi.tris, np.int64)
        assert tris.max() < n, f'{mi.name}: index out of range'
        q, uvt = _uv_transform(np.asarray(mi.uv, np.float32))
        q1, uvt1 = _uv_transform(np.asarray(mi.uv1, np.float32)) if mi.uv1 is not None else (q, uvt)
        nrm = np.zeros((n, 4), np.float16); nrm[:, :3] = mi.nrm
        tan = np.asarray(mi.tan, np.float16)
        rec = np.zeros(n, dtype=[('n', '<f2', 4), ('uv', '<u2', (4, 2)), ('t0', '<f2', 4), ('t1', '<f2', 4)])
        rec['n'] = nrm; rec['uv'] = q[:, None, :]; rec['uv'][:, 1] = q1; rec['t0'] = tan; rec['t1'] = tan
        assert rec.dtype.itemsize == ATTR_STRIDE
        positions.append(np.asarray(mi.pos, '<f4')); attrs.append(rec.tobytes())
        indices.append((tris + vstart).astype('<u4').ravel())
        uv_t = struct.pack('<4f', *uvt) + struct.pack('<4f', *uvt1) + struct.pack('<4f', *uvt) * 2 + \
            struct.pack('<4f', 0, 1, 0, 1)
        bucket = (BUCKET_TRANSPARENT | BUCKET_SHADOW | BUCKET_NOT_SHADOW | BUCKET_A2C) if mi.transparent \
            else (BUCKET_OPAQUE | BUCKET_NOT_SHADOW)
        if mi.bucket is not None:
            bucket = mi.bucket
        common = dict(bone=mi.bone, lod_flags=mi.lod_flags if mi.lod_flags is not None else ALL_LODS, lod_min=0, lod_max=255, bucket_order=0, skin_elems=0,
                      morph_targets=2 if morphs else 0, morph_damage=0, idx32=1, topology=4, ib_index=0, ib_offset=0,
                      ib_draw_offset=istart, base_vertex=0 if morphs else -vstart, index_count=len(tris) * 3,
                      prim_count=len(tris), acmr=_acmr(tris), ref_vertex_count=n, morph_buf=0 if morphs else -1,
                      skin_buf=-1, cbs=[], source_mesh=0)
        off = 0 if morphs else vstart      # morph models: absolute indices into streams that start at row 0
        me = Mesh(material=mi.material, bucket_flags=bucket, layout=0,
                  vbs=[(-1, 0, 12, off * 12), (0, 1, ATTR_STRIDE, off * ATTR_STRIDE)],
                  uv_transforms=uv_t, **common)
        meta = [Meta('Name', mi.name.encode('latin1')), Meta('BBox', _bbox(mi.pos))]
        mesh_blobs.append(Blob('Mesh', 1, 7, me.build(), meta))
        if shadows and mi.cast_shadow and not mi.transparent:
            sh = Mesh(material=mi.material, bucket_flags=BUCKET_OPAQUE | BUCKET_SHADOW, layout=1,
                      vbs=[(-1, 0, 12, off * 12)], uv_transforms=struct.pack('<4f', 0, 1, 0, 1) * 5, **common)
            mesh_blobs.append(Blob('Mesh', 1, 7, sh.build(), [Meta('Name', (mi.name + '_Shadow').encode('latin1')),
                                                             Meta('BBox', _bbox(mi.pos))]))
        vstart += n; istart += len(tris) * 3

    allpos = np.concatenate(positions)
    idx = np.concatenate(indices)
    blobs = [skel_blob] + list(materials) + mesh_blobs
    blobs.append(Blob('IndB', 1, 0, Buffer(len(idx), 4, 1, R32_UINT, idx.tobytes()).build(), [_id(0)]))
    blobs.append(Blob('VLay', 1, 1, full.build(), [_id(0)]))
    blobs.append(Blob('VLay', 1, 1, pos_only.build(), [_id(-1)]))
    blobs.append(Blob('VerB', 1, 0, Buffer(len(allpos), 12, 1, R32G32B32_FLOAT, allpos.tobytes()).build(), [_id(-1)]))
    blobs.append(Blob('VerB', 1, 0, Buffer(vstart, ATTR_STRIDE, len(full.elems) - 1, R16G16B16A16_FLOAT,
                                           b''.join(attrs)).build(), [_id(0)]))
    if morphs:   # per vertex: [pos delta A, pos delta B, normal delta A, normal delta B] as half4, .w = weight index
        d = np.concatenate([np.asarray(mi.morph, np.float32).reshape(len(mi.pos), 4, 4) if mi.morph is not None
                            else np.tile(ZERO_MORPH, (len(mi.pos), 1, 1)) for mi in meshes]).astype(np.float16)
        assert len(d) == vstart
        blobs.append(Blob('MBuf', 1, 0, Buffer(vstart, 32, 4, R16G16B16A16_FLOAT, d.tobytes()).build(), [_id(0)]))
        mrph = struct.pack('<H', 2) + b''.join(struct.pack('<i', 9) + f'weight[{i}]'.encode() for i in range(2))
        blobs.insert(1, Blob('Mrph', 0, 0, mrph, []))
    stats = ModelStats(meshes=len(mesh_blobs), buffers=4 if morphs else 3, layouts=2, materials=len(materials),
                       has_lod=1, lod_min=5, lod_max=-1, lod_flags=ALL_LODS)
    tref = sorted(set(tref_hashes))
    mmeta = [Meta('BBox', _bbox(allpos))]
    if tref:
        mmeta.append(Meta('TRef', struct.pack(f'<I{len(tref)}I', len(tref), *tref)))
    blobs.append(Blob('Modl', 1, 1, stats.build(), mmeta))
    # MatI blobs need ids 0..n-1 (meshes reference them by index)
    for i, mb in enumerate(materials):
        mb.metas = [m for m in mb.metas if m.tag != 'Id  '] + [_id(i)]
    return Bundle(1, 0, blobs).build()
