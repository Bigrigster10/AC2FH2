"""FH2 / FM5 .modelbin blobs: typed parse + serialize, and geometry decoding.

Blob versions in FH2/FM5 car models: Mesh 1.7, VLay 1.1, VerB/IndB/MBuf 1.0, Modl 1.1,
MatI 1.0 (nested Grub bundle, kept as bytes), Skel 0.0 and Mrph (kept as bytes).
Field layouts cross-checked with ForzaTech Studio (MIT) ForzaTools.Bundles/Blobs.
"""
import struct
from dataclasses import dataclass, field

import numpy as np

from .bundle import Blob, Bundle


class W:
    """tiny little-endian writer"""
    def __init__(self): self.b = bytearray()
    def __call__(self, fmt, *v): self.b += struct.pack('<' + fmt, *v); return self


class R:
    def __init__(self, d): self.d, self.p = d, 0
    def __call__(self, fmt):
        v = struct.unpack_from('<' + fmt, self.d, self.p); self.p += struct.calcsize('<' + fmt)
        return v if len(v) > 1 else v[0]
    def raw(self, n): v = self.d[self.p:self.p + n]; self.p += n; return bytes(v)
    def done(self): return self.p == len(self.d)


# ---------------------------------------------------------------- Mesh 1.7
@dataclass
class Mesh:
    material: int            # index into MatI blobs (-1 = none)
    bone: int                # rigid bone index
    lod_flags: int           # bitmask of LODs this mesh is drawn in
    lod_min: int
    lod_max: int
    bucket_flags: int        # 1 opaque, 2 decal, 4 transparent, 8 shadow, 0x10 notShadow, 0x20 a2c
    bucket_order: int
    skin_elems: int
    morph_targets: int
    morph_damage: int
    idx32: int
    topology: int            # 4 = triangle list
    ib_index: int
    ib_offset: int           # in indices
    ib_draw_offset: int
    base_vertex: int
    index_count: int
    prim_count: int
    acmr: float
    ref_vertex_count: int
    layout: int
    vbs: list                # [(index, slot, stride, offset_bytes)]
    morph_buf: int
    skin_buf: int
    cbs: list
    source_mesh: int
    uv_transforms: bytes     # 5 x float4

    @classmethod
    def parse(cls, d):
        r = R(d)
        material, bone, lod_flags, lmin, lmax, bflags, border, skin, morph, mdmg, i32, topo = \
            r('hhHBBHBBBBBH')
        ibi, ibo, ibd, bv, ic, pc = r('6i')
        acmr, rvc = r('fI')
        layout, nvb = r('ii')
        vbs = [r('iIII') for _ in range(nvb)]
        mb, sb = r('ii')
        ncb = r('i')
        cbs = list(r(f'{ncb}i')) if ncb > 1 else ([r('i')] if ncb == 1 else [])
        src = r('I')
        uvt = r.raw(80)
        assert r.done(), f'Mesh: {len(d) - r.p} trailing bytes'
        return cls(material, bone, lod_flags, lmin, lmax, bflags, border, skin, morph, mdmg, i32, topo,
                   ibi, ibo, ibd, bv, ic, pc, acmr, rvc, layout, [tuple(v) for v in vbs], mb, sb, cbs, src, uvt)

    def build(self):
        w = W()
        w('hhHBBHBBBBBH', self.material, self.bone, self.lod_flags, self.lod_min, self.lod_max, self.bucket_flags,
          self.bucket_order, self.skin_elems, self.morph_targets, self.morph_damage, self.idx32, self.topology)
        w('6i', self.ib_index, self.ib_offset, self.ib_draw_offset, self.base_vertex, self.index_count, self.prim_count)
        w('fI', self.acmr, self.ref_vertex_count)
        w('ii', self.layout, len(self.vbs))
        for v in self.vbs: w('iIII', *v)
        w('ii', self.morph_buf, self.skin_buf)
        w('i', len(self.cbs))
        for c in self.cbs: w('i', c)
        w('I', self.source_mesh)
        w.b += self.uv_transforms
        return bytes(w.b)


# ---------------------------------------------------------------- VLay 1.1
@dataclass
class Layout:
    names: list
    elems: list              # [(name_idx, sem_idx, slot, slot_class, fmt, offset, step)]
    packed: list             # packed (storage) format per element
    flags: int

    @classmethod
    def parse(cls, d):
        r = R(d)
        names = []
        for _ in range(r('H')):
            names.append(r.raw(r('i')).decode('ascii'))
        n = r('H')
        elems = [r('hhhhiii') for _ in range(n)]
        packed = [r('i') for _ in range(n)]
        flags = r('I')
        assert r.done()
        return cls(names, [tuple(e) for e in elems], packed, flags)

    def build(self):
        w = W()
        w('H', len(self.names))
        for s in self.names: w('i', len(s)); w.b += s.encode('ascii')
        w('H', len(self.elems))
        for e in self.elems: w('hhhhiii', *e)
        for p in self.packed: w('i', p)
        w('I', self.flags)
        return bytes(w.b)

    def semantic(self, e):
        return self.names[e[0]], e[1]

    def offsets(self):
        """byte offset of each element in its stream (-1 = D3D 'append aligned')"""
        cur, out = {}, []
        for e in self.elems:
            off = cur.get(e[2], 0) if e[5] == -1 else e[5]
            out.append(off)
            cur[e[2]] = off + DXGI[e[4]][0]
        return out


# ------------------------------------------------- VerB / IndB / MBuf 1.0
@dataclass
class Buffer:
    count: int
    stride: int
    sub_elems: int
    fmt: int
    data: bytes

    @classmethod
    def parse(cls, d):
        count, size, stride, sub, _pad, fmt = struct.unpack_from('<iiHBBi', d, 0)
        assert 16 + size == len(d), 'buffer size mismatch'
        return cls(count, stride, sub, fmt, bytes(d[16:16 + size]))

    def build(self):
        return struct.pack('<iiHBBi', self.count, len(self.data), self.stride, self.sub_elems, 0, self.fmt) + self.data


# ---------------------------------------------------------------- Modl 1.1
@dataclass
class ModelStats:
    meshes: int
    buffers: int
    layouts: int
    materials: int
    has_lod: int
    lod_min: int
    lod_max: int
    lod_flags: int

    @classmethod
    def parse(cls, d):
        m, b, l, mat, has, lmin, lmax, _p, lf = struct.unpack_from('<HHHHBbbBH', d, 0)
        assert len(d) == 14
        return cls(m, b, l, mat, has, lmin, lmax, lf)

    def build(self):
        return struct.pack('<HHHHBbbBH', self.meshes, self.buffers, self.layouts, self.materials,
                           self.has_lod, self.lod_min, self.lod_max, 0, self.lod_flags)


PARSERS = {'Mesh': Mesh, 'VLay': Layout, 'VerB': Buffer, 'IndB': Buffer, 'MBuf': Buffer, 'Modl': ModelStats}


# ------------------------------------------------------- vertex decoding
# DXGI format -> (bytes, numpy kind, components)
DXGI = {
    2: (16, 'f4', 4), 6: (12, 'f4', 3), 16: (8, 'f4', 2), 41: (4, 'f4', 1),
    10: (8, 'f2', 4), 34: (4, 'f2', 2), 54: (2, 'f2', 1),
    11: (8, 'u2n', 4), 13: (8, 's2n', 4), 35: (4, 'u2n', 2), 37: (4, 's2n', 2),
    28: (4, 'u1n', 4), 31: (4, 's1n', 4), 24: (4, 'u10', 4), 9: (8, 'u2', 4),
}


def decode(raw, count, stride, off, fmt):
    size, kind, comps = DXGI[fmt]
    a = np.frombuffer(raw, np.uint8)[: count * stride].reshape(count, stride)[:, off:off + size].copy()
    if kind in ('f4', 'f2'):
        return a.view('<' + kind).astype(np.float32)
    if kind == 'u2n': return a.view('<u2').astype(np.float32) / 65535
    if kind == 's2n': return np.maximum(a.view('<i2').astype(np.float32) / 32767, -1)
    if kind == 'u1n': return a.astype(np.float32) / 255
    if kind == 's1n': return np.maximum(a.view('i1').astype(np.float32) / 127, -1)
    if kind == 'u2': return a.view('<u2').astype(np.float32)
    v = a.view('<u4')[:, 0]
    return np.stack([v & 1023, (v >> 10) & 1023, (v >> 20) & 1023, v >> 30], 1).astype(np.float32) / 1023


@dataclass
class Model:
    bundle: Bundle
    meshes: list = field(default_factory=list)       # (blob, Mesh)
    layouts: dict = field(default_factory=dict)       # by list position; buffers by 'Id  ' metadata
    vbufs: dict = field(default_factory=dict)
    ibufs: dict = field(default_factory=dict)
    materials: dict = field(default_factory=dict)     # MatI blobs

    @classmethod
    def load(cls, data):
        b = Bundle.parse(data)
        m = cls(b)
        for blob in b.blobs:
            if blob.tag == 'Mesh': m.meshes.append((blob, Mesh.parse(blob.data)))
            elif blob.tag == 'VLay': m.layouts[len(m.layouts)] = Layout.parse(blob.data)  # by position
            elif blob.tag == 'VerB': m.vbufs[blob.id] = Buffer.parse(blob.data)
            elif blob.tag == 'IndB': m.ibufs[blob.id] = Buffer.parse(blob.data)
            elif blob.tag == 'MatI': m.materials[blob.id] = blob
        return m

    def geometry(self, blob_mesh):
        """-> dict(name, material, lod, pos, nrm, uv, tris) for one Mesh (vertices compacted)"""
        blob, me = blob_mesh
        lay = self.layouts[me.layout]
        ib = self.ibufs[me.ib_index]
        dt = '<u4' if ib.stride == 4 else '<u2'
        # indices start at ib_draw_offset; a stream element lives at offset + (index + base_vertex) * stride
        idx = np.frombuffer(ib.data, dt)[me.ib_draw_offset: me.ib_draw_offset + me.index_count].astype(np.int64) + me.base_vertex
        used, tris = np.unique(idx, return_inverse=True)
        out = {}
        for e, eoff in zip(lay.elems, lay.offsets()):
            fmt = e[4]
            sem, si = lay.semantic(e)
            vb = next((v for v in me.vbs if v[1] == e[2]), None)
            if vb is None or fmt not in DXGI: continue
            vi, _slot, stride, voff = vb
            raw = self.vbufs[vi].data[voff:]
            n = len(raw) // stride
            out[(sem, si)] = decode(raw, n, stride, eoff, fmt)[used]
        return dict(name=blob.name, material=me.material, lod=me.lod_flags, bucket=me.bucket_flags,
                    pos=out.get(('POSITION', 0)), nrm=out.get(('NORMAL', 0)), uv=out.get(('TEXCOORD', 0)),
                    attrs=out, tris=tris.reshape(-1, 3))
