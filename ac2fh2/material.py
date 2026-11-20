"""Material instances (MatI blob = nested Grub bundle: MATI library path + MTPR parameter overrides)."""
import struct
import zlib
from dataclasses import dataclass, field

from .bundle import Blob, Bundle, Meta

T_VECTOR, T_COLOR, T_FLOAT, T_BOOL, T_INT, T_SWIZZLE, T_TEX, T_SAMPLER, T_GRADIENT, T_RANGE, _, T_VEC2 = range(12)


def name_hash(s: str) -> int:
    return zlib.crc32(s.encode('latin1')) & 0xFFFFFFFF


def _read7(d, p):
    """.NET 7-bit length-prefixed string"""
    n = shift = 0
    while True:
        b = d[p]; p += 1
        n |= (b & 0x7F) << shift; shift += 7
        if not b & 0x80: break
    return d[p:p + n].decode('latin1'), p + n


def _write7(s: str) -> bytes:
    raw = s.encode('latin1'); n = len(raw); out = bytearray()
    while True:
        b = n & 0x7F; n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n: break
    return bytes(out) + raw


@dataclass
class Param:
    vmaj: int
    vmin: int
    hash: int
    type: int
    value: object           # tuple of floats / int / (path, path_hash) / (u, v[, filt]) / list

    @classmethod
    def read(cls, d, p):
        vmaj, vmin, h, t = struct.unpack_from('<BBIB', d, p); p += 7
        assert vmaj < 3, 'MTPR params v3+ not used by FH2'
        if t in (T_VECTOR, T_COLOR, T_SWIZZLE, T_RANGE):
            v = struct.unpack_from('<4f', d, p); p += 16
        elif t == T_FLOAT:
            v = struct.unpack_from('<f', d, p)[0]; p += 4
        elif t in (T_BOOL, T_INT):
            v = struct.unpack_from('<i', d, p)[0]; p += 4
        elif t == T_TEX:
            path, p = _read7(d, p)
            ph = None
            if vmaj >= 2:
                ph = struct.unpack_from('<I', d, p)[0]; p += 4
            v = (path, ph)
        elif t == T_SAMPLER:
            n = 3 if (vmaj >= 1 and vmin >= 1) else 2
            v = struct.unpack_from(f'<{n}i', d, p); p += 4 * n
        elif t == T_GRADIENT:
            n = struct.unpack_from('<I', d, p)[0]; p += 4
            v = [struct.unpack_from('<4f', d, p + 16 * i) for i in range(n)]; p += 16 * n
        elif t == T_VEC2:
            v = struct.unpack_from('<2f', d, p); p += 8
            if vmaj == 1: v = (v, d[p:p + 8]); p += 8
        else:
            raise ValueError(f'unknown shader parameter type {t}')
        return cls(vmaj, vmin, h, t, v), p

    def write(self) -> bytes:
        out = bytearray(struct.pack('<BBIB', self.vmaj, self.vmin, self.hash, self.type))
        t, v = self.type, self.value
        if t in (T_VECTOR, T_COLOR, T_SWIZZLE, T_RANGE): out += struct.pack('<4f', *v)
        elif t == T_FLOAT: out += struct.pack('<f', v)
        elif t in (T_BOOL, T_INT): out += struct.pack('<i', v)
        elif t == T_TEX:
            out += _write7(v[0])
            if self.vmaj >= 2: out += struct.pack('<I', v[1])
        elif t == T_SAMPLER: out += struct.pack(f'<{len(v)}i', *v)
        elif t == T_GRADIENT:
            out += struct.pack('<I', len(v))
            for c in v: out += struct.pack('<4f', *c)
        elif t == T_VEC2:
            if self.vmaj == 1: out += struct.pack('<2f', *v[0]) + v[1]
            else: out += struct.pack('<2f', *v)
        return bytes(out)


@dataclass
class MaterialInstance:
    name: str
    library_path: str                 # Game:\Media\cars\_library\materials\...\x.materialbin
    params: list = field(default_factory=list)
    trailer: bytes = bytes(12)
    blob: Blob = None                 # original, for metadata we don't model

    @classmethod
    def from_blob(cls, blob: Blob):
        if not blob.data:                      # empty placeholder slot
            return cls(blob.name or '', '', [], b'', blob)
        nb = Bundle.parse(blob.data)
        mati, mtpr = nb.first('MATI'), nb.first('MTPR')
        path, _ = _read7(mati.data, 0)
        d = mtpr.data
        count = d[0]; p = 1
        params = []
        for _ in range(count):
            prm, p = Param.read(d, p); params.append(prm)
        return cls(blob.name, path, params, d[p:], blob)

    def to_blob(self) -> Blob:
        if not self.blob.data:
            return self.blob
        nb = Bundle.parse(self.blob.data)
        mati, mtpr = nb.first('MATI'), nb.first('MTPR')
        mtpr.data = bytes([len(self.params)]) + b''.join(p.write() for p in self.params) + self.trailer
        name = self.name.encode('latin1')
        lib = self.library_path.replace('/', '\\').rsplit('\\', 1)[-1].rsplit('.', 1)[0]
        if mati.data != _write7(self.library_path):
            mati.set_meta('Name', lib.lower().encode('latin1'))   # inner name = library material name
        mati.data = _write7(self.library_path)
        out = Blob(self.blob.tag, self.blob.major, self.blob.minor, nb.build(),
                   [Meta(m.tag, m.data, m.version) for m in self.blob.metas])
        out.set_meta('Name', name)
        return out

    def textures(self):
        return [p for p in self.params if p.type == T_TEX]
