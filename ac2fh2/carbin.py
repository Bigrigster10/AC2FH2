"""FH2 .carbin (CarScene v5) reader/writer.

A carbin lists every render model of a car: the fixed parts (body, brakes, wheels) and the
upgrade groups (wing, bumpers, skirts, roll cage) whose upgrade ids match database rows.
Strings are u32-length-prefixed; integers little endian.
"""
import struct
from dataclasses import dataclass, field


class _R:
    def __init__(self, d): self.d, self.p = d, 0
    def raw(self, n):
        if n > len(self.d) - self.p: raise ValueError(f'truncated carbin at 0x{self.p:x}')
        v = self.d[self.p:self.p + n]; self.p += n; return bytes(v)
    def u(self, n=4, signed=False): return int.from_bytes(self.raw(n), 'little', signed=signed)
    def s(self): return self.raw(self.u()).decode('utf-8')
    def arr(self, f): return [f(self) for _ in range(self.u())]


class _W:
    def __init__(self): self.b = bytearray()
    def raw(self, v): self.b += v
    def u(self, v, n=4, signed=False): self.b += int(v).to_bytes(n, 'little', signed=signed)
    def s(self, v): e = v.encode('utf-8'); self.u(len(e)); self.b += e
    def arr(self, items, f):
        self.u(len(items))
        for it in items: f(self, it)


@dataclass
class AOMap:
    version: int
    path: str
    data: bytes

    @classmethod
    def read(cls, r):
        v = r.u(2)
        return cls(v, r.s(), r.raw(8 + (16 if v >= 2 else 3) + 1 + (2 if v >= 3 else 0)))

    def write(self, w):
        w.u(self.version, 2); w.s(self.path); w.raw(self.data)

    @classmethod
    def none(cls):
        """'no AO texture' entry, as the game uses on glass"""
        return cls(3, '', bytes.fromhex('ffffffffffffffff0000000000000000000000000000000001001f'))


@dataclass
class RenderModel:
    version: int                  # 15 FH2, 14 FM5
    path: str                     # game:\media\cars\<car>\scene\...\x.modelbin
    transform: bytes              # 16 float32
    lod: int
    bone_name: str
    bone_index: int
    flags: int
    draw_groups: int
    material_overrides: list      # [(name, data)]
    material_indices: list        # [(material name, value)]
    hash: bytes                   # b'' or 8 bytes
    unknown8: bytes
    ao_maps: list
    unknown10: bytes
    unknown11: bytes
    part_name: str
    guid: bytes
    unknown14: bytes
    unknown15: bytes
    upgrade_ids: list = None      # only for models inside upgrade groups

    @classmethod
    def read(cls, r):
        v = r.u(2)
        if v not in (14, 15): raise ValueError(f'unsupported render model version {v}')
        path = r.s(); tr = r.raw(64); lod = r.u(2); bone = r.s(); bi = r.u(2); fl = r.u(1)
        dg = r.u(signed=True)
        mo = r.arr(lambda r: (r.s(), r.raw(r.u())))
        mi = r.arr(lambda r: (r.s(), r.u()))
        h = r.raw(8) if r.u(1) else b''
        u8 = r.raw(4)
        ao = r.arr(AOMap.read)
        return cls(v, path, tr, lod, bone, bi, fl, dg, mo, mi, h, u8, ao, r.raw(1), r.raw(18), r.s(), r.raw(16),
                   r.raw(20), r.raw(4) if v >= 15 else b'')

    def write(self, w):
        w.u(self.version, 2); w.s(self.path); w.raw(self.transform); w.u(self.lod, 2); w.s(self.bone_name)
        w.u(self.bone_index, 2); w.u(self.flags, 1); w.u(self.draw_groups, signed=True)
        w.arr(self.material_overrides, lambda w, m: (w.s(m[0]), w.u(len(m[1])), w.raw(m[1])))
        w.arr(self.material_indices, lambda w, m: (w.s(m[0]), w.u(m[1])))
        w.u(1 if self.hash else 0, 1); w.raw(self.hash); w.raw(self.unknown8)
        w.arr(self.ao_maps, lambda w, a: a.write(w))
        w.raw(self.unknown10); w.raw(self.unknown11); w.s(self.part_name); w.raw(self.guid)
        w.raw(self.unknown14); w.raw(self.unknown15)

    @property
    def matrix(self):
        return struct.unpack('<16f', self.transform)


@dataclass
class Part:
    dispatch: int
    version: int
    part_type: int               # 2 body, 4 brakes, 43 wheels
    models: list
    data: bytes                  # 8 floats (part bounds)

    @classmethod
    def read(cls, r):
        d, v, t = r.u(1), r.u(2), r.u()
        return cls(d, v, t, r.arr(RenderModel.read), r.raw(32 if v >= 2 else 0))

    def write(self, w):
        w.u(self.dispatch, 1); w.u(self.version, 2); w.u(self.part_type)
        w.arr(self.models, lambda w, m: m.write(w)); w.raw(self.data)


@dataclass
class Upgrade:
    version: int
    level: int
    is_stock: int
    id: int
    car_body_id: int
    parent_is_stock: int
    data: bytes

    @classmethod
    def read(cls, r):
        v = r.u(2)
        return cls(v, r.u(1), r.u(1), r.u(signed=True), r.u(signed=True), r.u(1), r.raw(32 if v >= 2 else 0))

    def write(self, w):
        w.u(self.version, 2); w.u(self.level, 1); w.u(self.is_stock, 1); w.u(self.id, signed=True)
        w.u(self.car_body_id, signed=True); w.u(self.parent_is_stock, 1); w.raw(self.data)


@dataclass
class Group:
    version: int
    part_type: int               # 9 wing, 34 front bumper, 35 rear bumper, 36 hood, 37 skirts, 41 cage
    upgrades: list
    models: list

    @classmethod
    def read(cls, r):
        v = r.u(2); t = r.u(); ups = r.arr(Upgrade.read)
        def shared(r):
            ids = r.arr(lambda r: r.u()); m = RenderModel.read(r); m.upgrade_ids = ids; return m
        return cls(v, t, ups, r.arr(shared))

    def write(self, w):
        w.u(self.version, 2); w.u(self.part_type); w.arr(self.upgrades, lambda w, u: u.write(w))
        def shared(w, m):
            w.arr(m.upgrade_ids, lambda w, n: w.u(n)); m.write(w)
        w.arr(self.models, shared)


@dataclass
class CarScene:
    guid: bytes
    flags: int
    ordinal: int                 # car id
    media_name: str
    skeleton: str
    unknown: int
    parts: list = field(default_factory=list)
    groups: list = field(default_factory=list)

    @classmethod
    def parse(cls, data: bytes):
        r = _R(data)
        if r.u(2) != 5: raise ValueError('not an FH2 CarScene v5 carbin')
        c = cls(r.raw(16), r.u(1), r.u(), r.s(), r.s(), r.u(2))
        c.parts = r.arr(Part.read); c.groups = r.arr(Group.read)
        if r.p != len(data): raise ValueError('trailing bytes in carbin')
        return c

    def build(self) -> bytes:
        w = _W()
        w.u(5, 2); w.raw(self.guid); w.u(self.flags, 1); w.u(self.ordinal); w.s(self.media_name)
        w.s(self.skeleton); w.u(self.unknown, 2)
        w.arr(self.parts, lambda w, p: p.write(w)); w.arr(self.groups, lambda w, g: g.write(w))
        return bytes(w.b)

    def all_models(self):
        for p in self.parts: yield from p.models
        for g in self.groups: yield from g.models
