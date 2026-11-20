"""Assetto Corsa .kn5 reader (textures, materials, node tree, meshes)."""
import struct, sys
from dataclasses import dataclass, field

class R:
    def __init__(s, b): s.b, s.p = b, 0
    def u32(s): v = struct.unpack_from('<I', s.b, s.p)[0]; s.p += 4; return v
    def i32(s): v = struct.unpack_from('<i', s.b, s.p)[0]; s.p += 4; return v
    def f32(s): v = struct.unpack_from('<f', s.b, s.p)[0]; s.p += 4; return v
    def u8(s): v = s.b[s.p]; s.p += 1; return v
    def raw(s, n): v = s.b[s.p:s.p+n]; s.p += n; return v
    def str(s): return s.raw(s.u32()).decode('utf-8', 'replace')
    def fs(s, n): v = struct.unpack_from(f'<{n}f', s.b, s.p); s.p += 4*n; return v

@dataclass
class Material:
    name: str; shader: str; blend: int; alpha_tested: bool; depth_mode: int
    props: dict = field(default_factory=dict); textures: dict = field(default_factory=dict)

@dataclass
class Node:
    kind: int; name: str; children: list = field(default_factory=list)
    matrix: tuple = None; active: bool = True
    material: int = -1; verts: object = None; indices: object = None; skinned: bool = False

def read(path):
    r = R(open(path, 'rb').read())
    assert r.raw(6) == b'sc6969'
    ver = r.u32()
    if ver > 5: r.u32()
    textures = {}
    for _ in range(r.u32()):
        r.i32(); name = r.str(); textures[name] = r.raw(r.u32())
    mats = []
    for _ in range(r.u32()):
        m = Material(r.str(), r.str(), r.u8(), bool(r.u8()), r.i32())
        for _ in range(r.u32()):
            n = r.str(); v = r.f32(); r.fs(2); r.fs(3); r.fs(4); m.props[n] = v
        for _ in range(r.u32()):
            slot = r.str(); r.u32(); m.textures[slot] = r.str()
        if ver > 5 and False: pass
        mats.append(m)
    root = _node(r, ver)
    return ver, textures, mats, root

def _node(r, ver):
    import numpy as np
    kind = r.u32(); name = r.str(); nchild = r.u32(); active = bool(r.u8())
    n = Node(kind, name, active=active)
    if kind == 1:
        n.matrix = r.fs(16)
    elif kind in (2, 3):
        r.u8(); r.u8(); r.u8()                        # castShadows, visible, transparent
        if kind == 3:
            bones = [(r.str(), r.fs(16)) for _ in range(r.u32())]
        nv = r.u32()
        stride = 44 + (32 if kind == 3 else 0)        # pos3 nrm3 uv2 tan3 [+ weights4 idx4]
        n.verts = np.frombuffer(r.raw(nv*stride), dtype=np.float32).reshape(nv, stride//4)
        n.indices = np.frombuffer(r.raw(2*r.u32()), dtype=np.uint16)
        n.material = r.u32(); r.u32()                 # layer
        r.f32(); r.f32()                              # lodIn, lodOut
        r.fs(3); r.f32()                              # bounding sphere
        r.u8()                                        # renderable
        n.skinned = kind == 3
        if kind == 3: r.u32()
    for _ in range(nchild): n.children.append(_node(r, ver))
    return n

def walk(n, depth=0):
    yield depth, n
    for c in n.children: yield from walk(c, depth+1)

if __name__ == '__main__':
    ver, tex, mats, root = read(sys.argv[1])
    print('kn5 version', ver, '| textures', len(tex), '| materials', len(mats))
    meshes = [n for _, n in walk(root) if n.kind in (2, 3)]
    print('meshes', len(meshes), 'total tris', sum(len(m.indices)//3 for m in meshes))
    import collections
    print('shaders:', collections.Counter(m.shader for m in mats))
