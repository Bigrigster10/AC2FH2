"""Assetto Corsa data.acd unpacker (key derived from the car folder name)."""
import struct, sys, os

def _i32(x): x &= 0xffffffff; return x - (1 << 32) if x & 0x80000000 else x
def _div(a, b): q = abs(a) // abs(b); return _i32(q if (a >= 0) == (b >= 0) else -q)
def _mod(a, b): return _i32(a - _div(a, b) * b)

def key_for(car_id):
    s = [ord(c) for c in car_id]; n = len(s)
    k1 = 0
    for c in s: k1 = _i32(k1 + c)
    k2 = 0
    for i in range(0, n - 1, 2): k2 = _i32(_i32(k2 * s[i]) - s[i + 1])
    k3 = 0
    for i in range(1, n - 3, 3):
        k3 = _i32(k3 * s[i]); k3 = _div(k3, s[i + 1] + 0x1b); k3 = _i32(k3 - 0x1b - s[i - 1])
    k4 = 0x1683
    for i in range(1, n): k4 = _i32(k4 - s[i])
    k5 = 0x42
    for i in range(1, n - 4, 4): k5 = _i32(_i32(_i32((s[i] + 0xf) * k5) * (s[i - 1] + 0xf)) + 0x16)
    k6 = 0x65
    for i in range(0, n - 2, 2): k6 = _i32(k6 - s[i])
    k7 = 0xab
    for i in range(0, n - 2, 2): k7 = _mod(k7, s[i])
    k8 = 0xab
    for i in range(0, n - 1): k8 = _div(k8, s[i]); k8 = _i32(k8 + s[i + 1])
    return '-'.join(str(k & 0xff) for k in (k1, k2, k3, k4, k5, k6, k7, k8))

def unpack(path, car_id=None):
    car_id = car_id or os.path.basename(os.path.dirname(os.path.abspath(path)))
    key = key_for(car_id).encode()
    b = open(path, 'rb').read(); p = 0; files = {}
    if struct.unpack_from('<i', b, 0)[0] == -1111: p = 8
    while p < len(b):
        nl = struct.unpack_from('<I', b, p)[0]; p += 4
        name = b[p:p + nl].decode(); p += nl
        dl = struct.unpack_from('<I', b, p)[0]; p += 4
        enc = b[p:p + dl * 4:4]; p += dl * 4
        files[name] = bytes((e - key[i % len(key)]) & 0xff for i, e in enumerate(enc))
    return files

if __name__ == '__main__':
    files = unpack(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else None
    for n, d in files.items():
        print(f'{n:28s} {len(d):7d}  {d[:40]!r}')
        if out: os.makedirs(out, exist_ok=True); open(os.path.join(out, n), 'wb').write(d)
