"""Xbox One (GCN, 2D thin, P4_16x16, 8 banks) texture tiling for 16-byte block formats (BC7/BC3/BC5).
Formula from fh2-mod-platform docs/game-notes.md (Thumbnails)."""
import numpy as np

def b(v, n): return (v >> n) & 1

def block_offset(x, y, macro_per_row):
    t = 16 * (b(x,0) | b(y,0)<<1 | b(x,1)<<2 | b(y,1)<<3 | b(x,2)<<4 | b(y,2)<<5)
    t += 1024 * ((y // 32) * macro_per_row + (x // 64))
    pipe = (b(x,3) ^ b(y,3) ^ b(x,4)) | (b(x,4) ^ b(y,4)) << 1
    tx, ty = x // 32, y // 8
    bank = (b(tx,0) ^ b(ty,2)) | (b(tx,1) ^ b(ty,1) ^ b(ty,2)) << 1 | (b(tx,2) ^ b(ty,0)) << 2
    return (t & 255) | pipe << 8 | bank << 10 | (t >> 8) << 13

def offsets(wb, hb):
    """byte offset of every 16-byte block (row-major wb x hb) inside the tiled surface"""
    mpr = -(-wb // 64)
    ys, xs = np.mgrid[0:hb, 0:wb]
    f = np.vectorize(lambda x, y: block_offset(int(x), int(y), mpr))
    return f(xs, ys).ravel()

def surface_size(wb, hb):
    return (-(-wb // 64) * 64) * (-(-hb // 32) * 32) * 16

def detile(data, wb, hb):
    offs = offsets(wb, hb)
    src = np.frombuffer(data, np.uint8)
    out = np.empty((wb*hb, 16), np.uint8)
    for i, o in enumerate(offs): out[i] = src[o:o+16]
    return out.tobytes()

def tile(linear, wb, hb):
    offs = offsets(wb, hb)
    dst = np.zeros(surface_size(wb, hb), np.uint8)
    src = np.frombuffer(linear, np.uint8).reshape(-1, 16)
    for i, o in enumerate(offs): dst[o:o+16] = src[i]
    return dst.tobytes()
