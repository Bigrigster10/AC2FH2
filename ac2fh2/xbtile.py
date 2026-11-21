"""Xbox One (GCN) texture tiling for 16-byte block formats (BC7 / BC3 / BC5 / BC6H), whole mip chains.

Level layout (matches the game's swatchbins exactly):
  * the top level, and levels at least one macro tile (64 x 32 blocks) -> 2D thin: P4_16x16, 8 banks, macro tile
    64 x 32 blocks, 256-byte pipe interleave  (formula: fh2-mod-platform docs/game-notes.md)
  * smaller levels -> 1D thin: 8 x 8-block micro tiles in row order, at least one micro tile (1 KiB)
  * levels are stored back to back from the largest.
"""
import numpy as np


def _bit(v, n):
    return (v >> n) & 1


def _micro(x, y):
    """element index inside an 8x8 micro tile (16-byte elements)"""
    return (_bit(x, 0) | _bit(y, 0) << 1 | _bit(x, 1) << 2 | _bit(y, 1) << 3 | _bit(x, 2) << 4 | _bit(y, 2) << 5)


def _offsets_2d(wb, hb):
    mpr = -(-wb // 64)
    y, x = np.mgrid[0:hb, 0:wb]
    t = 16 * _micro(x, y) + 1024 * ((y // 32) * mpr + (x // 64))
    pipe = (_bit(x, 3) ^ _bit(y, 3) ^ _bit(x, 4)) | (_bit(x, 4) ^ _bit(y, 4)) << 1
    tx, ty = x // 32, y // 8
    bank = (_bit(tx, 0) ^ _bit(ty, 2)) | (_bit(tx, 1) ^ _bit(ty, 1) ^ _bit(ty, 2)) << 1 | (_bit(tx, 2) ^ _bit(ty, 0)) << 2
    return ((t & 255) | pipe << 8 | bank << 10 | (t >> 8) << 13).ravel()


def _offsets_1d(wb, hb):
    tpr = -(-wb // 8)
    y, x = np.mgrid[0:hb, 0:wb]
    return (1024 * ((y // 8) * tpr + (x // 8)) + 16 * _micro(x, y)).ravel()


def level_layout(wb, hb, level=0):
    """-> (offsets of each 16-byte block in the level, level size in bytes)
    The top level is always 2D thin (padded to whole macro tiles); smaller mips drop to 1D thin
    once they are narrower than one macro tile (64 x 32 blocks)."""
    if level == 0 or (wb >= 64 and hb >= 32):
        return _offsets_2d(wb, hb), (-(-wb // 64) * 64) * (-(-hb // 32) * 32) * 16
    return _offsets_1d(wb, hb), (-(-wb // 8) * 8) * (-(-hb // 8) * 8) * 16


def mip_dims(w, h, mips):
    out = []
    for i in range(mips):
        out.append((max(1, -(-max(1, w >> i) // 4)), max(1, -(-max(1, h >> i) // 4))))
    return out


def chain_size(w, h, mips):
    return sum(level_layout(wb, hb, i)[1] for i, (wb, hb) in enumerate(mip_dims(w, h, mips)))


def detile_chain(data, w, h, mips):
    """-> list of linear block data per mip"""
    out, base = [], 0
    src = np.frombuffer(data, np.uint8)
    for i, (wb, hb) in enumerate(mip_dims(w, h, mips)):
        offs, size = level_layout(wb, hb, i)
        idx = (base + offs)[:, None] + np.arange(16)
        out.append(src[idx].tobytes())
        base += size
    return out


def tile_chain(levels, w, h):
    """levels: linear 16-byte-block data per mip (largest first) -> tiled bytes"""
    parts = []
    for i, ((wb, hb), lin) in enumerate(zip(mip_dims(w, h, len(levels)), levels)):
        offs, size = level_layout(wb, hb, i)
        dst = np.zeros(size, np.uint8)
        blocks = np.frombuffer(lin, np.uint8).reshape(-1, 16)
        dst[(offs[:, None] + np.arange(16)).ravel()] = blocks.ravel()
        parts.append(dst.tobytes())
    return b''.join(parts)
