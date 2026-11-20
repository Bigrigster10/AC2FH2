"""Write / read FH2 .swatchbin textures (BC7, Xbox One tiled, full mip chain)."""
import io
import struct
import uuid

import numpy as np
from PIL import Image

from . import xbtile
from .bundle import Blob, Bundle, Meta

FMT_BC7_SRGB = bytes([0x2E, 0x01, 0x12, 0xB0])     # game diffuse textures
FMT_BC7_LINEAR = bytes([0x2E, 0x01, 0x00, 0xB0])   # masks / normal / data


def _guid_bytes(g: uuid.UUID) -> bytes:
    return g.bytes_le                                # .NET Guid byte order, as in the game files


def _mips(img: Image.Image):
    out = [img]
    while out[-1].width > 1 or out[-1].height > 1:
        w, h = max(1, out[-1].width // 2), max(1, out[-1].height // 2)
        out.append(out[-1].resize((w, h), Image.BOX))
    return out


def _bc7(img: Image.Image) -> bytes:
    import etcpak
    w, h = img.size
    pw, ph = -(-w // 4) * 4, -(-h // 4) * 4
    if (pw, ph) != (w, h):
        canvas = Image.new('RGBA', (pw, ph)); canvas.paste(img, (0, 0))
        # repeat the edge so block compression doesn't bleed black into the texture
        if pw > w: canvas.paste(img.crop((w - 1, 0, w, h)).resize((pw - w, h)), (w, 0))
        if ph > h: canvas.paste(canvas.crop((0, h - 1, pw, h)).resize((pw, ph - h)), (0, h))
        img = canvas
    return etcpak.compress_bc7(img.tobytes(), pw, ph)


def pow2_size(w, h, max_size=4096):
    p = lambda v: 1 << max(2, min(max_size.bit_length() - 1, int(round(np.log2(max(v, 4))))))
    return p(w), p(h)


def encode(img: Image.Image, srgb=True, guid: uuid.UUID = None, max_size=4096, mips=True) -> bytes:
    img = img.convert('RGBA')
    w, h = pow2_size(*img.size, max_size=max_size)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)
    levels = _mips(img) if mips else [img]
    data = xbtile.tile_chain([_bc7(l) for l in levels], w, h)
    guid = guid or uuid.uuid4()
    n = len(levels)
    txch = _guid_bytes(guid) + struct.pack('<HHHHHH', w, h, 1, w, h, 1) + bytes([n, n, 0]) + \
        (FMT_BC7_SRGB if srgb else FMT_BC7_LINEAR)
    assert len(txch) == 35
    return Bundle(1, 0, [Blob('TXCB', 2, 0, data, [Meta('TXCH', txch)])]).build()


def decode(swatch: bytes, mip=0) -> Image.Image:
    """BC7 swatchbins only (what this tool writes, and the game's diffuse textures)"""
    import texture2ddecoder as t
    b = Bundle.parse(swatch).first('TXCB'); m = b.meta('TXCH')
    w, h = struct.unpack_from('<HH', m, 16); n = m[28]
    if m[31] & 0x3F != 0x2E:
        raise ValueError('only BC7 swatchbins are decoded here')
    lv = xbtile.detile_chain(b.data, w, h, n)[mip]
    mw, mh = max(1, w >> mip), max(1, h >> mip)
    pw, ph = -(-mw // 4) * 4, -(-mh // 4) * 4
    return Image.frombytes('RGBA', (pw, ph), t.decode_bc7(lv, pw, ph), 'raw', 'BGRA').crop((0, 0, mw, mh))


def guid_name(stem: str, guid: uuid.UUID, kind='DIFF') -> str:
    """the game names car textures <name>_<KIND>_<guid>.swatchbin"""
    return f'{stem}_{kind}_{guid}.swatchbin'
