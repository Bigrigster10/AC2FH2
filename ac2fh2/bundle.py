"""ForzaTech 'Grub' bundle container (.modelbin, .swatchbin, materials).

Layout (little endian), FH2 / FM5 era (bundle version 1.0):
  header   : 'burG', u8 vmaj, u8 vmin, u16 blobCount, u32 headerSize, u32 totalSize
  blobInfo : 24 bytes each: tag[4] (stored reversed), u8 maj, u8 min, u16 metaCount,
             u32 metaOffset, u32 dataOffset, u32 size, u32 uncompressedSize
  metadata : per blob, metaCount entries of 8 bytes:
             tag[4] (reversed), u16 (size << 4 | version), u16 offset relative to the entry
             followed by the metadata payloads
  data     : blob payloads, each 4-byte aligned
"""
import struct
from dataclasses import dataclass, field

HEADER = struct.Struct('<4sBBHII')
INFO = struct.Struct('<4sBBHIIII')
META = struct.Struct('<4sHH')


def _tag_in(raw: bytes) -> str:
    return raw[::-1].decode('latin1')


def _tag_out(tag: str) -> bytes:
    return tag.encode('latin1')[::-1]


@dataclass
class Meta:
    tag: str
    data: bytes
    version: int = 0


@dataclass
class Blob:
    tag: str
    major: int
    minor: int
    data: bytes
    metas: list = field(default_factory=list)

    def meta(self, tag, default=None):
        for m in self.metas:
            if m.tag == tag:
                return m.data
        return default

    def set_meta(self, tag, data, version=0):
        for m in self.metas:
            if m.tag == tag:
                m.data = data
                return
        self.metas.append(Meta(tag, data, version))

    @property
    def id(self):
        d = self.meta('Id  ')
        return struct.unpack('<i', d[:4])[0] if d and len(d) >= 4 else None

    @property
    def name(self):
        d = self.meta('Name')
        return d.split(b'\0')[0].decode('latin1') if d is not None else None

    def __repr__(self):
        return f'<{self.tag} v{self.major}.{self.minor} {len(self.data)}B metas={[m.tag for m in self.metas]}>'


@dataclass
class Bundle:
    major: int = 1
    minor: int = 0
    blobs: list = field(default_factory=list)

    # ------------------------------------------------------------------ read
    @classmethod
    def parse(cls, buf: bytes) -> 'Bundle':
        magic, vmaj, vmin, count, _hdr, _total = HEADER.unpack_from(buf, 0)
        if magic != b'burG':
            raise ValueError('not a burG bundle')
        if (vmaj, vmin) != (1, 0):
            raise ValueError(f'unsupported bundle version {vmaj}.{vmin} (FH2 uses 1.0)')
        b = cls(vmaj, vmin)
        for i in range(count):
            tag, maj, mnr, mcount, moff, doff, size, _usize = INFO.unpack_from(buf, HEADER.size + INFO.size * i)
            metas = []
            for m in range(mcount):
                eo = moff + META.size * m
                mtag, packed, rel = META.unpack_from(buf, eo)
                ln, ver = packed >> 4, packed & 0xF
                metas.append(Meta(_tag_in(mtag), bytes(buf[eo + rel: eo + rel + ln]), ver))
            b.blobs.append(Blob(_tag_in(tag), maj, mnr, bytes(buf[doff:doff + size]), metas))
        return b

    # ----------------------------------------------------------------- write
    def build(self, pad_header: bool | None = None) -> bytes:
        if pad_header is None:  # textures (TXCB) start their data right after the header
            pad_header = not any(b.tag == 'TXCB' for b in self.blobs)
        n = len(self.blobs)
        out = bytearray(HEADER.size + INFO.size * n)
        meta_offs = []
        for blob in self.blobs:
            meta_offs.append(len(out))
            base = len(out)
            out += bytes(META.size * len(blob.metas))
            for j, m in enumerate(blob.metas):
                eo = base + META.size * j
                rel = len(out) - eo
                META.pack_into(out, eo, _tag_out(m.tag), (len(m.data) << 4) | m.version, rel)
                out += m.data
        if pad_header:
            _pad(out)
        header_size = len(out)
        data_offs = []
        for k, blob in enumerate(self.blobs):
            data_offs.append(len(out))
            out += blob.data
            if pad_header or k < n - 1:
                _pad(out)
        for i, blob in enumerate(self.blobs):
            INFO.pack_into(out, HEADER.size + INFO.size * i, _tag_out(blob.tag), blob.major, blob.minor,
                           len(blob.metas), meta_offs[i],
                           data_offs[i], len(blob.data), len(blob.data))
        HEADER.pack_into(out, 0, b'burG', self.major, self.minor, n, header_size, len(out))
        return bytes(out)

    # --------------------------------------------------------------- helpers
    def all(self, tag):
        return [b for b in self.blobs if b.tag == tag]

    def first(self, tag):
        return next((b for b in self.blobs if b.tag == tag), None)


def _pad(buf: bytearray, align=4):
    buf += bytes(-len(buf) % align)
