"""Material templates harvested at runtime from the user's own FH2 car zips (no game data ships with the tool),
and the Assetto Corsa -> FH2 material classifier."""
import copy
import os
import re
import zipfile
from dataclasses import dataclass

from .bundle import Bundle
from .material import MaterialInstance, T_TEX, name_hash

# kind -> (library material file, carbin category flag)
KINDS = {
    'paint':          ('carpaint.materialbin', 1),
    'glass':          ('gls_window.materialbin', 128),
    'chrome':         ('chrome_polished.materialbin', 0),
    'rubber':         ('rubbertrim.materialbin', 0),
    'black_plastic':  ('blkplastic_mgl.materialbin', 0),
    'textured':       ('interiorengine\\interior_lod1.materialbin', 0),
    'textured_alpha': ('badges\\mit_emblem_002.materialbin', 0),
}
KIND_LABELS = {
    'paint': 'Car paint (in-game colour)', 'glass': 'Glass / windows', 'chrome': 'Chrome',
    'rubber': 'Rubber / trim', 'black_plastic': 'Black plastic', 'textured': 'Textured (opaque)',
    'textured_alpha': 'Textured with transparency (badges, lights, decals)', 'skip': "Don't convert",
}
# parameters that take the converted Assetto Corsa diffuse / alpha texture
DIFFUSE_PARAMS = {name_hash('DiffuseATexture'), name_hash('CH1DiffuseTextureTexture')}
ALPHA_PARAMS = {name_hash('AlphaTexture'), name_hash('CH1AlphaTextureTexture')}


class TemplateLibrary:
    """MatI blobs by library path, collected from FH2 car zips"""

    def __init__(self):
        self.by_lib = {}

    def harvest(self, zip_paths, wanted=None):
        wanted = {v[0].lower() for v in KINDS.values()} if wanted is None else wanted
        for zp in zip_paths:
            if wanted <= set(self.by_lib):
                break
            with zipfile.ZipFile(zp) as z:
                for n in z.namelist():
                    if not n.lower().endswith('.modelbin'):
                        continue
                    try:
                        blobs = Bundle.parse(z.read(n)).all('MatI')
                    except ValueError:
                        continue
                    for b in blobs:
                        if not b.data:
                            continue
                        mi = MaterialInstance.from_blob(b)
                        key = mi.library_path.lower().split('materials\\')[-1]
                        if key in wanted and key not in self.by_lib:
                            self.by_lib[key] = b
        return [k for k in wanted if k not in self.by_lib]

    def instance(self, kind) -> MaterialInstance:
        blob = self.by_lib[KINDS[kind][0].lower()]
        return MaterialInstance.from_blob(copy.deepcopy(blob))


@dataclass
class MatChoice:
    ac_name: str
    kind: str
    diffuse: str = None        # AC texture name
    reason: str = ''


def classify(mat) -> MatChoice:
    n, sh = mat.name.lower(), mat.shader
    diff = mat.textures.get('txDiffuse')
    d = (diff or '').lower()
    if sh in ('ksTyres', 'ksBrakeDisc') or 'blur' in n:
        return MatChoice(mat.name, 'skip', diff, 'tyre / brake / blur (wheels come from the donor)')
    if re.search(r'light|lamp|reflector|lens|indicator|detail_glass', n):
        return MatChoice(mat.name, 'textured_alpha', diff, 'light lens')
    if sh == 'ksWindscreen' or re.search(r'glass|window|windscreen|windshield', n):
        if sh != 'ksWindscreen' and getattr(mat, 'blend', 1) == 0 and not getattr(mat, 'alpha_tested', False):
            # AC draws it opaque (e.g. 'black_glass' = frit borders / wiper cowl): must not become see-through
            return MatChoice(mat.name, 'black_plastic' if 'black' in n or 'dark' in n else 'textured', diff,
                             'named glass but opaque in AC')
        return MatChoice(mat.name, 'glass', diff, 'glass')
    if 'skin' in d or re.search(r'(^|_)(car)?paint|^body', n) and sh == 'ksPerPixelMultiMap':
        return MatChoice(mat.name, 'paint', diff, 'body paint')
    if 'chrome' in n:
        return MatChoice(mat.name, 'chrome', diff, 'name says chrome')
    if re.search(r'rubber|tyre|tire|weatherstrip|seal', n):
        return MatChoice(mat.name, 'rubber', diff, 'rubber')
    if not diff:
        return MatChoice(mat.name, 'black_plastic', None, 'no texture')
    if mat.blend == 1 or mat.alpha_tested:
        return MatChoice(mat.name, 'textured_alpha', diff, 'uses transparency')
    return MatChoice(mat.name, 'textured', diff, 'textured')


def car_relative(path: str) -> bool:
    """texture paths that point inside some car's own folder (not the shared _library)"""
    p = path.lower()
    return '\\media\\cars\\' in p and '\\_library\\' not in p
