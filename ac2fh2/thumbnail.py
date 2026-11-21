"""Autoshow thumbnail: a studio render of the AC car on white, 3/4 front view like the stock tiles.

Small software renderer (no GPU needed): perspective projection, painter's algorithm, per-triangle colour from the
material (paint colour / texture sample at the triangle's UV), simple key + fill light and a soft ground shadow.
Rendered at 2x and downsampled for anti-aliasing.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .library import classify

FLAT = {'glass': (38, 44, 52), 'chrome': (200, 202, 206), 'rubber': (28, 28, 30), 'black_plastic': (30, 30, 32)}


def _look(eye, target):
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f])


def render(car, paint_rgb, texture_image, size=(816, 459), ss=2, yaw_deg=-38, pitch_deg=12):
    """car: accar.ACCar in FH2 axes (Y up, Z forward, -X left). paint_rgb: 0..1. texture_image(name) -> PIL or None"""
    W, H = size[0] * ss, size[1] * ss
    tris, cols, kinds = [], [], []
    tex_cache = {}

    def tex(name):
        if name not in tex_cache:
            im = texture_image(name) if name else None
            tex_cache[name] = np.asarray(im.convert('RGB').resize((256, 256))) if im is not None else None
        return tex_cache[name]

    for m in car.meshes:
        mat = m.material
        if '/RIM_BLUR' in m.path.upper() or 'shadow' in m.path.lower():
            continue
        kind = classify(mat).kind
        if mat.shader == 'ksTyres':
            kind = 'rubber'
        elif kind == 'skip':
            continue
        T = m.pos[m.tris]
        tris.append(T); kinds += [kind] * len(T)
        if kind == 'paint':
            c = np.tile(np.array(paint_rgb) * 255, (len(T), 1))
        elif kind in FLAT:
            c = np.tile(FLAT[kind], (len(T), 1)).astype(float)
        else:
            a = tex(mat.textures.get('txDiffuse'))
            if a is None:
                c = np.tile((120, 120, 120), (len(T), 1)).astype(float)
            else:
                uv = m.uv[m.tris].mean(1) % 1.0
                c = a[(uv[:, 1] * 255).astype(int), (uv[:, 0] * 255).astype(int)].astype(float)
        cols.append(c)
    T = np.concatenate(tris).astype(np.float64); C = np.concatenate(cols)
    kinds = np.array(kinds)

    lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0)
    centre = (lo + hi) / 2; centre[1] = lo[1] + (hi[1] - lo[1]) * 0.42
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    d = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])   # towards camera
    radius = np.linalg.norm(hi - lo) / 2
    fov = np.radians(26)
    eye = centre + d * radius / np.sin(fov / 2) * 0.66
    R = _look(eye, centre)
    V = (T - eye) @ R.T                                     # camera space: x right, y up, z forward
    f = (H / 2) / np.tan(fov / 2)
    P = np.stack([f * V[..., 0] / V[..., 2], -f * V[..., 1] / V[..., 2]], -1)
    plo, phi = P.reshape(-1, 2).min(0), P.reshape(-1, 2).max(0)     # fit the car to ~86% x 78% of the tile
    k = min(0.86 * W / (phi - plo)[0], 0.78 * H / (phi - plo)[1])
    fit = lambda q: (q - (plo + phi) / 2) * k + np.array([W / 2, H * 0.52])
    P = fit(P)

    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    to_cam = eye - T.mean(1); to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
    front = (n * to_cam).sum(1) > 0                            # winding is consistent (accar keeps it): cull backs
    n[~front] *= -1
    key = np.array([-0.4, 0.8, 0.45]); key /= np.linalg.norm(key)
    fill = np.array([0.6, 0.3, -0.2]); fill /= np.linalg.norm(fill)
    lam = 0.55 + 0.5 * np.clip(n @ key, 0, 1) + 0.2 * np.clip(n @ fill, 0, 1)
    h = key + to_cam; h /= np.linalg.norm(h, axis=1, keepdims=True)
    gloss = np.isin(kinds, ['paint', 'chrome', 'glass'])
    spec = np.where(gloss, np.clip((n * h).sum(1), 0, 1) ** 40 * 160, 0)
    # studio reflection: bright ceiling softboxes, mid-grey walls, darker floor; stronger at glancing angles
    ndv = np.clip((n * to_cam).sum(1), 0, 1)
    r = 2 * ndv[:, None] * n - to_cam                         # view vector reflected about the normal
    env = np.where(r[:, 1] > 0.35, 240, np.interp(r[:, 1], [-1, 0, 0.35], [70, 150, 200]))
    F = np.select([kinds == 'chrome', kinds == 'paint', kinds == 'glass'],
                  [0.75, 0.12 + 0.6 * (1 - ndv) ** 3, 0.08 + 0.7 * (1 - ndv) ** 3], 0.0)
    base = C * lam[:, None]
    rgb = np.clip(base * (1 - F)[:, None] + (env * F)[:, None] + spec[:, None], 0, 255).astype(np.uint8)

    img = Image.new('RGB', (W, H), (246, 246, 246))
    # soft floor gradient + contact shadow (the car's footprint projected on the ground)
    grad = np.linspace(246, 226, H // 2).astype(np.uint8)
    img.paste(Image.fromarray(np.repeat(grad[:, None], W, 1)).convert('RGB'), (0, H - H // 2))
    G = T.reshape(-1, 3).copy(); G[:, 1] = lo[1]
    Gv = (G - eye) @ R.T
    gp = fit(np.stack([f * Gv[:, 0] / Gv[:, 2], -f * Gv[:, 1] / Gv[:, 2]], -1))
    sh = Image.new('L', (W, H), 0)
    from scipy.spatial import ConvexHull
    hull = gp[ConvexHull(gp).vertices]
    ImageDraw.Draw(sh).polygon([tuple(p) for p in hull], fill=150)
    sh = sh.filter(ImageFilter.GaussianBlur(ss * 10))
    img.paste((150, 150, 150), (0, 0), sh)

    order = np.argsort(-V[..., 2].mean(1))                  # far to near
    dr = ImageDraw.Draw(img)
    vis = (V[..., 2] > 0.05).all(1) & (front | (kinds == 'glass'))
    for i in order:
        if vis[i]:
            dr.polygon([tuple(P[i, 0]), tuple(P[i, 1]), tuple(P[i, 2])], fill=tuple(rgb[i]))
    return img.resize(size, Image.LANCZOS)
