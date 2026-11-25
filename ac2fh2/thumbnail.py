"""Autoshow thumbnail: a studio render of the AC car, 3/4 front view like the stock tiles, transparent background.

Small software renderer (numpy only, no GPU): perspective z-buffer rasteriser at 3x resolution, per-pixel smooth
normals + perspective-correct UVs, alpha-tested decals, one see-through glass layer, screen-space ambient occlusion and
a studio environment (overhead softbox + side strip lights) reflected in paint clear coat, chrome and glass. Lit in
linear light, tone mapped, downsampled for anti-aliasing.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .library import classify

FLAT = {'glass': (8, 10, 12), 'chrome': (235, 237, 240), 'rubber': (26, 26, 28), 'black_plastic': (24, 24, 26)}
TEX = 1024                     # textures sampled at up to this size


def _look(eye, target):
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f])


def _lin(c):
    return (np.asarray(c, np.float32) / 255.0) ** 2.2


def _env(r):
    """studio lighting seen along direction r (N,3) -> linear radiance (N,)"""
    y = r[:, 1]
    e = np.interp(y, [-1, -0.08, -0.02, 0.0, 0.25, 1], [0.02, 0.04, 0.12, 0.75, 0.5, 0.45])   # dark floor, bright horizon
    e = e + 3.5 * np.clip((y - 0.72) / 0.08, 0, 1)                       # big overhead softbox
    az = np.degrees(np.arctan2(r[:, 0], r[:, 2]))
    for a in (-75, 70, 160):                                            # tall strip lights around the car
        d = np.abs((az - a + 180) % 360 - 180)
        e = e + 2.2 * np.clip((9 - d) / 3, 0, 1) * ((y > -0.05) & (y < 0.6))
    return e


def _fragments(S, iz, W, H, budget=6_000_000):
    """yield (tri index, pixel index, inverse depth, l1, l2) for every covered pixel centre (chunked)"""
    x, y = S[..., 0], S[..., 1]
    x0 = np.clip(np.floor(x.min(1) - 0.5), 0, W - 1).astype(np.int64)
    x1 = np.clip(np.ceil(x.max(1) - 0.5), 0, W - 1).astype(np.int64)
    y0 = np.clip(np.floor(y.min(1) - 0.5), 0, H - 1).astype(np.int64)
    y1 = np.clip(np.ceil(y.max(1) - 0.5), 0, H - 1).astype(np.int64)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    area2 = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    ok = (x.max(1) >= 0) & (x.min(1) < W) & (y.max(1) >= 0) & (y.min(1) < H) & (np.abs(area2) > 1e-9)
    idx = np.nonzero(ok)[0]
    area = (w * h)[idx]
    starts = np.r_[0, np.cumsum(area)]
    s = 0
    while s < len(idx):
        e = max(s + 1, int(np.searchsorted(starts, starts[s] + budget, 'right')) - 1)
        e = min(e, len(idx))
        t = idx[s:e]; a = area[s:e]
        k = np.repeat(np.arange(len(t)), a)
        off = np.arange(a.sum()) - np.repeat(np.cumsum(a) - a, a)
        ti = t[k]
        px = x0[ti] + off % w[ti]; py = y0[ti] + off // w[ti]
        cx, cy = px + 0.5, py + 0.5
        X, Y = x[ti], y[ti]
        l0 = ((X[:, 1] - cx) * (Y[:, 2] - cy) - (X[:, 2] - cx) * (Y[:, 1] - cy)) / area2[ti]
        l1 = ((X[:, 2] - cx) * (Y[:, 0] - cy) - (X[:, 0] - cx) * (Y[:, 2] - cy)) / area2[ti]
        l2 = 1 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        ti, px, py, l0, l1, l2 = ti[inside], px[inside], py[inside], l0[inside], l1[inside], l2[inside]
        inv = l0 * iz[ti, 0] + l1 * iz[ti, 1] + l2 * iz[ti, 2]
        yield ti, py * W + px, inv, l1, l2
        s = e


def render(car, paint_rgb, texture_image, size=(816, 459), ss=3, yaw_deg=38, pitch_deg=12, shadow=True):
    """car: accar.ACCar in FH2 axes (Y up, Z forward, -X left). paint_rgb: 0..1. texture_image(name) -> PIL or None"""
    W, H = size[0] * ss, size[1] * ss
    verts, nrms, uvs, tris, tmat = [], [], [], [], []
    mats = []                                  # (kind, linear albedo texture (h,w,3) or None, alpha (h,w) or None, flat)
    cache = {}

    def tex(name):
        if name not in cache:
            im = texture_image(name) if name else None
            if im is not None:
                im = im.convert('RGBA')
                if max(im.size) > TEX:
                    im.thumbnail((TEX, TEX))
                a = np.asarray(im)
                cache[name] = (_lin(a[..., :3]), a[..., 3] / 255.0)
            else:
                cache[name] = (None, None)
        return cache[name]

    nv = 0
    for m in car.meshes:
        mat = m.material
        if '/RIM_BLUR' in m.path.upper() or 'shadow' in m.path.lower() or len(m.tris) == 0:
            continue
        kind = classify(mat).kind
        if mat.shader == 'ksTyres':
            kind = 'tyre'
        elif mat.shader == 'ksBrakeDisc':
            kind = 'textured'
        elif kind == 'skip':
            continue
        diff = mat.textures.get('txDiffuse')
        rgb, alpha = tex(diff) if kind in ('textured', 'textured_alpha', 'tyre', 'paint') else (None, None)
        cutout = alpha is not None and (mat.alpha_tested or mat.blend == 1 or kind == 'textured_alpha') \
            and kind != 'paint' and alpha.min() < 0.5
        flat = _lin(np.array(paint_rgb) * 255) if kind == 'paint' else _lin(FLAT.get(kind, (110, 110, 110)))
        mats.append((kind, rgb, alpha if cutout else None, flat))
        verts.append(m.pos); nrms.append(m.nrm); uvs.append(m.uv)
        tris.append(m.tris + nv); tmat.append(np.full(len(m.tris), len(mats) - 1)); nv += len(m.pos)
    Vw = np.concatenate(verts).astype(np.float64); N = np.concatenate(nrms).astype(np.float64)
    UV = np.concatenate(uvs).astype(np.float64); T = np.concatenate(tris); TM = np.concatenate(tmat)
    kinds = np.array([k[0] for k in mats])

    # camera: 3/4 front-right like the stock tiles, car fitted to ~86% x 78% of the tile
    lo, hi = Vw.min(0), Vw.max(0)
    centre = (lo + hi) / 2; centre[1] = lo[1] + (hi[1] - lo[1]) * 0.42
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    d = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    radius = np.linalg.norm(hi - lo) / 2
    fov = np.radians(26)
    eye = centre + d * radius / np.sin(fov / 2) * 0.66
    R = _look(eye, centre)
    C = (Vw - eye) @ R.T
    f = (H / 2) / np.tan(fov / 2)
    P = np.stack([f * C[:, 0] / C[:, 2], -f * C[:, 1] / C[:, 2]], -1)
    used = np.unique(T)
    plo, phi = P[used].min(0), P[used].max(0)
    k = min(0.86 * W / (phi - plo)[0], 0.78 * H / (phi - plo)[1])
    fit = lambda q: (q - (plo + phi) / 2) * k + np.array([W / 2, H * 0.52])
    S = fit(P)[T]
    iz = 1.0 / np.maximum(C[:, 2], 1e-3)[T]

    def uv_at(ti, l1, l2, inv):
        tv = T[ti]
        w0, w1, w2 = (1 - l1 - l2) * iz[ti, 0], l1 * iz[ti, 1], l2 * iz[ti, 2]
        return (UV[tv[:, 0]] * w0[:, None] + UV[tv[:, 1]] * w1[:, None] + UV[tv[:, 2]] * w2[:, None]) / inv[:, None]

    def sample(img, uv):
        h, w = img.shape[:2]
        u = (uv[:, 0] % 1.0) * (w - 1); v = (uv[:, 1] % 1.0) * (h - 1)
        return img[v.astype(np.int64), u.astype(np.int64)]

    def alpha_keep(ti, l1, l2, inv):
        keep = np.ones(len(ti), bool)
        mi = TM[ti]
        for m_ in np.unique(mi):
            a = mats[m_][2]
            if a is None:
                continue
            sel = mi == m_
            keep[sel] = sample(a, uv_at(ti[sel], l1[sel], l2[sel], inv[sel])) >= 0.5
        return keep

    # opaque z-buffer (largest 1/z wins), then the nearest glass layer in front of it
    zb = np.zeros(W * H); tb = np.full(W * H, -1); b1 = np.zeros(W * H); b2 = np.zeros(W * H)
    glass_t = kinds[TM] == 'glass'
    for pass_glass in (False, True):
        sel_t = np.nonzero(glass_t == pass_glass)[0]
        if pass_glass:
            gz = np.zeros(W * H); gt = np.full(W * H, -1); g1 = np.zeros(W * H); g2 = np.zeros(W * H)
        for ti, pix, inv, l1, l2 in _fragments(S[sel_t], iz[sel_t], W, H):
            ti = sel_t[ti]
            if not pass_glass:
                keep = alpha_keep(ti, l1, l2, inv)
                ti, pix, inv, l1, l2 = ti[keep], pix[keep], inv[keep], l1[keep], l2[keep]
            else:
                front = inv > zb[pix] * 1.0005
                ti, pix, inv, l1, l2 = ti[front], pix[front], inv[front], l1[front], l2[front]
            zz, tt, aa, bb = (gz, gt, g1, g2) if pass_glass else (zb, tb, b1, b2)
            o = np.lexsort((-inv, pix))
            pix, inv, ti, l1, l2 = pix[o], inv[o], ti[o], l1[o], l2[o]
            first = np.r_[True, pix[1:] != pix[:-1]]
            pix, inv, ti, l1, l2 = pix[first], inv[first], ti[first], l1[first], l2[first]
            win = inv > zz[pix]
            p = pix[win]
            zz[p], tt[p], aa[p], bb[p] = inv[win], ti[win], l1[win], l2[win]

    def shade(pix, ti, l1, l2, inv, glass=False):
        tv = T[ti]
        w0, w1, w2 = (1 - l1 - l2) * iz[ti, 0], l1 * iz[ti, 1], l2 * iz[ti, 2]
        n = (N[tv[:, 0]] * w0[:, None] + N[tv[:, 1]] * w1[:, None] + N[tv[:, 2]] * w2[:, None])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        pos = (Vw[tv[:, 0]] * w0[:, None] + Vw[tv[:, 1]] * w1[:, None] + Vw[tv[:, 2]] * w2[:, None]) / inv[:, None]
        v = eye - pos; v /= np.linalg.norm(v, axis=1, keepdims=True)
        flip = (n * v).sum(1) < 0                 # two-sided: back faces seen through gaps light like fronts
        n[flip] *= -1
        ndv = np.clip((n * v).sum(1), 0, 1)
        r = 2 * ndv[:, None] * n - v
        mi = TM[ti]
        alb = np.zeros((len(ti), 3), np.float32)
        for m_ in np.unique(mi):
            sel = mi == m_
            kind, rgb, _, flat = mats[m_]
            if rgb is not None and kind != 'paint':
                alb[sel] = sample(rgb, uv_at(ti[sel], l1[sel], l2[sel], inv[sel]))
            else:
                alb[sel] = flat
        kd = kinds[mi]
        alb[kd == 'tyre'] *= 0.55
        key = np.array([-0.35, 0.85, 0.4]); key /= np.linalg.norm(key)
        dif = 0.25 + 0.55 * (0.5 + 0.5 * n[:, 1]) + 0.7 * np.clip(n @ key, 0, 1)
        env = _env(r)
        fres = 0.04 + 0.96 * (1 - ndv) ** 5
        refl = np.select([kd == 'paint', kd == 'chrome', kd == 'glass', np.isin(kd, ['tyre', 'rubber'])],
                         [0.07 + 0.93 * fres, 0.8 + 0 * fres, 0.08 + 0.92 * fres, 0.3 * fres], 0.6 * fres)
        col = alb * dif[:, None] * (1 - refl)[:, None] + (env * refl)[:, None]
        if glass:
            a = np.clip(0.65 + 0.35 * fres + 0.2 * env * refl, 0, 1)
            return col, a
        return col, None

    img = np.zeros((W * H, 3), np.float32); alpha = np.zeros(W * H, np.float32)
    hit = np.nonzero(tb >= 0)[0]
    col, _ = shade(hit, tb[hit], b1[hit], b2[hit], zb[hit])
    # screen-space ambient occlusion from the depth buffer (camera z), softened
    Z = np.where(tb >= 0, 1 / np.maximum(zb, 1e-9), 1e6).reshape(H, W)
    occ = np.zeros((H, W), np.float32); nsamp = 0
    rad = max(2, int(0.012 * H))
    for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        for sc_ in (0.5, 1.0):
            dx, dy = int(round(np.cos(ang) * rad * sc_ * 2)), int(round(np.sin(ang) * rad * sc_ * 2))
            Zs = np.roll(np.roll(Z, dy, 0), dx, 1)
            dz = Z - Zs
            occ += ((dz > 0.01) & (dz < 0.35)).astype(np.float32)
            nsamp += 1
    ao = 1 - 0.65 * np.asarray(Image.fromarray((occ / nsamp * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(ss)), np.float32) / 255
    col *= ao.reshape(-1)[hit][:, None]
    img[hit] = col; alpha[hit] = 1
    gh = np.nonzero(gt >= 0)[0]
    if len(gh):
        gc, ga = shade(gh, gt[gh], g1[gh], g2[gh], gz[gh], glass=True)
        behind = alpha[gh] > 0                 # glass over the car's own interior, or over the background
        img[gh] = np.where(behind[:, None], gc * ga[:, None] + img[gh] * (1 - ga)[:, None], gc)
        alpha[gh] = np.where(behind, 1.0, ga)

    # tone map (soft shoulder) + sRGB
    x = img * 1.15
    x = x / (1 + 0.25 * x)
    out = np.clip(x, 0, 1) ** (1 / 2.2)
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., :3] = (out.reshape(H, W, 3) * 255 + 0.5).astype(np.uint8)
    rgba[..., 3] = (alpha.reshape(H, W) * 255 + 0.5).astype(np.uint8)
    car_img = Image.fromarray(rgba, 'RGBA')

    canvas = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    if shadow:   # faint contact shadow under the car (transparent tile like the stock Autoshow images)
        G = Vw[used].copy(); G[:, 1] = lo[1]
        Gv = (G - eye) @ R.T
        gp = fit(np.stack([f * Gv[:, 0] / Gv[:, 2], -f * Gv[:, 1] / Gv[:, 2]], -1))
        from scipy.spatial import ConvexHull
        hull = gp[ConvexHull(gp).vertices]
        sh = Image.new('L', (W, H), 0)
        ImageDraw.Draw(sh).polygon([tuple(p) for p in hull], fill=90)
        sh = sh.filter(ImageFilter.GaussianBlur(ss * 8))
        canvas.putalpha(sh)
    canvas.alpha_composite(car_img)
    return canvas.convert('RGBa').resize(size, Image.LANCZOS).convert('RGBA')   # premultiplied: no dark fringes
