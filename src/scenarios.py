"""Stage 7: the five scenarios, with ONE continuous strength axis, per-track weather and a fully seeded random stream (plan section 7).

  rain        albumentations 2.0.8 RandomRain
  fog         imagecorruptions 1.1.2 `fog` (vendored, see below)
  night       gamma + brightness + Gaussian noise (numpy)
  side_view   albumentations 2.0.8 Affine, anisotropic x-scale + y-shear, keep_ratio=False
  motion_blur imagecorruptions 1.1.2 `motion_blur` (vendored)

Strength is a "severity" s in [0, 5]. s = 0 is the identity. s = 1..5 are anchor rows (ANCHORS below); in between, every numeric
parameter is linearly interpolated (np.interp). For fog and motion blur the anchors are the five imagecorruptions severities, so s = 3
means "imagecorruptions severity 3" (motion-blur radius and sigma are HALVED because our images are 112 px, not the 224 px those constants
were made for). The other three scenarios use anchors written by me (a choice, logged): they only fix the shape of the axis, and the
mild/severe levels are then CHOSEN from Fit accuracy drops (src/stage7.py), not from these numbers.

Why vendored: imagecorruptions 1.1.2 needs `pkg_resources` (gone from current setuptools) and `scipy.ndimage.interpolation` (gone from
current SciPy) just to import, which is the plan's risk R7. Also it only accepts the integer severities 1..5 and uses the global
np.random, so it can neither sweep a continuous strength nor be seeded per track and per frame. The code below is a transcription of its
`fog`, `plasma_fractal`, `_motion_blur`, `shift` and kernel helpers (Apache-2.0, Hendrycks and Dietterich 2019; Michaelis et al. 2019),
with two edits: `np.float_` -> `float`, and np.random.* -> an explicit numpy Generator that we pass in.

Randomness (plan section 7, "per-track determinism"):
  * PER TRACK, drawn once, seeded by (track name, scenario, severity): rain slant, fog density factor, night gamma and brightness factors,
    motion-blur angle, side-view shear direction and size. So one drive-by shares one "weather".
  * PER FRAME, seeded by (track name, frame index, scenario, severity): rain drop positions, fog texture, night noise.
  Seeds are SHA-256 of a text key, so they do not depend on row order, batch size, process, or the global numpy state.

Usage as a library:  corrupt_batch(images_uint8, track_names, frame_idx, scenario, severity) -> uint8 array of the same shape.
"""
import hashlib
import math

import numpy as np

SCENARIOS = ["rain", "fog", "night", "side_view", "motion_blur"]          # fixed order (also the tie-break order in Stage 7)
GRID = [0.1, 0.2, 0.3, 0.4, 0.5] + [round(0.75 + 0.25 * i, 2) for i in range(18)]   # calibration grid: 0.1 ... 0.5 (fine, fog and rain bite early), then 0.75 ... 5.0
CAP = {"rain": 4.0, "fog": 5.0, "night": 4.0, "side_view": 5.0, "motion_blur": 5.0}  # strongest level a human can still recognise the sign at, set from the
                                                                                      # contact sheet BEFORE any accuracy was measured (plan 7: "cap at the strongest human-recognisable level")


def grid_for(scenario):
    return [g for g in GRID if g <= CAP[scenario] + 1e-9]
ANCHOR_S = [0, 1, 2, 3, 4, 5]
ANCHORS = {                                                               # value at s = 0 (identity), then s = 1..5
    "rain": {"drop_length": [4, 8, 12, 16, 22, 28], "drop_width": [1, 1, 1, 1, 1, 2],
             "brightness": [1.0, 0.97, 0.94, 0.9, 0.86, 0.82], "blur": [1, 3, 3, 3, 3, 5]},
    "fog": {"amount": [0.0, 1.5, 2.0, 2.5, 2.5, 3.0], "decay": [2.0, 2.0, 2.0, 1.7, 1.5, 1.4]},
    "night": {"gamma": [1.0, 1.3, 1.6, 2.0, 2.4, 2.8], "brightness": [1.0, 0.85, 0.7, 0.55, 0.42, 0.3],
              "noise": [0.0, 0.02, 0.03, 0.04, 0.055, 0.07]},
    "side_view": {"scale_x": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], "shear_deg": [0.0, 5, 10, 15, 20, 25]},
    "motion_blur": {"radius": [0.0, 5.0, 7.5, 7.5, 7.5, 10.0], "sigma": [1.0, 1.5, 2.5, 4.0, 6.0, 7.5]},
}
RAIN_TYPE_EDGES = [(1.5, "drizzle"), (2.5, "default")]                    # s below the edge -> that type; else "heavy" (albumentations: 28 / 37 / 112 drops at 112 px)
JITTER = {"fog_amount": (0.85, 1.15), "night_gamma": (0.9, 1.1), "night_brightness": (0.9, 1.1), "side_shear": (0.9, 1.1),
          "rain_slant": (-10.0, 10.0), "blur_angle": (-45.0, 45.0)}      # per-track U(a, b) draws (a multiplier, or degrees/pixels for slant and angle)


# ------------------------------------------------------------------ seeds
def _seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:8], "little")


def _sev(s):
    return f"{float(s):.2f}"


def track_rng(track, scenario, s):
    return np.random.default_rng(_seed("track", track, scenario, _sev(s)))


def frame_seed(track, frame, scenario, s):
    return _seed("frame", track, int(frame), scenario, _sev(s))


# ------------------------------------------------------------------ nominal parameters
def nominal(scenario, s):
    """Interpolated parameters at severity s (before the per-track jitter)."""
    assert scenario in ANCHORS and 0 <= s <= 5, (scenario, s)
    p = {k: float(np.interp(s, ANCHOR_S, v)) for k, v in ANCHORS[scenario].items()}
    if scenario == "rain":
        p["rain_type"] = next((t for edge, t in RAIN_TYPE_EDGES if s < edge), "heavy")
        p["drop_length"], p["drop_width"], p["blur"] = int(round(p["drop_length"])), int(round(p["drop_width"])), int(round(p["blur"]))
    if scenario == "motion_blur":
        p["radius"] = int(round(p["radius"]))
    return p


def track_params(scenario, track, s):
    """What one track's weather is: nominal parameters with this track's draws applied."""
    p = nominal(scenario, s)
    r = track_rng(track, scenario, s)
    if scenario == "rain":
        p["slant"] = float(r.uniform(*JITTER["rain_slant"]))
    elif scenario == "fog":
        p["amount"] = p["amount"] * float(r.uniform(*JITTER["fog_amount"]))
    elif scenario == "night":
        p["gamma"] = p["gamma"] * float(r.uniform(*JITTER["night_gamma"]))
        p["brightness"] = min(1.0, p["brightness"] * float(r.uniform(*JITTER["night_brightness"])))
    elif scenario == "side_view":
        p["shear_deg"] = p["shear_deg"] * float(r.uniform(*JITTER["side_shear"])) * (1.0 if r.random() < 0.5 else -1.0)
    elif scenario == "motion_blur":
        p["angle"] = float(r.uniform(*JITTER["blur_angle"]))
    return p


# ------------------------------------------------------------------ vendored imagecorruptions code (see module docstring)
def _next_pow2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _plasma_fractal(rng, mapsize=256, wibbledecay=3):
    """Diamond-square heightmap in [0, 1] (imagecorruptions.plasma_fractal, with an explicit Generator)."""
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=float)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        msize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:msize:stepsize, stepsize // 2:msize:stepsize]
        ulgrid = maparray[0:msize:stepsize, 0:msize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:msize:stepsize, stepsize // 2:msize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:msize:stepsize, 0:msize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay
    maparray -= maparray.min()
    return maparray / maparray.max()


def _fog(img, amount, decay, rng):
    """imagecorruptions.fog with (amount, decay) = the severity constants, continuous."""
    h, w = img.shape[:2]
    x = img.astype(float) / 255.0
    max_val = x.max()
    x = x + amount * _plasma_fractal(rng, _next_pow2(max(h, w)), decay)[:h, :w][..., None]
    return np.clip(x * max_val / (max_val + amount), 0, 1) * 255


def _shift(image, dx, dy):
    if dx < 0:
        shifted = np.roll(image, shift=image.shape[1] + dx, axis=1)
        shifted[:, dx:] = shifted[:, dx - 1:dx]
    elif dx > 0:
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:, :dx] = shifted[:, dx:dx + 1]
    else:
        shifted = image
    if dy < 0:
        shifted = np.roll(shifted, shift=image.shape[0] + dy, axis=0)
        shifted[dy:, :] = shifted[dy - 1:dy, :]
    elif dy > 0:
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy, :] = shifted[dy:dy + 1, :]
    return shifted


def _motion_blur(x, radius, sigma, angle):
    """imagecorruptions._motion_blur: average of shifted copies along `angle`, weights = one-sided Gaussian."""
    width = radius * 2 + 1
    k = np.exp(-np.arange(width) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    kernel = k / k.sum()
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])
    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i * point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i * point[1]) / hypot) - 0.5)
        if abs(dy) >= x.shape[0] or abs(dx) >= x.shape[1]:
            break
        blurred = blurred + kernel[i] * _shift(x, dx, dy)
    return blurred


# ------------------------------------------------------------------ albumentations transforms (built once per track, reseeded per frame)
def _alb():
    """Import albumentations and OpenCV. Two precautions: (1) no network version check at import (reproducible, works offline);
    (2) albumentations also imports torch (only for ToTensorV2, which we never use) and tolerates ImportError but NOT the OSError that Windows
    Application Control raises when it blocks torch's DLLs (WinError 4551). In that case torch is hidden from albumentations and the import retried,
    so the smoke test works on a machine where torch is blocked."""
    import os
    import sys
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    try:
        try:
            import albumentations as A
        except OSError:
            for m in [k for k in sys.modules if k == "albumentations" or k.startswith("albumentations.")]:
                del sys.modules[m]
            sys.modules["torch"] = None                              # makes `import torch` raise ImportError, which albumentations suppresses
            try:
                import albumentations as A
            finally:
                if sys.modules.get("torch", 0) is None:
                    del sys.modules["torch"]
        import cv2
    except Exception as e:                                           # pragma: no cover - environment message
        raise RuntimeError("Stage 7 needs albumentations==2.0.8 (and OpenCV): py -3.12 -m pip install albumentations==2.0.8") from e
    return A, cv2


def _rain_transform(p):
    A, _ = _alb()
    return A.RandomRain(slant_range=(p["slant"], p["slant"]), drop_length=p["drop_length"], drop_width=p["drop_width"],
                        drop_color=(200, 200, 200), blur_value=p["blur"], brightness_coefficient=p["brightness"],
                        rain_type=p["rain_type"], p=1.0)


def _side_transform(p):
    A, cv2 = _alb()
    return A.Affine(scale={"x": (p["scale_x"], p["scale_x"]), "y": (1.0, 1.0)}, shear={"x": (0.0, 0.0), "y": (p["shear_deg"], p["shear_deg"])},
                    keep_ratio=False, border_mode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_LINEAR, p=1.0)


def _to_u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def corrupt_one(img, scenario, s, track, frame, params=None, transform=None):
    """One uint8 HxWx3 frame -> corrupted uint8 frame. `params`/`transform` may be passed in to reuse a track's weather."""
    if s == 0:
        return img.copy()
    p = params if params is not None else track_params(scenario, track, s)
    fs = frame_seed(track, frame, scenario, s)
    if scenario in ("rain", "side_view"):
        tr = transform if transform is not None else (_rain_transform(p) if scenario == "rain" else _side_transform(p))
        tr.set_random_seed(fs)
        return np.ascontiguousarray(tr(image=img)["image"]).astype(np.uint8)
    rng = np.random.default_rng(fs)
    if scenario == "fog":
        return _to_u8(_fog(img, p["amount"], p["decay"], rng))
    if scenario == "night":
        x = img.astype(float) / 255.0
        y = p["brightness"] * np.power(x, p["gamma"]) + rng.normal(0.0, p["noise"], x.shape)
        return _to_u8(np.clip(y, 0, 1) * 255)
    if scenario == "motion_blur":
        if p["radius"] < 1:
            return img.copy()
        return _to_u8(np.clip(_motion_blur(img, p["radius"], p["sigma"], p["angle"]), 0, 255))
    raise ValueError(scenario)


def corrupt_batch(images, track_names, frames, scenario, s):
    """images (N,H,W,3) uint8; track_names, frames: length N. Row order, batch size and the global numpy state do not matter."""
    out = np.empty(images.shape, dtype=np.uint8)
    cache = {}
    for i in range(len(images)):
        t = track_names[i]
        if t not in cache:
            p = track_params(scenario, t, s) if s != 0 else None
            tr = (_rain_transform(p) if scenario == "rain" else _side_transform(p)) if (s != 0 and scenario in ("rain", "side_view")) else None
            cache[t] = (p, tr)
        p, tr = cache[t]
        out[i] = corrupt_one(np.asarray(images[i]), scenario, s, t, frames[i], p, tr)
    return out


def array_sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def versions():
    import numpy
    import scipy
    v = {"numpy": numpy.__version__, "scipy": scipy.__version__}
    try:
        A, cv2 = _alb()
        v.update(albumentations=A.__version__, opencv=cv2.__version__)
    except Exception:                                                 # pragma: no cover
        v.update(albumentations=None, opencv=None)
    return v


# ------------------------------------------------------------------ contact sheet
def contact_sheet(images, track_names, frames, rows, path, label_w=190):
    """rows: list of (label, scenario or None, severity). First row is usually (clean, None, 0). Writes a PNG with Pillow only."""
    from PIL import Image, ImageDraw
    n, h, w = len(images), images.shape[1], images.shape[2]
    sheet = Image.new("RGB", (label_w + n * w, len(rows) * h), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for r, (label, scenario, s) in enumerate(rows):
        d.text((6, r * h + h // 2 - 6), label, fill=(0, 0, 0))
        out = images if scenario is None else corrupt_batch(images, track_names, frames, scenario, s)
        for c in range(n):
            sheet.paste(Image.fromarray(out[c]), (label_w + c * w, r * h))
    sheet.save(path)
    return path
