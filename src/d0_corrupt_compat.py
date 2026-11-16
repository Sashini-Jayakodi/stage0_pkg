"""d0 (EXPLORATORY): imagecorruptions 1.1.2 on numpy 2, with one patched call.

imagecorruptions 1.1.2 allocates the fog heightmap with `np.float_`, which numpy 2.0 removed, so `fog` fails at run
time. The only change here: `plasma_fractal` is the package's own function with `np.float_` replaced by `np.float64`
(the two names are the same dtype in numpy < 2). Nothing else in the package is touched; the other four corruptions
run unmodified. `reference_check` shows the patched fog equals the unmodified package fog (with numpy's old alias
restored for the duration of the call) on given images, bit for bit.
"""
import numpy as np

import imagecorruptions
from imagecorruptions import corruptions as _corr

CORRUPTIONS = ("fog", "motion_blur", "gaussian_noise", "contrast", "brightness")   # mild-to-severe = severity 1..5
SEVERITIES = (1, 2, 3, 4, 5)
_ORIGINAL_PLASMA = _corr.plasma_fractal
PATCH_NOTE = ("imagecorruptions.corruptions.plasma_fractal: np.empty(..., dtype=np.float_) -> dtype=np.float64; "
              "no other line changed")


def plasma_fractal_f64(mapsize=256, wibbledecay=3):
    """Package code of plasma_fractal (imagecorruptions 1.1.2), dtype np.float_ -> np.float64. Same random draws."""
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)       # the one changed call
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize_ = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize_:stepsize, stepsize // 2:mapsize_:stepsize]
        ulgrid = maparray[0:mapsize_:stepsize, 0:mapsize_:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize_:stepsize, stepsize // 2:mapsize_:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize_:stepsize, 0:mapsize_:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def install():
    """Point the package's module-level name at the patched function (fog looks it up at call time)."""
    _corr.plasma_fractal = plasma_fractal_f64


def corrupt(img, name, severity):
    """uint8 HxWx3 in, uint8 HxWx3 out; the package's own entry point after install()."""
    assert name in CORRUPTIONS and severity in SEVERITIES
    install()
    return imagecorruptions.corrupt(img, corruption_name=name, severity=severity)


def corrupt_track(frames, name, severity, seed):
    """All frames of one track (frame order), one corruption and severity; the global numpy RNG (which the package
    uses) is seeded once per track, so a track's output does not depend on which process or order runs it."""
    np.random.seed(seed)
    return np.stack([corrupt(np.ascontiguousarray(f), name, severity) for f in frames])


def reference_check(images, seed=0):
    """Per image: patched fog (severity 1..5) == unmodified package fog with numpy's old alias np.float_ = np.float64
    restored for the call. Returns a list of booleans, one per image (all severities must match)."""
    out = []
    had = "float_" in np.__dict__
    for img in images:
        ok = True
        for s in SEVERITIES:
            install()
            np.random.seed(seed)
            a = imagecorruptions.corrupt(img, corruption_name="fog", severity=s)
            _corr.plasma_fractal = _ORIGINAL_PLASMA
            np.float_ = np.float64                                   # numpy < 2 alias, only for the reference call
            try:
                np.random.seed(seed)
                b = imagecorruptions.corrupt(img, corruption_name="fog", severity=s)
            finally:
                if not had:
                    del np.float_
                install()
            ok &= bool(a.dtype == b.dtype and np.array_equal(a, b))
        out.append(ok)
    return out
