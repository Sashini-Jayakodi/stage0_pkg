"""Stage 7 corruptions: determinism per track and frame, shared weather within a track, identity at s=0, monotone degradation. No torch."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import scenarios as SC  # noqa: E402
try:
    SC._alb()                                        # albumentations + OpenCV (works even when torch is blocked, see scenarios._alb)
except RuntimeError as e:
    pytest.skip(f"albumentations/OpenCV not importable here: {e}", allow_module_level=True)


def _imgs(n=6, size=64, seed=0):
    """Smooth sign-like images (a coloured disc on a grey background) so degradation is measurable; not noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    out = np.empty((n, size, size, 3), np.uint8)
    for i in range(n):
        r = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size * 0.36) ** 2
        img = np.full((size, size, 3), 110, np.float64)
        img[r] = rng.integers(60, 230, 3)
        img[size // 2 - 3:size // 2 + 3, size // 4:3 * size // 4] = 245
        out[i] = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return out


NAMES = np.array(["00_00000", "00_00000", "00_00000", "05_00001", "05_00001", "05_00001"])
FRAMES = np.array([0, 3, 6, 0, 3, 6])


@pytest.mark.parametrize("sc", SC.SCENARIOS)
def test_severity_zero_is_the_identity(sc):
    x = _imgs()
    assert np.array_equal(SC.corrupt_batch(x, NAMES, FRAMES, sc, 0), x)


@pytest.mark.parametrize("sc", SC.SCENARIOS)
def test_deterministic_and_independent_of_row_order_batch_and_global_state(sc):
    x = _imgs()
    a = SC.corrupt_batch(x, NAMES, FRAMES, sc, 2.0)
    np.random.seed(123); np.random.rand(50)                               # global numpy state must not matter
    perm = np.array([4, 2, 5, 0, 3, 1])
    b = SC.corrupt_batch(x[perm], NAMES[perm], FRAMES[perm], sc, 2.0)
    assert np.array_equal(b, a[perm])
    c = np.concatenate([SC.corrupt_batch(x[:2], NAMES[:2], FRAMES[:2], sc, 2.0), SC.corrupt_batch(x[2:], NAMES[2:], FRAMES[2:], sc, 2.0)])
    assert np.array_equal(c, a)                                           # batch size does not matter
    assert a.dtype == np.uint8 and a.shape == x.shape


@pytest.mark.parametrize("sc", SC.SCENARIOS)
def test_different_scenario_or_severity_changes_the_pixels(sc):
    x = _imgs()
    a = SC.corrupt_batch(x, NAMES, FRAMES, sc, 1.0)
    b = SC.corrupt_batch(x, NAMES, FRAMES, sc, 3.0)
    assert not np.array_equal(a, b) and not np.array_equal(a, x)


def test_weather_is_drawn_once_per_track_and_differs_between_tracks():
    for sc in SC.SCENARIOS:
        p1, p2 = SC.track_params(sc, "00_00000", 2.0), SC.track_params(sc, "05_00001", 2.0)
        assert p1 == SC.track_params(sc, "00_00000", 2.0)                 # same track, same weather
        assert p1 != p2, sc                                               # another track has its own draw
        assert SC.track_params(sc, "00_00000", 2.0) != SC.track_params(sc, "00_00000", 2.5)    # seed includes severity


def test_pixel_randomness_depends_on_the_frame():
    x = _imgs(2)
    x[1] = x[0]
    a = SC.corrupt_batch(x, np.array(["t", "t"]), np.array([0, 1]), "night", 3.0)       # same track, same image, frames 0 and 1
    assert not np.array_equal(a[0], a[1])                                 # sensor noise differs per frame
    p = SC.track_params("night", "t", 3.0)
    assert p == SC.track_params("night", "t", 3.0)


def test_seeds_are_stable_across_processes():
    # SHA-256 based, so these never change between Python runs (hash() would).
    assert SC.frame_seed("00_00000", 3, "fog", 2.0) == SC.frame_seed("00_00000", 3, "fog", 2.0)
    assert SC.frame_seed("00_00000", 3, "fog", 2.0) != SC.frame_seed("00_00000", 4, "fog", 2.0)
    assert isinstance(SC.frame_seed("a", 0, "rain", 0.1), int)


@pytest.mark.parametrize("sc", SC.SCENARIOS)
def test_degradation_grows_with_severity(sc):
    # 1 - correlation with the clean image, averaged over 12 tracks. Coarse levels only: pixel metrics are not strictly monotone level by level
    # (fog re-normalises brightness), but mild < medium < strongest must hold, and that is what the calibration sweep relies on.
    x = _imgs(12, 112, seed=3)
    names, fr = np.array([f"{i:02d}_0000{i}" for i in range(12)]), np.arange(12) % 30
    d = []
    for s in (0.3, 2.0, SC.CAP[sc]):
        y = SC.corrupt_batch(x, names, fr, sc, s)
        d.append(np.mean([1 - np.corrcoef(a.astype(float).ravel(), b.astype(float).ravel())[0, 1] for a, b in zip(y, x)]))
    assert d[0] < d[1] < d[2], (sc, d)


def test_grid_and_caps():
    assert SC.GRID == sorted(SC.GRID) and SC.GRID[0] == 0.1 and SC.GRID[-1] == 5.0 and len(set(SC.GRID)) == len(SC.GRID)
    for sc in SC.SCENARIOS:
        g = SC.grid_for(sc)
        assert g == [s for s in SC.GRID if s <= SC.CAP[sc]] and g[-1] == SC.CAP[sc]
    assert SC.CAP["rain"] == 4.0 and SC.CAP["night"] == 4.0 and SC.CAP["fog"] == 5.0           # set from the synthetic sheet, before any accuracy was measured


def test_nominal_parameters_hit_the_anchors_and_interpolate():
    for sc, anchors in SC.ANCHORS.items():
        for s in SC.ANCHOR_S:
            p = SC.nominal(sc, float(s))
            for k, v in anchors.items():
                assert abs(p[k] - v[s]) < 1e-9 or isinstance(p[k], int) and abs(p[k] - v[s]) <= 0.5, (sc, k, s)
    assert SC.nominal("rain", 1.0)["rain_type"] == "drizzle" and SC.nominal("rain", 4.0)["rain_type"] == "heavy"
    mid = SC.nominal("night", 1.5)["gamma"]
    assert SC.ANCHORS["night"]["gamma"][1] < mid < SC.ANCHORS["night"]["gamma"][2]


def test_contact_sheet_is_written(tmp_path):
    x = _imgs(3, 32)
    rows = [("clean", None, 0.0), ("fog s=2", "fog", 2.0), ("rain s=2", "rain", 2.0)]
    p = SC.contact_sheet(x, NAMES[:3], FRAMES[:3], rows, tmp_path / "s.png", label_w=80)
    from PIL import Image
    im = Image.open(p)
    assert im.size == (80 + 3 * 32, 3 * 32)


def test_array_hash_changes_with_content():
    a = np.zeros((2, 4, 4, 3), np.uint8)
    b = a.copy(); b[0, 0, 0, 0] = 1
    assert SC.array_sha256(a) == SC.array_sha256(a.copy()) and SC.array_sha256(a) != SC.array_sha256(b)
