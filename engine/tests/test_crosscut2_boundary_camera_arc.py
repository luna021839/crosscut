'''Two-sided frozen hold camera paths: exact strips, one path, no clamping.'''
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from crosscut2.boundary_camera_arc import build, realizable  # noqa: E402


def rows(n, row):
    return np.tile(np.asarray(row, float), (n, 1))


@pytest.mark.parametrize('length', [12, 13])
def test_twelve_and_thirteen_frame_overlap_requires_equality(length):
    overlap = 14 - length
    entry = rows(7, (1.2, .5, .5))
    leave = np.vstack([rows(overlap, (1.2, .5, .5)), rows(7 - overlap, (1.3, .52, .5))])
    base = rows(length, (1., .5, .5))
    arc = build(entry, leave, base, cap=1.5)
    assert arc is not None and arc.kind == 'overlap'
    assert np.array_equal(arc.values[:7], entry)
    assert np.array_equal(arc.values[length - 7:], leave)
    bad = leave.copy()
    bad[0, 0] = 1.21
    assert build(entry, bad, base, cap=1.5) is None


def test_short_final_hold_needs_identical_strips():
    strip = rows(5, (1.1, .5, .5))
    arc = build(strip, strip, rows(5, (1., .5, .5)), cap=1.5)
    assert arc is not None and np.array_equal(arc.values, strip)
    other = strip.copy()
    other[2, 1] = .51
    assert build(strip, other, rows(5, (1., .5, .5)), cap=1.5) is None


@pytest.mark.parametrize('length', [20, 40])
def test_entry_zoom_one_exit_zoom_one_point_five_is_one_path(length):
    entry, leave = rows(7, (1., .5, .5)), rows(7, (1.5, .5, .5))
    base = rows(length, (1.25, .5, .5))
    arc = build(entry, leave, base, cap=1.5)
    assert arc is not None and arc.values.shape == (length, 3)
    assert np.array_equal(arc.values[:7], entry)
    assert np.array_equal(arc.values[-7:], leave)
    zoom = arc.values[:, 0]
    assert np.all(np.diff(zoom) >= -1e-12) and zoom.max() <= 1.5 + 1e-12
    assert arc.kind == ('connectors' if length >= 38 else 'bridge')
    if length >= 38:
        assert np.array_equal(arc.values[18:length - 18], base[18:length - 18])
        assert not arc.changed[18:length - 18].any()


@pytest.mark.parametrize('length', [1, 3, 7, 12, 13, 14, 20, 37, 38, 60])
def test_constant_camera_is_preserved_exactly(length):
    n = min(7, length)
    value = (1.4, .55, .45)
    arc = build(rows(n, value), rows(n, value), rows(length, value), cap=2.)
    assert arc is not None
    assert np.array_equal(arc.values, rows(length, value))
    assert not arc.changed.any()


@pytest.mark.parametrize('length', [14, 15, 25, 37, 38, 50])
def test_endpoint_strips_are_exact_and_long_interior_is_baseline(length):
    k = np.arange(length)
    base = np.column_stack((1.5 + .05 * np.sin(k / 5), .5 + .02 * np.cos(k / 7),
                            .5 - .02 * np.sin(k / 9)))
    entry = np.column_stack((np.linspace(1.4, 1.46, 7), np.linspace(.5, .53, 7), np.full(7, .49)))
    leave = np.column_stack((np.linspace(1.6, 1.55, 7), np.full(7, .51), np.linspace(.48, .5, 7)))
    arc = build(entry, leave, base, cap=2.)
    assert arc is not None
    assert np.array_equal(arc.values[:7], entry)
    assert np.array_equal(arc.values[-7:], leave)
    assert realizable(arc.values, 2.)
    assert 0 in arc.knots and length - 1 in arc.knots
    if length >= 38:
        assert np.array_equal(arc.values[18:length - 18], base[18:length - 18])


def test_entry_slope_continues_into_connector():
    entry = np.column_stack((np.linspace(1., 1.06, 7), np.full(7, .5), np.full(7, .5)))
    arc = build(entry, rows(7, (1., .5, .5)), rows(40, (1., .5, .5)), cap=2.)
    assert arc is not None and arc.kind == 'connectors'
    step6 = arc.values[6, 0] - arc.values[5, 0]
    step7 = arc.values[7, 0] - arc.values[6, 0]
    assert abs(step7 - step6) < .5 * abs(step6)


def test_cap_padding_and_pan_violations_return_none():
    base = rows(30, (1.2, .5, .5))
    ok = rows(7, (1.2, .5, .5))
    assert build(ok, ok, base, cap=1.5) is not None
    assert build(ok, rows(7, (1.6, .5, .5)), base, cap=1.5) is None
    assert build(rows(7, (.9, .5, .5)), ok, base, cap=1.5) is None
    assert build(rows(7, (1.2, .3, .5)), ok, base, cap=1.5) is None
    assert build(ok, rows(7, (2., .72, .28)), base, cap=2.) is None
    bad_base = rows(60, (1.2, .5, .5))
    bad_base[30, 1] = .3
    assert build(ok, ok, bad_base, cap=1.5) is None


def test_connector_overshoot_is_rejected_not_clamped():
    entry = np.column_stack((np.linspace(1.2, 1.5, 7), np.full(7, .5), np.full(7, .5)))
    leave, base = rows(7, (1.5, .5, .5)), rows(40, (1.5, .5, .5))
    assert build(entry, leave, base, cap=1.5) is None
    arc = build(entry, leave, base, cap=2.)
    assert arc is not None and arc.values[:, 0].max() > 1.5


def test_values_are_frozen():
    arc = build(rows(7, (1.2, .5, .5)), rows(7, (1.3, .5, .5)), rows(30, (1.2, .5, .5)), cap=1.5)
    assert arc is not None
    with pytest.raises(ValueError):
        arc.values[0, 0] = 1.
    with pytest.raises(ValueError):
        arc.changed[0] = True


def test_malformed_inputs_raise():
    strip, base = rows(7, (1.2, .5, .5)), rows(30, (1.2, .5, .5))
    with pytest.raises(ValueError):
        build(strip[:6], strip, base, cap=1.5)
    with pytest.raises(ValueError):
        build(strip * np.nan, strip, base, cap=1.5)
    with pytest.raises(ValueError):
        build(strip, strip, base[:, :2], cap=1.5)
    with pytest.raises(ValueError):
        build(strip, strip, base, cap=.5)
    with pytest.raises(ValueError):
        build(strip, strip, base, cap=float('nan'))
    with pytest.raises(ValueError):
        build(strip, strip, base, cap=1.5, connector_frames=0)
