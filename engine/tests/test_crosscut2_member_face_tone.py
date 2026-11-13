import cv2
import numpy as np
import pytest

from crosscut2.member_face_tone import fit, correct
from crosscut2.member_clothing_replace import composite


POINTS = np.array([[32, 23], [48, 23], [40, 30], [35, 37], [45, 37]], np.float32)


def pictures():
    a = np.full((100, 80, 3), (140, 170, 195), np.uint8)
    b = np.full_like(a, (165, 170, 190))
    person = np.zeros((100, 80), np.float32)
    person[5:95, 17:63] = 1
    return a, b, person


def test_both_faces_share_a_reference_without_matching_the_clothes_or_stage():
    a, b, person = pictures()
    delta = fit([a]*25, [b]*25, [POINTS]*25, [POINTS]*25)
    aa = correct(a, POINTS, person, delta[12], .5)
    bb = correct(b, POINTS, person, delta[12], -.5)
    lab = lambda image: cv2.cvtColor(image.astype(np.float32)/255, cv2.COLOR_BGR2LAB)[31, 34]
    assert np.linalg.norm(lab(aa)-lab(bb)) < np.linalg.norm(lab(a)-lab(b))*.3
    assert np.array_equal(aa[70:], a[70:])  # garment pixels
    assert np.array_equal(aa[:, :15], a[:, :15])  # stage pixels
    assert np.array_equal(correct(a, POINTS, person, delta[0], 0), a)


def test_short_window_fit_rejects_one_bad_cheek_sample_without_pulsing():
    a, b, _ = pictures()
    values = [b.copy() for _ in range(25)]
    values[12][:] = (40, 70, 140)
    regular = fit([a]*25, [b]*25, [POINTS]*25, [POINTS]*25)
    measured = fit([a]*25, values, [POINTS]*25, [POINTS]*25)
    np.testing.assert_allclose(measured, regular, atol=1e-5)


def test_original_endpoints_and_all_pixels_outside_face_are_unchanged():
    a, b, person = pictures()
    garment = np.zeros_like(person); garment[60:95] = person[60:95]
    delta = fit([a]*25, [b]*25, [POINTS]*25, [POINTS]*25)
    args = dict(incoming_garment=garment, outgoing_person=person, incoming_person=person,
        incoming_to_outgoing=np.eye(2, 3, dtype=np.float32), incoming_face_landmarks=POINTS,
        outgoing_face_landmarks=POINTS)
    for progress in (0, 1):
        expected = a if progress == 0 else b
        assert np.array_equal(composite(a, b, progress=progress, face_tone_delta=delta[0], **args), expected)
    old = composite(a, b, progress=.75, **args)
    new = composite(a, b, progress=.75, face_tone_delta=delta[18], **args)
    assert np.array_equal(old[70:], new[70:])
    assert np.array_equal(old[:, :15], new[:, :15])


def test_invalid_or_unobserved_colour_inputs_are_not_silently_fabricated():
    a, b, person = pictures()
    with pytest.raises(ValueError):fit([a]*3, [b]*4, [POINTS]*3, [POINTS]*3)
    with pytest.raises(ValueError):fit([a]*3, [b]*3, [POINTS*0]*3, [POINTS]*3)
    with pytest.raises(ValueError):correct(a, POINTS, person, [99, 0, 0], .5)
    with pytest.raises(ValueError):correct(a, POINTS, person, [0, np.nan, 0], .5)


def test_extreme_source_light_is_bounded_instead_of_clipping_skin_to_a_new_colour():
    a, _, _ = pictures()
    blue = np.full_like(a, (220, 100, 100))
    delta = fit([a]*3, [blue]*3, [POINTS]*3, [POINTS]*3)
    assert np.all(np.abs(delta) <= [18, 14, 14])
