import numpy as np
import pytest

from crosscut2.member_wardrobe_layer import phases, similarity, composite, target_parts


def test_garment_changes_before_background():
    garment, background = phases(.6)
    assert garment > .9 and background == 0.
    assert phases(0.) == (0., 0.)
    assert phases(1.) == (1., 1.)


def test_actual_point_registration():
    a = np.array([[10, 10], [20, 10], [15, 15], [10, 20], [20, 20]], np.float32)
    b = a*1.1+[7, -2]
    fit = similarity(a, b)
    assert np.allclose(a@fit[:, :2].T+fit[:, 2], b, atol=1e-5)
    with pytest.raises(ValueError):
        similarity(np.zeros((5, 2)), np.zeros((5, 2)))


def test_measured_clothes_not_whole_screen_or_face():
    a = np.zeros((80, 60, 3), np.uint8); a[:] = (200, 10, 20)
    b = np.zeros_like(a); b[:] = (20, 200, 10)
    labels = np.zeros((80, 60), np.uint8)
    labels[10:25, 25:35] = 13; labels[25:65, 20:40] = 5
    g, p = target_parts(labels, instance_mask=(labels>0))
    assert g[45, 30] == 1 and g[17, 30] == 0 and g[45, 0] == 0
    identity = np.eye(2, 3, dtype=np.float32)
    kwargs = dict(incoming_garment=g, outgoing_person=p,
        incoming_person=p, incoming_to_outgoing=identity)
    result = composite(a, b, progress=.6, **kwargs)
    assert np.array_equal(result[17, 30], a[17, 30])  # no double-exposed face
    assert np.array_equal(result[45, 0], a[45, 0])  # outgoing stage retained
    assert result[45, 30, 1] > 180  # new outfit is already visible
    assert np.array_equal(composite(a, b, progress=0., **kwargs), a)
    assert np.array_equal(composite(a, b, progress=1., **kwargs), b)


def test_other_visible_head_has_separate_garments():
    labels = np.zeros((90, 100), np.uint8)
    labels[5:20, 15:30] = 13; labels[20:85, 10:35] = 5
    labels[5:20, 70:85] = 13; labels[20:85, 65:90] = 5
    instance=(labels>0);instance[:,50:]=False
    g, _ = target_parts(labels, instance_mask=instance)
    assert g[50, 20] == 1 and g[50, 75] == 0


def test_outgoing_skirt_cannot_import_incoming_stage():
    a=np.zeros((80,60,3),np.uint8);a[:]=(10,10,10)
    b=np.zeros_like(a);b[:]=(255,0,255)
    outgoing=np.zeros((80,60),np.float32);outgoing[30:70,20:40]=1
    missing=np.zeros_like(outgoing)
    result=composite(a,b,incoming_garment=missing,
        outgoing_person=outgoing,incoming_person=missing,
        incoming_to_outgoing=np.eye(2,3,dtype=np.float32),progress=.6)
    assert np.array_equal(result,a)
