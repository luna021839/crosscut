from types import SimpleNamespace
import numpy as np
from crosscut import render_frame_schedule
from crosscut2.event_holds import check_selected_clock
from crosscut2.fixed_holds import State


def verify(monkeypatch,*,tick=15,key=15,observed=True,native_fps=30.,person=None):
    monkeypatch.setattr(render_frame_schedule,'probe_render_frame_clock',
        lambda path:SimpleNamespace(frame_rate=native_fps,frame_count=1000))
    plan=SimpleNamespace(sources=['fixture.mp4'],fps=30.,segments=[SimpleNamespace(
        source_start=0.,source_playback_rate=1.,editorial_transform=None)])
    banks=[dict(native_indices=np.array([[0,0],[round(29*native_fps/30)]*2]))]
    ports=[dict(frames=np.array([tick]),native_event_keys=[[key]],observed=np.array([observed]))]
    if person is not None:
        ports[0].update(person_native_pair_measured=np.array([True]),person_native_ordinals=np.array([person]))
    return check_selected_clock(plan,np.array([0,30]),[(0,1,0)],[State(0,None,'native')],banks,native_ports=ports)


def test_measured_retained_native_flanks_use_the_real_renderer_schedule(monkeypatch):
    r=verify(monkeypatch)
    assert r['retained_native_clock_verified'] and r['source_observation_ordinals_exact']
    assert r['retained_native_events_clock_checked']==1 and r['measured_retained_native_inputs_checked']==2


def test_shifted_native_mapping_cannot_pass_grid_only_clock_gate(monkeypatch):
    r=verify(monkeypatch,tick=16)
    assert not r['retained_native_clock_verified'] and not r['source_observation_ordinals_exact']
    assert any(d['kind']=='native_first_incoming_mapping' for d in r['differing_inputs'])


def test_skipped_native_raster_can_be_unknown_but_correctly_located(monkeypatch):
    r=verify(monkeypatch,tick=8,key=15,observed=False,native_fps=60.)
    assert r['retained_native_clock_verified'] and r['measured_retained_native_inputs_checked']==0
    assert r['retained_native_events_clock_checked']==1


def test_skipped_raster_cannot_be_misreported_as_an_exact_measured_pair(monkeypatch):
    r=verify(monkeypatch,tick=8,key=15,observed=True,native_fps=60.)
    assert not r['retained_native_clock_verified']
    assert any(d['kind']=='measured_retained_native_flank' for d in r['differing_inputs'])


def test_native_at_authored_hold_boundary_is_not_double_counted(monkeypatch):
    r=verify(monkeypatch,tick=0,key=0)
    assert r['retained_native_clock_verified'] and r['retained_native_events_clock_checked']==0


def test_person_extent_can_use_verified_actually_exposed_skipped_pair(monkeypatch):
    r=verify(monkeypatch,tick=8,key=15,observed=False,native_fps=60.,person=[14,16])
    assert r['retained_native_clock_verified'] and r['measured_person_retained_native_inputs_checked']==2
    assert r['measured_retained_native_inputs_checked']==0


def test_nearby_person_pair_cannot_pass_real_renderer_clock_gate(monkeypatch):
    r=verify(monkeypatch,tick=8,key=15,observed=False,native_fps=60.,person=[14,15])
    assert not r['retained_native_clock_verified']
    assert any(d['kind']=='person_retained_native_flank' for d in r['differing_inputs'])
