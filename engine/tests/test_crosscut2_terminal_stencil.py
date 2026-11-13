import numpy as np
from crosscut2.terminal_stencil import new_stencil,eligible,observe


def pixels():return np.full((540,960,3),120,np.uint8)


def test_new_large_opaque_ending_component_is_observed_without_a_neural_classifier():
    a,b=pixels(),pixels();b[280:410,350:600]=0
    result=new_stencil(a,b)
    assert result['new_opaque_ending_stencil'] and result['largest_new_component_fraction']>.04
    assert not result['semantic_logo_identity_certified']


def test_static_black_bars_and_a_dark_performance_are_not_an_ending_source_veto():
    a=pixels();a[:135]=0;a[220:350,350:600]=0
    assert not new_stencil(a,a.copy())['new_opaque_ending_stencil']
    assert not new_stencil(np.zeros_like(a),np.zeros_like(a))['new_opaque_ending_stencil']


def test_ending_filter_only_removes_observed_new_stencil_when_clean_finish_exists():
    bank=lambda known,bad:dict(terminal_stencil_observation=dict(known=known,new_opaque_ending_stencil=bad))
    assert eligible([bank(True,True),bank(True,False),{}]).tolist()==[False,True,True]
    assert eligible([bank(True,True),bank(True,True),{}]).all()
    assert eligible([{},{}]).all()


def test_actual_arc_domain_filters_only_final_hold_not_the_entire_source():
    from test_crosscut2_boundary_arc_costs import arc_fixture
    nodes,entry,exit,arcs,_=arc_fixture(False)
    for s,cat in enumerate(arcs.cats):cat.bank['terminal_stencil_observation']=dict(known=True,new_opaque_ending_stencil=s==0)
    _,allowed,_=arcs(0,len(nodes)-1)
    assert not allowed[[p.source==0 for p in entry[0]]].any()
    assert allowed[[p.source==1 for p in entry[0]]].any()
    assert arcs(0,1)[1][[p.source==0 for p in entry[0]]].any()


def test_different_native_shots_are_unknown_not_a_fabricated_graphic_detection(monkeypatch):
    from crosscut2 import terminal_stencil as module
    monkeypatch.setattr(module.sparse,'probe',lambda *a,**k:(_ for _ in ()).throw(AssertionError('must not probe')))
    bank=dict(native_indices=np.array([[89,90]]),native_cut_frames=np.array([75]))
    row=observe(['source.mp4'],[bank],[dict(fps=30.,frame_count=200)])[0]
    assert not row['known'] and not row['new_opaque_ending_stencil']


def test_optional_unsupported_seek_does_not_drop_source_or_stop_the_edit(monkeypatch):
    from crosscut2 import terminal_stencil as module
    def unsupported(*a,**k):raise module.sparse.UnsupportedSeek('unproven packet grid')
    monkeypatch.setattr(module.sparse,'probe',unsupported)
    bank=dict(native_indices=np.array([[89,90]]),native_cut_frames=np.array([],int))
    row=observe(['source.mp4'],[bank],[dict(fps=30.,frame_count=200)])[0]
    assert not row['known'] and eligible([dict(terminal_stencil_observation=row)]).all()
