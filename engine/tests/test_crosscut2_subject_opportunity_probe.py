from types import SimpleNamespace
import numpy as np
from crosscut2.subject_opportunity_probe import candidates,observation_limits,observer,provisional_candidates


def test_one_explicit_full_programme_allowance_can_cover_more_than_twelve_moments():
    old=observation_limits(12,180.)
    assert old==dict(face_limit=768,pose_limit=512,raster_limit=2048,recognition_limit=4096,seconds=180.)
    whole=observation_limits(64,600.)
    assert whole==dict(face_limit=4096,pose_limit=2048,raster_limit=8192,recognition_limit=16384,seconds=600.)
    # More measured opportunities never require more final cuts or sources.
    cats=[SimpleNamespace(subjects=SimpleNamespace(sample=lambda *a:None)) for _ in range(2)]
    requests={(n,0,1,0,0):.5 for n in range(1,71)}
    events=SimpleNamespace(bindings={},missing_action_requests=requests,episodes=np.arange(72),nodes=np.arange(72)*30)
    selected=candidates(cats,events,maximum=64,explore_missing_action=True)
    assert len(selected)==64 and min(k[0] for k in selected)==1 and max(k[0] for k in selected)>=69
    assert not events.bindings


def test_unbounded_observation_request_fails_before_reading_a_model_or_source():
    import pytest
    for centres,seconds in ((True,180.),(-1,180.),(65,180.),(12,True),(12,0.),(12,np.inf),(12,901.)):
        with pytest.raises(ValueError,match='bounded opportunity'):
            observer('not-opened',sources=[],source_hashes=[],identities=[],cache='not-opened',
                authorized_root='not-opened',device=0,maximum_centres=centres,maximum_seconds=seconds)


def test_preselection_probe_ranks_fixed_raw_opportunities_not_selected_movie_cuts():
    cats=[SimpleNamespace(subjects=SimpleNamespace(sample=lambda *a:None)) for _ in range(3)]
    bindings={(1,0,1,0,0):.8,(2,0,2,0,0):.9,(3,1,2,0,0):.7}
    events=SimpleNamespace(bindings=bindings,episodes=np.array([0,1,1,2]))
    assert candidates(cats,events,maximum=2)==[(2,0,2,0,0),(3,1,2,0,0)]
    # Repeated camera variants / pairs in one music episode do not consume
    # several observation centres. No final programme even exists here.
    events.bindings[(2,2,0,0,0)]=.89
    assert candidates(cats,events,maximum=2)==[(2,0,2,0,0),(3,1,2,0,0)]


def test_already_observed_pair_does_not_trigger_new_source_inference():
    cats=[SimpleNamespace(subjects=SimpleNamespace(sample=lambda *a:object())) for _ in range(2)]
    events=SimpleNamespace(bindings={(1,0,1,0,0):.8},episodes=np.array([0,1,2]))
    assert candidates(cats,events)==[]
    assert candidates(cats,events,include_observed=True)==[(1,0,1,0,0)]


def test_exploration_acquires_missing_action_without_requiring_positive_action_first():
    cats=[SimpleNamespace(subjects=SimpleNamespace(sample=lambda *a:None)) for _ in range(2)]
    key=(1,0,1,0,0)
    events=SimpleNamespace(bindings={},missing_action_requests={key:.4},
        episodes=np.array([0,1,2]),nodes=np.array([0,30,60]))
    assert candidates(cats,events)==[]
    assert candidates(cats,events,explore_missing_action=True)==[key]
    assert not events.bindings
    cats[0].subjects.sample=cats[1].subjects.sample=lambda *a:object()
    assert candidates(cats,events,include_observed=True,explore_missing_action=True)==[]


def test_exploration_spreads_one_finite_allowance_over_the_whole_music_clock():
    cats=[SimpleNamespace(subjects=SimpleNamespace(sample=lambda *a:None)) for _ in range(2)]
    requests={(n,0,1,0,0):value for n,value in ((1,.9),(2,.8),(3,.7),(5,.2),(8,.1))}
    events=SimpleNamespace(bindings={},missing_action_requests=requests,
        episodes=np.arange(10),nodes=np.arange(10)*30)
    actual=candidates(cats,events,maximum=3,explore_missing_action=True)
    assert actual==[(1,0,1,0,0),(3,0,1,0,0),(8,0,1,0,0)]
    assert len(actual)==3 and not events.bindings
    assert candidates(cats,events,maximum=0,explore_missing_action=True)==[]


def test_exploration_cannot_silently_launch_a_subject_model_from_another_cli_path(monkeypatch):
    import pytest
    from crosscut2.stage_mix import main
    monkeypatch.setattr('sys.argv',['stage_mix','--source-plan','unused','--analysis','unused','--cache','unused',
        '--output','unused','--models','unused','--explore-missing-action'])
    with pytest.raises(ValueError,match='explicit bounded subject opportunity probe'):main()


def test_expanded_cli_allowance_requires_an_explicit_observation_spec(monkeypatch):
    import pytest
    from crosscut2.stage_mix import main
    base=['stage_mix','--source-plan','unused','--analysis','unused','--cache','unused',
        '--output','unused','--models','unused']
    for option,value in (('--opportunity-observation-centres','64'),
            ('--maximum-opportunity-perception-seconds','600')):
        monkeypatch.setattr('sys.argv',[*base,option,value])
        with pytest.raises(ValueError,match='explicit bounded subject opportunity probe'):main()


def test_provisional_endpoints_only_request_missing_actual_addresses_not_a_final_edit():
    import pytest
    banks=[SimpleNamespace(bank={'heads':np.zeros((7,2,8,4))},
        subjects=SimpleNamespace(sample=lambda *a:None)) for _ in range(2)]
    def seam(t):return dict(output_frame=t,left_source=0,right_source=1,left_head_index=0,right_head_index=1)
    rows=[seam(t) for t in (10,20,30,40,50)]
    actual=provisional_candidates(banks,np.arange(7)*10,rows,maximum=3)
    assert actual==[(1,0,1,0,1),(3,0,1,0,1),(5,0,1,0,1)]
    assert provisional_candidates(banks,np.arange(7)*10,rows,maximum=0)==[]
    banks[0].subjects.sample=banks[1].subjects.sample=lambda *a:object()
    assert provisional_candidates(banks,np.arange(7)*10,rows,maximum=64)==[]
    with pytest.raises(ValueError,match='actual grid'):
        provisional_candidates(banks,np.arange(7)*10,[seam(11)],maximum=64)
    assert rows==[seam(t) for t in (10,20,30,40,50)]


def test_provisional_cli_cannot_silently_authorize_inference(monkeypatch):
    import pytest
    from crosscut2.stage_mix import main
    monkeypatch.setattr('sys.argv',['stage_mix','--source-plan','unused','--analysis','unused','--cache','unused',
        '--output','unused','--models','unused','--provisional-subject-observation-plan','unused'])
    with pytest.raises(ValueError,match='explicit bounded subject opportunity probe'):main()


def test_provisional_clock_change_is_rejected_before_opening_models(tmp_path):
    import json
    import pytest
    spec=tmp_path/'spec.json';plan=tmp_path/'provisional.json'
    spec.write_text(json.dumps(dict(sources=[{'path':'not-opened.mp4'}],fps=30.)))
    plan.write_text(json.dumps(dict(source_sha256=['source-hash'],source_file_identities=[[1,2,3,4]],
        render_plan=dict(sources=['not-opened.mp4'],fps=30.,master_audio_start=4.),seams=[])))
    for master in (None,True,0.,float('nan')):
        with pytest.raises(ValueError,match='physical source/clock'):
            observer(spec,sources=['not-opened.mp4'],source_hashes=['source-hash'],identities=[(1,2,3,4)],
                cache=tmp_path/'cache',authorized_root=tmp_path,device=0,provisional_plan=plan,master_start=master)


def test_opportunity_observer_finishes_before_final_domain_and_actual_costs(monkeypatch):
    from crosscut2 import played_pose_port_programme as module
    from test_crosscut2_preference_integration import producer
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    stages=[];old_make=module.make;old_arcs=module.ArcCosts
    def make(*args,**kw):
        stages.append('domain');return old_make(*args,**kw)
    def observed(**kw):
        assert kw['opportunities'].bindings is not None and kw['catalogues']
        stages.append('raw_observations');return dict(source_NN_calls=0,fixture_only=True)
    def arcs(*args,**kw):
        assert stages==['domain','raw_observations','domain']
        return old_arcs(*args,**kw)
    monkeypatch.setattr(module,'make',make);monkeypatch.setattr(module,'ArcCosts',arcs)
    result=module.select(*args,**kw,native_ports=native,music=[0,1,1,0],
        prior=module.PlayedPosePortPrior(editorial_opportunities=True),opportunity_observer=observed)
    assert result[4]['opportunity_subject_observations']['fixture_only']
