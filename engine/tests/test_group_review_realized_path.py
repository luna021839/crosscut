import json
import pytest
from crosscut.group_review_realized_path import (RealizedReviewVariant,
    select_realized_review_path,incident_tick_union,require_paired_motion_adoption)


def variant(a,b,camera='identity',actor='actor',utility=0.,changed=0):
    return RealizedReviewVariant(a,b,json.dumps(dict(camera_identity=camera,
        sampler_identity='native-sampler',actor_path_identity=actor,source_identity='source',
        quality_approved=False)),utility,changed)


def edge(a,b,cost=0.,allowed=True):
    return dict(left_identity=a.identity,right_identity=b.identity,boundary_tick=b.first,
        cost=cost,allowed=allowed)


def test_both_adjacent_actual_cameras_participate_in_whole_path_choice():
    a,b=variant(0,10),variant(0,10,'zoom',utility=2.,changed=10)
    c,d=variant(10,20),variant(10,20,'zoom',utility=2.,changed=10)
    measured=[]
    def evaluate(x,y):
        measured.append((x.payload['camera_identity'],y.payload['camera_identity']))
        return edge(x,y,cost=10. if x is b and y is d else 0.)
    path=select_realized_review_path([[a,b],[c,d]],evaluate,output_frame_count=20)
    assert path['utility']==2.
    assert ('zoom','zoom') in measured and ('identity','zoom') in measured
    assert not path['quality_approved']


def test_boundary_tick_choice_is_joint_and_cannot_omit_or_overlap_output():
    layers=[[variant(0,9),variant(0,10)],[variant(9,20,utility=2.),variant(10,20)]]
    path=select_realized_review_path(layers,edge,output_frame_count=20)
    assert [(v.first,v.end) for v in path['variants']]==[(0,9),(9,20)]
    with pytest.raises(ValueError,match='compatible'):
        select_realized_review_path([[variant(0,9)],[variant(10,20)]],edge,output_frame_count=20)


def test_serialization_cannot_refit_a_camera_or_replace_an_actor_after_select():
    v=variant(0,10);value=v.payload
    v.assert_unchanged(value)
    for key in ('camera_identity','sampler_identity','actor_path_identity'):
        changed=dict(value,**{key:'another'})
        with pytest.raises(ValueError,match='after selection'):v.assert_unchanged(changed)
    value['camera_identity']='mutable copy'
    assert v.payload['camera_identity']=='identity'


def test_exit_and_original_only_and_candidate_only_cuts_share_incident_contract():
    assert incident_tick_union(20,30,(22,),(25,),output_frame_count=100)==(20,22,25,30)


@pytest.mark.parametrize('verdict',['WORSE','INCONCLUSIVE'])
def test_paired_verdict_blocks_encode_even_when_support_passes(verdict):
    with pytest.raises(ValueError,match='permit encode'):
        require_paired_motion_adoption(comparison=dict(verdict=verdict),support_states=['PASS'],
            motion_verdicts=['NONREGRESSING'])


@pytest.mark.parametrize('motion',['REGRESSING','UNMEASURED'])
def test_more_closeup_cannot_purchase_known_motion_regression_or_missing_exit(motion):
    with pytest.raises(ValueError,match='permit encode'):
        require_paired_motion_adoption(comparison=dict(verdict='BETTER'),support_states=['PASS'],
            motion_verdicts=['NONREGRESSING',motion])


def test_unknown_support_string_has_no_original_provenance_and_cannot_approve():
    with pytest.raises(ValueError,match='frozen manifest'):
        require_paired_motion_adoption(comparison=dict(verdict='BETTER'),
            support_states=['UNKNOWN'],motion_verdicts=['NONREGRESSING'])


@pytest.mark.parametrize('support,motion',[([],['NONREGRESSING']),(['PASS'],[]),([],[])])
def test_no_observations_cannot_vacuously_permit_encode(support,motion):
    with pytest.raises(ValueError,match='explicit measured'):
        require_paired_motion_adoption(comparison=dict(verdict='BETTER'),
            support_states=support,motion_verdicts=motion)


def test_false_neighbor_receipt_and_unbounded_work_cannot_enter_the_beam():
    with pytest.raises(ValueError,match='actual neighbors'):
        select_realized_review_path([[variant(0,10)],[variant(10,20)]],
            lambda a,b:dict(edge(a,b),right_identity=a.identity),output_frame_count=20)
    with pytest.raises(ValueError,match='state budget'):
        select_realized_review_path([[variant(0,10),variant(0,10,'zoom')]],
            edge,output_frame_count=10,maximum_states=1)
