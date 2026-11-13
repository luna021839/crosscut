"""Synthetic contract counterexamples; not native/media quality evidence."""
from copy import deepcopy
import pytest
from crosscut.group_review_camera_repair import FramingDescriptor,fit_whole_camera_track
from crosscut.group_review_program_evaluator import (ReviewObjective,evaluate_review_program,
    consume_frozen_part_duties,freeze_empty_part_duties,bind_review_part_duties)
from crosscut.group_review_native_window import one_to_one_appearance_hypotheses,cosine
from crosscut.render import EditorialTransformKeyframe

def face(h=.1,x=.45,y=.3):
    box=[x,y,h/2,h]
    return dict(projection=dict(heads=[dict(observation_id=0,projected_box_xywh=box,visible_fraction=1.)],people=[]),
        native_observation=dict(heads=[dict(observation_id=0,box_xywh=box)],people=[],frame_size_wh=[40,24]))

def descriptor(h=.1):return FramingDescriptor((.5,.4),h,h/2,None,0,False)


def d(pixels=120.):
    return FramingDescriptor((.5,.4),pixels/1080,pixels/2160,None,0,False,None,pixels,pixels)


def evaluate(values,*,head=None,states=None):
    return evaluate_review_program(objective=ReviewObjective(),descriptors=values,
        pixel_features=[dict(detail_risk=.1)]*len(values),left=head or d(30),right=head or d(30),
        support_verdicts=[dict(state=s) for s in (states or ['PASS']*len(values))],
        association=dict(ambiguity_risk=.1),authority='selected_fresh')


def token(parts=None,head=True):
    return dict(contract='frozen_original_review_part_duties_v1',
        rows={'1':dict(head_required=head,parts=parts or {})})


def test_readable_close_style_tradeoff_remains_eligible():
    result=evaluate([d(180)]*3)
    assert result['eligible_review'] and result['incident_seam_costs'][0]>0
    assert not result['style_geometry_is_hard_gate'] and not result['quality_approved']


def test_mid_window_missing_is_not_removed_or_beneficial():
    full=evaluate([d()]*5);missing=evaluate([d(),None,None,None,d()])
    assert missing['soft_rank']>full['soft_rank']
    assert missing['actor_missing_fraction']==.6 and missing['longest_actor_missing_fraction']==.6
    assert missing['effective_detail_pixels'][2] is None


def test_known_original_required_missing_is_unknown_not_vacuous_pass():
    row=face();row['output_tick']=1;row['projection']['heads']=[];row['native_observation']['heads']=[]
    result=consume_frozen_part_duties(token({'left_arm':None}),row,candidate_head_ids=(),key=EditorialTransformKeyframe(0.,1.))
    assert result['state']=='UNKNOWN'
    assert not evaluate([d()],states=[result['state']])['eligible_review']


def test_known_original_head_loss_is_hard_not_style():
    row=face(.2,.001);row['output_tick']=1
    result=consume_frozen_part_duties(token(),row,candidate_head_ids=(0,),key=EditorialTransformKeyframe(0.,1.75))
    assert result['state']=='FAIL'
    assert not evaluate([d()],states=['FAIL'])['eligible_review']


def test_nonfocal_token_cannot_delete_focal_or_formation():
    for role in ('focal_head','focal_head_and_observed_arms','required_formation'):
        with pytest.raises(ValueError):freeze_empty_part_duties(original_role_program=dict(epochs=[dict(role=role)]),first=1,end=3)
    value=freeze_empty_part_duties(original_role_program=dict(epochs=[dict(role='observed_body_composition')]),first=1,end=3)
    assert not value['original_composition_inventory_certified']


def test_whole_camera_fit_consumes_actual_unequal_tick_phases():
    rows=[face() for _ in range(3)]
    for row,tick in zip(rows,(100,119,137)):
        row['output_tick']=tick;row['query']=dict(camera=dict(output_fps=29.97))
    result=fit_whole_camera_track(rows,descriptor(.15),descriptor(.15),duration=37/29.97)
    assert result is not None
    with pytest.raises(ValueError):fit_whole_camera_track(rows,descriptor(),descriptor(),duration=37/29.97,sample_times=(0.,.5,.9))


def embeddings(*ids):
    return dict(heads=[dict(observation_id=i,embedding=[1. if n==i else 0. for n in range(128)]) for i in ids])


def test_one_to_one_hypotheses_preserve_unmatched_and_order():
    a=embeddings(0,1,2);b=embeddings(1,0)
    before=deepcopy((a,b));result=one_to_one_appearance_hypotheses(a,b,objective=ReviewObjective())
    assert result['unique_review_hypotheses']==[(0,0),(1,1)]
    assert result['unmatched_source_head_ids']==[2] and not result['actor_identity_certified']
    b['heads'].reverse()
    assert one_to_one_appearance_hypotheses(a,b,objective=ReviewObjective())==result
    assert before[0]==a


def test_same_pose_or_identical_appearance_does_not_resolve_two_people():
    a=embeddings(0,1);b=embeddings(0,1);b['heads'][1]['embedding']=b['heads'][0]['embedding']
    result=one_to_one_appearance_hypotheses(a,b,objective=ReviewObjective())
    assert result['unique_review_hypotheses']==[] and result['unmatched_source_head_ids']==[0,1]
    assert cosine(None,b['heads'][0]['embedding']) is None


def test_same_objective_and_full_interior_fail_survives_parent_final():
    base=dict(objective=ReviewObjective(),descriptors=[d(),None,d()],pixel_features=[None]*3,
        left=d(),right=d(),support_verdicts=[dict(state='PASS'),dict(state='UNKNOWN'),dict(state='PASS')],
        association=dict(ambiguity_risk=1.))
    results=[evaluate_review_program(**base,authority=name) for name in ('selected_fresh','parent_fresh','final_fresh')]
    assert len({r['objective_sha256'] for r in results})==1
    assert all(not r['eligible_review'] for r in results)


def test_objective_rejects_nan_and_unbounded_shortlist():
    with pytest.raises(ValueError):ReviewObjective(readability_weight=float('nan'))
    with pytest.raises(ValueError):ReviewObjective(maximum_shortlist_sources=3)


def test_bound_duty_snapshot_matches_raw_verdict_without_rehashing_all_rows(monkeypatch):
    raw=token();bound=bind_review_part_duties(raw);row=face();row['output_tick']=1
    expected=consume_frozen_part_duties(raw,row,candidate_head_ids=(0,),key=EditorialTransformKeyframe(0.,1.))
    def no_rehash(value):raise AssertionError('full duty hash repeated per tick')
    monkeypatch.setattr('crosscut.group_review_program_evaluator.digest',no_rehash)
    for _ in range(20):
        actual=consume_frozen_part_duties(bound,row,candidate_head_ids=(0,),key=EditorialTransformKeyframe(0.,1.))
        assert actual==expected


def test_bound_duties_cannot_be_weakened_by_mutating_original_or_returned_row():
    raw=token({'left_arm':None});bound=bind_review_part_duties(raw)
    raw['rows']['1']['parts'].clear();bound.row(1)['parts'].clear()
    assert bound.row(1)['parts']=={'left_arm':None}
    assert bound.row(0) is None and bound.row(2) is None


def test_bound_duty_snapshot_rejects_output_gaps_and_wrong_contract():
    raw=token();raw['rows']['3']=deepcopy(raw['rows']['1'])
    with pytest.raises(ValueError,match='contiguous'):bind_review_part_duties(raw)
    raw=token();raw['contract']='borrowed'
    with pytest.raises(ValueError,match='frozen contract'):bind_review_part_duties(raw)
