from copy import deepcopy
from dataclasses import replace
import json
import pytest
from types import SimpleNamespace
from crosscut.group_performer_family_owner import digest
from crosscut.group_review_realized_path import RealizedReviewVariant,validate_restored_review_variant


def fixture():
    raw=dict(request_source_index=2,native_frame_index=100,source_file_identity=[1,2,3,4,5])
    view=dict(row=dict(native_observation=raw),descriptor=SimpleNamespace(frame_local_head_id=7),
        near=1.,detail_risk=.1)
    selected={0:view}
    bindings=[dict(output_tick=0,state='HEAD_OCCURRENCE',head_id=7,native_frame=100)]
    payload=dict(camera_identity='camera',sampler_identity='sampler',
        actor_path_identity=digest([(0,7,digest(raw))]),source_identity=[1,2,3,4,5],
        source=2,phase_frames=0,keep=False,bindings=bindings,near_fraction=1.,actor_path=None,quality_approved=False)
    v=RealizedReviewVariant(0,10,json.dumps(payload),10/30*(4.5-.45*.1),10)
    meta=dict(bindings=deepcopy(bindings),source=2,phase=0,keep=False,phase_scope=[0,16],validated_plan_fps=30.,
        evaluation_identity=v.evaluation_identity)
    return v,meta,selected


def validate(v,meta,selected):
    return validate_restored_review_variant(v,meta,selected,camera_identity='camera',sampler_identity='sampler',
        expected_source_identity=[1,2,3,4,5],output_frame_count=100)


def test_realized_cache_recomputes_actual_actor_and_evaluation():
    v,m,s=fixture();report=validate(v,m,s)
    assert report['actual_restored_actor_verified']


def test_model_metadata_head_substitution_is_rejected_with_unchanged_physical_hash():
    v,m,s=fixture();m['bindings'][0]['head_id']=8
    with pytest.raises(ValueError,match='metadata disagrees'):validate(v,m,s)


def test_same_native_other_head_or_raw_parent_cannot_restore_as_sealed_actor():
    v,m,s=fixture();s[0]['descriptor']=SimpleNamespace(frame_local_head_id=8)
    with pytest.raises(ValueError,match='occurrence/native'):validate(v,m,s)
    v,m,s=fixture();s[0]['row']['native_observation']['changed_raw']=True
    with pytest.raises(ValueError,match='actual actor/raw'):validate(v,m,s)


@pytest.mark.parametrize('field,value',[('utility',999.),('changed_frames',0)])
def test_selection_score_or_budget_mutation_is_not_hidden_by_physical_identity(field,value):
    v,m,s=fixture();changed=replace(v,**{field:value})
    assert changed.identity==v.identity
    with pytest.raises(ValueError,match='utility or changed-frame'):validate(changed,m,s)


def test_phase_scope_and_source_metadata_must_match_sealed_sampler():
    for key,value in [('phase',1),('phase_scope',[0,20]),('source',3),('keep',True)]:
        v,m,s=fixture();m[key]=value
        with pytest.raises(ValueError,match='metadata disagrees'):validate(v,m,s)
