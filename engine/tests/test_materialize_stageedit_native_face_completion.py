from dataclasses import replace
import pytest
from test_stage_edit_native_face_completion import fixture
from crosscut.stage_edit_native_face_completion import augment_missing_native_faces
from crosscut.stage_edit_face_cues import schema_with_face_cues
from crosscut.stage_edit_feature_materializer import StageEditFeatureSnapshot
from materialize_stageedit_native_face_completion import require_input_only_change


def test_full_candidate_census_keeps_clock_context_source_pose_and_gate_unchanged():
    old,face,raw,report=fixture()
    new=augment_missing_native_faces(old,face,report,raw)
    assert require_input_only_change(old,new,schema_with_face_cues())==1


@pytest.mark.parametrize('kind',['clock','gate','unrelated_feature','target_center'])
def test_census_rejects_unrelated_or_target_member_influence(kind):
    old,face,raw,report=fixture();schema=schema_with_face_cues()
    new=augment_missing_native_faces(old,face,report,raw)
    token=new.sequence.tokens[0];c=token.source_candidates[0]
    if kind=='clock':token=replace(token,master_time_seconds=.001)
    elif kind=='gate':token=replace(token,source_candidates=(replace(c,hard_gate_passed=False),))
    else:
        values=list(c.features);masks=list(c.missing_mask)
        j=0 if kind=='unrelated_feature' else schema.source_feature_names.index('target_center_x')
        values[j]=.3;masks[j]=False
        token=replace(token,source_candidates=(replace(c,features=tuple(values),missing_mask=tuple(masks)),))
    if kind=='target_center':
        with pytest.raises(ValueError,match='group-no-target'):
            StageEditFeatureSnapshot.build(replace(new.sequence,tokens=(token,)),schema=schema,
                provenance=new.provenance,causal_sensing_binding=new.causal_sensing_binding)
        return
    broken=StageEditFeatureSnapshot.build(replace(new.sequence,tokens=(token,)),schema=schema,
        provenance=new.provenance,causal_sensing_binding=new.causal_sensing_binding)
    with pytest.raises(ValueError):require_input_only_change(old,broken,schema)
