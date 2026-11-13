"""Refreshing cached identity scalars must not move source pixels or labels."""
from dataclasses import replace

import pytest

from tests.test_ive_performer_state import _solo_inputs, _provenance, _mass
from crosscut.ive import IVEMember
from crosscut.ive_performer_state import (
    IVEPerformerStateArtifact, LEGACY_CENTER_DISTRIBUTION,
    STAGE_EDIT_IVE_PERFORMER_STATE_FEATURE_NAMES, adapt_ive_performer_state_for_stage_edit,
    build_ive_performer_state_artifact, _artifact_unsigned_payload, _artifact_content_sha256,
)
from crosscut.stage_edit_feature_materializer import (
    StageEditFeatureSnapshot, StageEditCausalSensingBinding, StageEditFeatureProvenance,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA as SCHEMA, StageEditSequence, StageEditToken, SourceCandidateToken,
)


def fixture():
    graph, grammar, evidence = _solo_inputs()
    current = build_ive_performer_state_artifact(graph, grammar, performer_evidence=evidence, provenance=_provenance())
    old_record = replace(current.records[0], center_member=_mass(IVEMember.GAEUL, known=.1, unknown=.9),
                         center_member_known_mass=.1)
    unsigned = _artifact_unsigned_payload(records=(old_record,), provenance=current.provenance,
                                         input_binding=current.input_binding, center_distribution=LEGACY_CENTER_DISTRIBUTION)
    previous = IVEPerformerStateArtifact((old_record,), current.provenance, current.input_binding,
        _artifact_content_sha256(unsigned), center_distribution_contract=LEGACY_CENTER_DISTRIBUTION)
    profile = adapt_ive_performer_state_for_stage_edit(old_record)
    values, mask = [0.] * SCHEMA.raw_source_dimension, [True] * SCHEMA.raw_source_dimension
    for name, value, missing in zip(profile.feature_names, profile.values, profile.missing_mask, strict=True):
        index = SCHEMA.source_feature_names.index(name)
        values[index], mask[index] = value, missing
    candidate = SourceCandidateToken('source-a', tuple(values), tuple(mask))
    token = StageEditToken(0., (0.,) * len(SCHEMA.music_feature_names), (0.,) * len(SCHEMA.section_feature_names),
                          (candidate,), (0.,) * len(SCHEMA.recent_edit_feature_names), (0.,) * len(SCHEMA.style_feature_names),
                          missing_mask=(True,) * SCHEMA.raw_context_dimension)
    binding = StageEditCausalSensingBinding(shooting_grammar_content_sha256='1' * 64,
        member_state_graph_content_sha256='2' * 64, ive_performer_state_content_sha256=previous.content_sha256)
    provenance = StageEditFeatureProvenance(upstream_sha256=(('causal_shooting_grammar', '1' * 64),
        ('member_state_graph', '2' * 64), ('ive_performer_state', previous.content_sha256)))
    snapshot = StageEditFeatureSnapshot.build(StageEditSequence('numeric-source-cache', (token,)),
                                             provenance=provenance, causal_sensing_binding=binding)
    return snapshot, previous, current


def test_refresh_changes_only_performer_scalars_and_hashes_not_clock_pose_or_other_features():
    snapshot, previous, current = fixture()
    original_payload = snapshot.to_dict()
    refreshed = snapshot.refresh_ive_performer_state(previous, current)
    assert snapshot.to_dict() == original_payload
    assert refreshed.snapshot_sha256 != snapshot.snapshot_sha256
    assert refreshed.causal_sensing_binding.ive_performer_state_content_sha256 == current.content_sha256
    a, b = snapshot.sequence.tokens[0], refreshed.sequence.tokens[0]
    assert a.master_time_seconds == b.master_time_seconds
    assert a.music_features == b.music_features and a.section_features == b.section_features
    ac, bc = a.source_candidates[0], b.source_candidates[0]
    assert ac.source_id == bc.source_id and ac.temporal_pose_sidecar is bc.temporal_pose_sidecar
    for index, name in enumerate(SCHEMA.source_feature_names):
        if name not in STAGE_EDIT_IVE_PERFORMER_STATE_FEATURE_NAMES:
            assert ac.features[index] == bc.features[index]
            assert ac.missing_mask[index] == bc.missing_mask[index]
    index = SCHEMA.source_feature_names.index('ive_center_member_known_mass')
    assert ac.features[index] == .1 and bc.features[index] > .9
    assert StageEditFeatureSnapshot.from_dict(refreshed.to_dict()).snapshot_sha256 == refreshed.snapshot_sha256


def test_refresh_rejects_wrong_previous_artifact_and_different_causal_input_binding():
    snapshot, previous, current = fixture()
    with pytest.raises(ValueError, match='bound causal inputs'):
        snapshot.refresh_ive_performer_state(current, previous)
    binding = replace(current.input_binding, member_state_graph_sha256='9' * 64)
    unsigned = _artifact_unsigned_payload(records=current.records, provenance=current.provenance, input_binding=binding)
    foreign = IVEPerformerStateArtifact(current.records, current.provenance, binding, _artifact_content_sha256(unsigned))
    with pytest.raises(ValueError, match='bound causal inputs'):
        snapshot.refresh_ive_performer_state(previous, foreign)
