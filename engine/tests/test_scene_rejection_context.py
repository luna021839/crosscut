"""Rejections must survive fallback without claiming real media validation."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from crosscut import group_performer_family_owner as family
from crosscut import group_scene_source_owner as scene
from crosscut import group_source_changing_owner as source
from crosscut.group_joint_source_camera import JointGroupBudgetExhausted


@pytest.mark.parametrize('cause', ['no_audit', 'pixel_budget', 'asset'])
def test_scene_rejection_archive_keeps_exact_typed_cause(cause):
    context = NS(source_derivative_binding=None if cause == 'no_audit' else {},
                 domain=NS(eligible_indices=tuple(range(8))))
    error = (JointGroupBudgetExhausted('joint GROUP pose_evidence_bytes work budget reached')
             if cause == 'pixel_budget' else OSError('native sensor asset unavailable'))
    with patch.object(family, '_fresh'), patch.object(source, 'installed_native_assets', side_effect=error):
        result = scene.propose_scene_changes(context)
    assert isinstance(result, family.PerformerFamilyRejection)
    assert result.evidence['error'] == result.error
    assert result.evidence['error_type'] == ('ValueError' if cause == 'no_audit' else type(error).__name__)
    assert result.evidence['quality_approved'] is False


@dataclass(frozen=True)
class _Plan:
    segments: tuple = ()


@dataclass(frozen=True)
class _Bundle:
    render_plan: _Plan = _Plan()
    boundary_look_match: object = None
    phrase_times: tuple = ()


def test_normal_fallback_retains_scene_cause_without_new_sensing():
    context = NS(incumbent=_Bundle(), domain=NS(eligible_indices=(0,),
        audio_offsets=(0.,), source_timewarps=(None,), scene_cut_times=((),)),
        model_path=Path('explicit-no-model.bin'), source_derivative_binding={})
    with patch.object(family, '_fresh'), patch.object(source, 'installed_native_assets',
            side_effect=JointGroupBudgetExhausted('joint GROUP pose_evidence_bytes work budget reached')):
        first = scene.propose_scene_changes(context)
    with patch.object(family, '_fresh'), patch.object(scene, 'propose_scene_changes', return_value=first), \
            patch.object(source, 'propose_source_changes', return_value=family.PerformerFamilyRejection(
                'legacy source proposal unavailable', {})), \
            patch('crosscut.pipeline._review_first_group_motion_camera_bundle', return_value=None):
        result = family.propose_performer_family.__wrapped__(context)
    cause = result.evidence['source_search']['scene_search']
    assert cause['error'] == first.error
    assert cause['error_type'] == 'JointGroupBudgetExhausted'
    assert cause['quality_approved'] is False
