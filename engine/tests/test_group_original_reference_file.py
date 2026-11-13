"""Complete request/metadata transport. Numeric fixtures are not media QA."""
from dataclasses import replace
from hashlib import sha256
import json

import pytest

from crosscut.group_external_original_reference import GroupExternalOriginalReference
from crosscut.group_original_reference_file import (original_reference_file_payload,
    write_original_reference_file, load_original_reference_file, resolve_request_original_reference,
    _canonical)
from crosscut.request import AnalysisRequest, GroupOriginalReferenceInput
from crosscut.ive import EditMode, SourceMode
from tests.test_group_native_motion_bundle import synthetic_bundle


@pytest.fixture
def reference(tmp_path):
    old, domain = synthetic_bundle()
    sources = tuple(tmp_path / f'numeric-unobserved-{i}.mp4' for i in range(8))
    for source in sources:
        source.write_bytes(b'NUMERIC_IDENTITY_FIXTURE_NOT_A_VIDEO')
    old = replace(old, render_plan=replace(old.render_plan, sources=sources,
        output=tmp_path / 'never-rendered.mp4'), candidate_origin='ordinary_visual_baseline',
        boundary_look_match=replace(old.boundary_look_match, enabled=True, blocked_reason=None))
    domain = replace(domain, request_sources=sources)
    return GroupExternalOriginalReference(old, domain, ((0, 30), (90, 120)))


def saved_input(reference, tmp_path):
    path = tmp_path / 'complete-original.json'
    content_sha = write_original_reference_file(reference, path, song='XOXZ')
    return GroupOriginalReferenceInput(path, content_sha)


def request_for(reference, descriptor, tmp_path, **changes):
    payload = dict(song='XOXZ', edit_mode='group', source_mode='automatic', sources=[],
        output_directory=str(tmp_path / 'output'), group_external_original_reference=descriptor.as_dict())
    return AnalysisRequest.from_dict({**payload, **changes})


def test_complete_original_round_trip_preserves_every_family_and_protected_ticks(reference, tmp_path):
    descriptor = saved_input(reference, tmp_path)
    loaded = load_original_reference_file(descriptor.file, expected_sha256=descriptor.sha256,
        song='XOXZ', request_sources=reference.source_domain.request_sources)
    assert loaded.original_bundle == reference.original_bundle
    assert loaded.source_domain == reference.source_domain
    assert loaded.to_dict() == reference.to_dict()
    assert loaded.protected_output_intervals == ((0, 30), (90, 120))
    for field in ('face_match_plan', 'boundary_repairs', 'editorial_motion_plan', 'boundary_look_match',
            'section_plan', 'candidate_lineage', 'stage_edit_exact_shot_clock', 'dynamic_crop_application'):
        assert getattr(loaded.original_bundle, field) == getattr(reference.original_bundle, field)
    assert loaded.to_dict()['quality_approved'] is False


def test_normal_request_materialization_and_analysis_entry_own_the_complete_original(reference, tmp_path, monkeypatch):
    import crosscut.pipeline as pipeline
    descriptor = saved_input(reference, tmp_path)
    request = request_for(reference, descriptor, tmp_path)
    assert request.group_external_original_reference == descriptor
    assert AnalysisRequest.from_dict(request.as_dict()) == request
    request = request.with_materialized_sources(reference.source_domain.request_sources)
    assert request.group_external_original_reference == descriptor

    class ReachedNormalAnalysisPort(Exception):
        pass

    def inspect_port(actual_request, reporter, **kwargs):
        original = kwargs['group_external_original_reference']
        assert original.original_bundle == reference.original_bundle
        assert original.source_domain.eligible_indices == (0, 2, 3, 5)
        assert actual_request.source_mode is SourceMode.AUTOMATIC
        assert original.protected_output_intervals == reference.protected_output_intervals
        original.validate_baseline(original.original_bundle, original.original_bundle.render_plan)
        raise ReachedNormalAnalysisPort

    # This spy checks the real public request handoff, not rendering or NN.
    monkeypatch.setattr(pipeline, '_analyze_impl', inspect_port)
    with pytest.raises(ReachedNormalAnalysisPort):
        pipeline.analyze(request, object())


def test_nonzero_audio_offsets_are_not_replaced_by_numeric_replay_zeroes(reference, tmp_path):
    from crosscut.timewarp import TimeWarpControlPoint, TimeWarpConfig, _scalar_fallback
    warp = replace(_scalar_fallback(global_offset_seconds=.1, master_duration_seconds=10.,
        source_duration_seconds=11., anchors=(), config=TimeWarpConfig(),
        reason='NUMERIC_TRANSPORT_ONLY'), scalar_fallback=False,
        fallback_reason=None, control_points=(TimeWarpControlPoint(0., .1), TimeWarpControlPoint(10., 10.1)))
    domain = replace(reference.source_domain, audio_offsets=(.1, .2, .3, .4),
        source_timewarps=(warp, None, None, None))
    # This test verifies the complete state codec, independently of a render
    # whose scalar source requests were authored on the original zero domain.
    shifted = GroupExternalOriginalReference(reference.original_bundle, domain,
        reference.protected_output_intervals)
    payload = original_reference_file_payload(shifted, song='XOXZ')
    from crosscut.group_original_reference_file import _state_codec
    actual = _state_codec(decode=True)(payload['complete_source_domain_state'])
    assert actual == domain
    assert actual.audio_offsets != (0.,)*4
    assert actual.source_timewarps[0].control_points[1].source_seconds == 10.1


def test_new_output_only_is_a_relocation_not_a_new_original(reference, tmp_path):
    descriptor = saved_input(reference, tmp_path)
    original = load_original_reference_file(descriptor.file, expected_sha256=descriptor.sha256,
        song='XOXZ', request_sources=reference.source_domain.request_sources)
    moved = original.baseline_for_output(tmp_path / 'new-output.mp4')
    original.validate_baseline(moved, moved.render_plan)
    assert moved.to_dict() == original.original_bundle.to_dict()


def test_complete_variable_programme_owner_survives_file_request_without_becoming_neutral(reference, tmp_path):
    from crosscut.group_native_motion_metadata import complete_native_motion_metadata
    from crosscut.group_joint_native_preparation import _plan_identity
    from tests.test_group_native_motion_metadata import variable_programme_complete
    context, selected, metadata, domain = variable_programme_complete()
    sources = reference.source_domain.request_sources
    context.incumbent = replace(context.incumbent,
        render_plan=replace(context.incumbent.render_plan, sources=sources,
            output=tmp_path / 'variable-programme.mp4'))
    domain = replace(domain, request_sources=sources)
    selected = replace(selected, render_plan=replace(selected.render_plan, sources=sources,
        output=context.incumbent.render_plan.output))
    evidence = json.loads(metadata.evidence_json)
    evidence.update(original_plan_identity=_plan_identity(context.incumbent.render_plan),
        selected_plan_identity=_plan_identity(selected.render_plan))
    metadata = replace(metadata, evidence_json=json.dumps(evidence))
    bundle = complete_native_motion_metadata(context=context, selected=selected,
        metadata=metadata, domain=domain)
    original = GroupExternalOriginalReference(bundle, domain, ())
    descriptor = saved_input(original, tmp_path)
    restored = load_original_reference_file(descriptor.file, expected_sha256=descriptor.sha256,
        song='XOXZ', request_sources=sources)
    actual = restored.original_bundle
    assert actual.to_dict() == bundle.to_dict()
    assert actual.native_motion_production == bundle.native_motion_production
    assert actual.native_motion_production.selection_scope == 'variable_programme'
    assert len(actual.native_motion_production.selected_shots) == 3
    assert len(actual.visual_timeline.choices) == 9
    assert actual.performer_family_review is not None
    assert all(frame.frame_local_head_id is None for shot in actual.native_motion_production.selected_shots
        for frame in shot.actor_frames)
    assert actual.native_motion_production.to_dict()['quality_approved'] is False


@pytest.mark.parametrize('changes', [dict(review_first_enabled=False), dict(acquire_only=True),
    dict(edit_mode='member', target_member='가을', aspect_ratio='9:16')])
def test_reference_request_cannot_silently_enter_member_or_acquisition_only(reference, tmp_path, changes):
    descriptor = saved_input(reference, tmp_path)
    with pytest.raises(ValueError, match='review-first GROUP render'):
        request_for(reference, descriptor, tmp_path, **changes)


@pytest.mark.parametrize('value', [None, True, {}, {'file': 'missing'},
    {'file': 'missing', 'sha256': 'a'*64, 'protected': []}])
def test_descriptor_requires_exact_fields_not_implicit_default_authority(value):
    with pytest.raises(ValueError):
        AnalysisRequest.from_dict(dict(song='XOXZ', edit_mode='group', source_mode='automatic',
            sources=[], group_external_original_reference=value))


def test_changed_file_and_changed_live_source_are_rejected_before_authoring(reference, tmp_path):
    descriptor = saved_input(reference, tmp_path)
    original_bytes = descriptor.file.read_bytes()
    descriptor.file.write_bytes(original_bytes + b' ')
    with pytest.raises(ValueError, match='file content changed'):
        load_original_reference_file(descriptor.file, expected_sha256=descriptor.sha256,
            song='XOXZ', request_sources=reference.source_domain.request_sources)
    descriptor.file.write_bytes(original_bytes)
    reference.source_domain.eligible_sources[0].write_bytes(b'CHANGED_SOURCE_IDENTITY')
    with pytest.raises(ValueError, match='policy, source identity or full metadata changed'):
        load_original_reference_file(descriptor.file, expected_sha256=descriptor.sha256,
            song='XOXZ', request_sources=reference.source_domain.request_sources)


def test_source_order_song_and_competing_explicit_original_are_not_ignored(reference, tmp_path):
    descriptor = saved_input(reference, tmp_path)
    for sources, song in ((tuple(reversed(reference.source_domain.request_sources)), 'XOXZ'),
            (reference.source_domain.request_sources, 'OTHER SONG')):
        with pytest.raises(ValueError):
            load_original_reference_file(descriptor.file, expected_sha256=descriptor.sha256,
                song=song, request_sources=sources)
    request = request_for(reference, descriptor, tmp_path).with_materialized_sources(
        reference.source_domain.request_sources)
    other = GroupExternalOriginalReference(reference.original_bundle, reference.source_domain, ((0, 29),))
    with pytest.raises(ValueError, match='differs from the saved request'):
        resolve_request_original_reference(request, other)


@pytest.mark.parametrize('mutation', ['drop_family', 'foreign_type', 'wrong_declared_type', 'quality'])
def test_self_consistent_inner_changes_do_not_become_complete_authority(reference, tmp_path, mutation):
    payload = original_reference_file_payload(reference, song='XOXZ')
    if mutation == 'drop_family':
        payload['complete_bundle_state']['fields'].pop('face_match_plan')
    elif mutation == 'foreign_type':
        payload['complete_bundle_state']['$type'] = 'subprocess.Popen'
    elif mutation == 'wrong_declared_type':
        payload['complete_bundle_state']['fields']['candidate_index'] = True
    else:
        payload['professional_quality_approved'] = True
    path = tmp_path / 'changed-inner-original.json'
    data = _canonical(payload).encode()
    path.write_bytes(data)
    with pytest.raises(ValueError):
        load_original_reference_file(path, expected_sha256=sha256(data).hexdigest(),
            song='XOXZ', request_sources=reference.source_domain.request_sources)


def test_numeric_render_plan_alone_is_not_a_complete_original(reference, tmp_path):
    path = tmp_path / 'numeric-only.json'
    data = json.dumps(reference.original_bundle.render_plan.to_json()).encode()
    path.write_bytes(data)
    with pytest.raises(ValueError, match='transport contract'):
        load_original_reference_file(path, expected_sha256=sha256(data).hexdigest(),
            song='XOXZ', request_sources=reference.source_domain.request_sources)


def test_existing_snapshot_is_never_overwritten(reference, tmp_path):
    descriptor = saved_input(reference, tmp_path)
    data = descriptor.file.read_bytes()
    with pytest.raises(FileExistsError):
        write_original_reference_file(reference, descriptor.file, song='XOXZ')
    assert descriptor.file.read_bytes() == data


def test_legacy_requests_do_not_add_or_activate_an_external_original():
    request = AnalysisRequest.from_dict(dict(song='XOXZ', edit_mode='group',
        source_mode='automatic', sources=[]))
    assert 'group_external_original_reference' not in request.as_dict()
    assert resolve_request_original_reference(request) is None
