"""Normal, fixed-clock performer camera + retained family transaction.

The existing projected pose/head proposer owns camera selection. This owner
rebuilds affected descriptors and look, keeps source-space face recipes only
under an unchanged source/clock/base-crop preimage, and publishes ONE complete
bundle after a fresh independent worker replay. It is NOT a source-changing
owner, identity certificate, or final editorial quality approval.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Any, Mapping

from .group_joint_observations import GroupJointObservationSession, _file_identity
from .group_joint_source_camera import JointGroupWorkBudget
from .group_source_domain import GroupSourceDomain
from .optional_native_worker import current_optional_deadline, isolated_optional, optional_native_scope
from .render import segment_frame_counts
from .render_boundary_clock import native_frame_clocks_for_output, hard_cut_boundary_sample_clocks
from .render_frame_schedule import probe_render_frame_clock
from .prescribed_render_geometry import PrescribedGeometrySampler
from .post_render_perceptual_quality import _reference_view_for_span
from .visual_boundary_repair import _thaw_audit_value

CONTRACT = 'normal_performer_family_camera_owner_v1'
SOURCE_CONTRACT = 'normal_source_changing_family_owner_v1'
FAMILY_CONTRACTS = (CONTRACT, SOURCE_CONTRACT)
QUALITY_CODE = 'performer_family_quality_unapproved'
MAX_OUTPUT_FRAMES = 36000
MAX_REFRESHED_CELLS = 4
MAX_NATIVE_READS = 160
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
LOOK_FAMILY_BINDING = 'retained_original_boundary_look_family_v1'
UNRELATED_TRAINING_ENTRYPOINTS = frozenset({
    'stage_edit_dynamic_crop_capacity_probe.py',
    'stage_edit_release_fold_launcher.py',
})


def plain(value):
    from .group_scene_native_storage import expand_native_evidence
    return expand_native_evidence(_thaw_audit_value(value))


def digest(value):
    return sha256(json.dumps(plain(value), sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def plan_payload(plan):
    value = dict(plan.to_json() if hasattr(plan, 'to_json') else plan)
    value.pop('output', None)  # Immutable final-QA snapshot relocates only output.
    return value


def metadata_payload(bundle):
    value = bundle.to_dict()
    value.pop('performer_family_review', None)
    return value


def boundary_look_inventory(look, plan):
    """Bind actual metadata AND renderer tracks; absence is not a disabled fit.

    Static source looks remain independent and are also bound, not cleared.
    An absent metadata object with a live residual track is corruption, never
    permission to silently erase the renderer's existing color recipe.
    """
    rows=plan_payload(plan)['segments']
    tracks=[row.get('boundary_look_track') for row in rows]
    metadata=None if look is None else plain(look.to_dict() if hasattr(look,'to_dict') else look)
    if metadata is None:
        if any(track is not None for track in tracks):
            raise ValueError('original boundary look metadata absent but renderer tracks remain')
        mode='ORIGINAL_ABSENT'
    else:
        if (not isinstance(metadata,dict) or metadata.get('enabled') is not True
                or digest(metadata.get('segment_tracks'))!=digest(tracks)):
            raise ValueError('retained look family is disabled or not the actual renderer track inventory')
        mode='ENABLED'
    return dict(mode=mode,segment_count=len(rows),metadata_sha256=digest(metadata),
        renderer_tracks_sha256=digest(tracks),
        static_source_looks_sha256=digest([row.get('look_adjustment') for row in rows]))


def retained_applied_look_edges(bundle):
    # Numeric-only callers can still carry a declared decision inventory. The
    # native owner independently validates its complete family at freeze and
    # completion; None must already agree with every actual renderer track.
    if bundle.boundary_look_match is None:
        boundary_look_inventory(None,bundle.render_plan)
        return ()
    return tuple(sorted(d.boundary_index for d in bundle.boundary_look_match.decisions if d.applied))


def _bind_look_refresh(old,plan,look,evidence):
    return dict(evidence,retained_family=dict(contract=LOOK_FAMILY_BINDING,
        original=boundary_look_inventory(old.boundary_look_match,old.render_plan),
        final=boundary_look_inventory(look,plan),
        original_plan_sha256=digest(plan_payload(old.render_plan)),
        original_metadata_sha256=digest(metadata_payload(old)),
        final_plan_sha256=digest(plan_payload(plan)),quality_approved=False))


def validate_look_refresh(plan,look,receipt,*,original_bundle=None):
    """Consumer checks complete color-family mode, not a caller's absent flag."""
    audit=plain(receipt.get('look_refresh'));binding=audit.get('retained_family') if isinstance(audit,dict) else None
    final=boundary_look_inventory(look,plan)
    original=binding.get('original') if isinstance(binding,dict) else None
    if (not isinstance(original,dict) or binding.get('contract')!=LOOK_FAMILY_BINDING
            or binding.get('quality_approved') is not False or binding.get('final')!=final
            or original.get('mode')!=final['mode'] or original.get('segment_count')!=final['segment_count']
            or binding.get('original_plan_sha256')!=receipt.get('original_plan_sha256')
            or binding.get('original_metadata_sha256')!=receipt.get('original_metadata_sha256')
            or binding.get('final_plan_sha256')!=digest(plan_payload(plan))):
        raise ValueError('retained original boundary look family binding changed')
    if final['mode']=='ORIGINAL_ABSENT':
        if (original.get('metadata_sha256')!=digest(None)
                or original.get('renderer_tracks_sha256')!=digest([None]*len(plan.segments))
                or audit.get('refreshed_edges')!=[] or audit.get('projected_samples')!=[]
                or audit.get('matched_production_renderer_used') is not False
                or audit.get('quality_approved') is not False
                or audit.get('incident_edges_without_boundary_look')!=receipt.get('incident_edges')
                or 'proposed' in audit or 'verified' in audit):
            raise ValueError('originally absent boundary look acquired a fabricated fit/render proof')
    elif audit.get('matched_production_renderer_used') is not True:
        raise ValueError('enabled boundary look lost its actual matched-render refresh')
    if original_bundle is not None and (original!=boundary_look_inventory(
            original_bundle.boundary_look_match,original_bundle.render_plan)
            or binding['original_metadata_sha256']!=digest(metadata_payload(original_bundle))
            or binding['original_plan_sha256']!=digest(plan_payload(original_bundle.render_plan))):
        raise ValueError('retained boundary look differs from external original authority')


def _stable(value):
    """Execution time is diagnostic, never an authoritative sensor value."""
    if isinstance(value, Mapping):
        return {k: _stable(v) for k, v in value.items() if k not in
                {'elapsed_seconds', 'wall_seconds', 'worker_pid'}}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    return value


def _file_hash(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dependencies(model_path=None):
    # Pin the complete shipped Python sensor/renderer/consumer code, not just
    # a digest of the owner's wrapper. No directory outside this package read.
    root = Path(__file__).parent
    values = {str(p.relative_to(root)): _file_hash(p)
        for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts
        and p.name not in UNRELATED_TRAINING_ENTRYPOINTS}
    # Training/release launchers are not dependencies of this render owner.
    # Keep the exclusion explicit; do not read out-of-scope entry points just
    # to hash the package. All actual sensors and consumers remain bound.
    values['@unrelated_training_entrypoints_excluded'] = digest(sorted(UNRELATED_TRAINING_ENTRYPOINTS))
    # Also bind installed sensor assets, including ABSENCE. A newly appearing
    # detector or modified secondary weight cannot silently change the replay.
    for label, directory in (('@package_models', root/'models'),
            ('@pose_directory', Path(model_path).parent if model_path is not None else root/'models')):
        values[label] = digest({str(p.relative_to(directory)): _file_hash(p)
            for p in sorted(directory.rglob('*')) if p.is_file()
            and p.suffix.lower() in {'.onnx','.pth','.xml','.pt'}}) if directory.is_dir() else 'missing'
    import cv2
    import numpy
    values['@runtime_versions'] = digest({'opencv':cv2.__version__,'numpy':numpy.__version__})
    return values


def installed_pose_model(cache_directory):
    """Resolve the installed pinned asset only; this owner NEVER downloads."""
    from .body_pose_2d import RTMPOSE_BODY7_MODEL_SHA256
    if cache_directory is None:
        raise ValueError('performer owner requires an already installed pinned pose model')
    path = Path(cache_directory) / 'models' / 'source-first' / 'rtmpose_body7.onnx'
    if not path.is_file() or _file_hash(path) != RTMPOSE_BODY7_MODEL_SHA256:
        raise ValueError('installed performer pose model is missing or has the wrong digest; no download attempted')
    return path


@dataclass(frozen=True, slots=True)
class PerformerFamilyContext:
    incumbent: Any
    domain: GroupSourceDomain
    source_identities: tuple
    code_dependencies: tuple
    model_path: Path
    model_identity: tuple
    model_sha256: str
    token: Any
    deadline: float
    sync_scores: tuple[float, ...]
    cache_directory: Path | None = None
    derived_cache_directory: Path | None = None
    source_derivative_binding: Any = None
    external_original_reference: Any = None


def _supported(bundle):
    # No suppression of face/alignment/look. These are precisely the supported
    # retained families. Other authority is either preserved by its own binder
    # or explicitly declined; it is never moved to an unvalidated alias.
    unsupported = [name for name in ('normal_interval_review', 'performer_family_review',
        'dynamic_crop_application', 'output_geometry_deferral', 'splice_graph_authorization',
        'stage_edit_dynamic_crop_version', 'dynamic_crop_motion_timeline_v3', 'cut_local_sync')
        if getattr(bundle, name, None) is not None]
    if (bundle.render_plan.authorized_shot_video_clocks
            or any(t.kind.uses_overlap for t in bundle.render_plan.transitions)):
        unsupported.append('local_clock_or_overlap_needs_its_own_refresh_owner')
    if bundle.editorial_motion_plan is None or bundle.editorial_motion_plan.boundary_subject_alignment_plan is None:
        unsupported.append('existing_complete_subject_alignment_required')
    if bundle.face_match_plan is None:
        unsupported.append('existing_source_face_recipe_required')
    if unsupported:
        raise ValueError('performer family owner unsupported retained authority: ' + ', '.join(unsupported))
    boundary_look_inventory(bundle.boundary_look_match,bundle.render_plan)
    if bundle.render_plan.aspect_ratio.value != '16:9':
        raise ValueError('performer owner supports only landscape GROUP')
    if bundle.render_plan.output_frame_count > MAX_OUTPUT_FRAMES:
        raise ValueError('performer owner numeric schedule exceeds its explicit frame budget')


def freeze_context(*, controller, request, eligible_indices, offsets, timewarps, scene_cut_times, sync_scores, source_derivatives=None):
    from .pipeline import _validate_review_first_group_bundle
    incumbent = controller.incumbent
    _supported(incumbent)
    _validate_review_first_group_bundle(incumbent, eligible_indices=eligible_indices,
        offsets=offsets, timewarps=timewarps, common_start=incumbent.render_plan.master_audio_start,
        phrase_times=incumbent.phrase_times)
    deadline = current_optional_deadline()
    if deadline is None or deadline <= monotonic():
        raise ValueError('performer family owner requires a shared transaction deadline')
    model = installed_pose_model(getattr(request, 'source_cache_directory', None))
    if (len(sync_scores) != len(eligible_indices)
            or any(type(v) not in (float,int) or not math.isfinite(v) for v in sync_scores)):
        raise ValueError('performer refresh needs caller supplied synchronization scores')
    sources = incumbent.render_plan.sources
    identities = tuple(_file_identity(sources[i]) for i in eligible_indices)
    clocks = tuple(probe_render_frame_clock(sources[i],
        timeout_seconds=max(.01, min(10., deadline-monotonic()))) for i in eligible_indices)
    if identities != tuple(_file_identity(sources[i]) for i in eligible_indices):
        raise ValueError('performer source changed during native clock preparation')
    domain = GroupSourceDomain(tuple(sources), tuple(eligible_indices), tuple(offsets),
        tuple(timewarps), tuple(tuple(v) for v in scene_cut_times), clocks)
    external = getattr(controller, 'external_original_reference', None)
    if external is not None:
        external.validate_source_domain(domain)
        external.validate_candidate(incumbent, incumbent.render_plan)
    from .group_scene_source_owner import freeze_derivatives
    derivative_binding = freeze_derivatives(source_derivatives, domain, identities)
    return PerformerFamilyContext(incumbent, domain, identities, tuple(sorted(dependencies(model).items())),
        model, _file_identity(model), _file_hash(model), controller.transaction_token(), deadline, tuple(sync_scores),
        getattr(request, 'source_cache_directory', None),
        getattr(request, 'visual_feature_derived_cache_directory', None), derivative_binding, external)


def _fresh(context):
    if monotonic() >= context.deadline:
        raise ValueError('performer family transaction deadline expired')
    if (context.source_identities != tuple(_file_identity(p) for p in context.domain.eligible_sources)
            or context.code_dependencies != tuple(sorted(dependencies(context.model_path).items()))
            or context.model_identity != _file_identity(context.model_path)
            or context.model_sha256 != _file_hash(context.model_path)):
        raise ValueError('performer family live source/model/code preimage changed')
    external = getattr(context, 'external_original_reference', None)
    if external is not None:
        external.validate_source_domain(context.domain)
        external.validate_candidate(context.incumbent, context.incumbent.render_plan)


def external_reference_binding(context, candidate):
    external = getattr(context, 'external_original_reference', None)
    if external is None:
        return None
    external.validate_source_domain(context.domain)
    external.validate_candidate(candidate, candidate.render_plan)
    return external.to_dict()


def validate_external_reference_binding(candidate, receipt, *, context=None):
    from .group_external_original_reference import validate_external_original_reference_payload
    policy = plain(receipt.get('external_original_reference'))
    if context is not None and policy != external_reference_binding(context, candidate):
        raise ValueError('child changed the external original policy before fresh parent admission')
    if policy is not None and receipt.get('external_original_selected_boundary_look_metadata') != (
            None if candidate.boundary_look_match is None else plain(candidate.boundary_look_match.to_dict())):
        raise ValueError('external policy selected look differs from the actual complete candidate')
    validate_external_original_reference_payload(policy, plan=candidate.render_plan,
        look=candidate.boundary_look_match, source_domain_payload=receipt['source_domain'],
        source_identities=receipt['source_identities'],
        live_clocks=None if context is None else context.domain.clocks_by_request)


def _schedule(plan, domain):
    rows = native_frame_clocks_for_output(plan, domain.clocks_by_request, range(plan.output_frame_count))
    if len(rows) != plan.output_frame_count or any(row.terminal_clone for row in rows.values()):
        raise ValueError('performer complete native output schedule is missing or terminal-cloned')
    return rows


def _schedule_digest(rows):
    return digest([asdict(rows[tick]) for tick in sorted(rows)])


def changed_scope(before, after, domain):
    """No source changes or synthetic cuts hidden in a metadata refresh."""
    from .group_actual_shots import visible_group_cut_inventory
    if (before.sources != after.sources or before.transitions != after.transitions
            or len(before.segments) != len(after.segments)
            or segment_frame_counts(before) != segment_frame_counts(after)
            or replace(after, segments=before.segments, output=before.output) != before):
        raise ValueError('performer owner changed source domain, global geometry, cuts or output clock')
    changed = []
    for index, (old, new) in enumerate(zip(before.segments, after.segments, strict=True)):
        # Only editorial camera and dependent look may change. In particular,
        # raw face observations remain valid ONLY with this pinned preimage.
        if replace(new, editorial_transform=old.editorial_transform,
                boundary_look_track=old.boundary_look_track) != old:
            raise ValueError('performer owner source/clock/base face crop/static look changed')
        if new.editorial_transform != old.editorial_transform:
            changed.append(index)
    if not changed or len(changed) > 2 or max(changed)-min(changed) > 1:
        raise ValueError('performer owner needs one bounded one/two-cell camera change')
    if _schedule_digest(_schedule(before, domain)) != _schedule_digest(_schedule(after, domain)):
        raise ValueError('performer camera changed exact whole-output native sampling')
    if visible_group_cut_inventory(before, domain) != visible_group_cut_inventory(after, domain):
        raise ValueError('performer camera changed an actual cut event')
    refreshed = tuple(range(max(0, min(changed)-1), min(len(before.segments), max(changed)+2)))
    if len(refreshed) > MAX_REFRESHED_CELLS:
        raise ValueError('performer family local refresh budget exceeded')
    edges = tuple(i for i in range(len(before.transitions))
        if i in changed or i+1 in changed)
    return tuple(changed), refreshed, edges


def _frame_offsets(plan):
    offsets = [0]
    for count in segment_frame_counts(plan):
        offsets.append(offsets[-1]+count)
    return offsets


def _project(session, plan, sample, *, offsets=None):
    offsets = _frame_offsets(plan) if offsets is None else offsets
    index = bisect_right(offsets, sample.output_frame_index)-1
    segment = plan.segments[index]
    raw = session.native_pixels(sample.source_index, sample.native_frame_index)
    sampler = PrescribedGeometrySampler(SimpleNamespace(), segment, master_start=segment.master_start)
    width = min(raw.shape[1], 854)
    image = _reference_view_for_span(raw, sampler.span,
        relative_time=(sample.output_frame_index-offsets[index])/plan.fps,
        size=(width, max(1, round(width*9/16))),
        render_size=(plan.width, plan.height))
    return image, dict(output_frame=sample.output_frame_index, segment_index=index,
        request_source_index=sample.source_index, native_frame=sample.native_frame_index,
        native_pixels_sha256=sha256(raw.tobytes()).hexdigest(),
        projected_pixels_sha256=sha256(image.tobytes()).hexdigest())


def observe_scope(plan, domain, refreshed, edges, model_path, session):
    """Fresh source-projected face + four-frame observed pose/head, not claims."""
    from .group_motion_camera_refinement import measured_output_face_box, measure_projected_motion
    from .closeup_gesture import CloseupGestureProvider
    offsets = _frame_offsets(plan)
    ticks = set()
    for index in refreshed:
        ticks.update((offsets[index], (offsets[index]+offsets[index+1]-1)//2, offsets[index+1]-1))
    edge_samples = {}
    for edge in edges:
        # Same-native canonical seams have no fabricated cross-camera pose.
        if plan.segments[edge].source_index == plan.segments[edge+1].source_index:
            continue
        samples = hard_cut_boundary_sample_clocks(plan, domain.clocks_by_request, boundary_index=edge)
        for first, last in ((samples[0], samples[1]), (samples[2], samples[3])):
            rate = float(domain.clocks_by_request[first.source_index].frame_rate)
            if any(first.native_frame_index/rate < cut <= last.native_frame_index/rate
                    for cut in domain.cuts_for_request(first.source_index)):
                raise ValueError('performer incident observation straddles a broadcast cut')
        edge_samples[edge] = samples
        ticks.update(s.output_frame_index for s in samples)
    schedule = native_frame_clocks_for_output(plan, domain.clocks_by_request, sorted(ticks))
    pixels, records = {}, {}
    for tick, sample in sorted(schedule.items()):
        image, row = _project(session, plan, sample, offsets=offsets)
        face = measured_output_face_box(image)
        row['projected_global_face_bbox_xywh'] = list(face) if face is not None else None
        pixels[tick], records[tick] = image, row
    provider = CloseupGestureProvider(plan.sources, model_path, verify_native_global_focal=True)
    observed_edges = []
    try:
        for edge, samples in sorted(edge_samples.items()):
            frames = tuple(pixels[s.output_frame_index] for s in samples)
            measured = measure_projected_motion(frames, samples, provider, output_fps=plan.fps)
            observed_edges.append(dict(boundary_index=edge,
                native_clock=[asdict(s) for s in samples],
                motion=measured.to_dict() if measured is not None else None,
                uncertainty_not_imputed=measured is None,
                cross_camera_identity_certified=False))
    finally:
        provider.close()
    return dict(samples=[records[t] for t in sorted(records)], edges=observed_edges,
        basis='fresh_native_pixels_projected_by_final_camera_before_look',
        quality_approved=False, decoded_output_quality_certified=False)


class ProjectedLookCache:
    """Scalar cache keyed by segment geometry AND exact native pixels.

    It deliberately does not touch raw/derived source-feature caches. Static
    source look is applied once by the existing analyzer, boundary ramps zeroed
    only in this analysis view and freshly rebuilt before a candidate exists.
    """
    def __init__(self, session, domain):
        self.session, self.domain = session, domain
        self.decoded_frame_count = self.cache_hit_count = 0
        self.rows, self.evidence = {}, []

    def measure_many_for_segment(self, plan, index, times):
        from .source_look_match import measure_frame_look
        segment = plan.segments[index]
        clock = self.domain.clocks_by_request[segment.source_index]
        adapter = PrescribedGeometrySampler(SimpleNamespace(), segment, master_start=segment.master_start)
        result = {}
        for micros in sorted({int(round(float(t)*1_000_000)) for t in times}):
            seconds = micros/1_000_000
            native = math.floor(seconds*float(clock.frame_rate)+.5)
            key = (index, adapter.observation_geometry_identity, native, micros)
            if key in self.rows:
                self.cache_hit_count += 1
                result[micros] = self.rows[key]
                continue
            if not 0 <= native < clock.frame_count:
                self.rows[key] = result[micros] = None
                self.evidence.append(dict(segment_index=index, microseconds=micros,
                    reason='synchronized_source_handle_unavailable'))
                continue
            raw = self.session.native_pixels(segment.source_index, native)
            relative = min(segment.duration, max(0., (seconds-segment.source_start)/segment.source_playback_rate))
            width = min(raw.shape[1], 854)
            image = _reference_view_for_span(raw, adapter.span, relative_time=relative,
                size=(width, max(1, round(width*9/16))),
                render_size=(plan.width, plan.height))
            metric = measure_frame_look(image)
            self.decoded_frame_count += 1
            self.rows[key] = result[micros] = metric
            self.evidence.append(dict(segment_index=index, request_source_index=segment.source_index,
                microseconds=micros, native_frame=native, relative_seconds=relative,
                geometry_sha256=adapter.observation_geometry_identity,
                raw_pixels_sha256=sha256(raw.tobytes()).hexdigest(),
                projected_pixels_sha256=sha256(image.tobytes()).hexdigest(),
                metric=asdict(metric)))
        return result


def _look_value(track, time):
    # Use the renderer's own evaluator, not a guessed interpolation equation.
    from .render import _boundary_look_values
    return _boundary_look_values(track, time)


def refresh_look(old_bundle, camera_plan, edges, session, domain):
    original=boundary_look_inventory(old_bundle.boundary_look_match,old_bundle.render_plan)
    if original['mode']=='ORIGINAL_ABSENT':
        # Do not fit, encode, clear static source looks, or fabricate a look
        # result just to admit an original programme which never had one.
        boundary_look_inventory(None,camera_plan)
        return camera_plan,None,_bind_look_refresh(old_bundle,camera_plan,None,dict(
            refreshed_edges=[],incident_edges_without_boundary_look=list(edges),projected_samples=[],
            original_absence_preserved=True,matched_production_renderer_used=False,quality_approved=False))
    from .boundary_look_match import (analyze_boundary_look_match,
        verify_boundary_look_match_render_domain, _tracks_from_decisions,
        _boundary_pair_render_plan, BoundaryLookRenderAdmissionConfig)
    previous = old_bundle.boundary_look_match
    cache = ProjectedLookCache(session, domain)
    clean = replace(camera_plan, segments=tuple(replace(s, boundary_look_track=None)
        for s in camera_plan.segments))
    proposed = analyze_boundary_look_match(clean, group_16x9_safe=True,
        config=previous.analysis_config, boundary_indices=edges, projected_cache=cache)
    fresh = verify_boundary_look_match_render_domain(clean, proposed,
        config=previous.render_admission_config, maximum_verified_boundaries=max(1,len(edges)),
        clipping_backoff_enabled=True)
    prior_by_edge = {d.boundary_index:d for d in previous.decisions}
    if len(prior_by_edge) != len(previous.decisions):
        raise ValueError('retained look decisions have duplicate boundary ownership')
    fresh_by_edge = {d.boundary_index:d for d in fresh.decisions}
    if set(fresh_by_edge) != set(edges):
        raise ValueError('fresh look did not cover every affected edge')
    for edge, decision in fresh_by_edge.items():
        old = prior_by_edge.get(edge)
        if old is not None and old.applied and not decision.applied:
            raise ValueError('existing active look failed fresh admission; whole candidate retained instead of stripping look')
        if decision.applied and (decision.render_verification is None or not decision.render_verification.passed):
            raise ValueError('look candidate lacks actual matched-render admission')
    merged = dict(prior_by_edge); merged.update(fresh_by_edge)
    decisions = tuple(merged[i] for i in sorted(merged))
    tracks = _tracks_from_decisions(clean, decisions, previous.analysis_config)
    result = replace(previous, decisions=decisions, segment_tracks=tracks,
        decoded_frame_count=previous.decoded_frame_count+fresh.decoded_frame_count,
        cache_hit_count=previous.cache_hit_count+fresh.cache_hit_count)
    # Verification observes four exact endpoint frames. Recomposition must be
    # identical there to the individually encoded recipe, for EVERY old/fresh
    # applied edge. Otherwise an opposite short-shot ramp invalidates the proof.
    counts = segment_frame_counts(clean)
    admission = previous.render_admission_config or BoundaryLookRenderAdmissionConfig()
    for decision in decisions:
        if not decision.applied:
            continue
        index = decision.boundary_index
        pair = _boundary_pair_render_plan(clean, decision, output=clean.output,
            adjusted=True, config=previous.analysis_config, admission=admission)
        for cell, local, ticks in ((index,0,(counts[index]-2,counts[index]-1)),
                                  (index+1,1,(0,1))):
            for tick in ticks:
                if _look_value(tracks[cell],tick/clean.fps) != _look_value(
                        pair.segments[local].boundary_look_track,tick/clean.fps):
                    raise ValueError('recomposed short-shot look differs on an encoded proof window')
    if (previous.render_admission_encode_settings_sha256 is not None
            and fresh.render_admission_encode_settings_sha256 != previous.render_admission_encode_settings_sha256):
        raise ValueError('retained/fresh look use incompatible encode settings')
    refreshed_plan=result.apply(clean)
    return refreshed_plan, result, _bind_look_refresh(old_bundle,refreshed_plan,result,dict(refreshed_edges=list(edges),
        projected_samples=cache.evidence,
        proposed=proposed.to_dict(), verified=fresh.to_dict(),
        old_family_retained_in_predecessor=True,
        disjoint_decisions_preserved=True,
        matched_production_renderer_used=True))


def _refresh_face(old, candidate_plan, observations, refreshed):
    rows = []
    for index, row in enumerate(old.segments):
        if index in refreshed:
            # Keep the original source-measured fit and policy. A camera-only
            # edit does not make its source-space value a final output value.
            payload = dict(basis=observations['basis'],
                samples=[r for r in observations['samples'] if r['segment_index']==index],
                raw_source_recipe_preserved_under_equal_clock_and_base_crop=True,
                quality_approved=False)
            row = replace(row, projected_output_observations=payload)
        if row.pan_crop != candidate_plan.segments[index].pan_crop:
            raise ValueError('source face recipe is not the actual candidate base pan')
        rows.append(row)
    return replace(old, segments=tuple(rows))


def complete_candidate(context, camera_bundle):
    """Recompute dependencies of an already proposed, dense-bound camera."""
    from .pipeline import _validate_review_first_group_bundle
    from .group_source_span_proposal import refresh_fixed_group_render_observations
    from .group_boundary_evidence_preservation import preserve_boundary_evidence
    from .timeline_review import review_timeline
    old = context.incumbent
    changed, refreshed, edges = changed_scope(old.render_plan, camera_bundle.render_plan, context.domain)
    if any(old.visual_timeline.choices[i].internal_scene_cut_master_seconds for i in changed):
        raise ValueError('changed coalesced cell needs complete internal-epoch performer refresh')
    # Never silently reinterpret a learned or transition-specific semantic row.
    if any(old.visual_timeline.choices[i].transition_evidence_authority_from_previous is not None
            for i in refreshed):
        raise ValueError('affected transition-specific evidence needs its own fresh producer')
    budget = JointGroupWorkBudget(absolute_deadline=context.deadline, maximum_new_decodes=MAX_NATIVE_READS)
    session = GroupJointObservationSession(domain=context.domain, budget=budget)
    try:
        old_observations = observe_scope(old.render_plan, context.domain, refreshed, edges, context.model_path, session)
        observations = observe_scope(camera_bundle.render_plan, context.domain, refreshed, edges, context.model_path, session)
        # Full per-cell descriptor producer: recompute entry/middle/exit and
        # incident edges in final prescribed camera coordinates, never copy
        # source descriptors under a new field name.
        timeline = refresh_fixed_group_render_observations(plan=camera_bundle.render_plan,
            clock_timeline=old.visual_timeline, first_segment_index=refreshed[0],
            last_segment_index=refreshed[-1], source_domain=context.domain,
            sync_scores=context.sync_scores,
            feature_cache_directory=None,
            feature_derived_cache_directory=None,
            require_persistent_feature_cache=False)
        plan, look, look_evidence = refresh_look(old, camera_bundle.render_plan, edges, session, context.domain)
        repairs, affine, preserved = preserve_boundary_evidence(old_plan=old.render_plan,new_plan=plan,
            old_timeline=old.visual_timeline,new_timeline=timeline,
            boundary_repairs=old.boundary_repairs,boundary_affine_plan=old.boundary_affine_plan)
        face = _refresh_face(old.face_match_plan, plan, observations, refreshed)
        review = review_timeline(timeline, available_source_count=len(context.domain.eligible_indices))
        candidate = replace(camera_bundle, render_plan=plan, visual_timeline=timeline,
            timeline_review=review, pre_render_score=float(review.weighted_score),
            face_match_plan=face, boundary_look_match=look,
            boundary_repairs=repairs, boundary_affine_plan=affine,
            candidate_origin=CONTRACT, cadence_profile=CONTRACT,
            candidate_lineage=(*old.candidate_lineage,
                *tuple(_stable(row) for row in camera_bundle.candidate_lineage[len(old.candidate_lineage):]),
                dict(stage=CONTRACT, changed_cells=list(changed), refreshed_cells=list(refreshed),
                    incident_edges=list(edges), source_changes_supported=False,
                    retained_raw_face_fit_not_relabelled=True, quality_approved=False)))
        # Unaffected rows and actual rendering fields must be byte-for-byte
        # identical. Old aggregate skipped-reason counts are planning history;
        # preserve them, while final proof/event counters come from fresh dense
        # decisions and the existing final renderer event calculator.
        for i in range(len(plan.segments)):
            if i not in refreshed:
                if (plan.segments[i] != old.render_plan.segments[i]
                        or timeline.choices[i] != old.visual_timeline.choices[i]
                        or face.segments[i] != old.face_match_plan.segments[i]
                        or candidate.editorial_motion_plan.segments[i] != old.editorial_motion_plan.segments[i]):
                    raise ValueError('performer transaction changed an unrelated family row')
        receipt = dict(contract=CONTRACT, quality_approved=False,
            source_changes_supported=False, cross_camera_identity_certified=False,
            body_phase_certified=False, decoded_output_quality_certified=False,
            changed_cells=list(changed), refreshed_cells=list(refreshed), incident_edges=list(edges),
            plan_sha256=digest(plan_payload(plan)), metadata_sha256=digest(metadata_payload(candidate)),
            original_plan_sha256=digest(plan_payload(old.render_plan)),
            original_metadata_sha256=digest(metadata_payload(old)),
            external_original_reference=external_reference_binding(context, candidate),
            external_original_selected_boundary_look_metadata=None if candidate.boundary_look_match is None
                else plain(candidate.boundary_look_match.to_dict()),
            source_domain=context.domain.to_dict(), synchronization_scores=list(context.sync_scores), source_identities=[list(v) for v in context.source_identities],
            dependencies=dict(context.code_dependencies), model_path=str(context.model_path),
            model_identity=list(context.model_identity), model_sha256=context.model_sha256,
            global_native_schedule_sha256=_schedule_digest(_schedule(plan,context.domain)),
            before_observations=old_observations, final_observations=observations,
            look_refresh=look_evidence, boundary_evidence_preservation=preserved,
            final_quality_status='UNAPPROVED_PERFORMER_REVIEW_CANDIDATE')
        candidate = replace(candidate, performer_family_review=receipt)
        _validate_review_first_group_bundle(candidate, eligible_indices=context.domain.eligible_indices,
            offsets=context.domain.audio_offsets,timewarps=context.domain.source_timewarps,
            common_start=old.render_plan.master_audio_start,phrase_times=old.phrase_times)
        _fresh(context)
        return candidate
    finally:
        session.close()


@dataclass(frozen=True, slots=True)
class PerformerFamilyRejection:
    error: str
    evidence: Any


@isolated_optional()
def propose_performer_family(context):
    from .pipeline import _review_first_group_motion_camera_bundle
    from .group_source_changing_owner import propose_source_changes
    _fresh(context)
    # Normal app source-changing producer, not an opt-in contrast mode. The
    # complete source/clock/camera family transaction runs before the legacy
    # fixed-source refinement; no single-source success is a prerequisite.
    from .group_scene_source_owner import propose_scene_changes
    scene_candidate = propose_scene_changes(context)
    if not isinstance(scene_candidate, PerformerFamilyRejection):
        return scene_candidate
    if context.source_derivative_binding is not None:
        # The normal source/scene path now owns frozen original duties. The
        # legacy siblings cannot yet consume that contract or the same request
        # ledger; they are UNSUPPORTED, not an automatic new-budget retry.
        # Keep the exact rejection/partial history for the normal controller.
        return PerformerFamilyRejection(scene_candidate.error,dict(
            contract='normal_scene_fallback_contract_unavailable_v1',
            fallback_status='UNSUPPORTED_ORIGINAL_DUTY_REQUEST_LEDGER',
            legacy_proposal_called=False,new_budget_retry_started=False,
            original_duties_waived=False,scene_search=scene_candidate.evidence,
            quality_approved=False))
    source_candidate = propose_source_changes(context)
    if not isinstance(source_candidate, PerformerFamilyRejection):
        receipt = plain(source_candidate.performer_family_review)
        receipt['scene_search_rejection'] = dict(error=scene_candidate.error,evidence=scene_candidate.evidence)
        return replace(source_candidate,performer_family_review=receipt)
    source_candidate = PerformerFamilyRejection(source_candidate.error,
        dict(source_search=source_candidate.evidence,scene_search=scene_candidate.evidence))
    old = context.incumbent
    # An internal geometry-only work view is not an adopted bundle. Full old
    # look stays in context; complete_candidate MUST put fresh/preserved look
    # back and prove that all families still drive the renderer before CAS.
    work = replace(old, boundary_look_match=None,
        render_plan=replace(old.render_plan, segments=tuple(replace(s,boundary_look_track=None)
            for s in old.render_plan.segments)))
    request = SimpleNamespace(source_cache_directory=None,
        visual_feature_derived_cache_directory=None)
    candidate = _review_first_group_motion_camera_bundle(work, request=request,
        eligible_indices=context.domain.eligible_indices, offsets=context.domain.audio_offsets,
        timewarps=context.domain.source_timewarps,scene_cut_times=context.domain.scene_cut_times,
        phrase_times=old.phrase_times,model_path=context.model_path,
        allow_retiming=False,require_incident_edge_replay=True,maximum_boundaries=4,maximum_accepted_boundaries=1)
    if candidate is None or candidate.render_plan == work.render_plan:
        return PerformerFamilyRejection('no complete observed performer camera candidate',
            {**(candidate.to_dict() if candidate is not None else {}), 'source_search': source_candidate.evidence})
    try:
        completed = complete_candidate(context, candidate)
        receipt = plain(completed.performer_family_review)
        receipt['source_search_rejection'] = dict(error=source_candidate.error, evidence=source_candidate.evidence)
        return replace(completed, performer_family_review=receipt)
    except Exception as exc:
        # A typed outcome survives the native-worker IPC (exception attributes
        # do not). The parent archives the FULL proposed family before rejection.
        return PerformerFamilyRejection(f'{type(exc).__name__}: {exc}',
            {**candidate.to_dict(), 'source_search': source_candidate.evidence})


def validate_bundle_binding(bundle):
    receipt = plain(getattr(bundle, 'performer_family_review', None))
    if receipt is not None and receipt.get('contract') == SOURCE_CONTRACT:
        from .group_source_changing_owner import validate_binding
        return validate_binding(bundle)
    if receipt is None:
        if bundle.candidate_origin in FAMILY_CONTRACTS or getattr(bundle.editorial_motion_plan, 'source_camera_refresh', None) is not None:
            raise ValueError('performer complete bundle receipt was removed')
        return
    if (receipt.get('contract') != CONTRACT or bundle.candidate_origin != CONTRACT
            or receipt.get('quality_approved') is not False
            or receipt.get('source_changes_supported') is not False
            or receipt.get('cross_camera_identity_certified') is not False
            or receipt.get('body_phase_certified') is not False
            or receipt.get('decoded_output_quality_certified') is not False
            or receipt.get('plan_sha256') != digest(plan_payload(bundle.render_plan))
            or receipt.get('metadata_sha256') != digest(metadata_payload(bundle))):
        raise ValueError('performer complete renderer/family/uncertainty binding changed')
    validate_look_refresh(bundle.render_plan,bundle.boundary_look_match,receipt)
    validate_external_reference_binding(bundle, receipt)


def validate_parent(candidate, context):
    if plain(candidate.performer_family_review).get('contract') == SOURCE_CONTRACT:
        from .group_source_changing_owner import validate_parent as validate_source_parent
        return validate_source_parent(candidate, context)
    _fresh(context)
    validate_bundle_binding(candidate)
    changed_scope(context.incumbent.render_plan,candidate.render_plan,context.domain)
    receipt = plain(candidate.performer_family_review)
    validate_look_refresh(candidate.render_plan,candidate.boundary_look_match,receipt,
        original_bundle=context.incumbent)
    validate_external_reference_binding(candidate, receipt, context=context)
    if (receipt['source_domain'] != context.domain.to_dict()
            or receipt['source_identities'] != [list(v) for v in context.source_identities]
            or receipt['dependencies'] != dict(context.code_dependencies)
            or receipt['synchronization_scores'] != list(context.sync_scores)
            or receipt['original_metadata_sha256'] != digest(metadata_payload(context.incumbent))):
        raise ValueError('performer parent source/metadata preimage differs')
    replay = propose_performer_family(context)  # NEW isolated worker, fresh native pixels/model outputs.
    if isinstance(replay, PerformerFamilyRejection):
        raise ValueError('performer parent independent replay rejected: '+replay.error)
    if candidate.to_dict() != replay.to_dict() or candidate.render_plan != replay.render_plan:
        raise ValueError('performer parent independent complete sensor/family replay disagrees')
    _fresh(context)


def attempt_performer_family(*, controller, request, eligible_indices, offsets, timewarps,
        scene_cut_times, sync_scores, archive_path, source_derivatives=None):
    from .pipeline import _write_analysis_manifest_atomically
    def transaction():
        old = controller.incumbent
        archive = dict(contract=CONTRACT,status='preparing',quality_approved=False,
            original_bundle=old.to_dict(),original_plan=old.render_plan.to_json())
        def publish():
            # Partial graphs can own frame-cache read-through mappings, unlike
            # the expanded public bundle JSON. Retain their exact wire views
            # at this separate durable seam; never stringify/drop raw evidence.
            from .group_scene_native_blob_storage import pack_owned_native_evidence
            payload=pack_owned_native_evidence(archive)
            if len(json.dumps(payload,sort_keys=True,allow_nan=False).encode()) > MAX_ARCHIVE_BYTES:
                raise ValueError('performer complete archive exceeds explicit byte budget')
            _write_analysis_manifest_atomically(Path(archive_path),payload)
        publish()  # Even a missing sensor asset leaves the complete predecessor.
        try:
            context = freeze_context(controller=controller,request=request,eligible_indices=eligible_indices,
                offsets=offsets,timewarps=timewarps,scene_cut_times=scene_cut_times,sync_scores=sync_scores,
                source_derivatives=source_derivatives)
            archive['transaction_token']=asdict(context.token)
            candidate = propose_performer_family(context)
            if isinstance(candidate, PerformerFamilyRejection):
                error = ValueError(candidate.error)
                error.candidate_evidence = candidate.evidence
                raise error
            archive.update(status='pending_independent_parent_replay',candidate_bundle=candidate.to_dict(),
                candidate_plan=candidate.render_plan.to_json())
            publish()
            def parent_checked(value):
                proof=validate_parent(value,context)
                if proof is not None:
                    archive['independent_parent_replay']=proof
                    publish()
            controller.compare_and_promote(candidate,plan=candidate.render_plan,authority=candidate.candidate_origin,
                token=context.token,validate=parent_checked)
            # The pre-CAS durable record is authoritative. Journal owns commit.
            return candidate
        except Exception as exc:
            archive.update(status='rejected_incumbent_preserved',error_type=type(exc).__name__,error=str(exc),
                producer_rejection_evidence=getattr(exc,'candidate_evidence',None))
            publish()
            raise
    return controller.run_optional('normal_performer_retained_family_owner',transaction)


def validate_serialized_metadata(complete):
    if not isinstance(complete,Mapping):
        raise ValueError('performer selected complete metadata missing')
    metadata = dict(complete)
    receipt = metadata.pop('performer_family_review',None)
    if (not isinstance(receipt,Mapping) or receipt.get('contract') not in FAMILY_CONTRACTS
            or complete.get('candidate_origin') != receipt.get('contract')
            or receipt.get('source_changes_supported') is not (receipt.get('contract') == SOURCE_CONTRACT)
            or receipt.get('contract') == SOURCE_CONTRACT and receipt.get('source_camera_refresh') !=
                (complete.get('editorial_motion') or {}).get('source_camera_refresh')
            or receipt.get('metadata_sha256') != digest(metadata)
            or receipt.get('quality_approved') is not False
            or receipt.get('decoded_output_quality_certified') is not False
            or receipt.get('body_phase_certified') is not False
            or receipt.get('cross_camera_identity_certified') is not False):
        raise ValueError('performer selected metadata, family or uncertainty changed')
    if receipt.get('contract') == SOURCE_CONTRACT:
        from .group_source_changing_owner import validate_dense_uncertainty
        validate_dense_uncertainty(plain(receipt.get('source_camera_refresh')))
    # Complete metadata intentionally contains final_render_segments, not a
    # RenderPlan. Compare its non-lossy residual/color inventory here; the
    # final consumer checks the actual complete RenderPlan separately below.
    audit=plain(receipt.get('look_refresh'))
    binding=audit.get('retained_family') if isinstance(audit,dict) else None
    final=binding.get('final') if isinstance(binding,dict) else None
    rows=complete.get('final_render_segments')
    if not isinstance(rows,list) or not isinstance(final,dict):
        raise ValueError('performer selected metadata lost its actual boundary look inventory')
    tracks=[]
    for row in rows:
        track=row.get('boundary_look_track') if isinstance(row,Mapping) else None
        if track is not None:
            track=dict(track);track.pop('post_global_residual',None)
        tracks.append(track)
    look=complete.get('boundary_look_match')
    mode='ORIGINAL_ABSENT' if look is None else 'ENABLED'
    if (final.get('mode')!=mode or final.get('segment_count')!=len(rows)
            or final.get('metadata_sha256')!=digest(look)
            or final.get('renderer_tracks_sha256')!=digest(tracks)
            or look is None and any(t is not None for t in tracks)
            or look is not None and (not isinstance(look,Mapping) or look.get('enabled') is not True
                or digest(look.get('segment_tracks'))!=digest(tracks))):
        raise ValueError('performer selected metadata boundary look differs from the actual track inventory')
    if receipt.get('external_original_reference') is not None and (
            receipt.get('external_original_selected_boundary_look_metadata') != look
            or receipt['external_original_reference'].get('mode') != (
                'ABSENT_PRESERVED' if look is None else 'ACTIVE_EXISTING')):
        raise ValueError('selected metadata lost its external original boundary-look policy')
    return receipt


@isolated_optional()
def _final_live_replay(plan_json, receipt, deadline):
    from fractions import Fraction
    from .render_plan_serialization import render_plan_from_json
    from .render_frame_schedule import VerifiedRenderFrameClock
    plan = render_plan_from_json(plan_json)
    if digest(plan_payload(plan)) != receipt['plan_sha256'] or dependencies(receipt['model_path']) != receipt['dependencies']:
        raise ValueError('performer final camera or code differs from admitted complete bundle')
    raw = receipt['source_domain'];indices=tuple(raw['eligible_request_indices'])
    if not indices or len(set(indices)) != len(indices) or any(
            type(i) is not int or not 0 <= i < len(plan.sources) for i in indices):
        raise ValueError('performer final request source domain is invalid')
    if raw['eligible_sources'] != [str(plan.sources[i]) for i in indices]:
        raise ValueError('performer final eligible source mapping changed')
    clocks=[]
    for i, row in zip(indices,raw['render_frame_clocks'],strict=True):
        expected=VerifiedRenderFrameClock(Fraction(row['frame_rate']),row['frame_count'],row['packet_grid_sha256'],row['grid_basis'])
        observed=probe_render_frame_clock(plan.sources[i],timeout_seconds=max(.01,min(10.,deadline-monotonic())))
        if observed!=expected:
            raise ValueError('performer final native PTS clock changed')
        clocks.append(observed)
    # Only fixed source sample clocks are needed here; actual source/reference
    # alignment remains owned by the complete bundle's existing clock family.
    domain=GroupSourceDomain(tuple(plan.sources),indices,(0.,)*len(indices),(None,)*len(indices),
        tuple(tuple(c) for c in raw['scene_cut_times']),tuple(clocks))
    if ([list(_file_identity(p)) for p in domain.eligible_sources] != receipt['source_identities']
            or list(_file_identity(Path(receipt['model_path']))) != receipt['model_identity']
            or _file_hash(receipt['model_path']) != receipt['model_sha256']
            or _schedule_digest(_schedule(plan,domain)) != receipt['global_native_schedule_sha256']):
        raise ValueError('performer final live source/model/native inventory changed')
    from .group_external_original_reference import validate_external_original_reference_payload
    validate_external_original_reference_payload(plain(receipt.get('external_original_reference')),
        plan=plan, look=plain(receipt.get('external_original_selected_boundary_look_metadata')),
        source_domain_payload=raw, source_identities=receipt['source_identities'],
        live_clocks=domain.clocks_by_request)
    session=GroupJointObservationSession(domain=domain,budget=JointGroupWorkBudget(
        absolute_deadline=deadline,maximum_new_decodes=MAX_NATIVE_READS))
    try:
        observed=observe_scope(plan,domain,tuple(receipt['refreshed_cells']),tuple(receipt['incident_edges']),
            Path(receipt['model_path']),session)
        if observed!=receipt['final_observations']:
            raise ValueError('performer final fresh observations differ from parent-admitted semantics/geometry')
    finally:
        session.close()
    if [list(_file_identity(p)) for p in domain.eligible_sources] != receipt['source_identities']:
        raise ValueError('performer source changed during final replay')
    result=dict(contract=CONTRACT,binding_valid=True,source_live_replay=True,quality_approved=False,
        decoded_output_quality_certified=False,plan_sha256=receipt['plan_sha256'],
        metadata_sha256=receipt['metadata_sha256'],sampled_native_pixel_count=len(observed['samples']),
        final_quality_status='UNAPPROVED_PERFORMER_REVIEW_CANDIDATE')
    result['evidence_sha256']=digest(result)
    return result


def replay_final_sources(plan_json,receipt):
    deadline=min(current_optional_deadline() or float('inf'),monotonic()+45.)
    with optional_native_scope(deadline):
        if plain(receipt).get('contract') == SOURCE_CONTRACT:
            from .group_source_changing_owner import final_live_replay
            return final_live_replay(plan_json,plain(receipt),deadline)
        return _final_live_replay(plan_json,plain(receipt),deadline)


def consume_performer_family_qa(analysis,plan_json,selection):
    render=analysis.get('render') or {};complete=selection.get('complete_candidates') or ()
    receipt=render.get('performer_family_review')
    marked=(render.get('candidate_origin') in FAMILY_CONTRACTS or selection.get('selected_path_authority') in FAMILY_CONTRACTS
        or (render.get('editorial_motion') or {}).get('source_camera_refresh') is not None
        or any(isinstance(c,Mapping) and (c.get('performer_family_review') is not None
            or (c.get('editorial_motion') or {}).get('source_camera_refresh') is not None) for c in complete))
    if receipt is None and not marked:
        return None
    if len(complete)!=1 or validate_serialized_metadata(complete[0])!=receipt:
        raise ValueError('performer final selection/render receipt copies are missing or differ')
    selected=complete[0]
    from .render_plan_serialization import render_plan_from_json
    validate_look_refresh(render_plan_from_json(plan_json),selected.get('boundary_look_match'),receipt)
    for key in ('face_match','editorial_motion','boundary_look_match','group_transitions','final_render_segments'):
        if render.get(key)!=selected.get(key):
            raise ValueError('performer actual render has stale '+key+' family')
    timeline=dict(analysis.get('visual_timeline') or {});timeline.pop('index_domain',None)
    if timeline!=selected['visual_timeline']:
        raise ValueError('performer actual render has a stale projected timeline')
    return replay_final_sources(plan_json,receipt)


def qa_metrics_valid(metrics):
    if not isinstance(metrics,Mapping):return False
    value=dict(metrics);identity=value.pop('evidence_sha256',None)
    return bool(value.get('contract') in FAMILY_CONTRACTS and value.get('binding_valid') is True
        and value.get('source_live_replay') is True and value.get('quality_approved') is False
        and value.get('decoded_output_quality_certified') is False
        and value.get('final_quality_status')=='UNAPPROVED_PERFORMER_REVIEW_CANDIDATE'
        and (value.get('contract') != SOURCE_CONTRACT or (
            value.get('semantic_raw_v2_admission') is False
            and value.get('cross_camera_identity_certified') is False
            and value.get('body_phase_certified') is False
            and type(value.get('dense_native_output_tick_count')) is int
            and 1 <= value['dense_native_output_tick_count'] <= 600))
        and type(value.get('sampled_native_pixel_count')) is int
        and 1<=value['sampled_native_pixel_count']<=MAX_NATIVE_READS and identity==digest(value))
