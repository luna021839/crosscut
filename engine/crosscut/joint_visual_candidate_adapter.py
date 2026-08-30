"""Production bridge from the visual lattice to the joint boundary DP.

The visual scorer deliberately retains a transient optimizer workspace.  This
module consumes that workspace without rebuilding descriptors, runs the three
existing professional cadence profiles through the joint source/offset/crop
selector, and keeps the exact decoded proof selected by the DP.

Only scalar identities and reviewer data cross the audit boundary.  Decoded
descriptors, appearance embeddings, and transform-path samples remain attached
to process-local provider materializations and are never serialized here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .ive import AspectRatio
from .editorial_motion import EditorialMotionPlan, EditorialMotionSegment
from .joint_boundary_provider import (
    BoundaryMaterializationProof,
    DecodedBoundaryOptionProvider,
    JointSelectionMaterializationProof,
    SegmentTransformPathMaterializationProof,
)
from .joint_timeline import (
    AuthorizedBoundaryOption,
    BoundaryMeasurementKey,
    BoundaryOptionEvidenceScope,
    BoundaryOptionSet,
    JOINT_BOUNDARY_FRAME_OFFSETS,
    JOINT_SHARED_BRIDGE_CACHE_MAXIMUM_ENTRIES,
    JointTimelineProgress,
    JointTimelineProgressCallback,
    JointTimelineResult,
    JointTimelineSharedBridgeEvidenceCache,
    select_joint_timeline_candidates,
)
from .timeline import (
    comparable_group_formation_mismatch,
    cross_source_layout_mismatch,
    face_geometry_minimax_mismatch,
    NoCompleteTimelinePath,
    TimelineCandidate,
    TimelineChoice,
    TimelineResult,
)
from .timeline_no_good import TimelineNoGoodSet
from .timeline_review import review_timeline
from .render import (
    RenderPlan,
    RenderSegment,
    RenderTransition,
    aligned_render_segment,
    output_size,
)
from .visual_score import (
    VisualTimelineCandidateRecord,
    VisualTimelineCandidateSet,
    VisualTimelineChoice,
    VisualTimelineOptimizerCarrier,
    VisualTimelineOptimizerSeed,
    VisualTimelinePlan,
    _candidate_review_summary,
    _embedding_rms_distance,
    _master_clock_time,
    _missing_candidate_face_metadata,
    _missing_candidate_subject_metadata,
    _review_ranked_candidate_indices,
    _role_conditioned_visual_controls,
    _safe_cadence_profiles,
    _source_clock_time,
)


JOINT_VISUAL_CANDIDATE_ADAPTER_BASIS = (
    "joint_visual_candidate_adapter_v1+"
    "three_existing_safe_cadence_profiles_v1+"
    "exact_source_shot_proof_variant_coalescing_v2+"
    "mathematical_unique_path_floor_minimum_eight_render_variants_v2+"
    "shifted_scene_cut_clock_rematerialization_v1+"
    "workspace_only_optimizer_seed_v1+"
    "selection_scalar_exact_rehydration_v1+"
    "finite_exact_boundary_nogood_retry_v1"
)
JOINT_EDITORIAL_MOTION_AUTHORITY_BASIS = (
    "decoded_joint_boundary_exact_renderer_track_editorial_motion_authority_v1"
)
MINIMUM_PRODUCTION_JOINT_CANDIDATES = 8
MAXIMUM_PRODUCTION_PROOF_VARIANTS_PER_PATH = 2
MINIMUM_PRODUCTION_UNIQUE_SOURCE_SHOT_PATHS = math.ceil(
    MINIMUM_PRODUCTION_JOINT_CANDIDATES
    / MAXIMUM_PRODUCTION_PROOF_VARIANTS_PER_PATH
)
MAXIMUM_PROFILE_JOINT_CANDIDATES = 8
_SCENE_CUT_INTERIOR_EPSILON_SECONDS = 1e-6


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_shot_path_signature(
    result: JointTimelineResult,
) -> tuple[tuple[int, str, str], ...]:
    """Return the exact editorial path, intentionally excluding proof choice."""

    return tuple(
        (candidate.segment_index, candidate.source_id, candidate.shot_id)
        for candidate in result.candidates
    )


def _proof_variant_signature(result: JointTimelineResult) -> tuple[str, ...]:
    return tuple(option.proof_identity for option in result.boundary_options)


def _render_variant_signature(
    result: JointTimelineResult,
) -> tuple[tuple[object, ...], ...]:
    """Identify the exact boundary clocks and renderer-authority transforms."""

    return tuple(
        (
            option.frame_offset,
            option.outgoing_transform.transform_identity,
            option.outgoing_transform.crop_xywh,
            option.outgoing_transform.rotation_degrees,
            option.outgoing_transform.track_id,
            option.outgoing_transform.subject_basis,
            option.incoming_transform.transform_identity,
            option.incoming_transform.crop_xywh,
            option.incoming_transform.rotation_degrees,
            option.incoming_transform.track_id,
            option.incoming_transform.subject_basis,
        )
        for option in result.boundary_options
    )


@dataclass(frozen=True, slots=True)
class JointVisualSelectionRecord:
    """One coalesced joint result before render-track materialization."""

    cadence_profile: str
    joint_result: JointTimelineResult = field(repr=False)
    shifted_timeline_result: TimelineResult = field(repr=False)
    source_shot_path: tuple[tuple[int, str, str], ...]
    proof_variant_signature: tuple[str, ...]
    role_control_audit: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cadence_profile, str)
            or not self.cadence_profile
            or self.cadence_profile != self.cadence_profile.strip()
        ):
            raise ValueError("cadence_profile must be an exact identifier")
        if not isinstance(self.joint_result, JointTimelineResult):
            raise TypeError("joint_result must be a JointTimelineResult")
        if not isinstance(self.shifted_timeline_result, TimelineResult):
            raise TypeError("shifted_timeline_result must be a TimelineResult")
        expected_path = _source_shot_path_signature(self.joint_result)
        if tuple(self.source_shot_path) != expected_path:
            raise ValueError("source_shot_path does not match joint_result")
        expected_proofs = _proof_variant_signature(self.joint_result)
        if tuple(self.proof_variant_signature) != expected_proofs:
            raise ValueError("proof_variant_signature does not match joint_result")
        if tuple(
            (item.segment_index, item.source_id, item.shot_id)
            for item in self.shifted_timeline_result.candidates
        ) != expected_path:
            raise ValueError("shifted timeline changed the exact source/shot path")
        audit = self.role_control_audit
        object.__setattr__(
            self,
            "role_control_audit",
            None if audit is None else MappingProxyType(dict(audit)),
        )

    @property
    def optimizer_score(self) -> float:
        return self.shifted_timeline_result.total_score

    @property
    def privacy_safe_signature(self) -> str:
        return "joint-visual-path:" + _sha256_json(
            {
                "contract": "joint-visual-source-shot-proof-signature-v1",
                "source_shot_path": [list(item) for item in self.source_shot_path],
                "boundary_proof_identities": list(
                    self.proof_variant_signature
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class JointVisualSelectionSet:
    """Round-robin profile output with exact decoded proof variants retained."""

    candidate_set: VisualTimelineOptimizerCarrier = field(repr=False)
    selections: tuple[JointVisualSelectionRecord, ...]
    candidate_records: tuple[VisualTimelineCandidateRecord, ...]
    selected_candidate_index: int
    profile_failures: tuple[tuple[str, NoCompleteTimelinePath], ...] = field(
        default=(),
        repr=False,
    )
    role_control_audits: tuple[Mapping[str, Any], ...] = ()
    shared_option_evidence_cache_audit: Mapping[str, Any] = field(
        default_factory=dict
    )
    shared_bridge_evidence_cache_audit: Mapping[str, Any] = field(
        default_factory=dict
    )
    analysis_basis: str = JOINT_VISUAL_CANDIDATE_ADAPTER_BASIS

    def __post_init__(self) -> None:
        if not isinstance(
            self.candidate_set,
            (VisualTimelineCandidateSet, VisualTimelineOptimizerSeed),
        ):
            raise TypeError(
                "candidate_set must carry a visual optimizer workspace"
            )
        selections = tuple(self.selections)
        records = tuple(self.candidate_records)
        if len(selections) < MINIMUM_PRODUCTION_JOINT_CANDIDATES:
            raise ValueError("joint selection set requires at least eight variants")
        if len(selections) != len(records):
            raise ValueError("joint selections and candidate records must align")
        if tuple(record.candidate_index for record in records) != tuple(
            range(len(records))
        ):
            raise ValueError("joint candidate indices must be contiguous")
        if any(
            record.timeline_result is not selection.shifted_timeline_result
            for record, selection in zip(records, selections, strict=True)
        ):
            raise ValueError("candidate records lost their shifted joint timeline")
        paths = tuple(selection.source_shot_path for selection in selections)
        unique_path_count = len(set(paths))
        if unique_path_count < MINIMUM_PRODUCTION_UNIQUE_SOURCE_SHOT_PATHS:
            raise ValueError(
                "joint selection set does not retain the mathematical minimum "
                "of unique source/shot paths"
            )
        supplement_count = len(paths) - unique_path_count
        required_supplement_count = max(
            0,
            MINIMUM_PRODUCTION_JOINT_CANDIDATES - unique_path_count,
        )
        if supplement_count != required_supplement_count:
            raise ValueError(
                "joint selection set retained an invalid proof supplement count"
            )
        path_multiplicity = {
            path: paths.count(path)
            for path in set(paths)
        }
        if any(
            count > MAXIMUM_PRODUCTION_PROOF_VARIANTS_PER_PATH
            for count in path_multiplicity.values()
        ):
            raise ValueError(
                "joint selection set retained too many variants for one path"
            )
        proof_variants = tuple(
            (selection.source_shot_path, selection.proof_variant_signature)
            for selection in selections
        )
        if len(set(proof_variants)) != len(proof_variants):
            raise ValueError("joint selection set contains a duplicate proof variant")
        render_variants = tuple(
            (
                selection.source_shot_path,
                _render_variant_signature(selection.joint_result),
            )
            for selection in selections
        )
        if len(set(render_variants)) != len(render_variants):
            raise ValueError("joint selection set contains a duplicate render variant")
        if not 0 <= self.selected_candidate_index < len(records):
            raise ValueError("selected joint candidate index is outside the set")
        selected = tuple(
            record.candidate_index for record in records if record.selected
        )
        if selected != (self.selected_candidate_index,):
            raise ValueError("joint selection set must mark one selected record")
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "candidate_records", records)
        object.__setattr__(
            self,
            "role_control_audits",
            tuple(MappingProxyType(dict(item)) for item in self.role_control_audits),
        )
        object.__setattr__(
            self,
            "shared_option_evidence_cache_audit",
            MappingProxyType(dict(self.shared_option_evidence_cache_audit)),
        )
        object.__setattr__(
            self,
            "shared_bridge_evidence_cache_audit",
            MappingProxyType(dict(self.shared_bridge_evidence_cache_audit)),
        )

    @property
    def ranked_records(self) -> tuple[VisualTimelineCandidateRecord, ...]:
        return tuple(sorted(self.candidate_records, key=lambda item: item.review_rank))

    @property
    def unique_source_shot_path_count(self) -> int:
        return len({selection.source_shot_path for selection in self.selections})

    @property
    def proof_variant_supplement_count(self) -> int:
        return len(self.selections) - self.unique_source_shot_path_count

    def selection_for_candidate(
        self,
        candidate_index: int,
    ) -> JointVisualSelectionRecord:
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or not 0 <= candidate_index < len(self.selections)
        ):
            raise IndexError("joint candidate index is outside the selection set")
        return self.selections[candidate_index]

    def to_dict(self) -> dict[str, object]:
        """Serialize only scalar review/audit data, never transient descriptors."""

        return {
            "analysis_basis": self.analysis_basis,
            "candidate_count": len(self.candidate_records),
            "unique_source_shot_path_count": self.unique_source_shot_path_count,
            "proof_variant_supplement_count": (
                self.proof_variant_supplement_count
            ),
            "selected_candidate_index": self.selected_candidate_index,
            "candidates": [
                {
                    **record.to_dict(),
                    "joint_path_signature": selection.privacy_safe_signature,
                    "boundary_frame_offsets": [
                        option.frame_offset
                        for option in selection.joint_result.boundary_options
                    ],
                    "boundary_proof_count": len(
                        selection.joint_result.boundary_options
                    ),
                }
                for record, selection in zip(
                    self.candidate_records,
                    self.selections,
                    strict=True,
                )
            ],
            "profile_failures": [
                {
                    "cadence_profile": profile,
                    "no_complete_path": failure.to_dict(),
                }
                for profile, failure in self.profile_failures
            ],
            "role_control_audits": [dict(item) for item in self.role_control_audits],
            "shared_option_evidence_cache": dict(
                self.shared_option_evidence_cache_audit
            ),
            "shared_bridge_evidence_cache": dict(
                self.shared_bridge_evidence_cache_audit
            ),
            "raw_visual_descriptors_persisted": False,
            "raw_candidate_embeddings_persisted": False,
            "biometric_model_vectors_persisted": False,
        }


def _copy_endpoint_fields(
    base: TimelineCandidate,
    measured: TimelineCandidate,
    *,
    prefix: str,
) -> TimelineCandidate:
    replacements = {
        item.name: getattr(measured, item.name)
        for item in fields(TimelineCandidate)
        if item.name.startswith(prefix)
    }
    return replace(base, **replacements)


def _shift_joint_timeline_result(
    result: JointTimelineResult,
    *,
    fps: float,
    candidate_set: VisualTimelineOptimizerCarrier,
) -> TimelineResult:
    """Apply each selected boundary offset to both adjacent segment clocks."""

    if not isinstance(result, JointTimelineResult):
        raise TypeError("result must be a JointTimelineResult")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    choices = result.timeline.choices
    options = result.boundary_options
    if len(options) != max(0, len(choices) - 1):
        raise ValueError("joint result boundary option count is inconsistent")
    shifted_choices: list[TimelineChoice] = []
    for index, choice in enumerate(choices):
        candidate = choice.candidate
        if index > 0:
            incoming = options[index - 1].scoring_incoming
            if incoming is None:  # pragma: no cover - option invariant
                raise RuntimeError("joint incoming scoring candidate is unavailable")
            candidate = _copy_endpoint_fields(candidate, incoming, prefix="entry_")
        if index < len(options):
            outgoing = options[index].scoring_outgoing
            if outgoing is None:  # pragma: no cover - option invariant
                raise RuntimeError("joint outgoing scoring candidate is unavailable")
            candidate = _copy_endpoint_fields(candidate, outgoing, prefix="exit_")
        start = candidate.start_seconds + (
            options[index - 1].frame_offset / fps if index > 0 else 0.0
        )
        end = candidate.end_seconds + (
            options[index].frame_offset / fps if index < len(options) else 0.0
        )
        if end - start < 2.0 / fps - 1e-9:
            raise ValueError("joint offset produced a segment shorter than two frames")
        source_index = int(candidate.source_id)
        if not 0 <= source_index < len(candidate_set.sources):
            raise ValueError("joint candidate source_id is outside the candidate set")
        if candidate_set.scene_cut_times:
            source_start = _source_clock_time(
                start,
                source_index=source_index,
                audio_offsets=candidate_set.audio_offsets,
                source_timewarps=candidate_set.source_timewarps,
            )
            source_end = _source_clock_time(
                end,
                source_index=source_index,
                audio_offsets=candidate_set.audio_offsets,
                source_timewarps=candidate_set.source_timewarps,
            )
            internal_scene_cut_master_seconds = tuple(
                master_cut
                for source_cut in candidate_set.scene_cut_times[source_index]
                if source_start + _SCENE_CUT_INTERIOR_EPSILON_SECONDS
                < source_cut
                < source_end - _SCENE_CUT_INTERIOR_EPSILON_SECONDS
                for master_cut in (
                    _master_clock_time(
                        source_cut,
                        source_index=source_index,
                        audio_offsets=candidate_set.audio_offsets,
                        source_timewarps=candidate_set.source_timewarps,
                    ),
                )
                if start + _SCENE_CUT_INTERIOR_EPSILON_SECONDS
                < master_cut
                < end - _SCENE_CUT_INTERIOR_EPSILON_SECONDS
            )
        else:
            internal_scene_cut_master_seconds = tuple(
                value
                for value in candidate.internal_scene_cut_master_seconds
                if start + _SCENE_CUT_INTERIOR_EPSILON_SECONDS
                < value
                < end - _SCENE_CUT_INTERIOR_EPSILON_SECONDS
            )
        candidate = replace(
            candidate,
            start_seconds=start,
            end_seconds=end,
            internal_scene_cut_master_seconds=(
                internal_scene_cut_master_seconds
            ),
        )
        shifted_choices.append(replace(choice, candidate=candidate))
    for left, right in zip(shifted_choices, shifted_choices[1:]):
        if not math.isclose(
            left.candidate.end_seconds,
            right.candidate.start_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError("joint offset materialization is not contiguous")
    return TimelineResult(
        mode=result.timeline.mode,
        target_member=result.timeline.target_member,
        choices=tuple(shifted_choices),
        total_score=result.total_score,
    )


def _coalesce_profile_results(
    profile_results: Sequence[tuple[str, Sequence[JointTimelineResult]]],
) -> tuple[
    dict[str, tuple[JointTimelineResult, ...]],
    dict[str, tuple[JointTimelineResult, ...]],
]:
    """Retain unique path primaries plus exact alternate render variants."""

    profile_order = {
        profile: index for index, (profile, _results) in enumerate(profile_results)
    }
    variant_winners: dict[
        tuple[
            tuple[tuple[int, str, str], ...],
            tuple[str, ...],
        ],
        tuple[str, JointTimelineResult],
    ] = {}
    for profile, results in profile_results:
        for result in results:
            signature = (
                _source_shot_path_signature(result),
                _proof_variant_signature(result),
            )
            previous = variant_winners.get(signature)
            if previous is None:
                variant_winners[signature] = (profile, result)
                continue
            previous_profile, previous_result = previous
            current_key = (result.total_score, -profile_order[profile])
            previous_key = (
                previous_result.total_score,
                -profile_order[previous_profile],
            )
            if current_key > previous_key:
                variant_winners[signature] = (profile, result)

    primary_winners: dict[
        tuple[tuple[int, str, str], ...],
        tuple[str, JointTimelineResult],
    ] = {}
    for profile, result in variant_winners.values():
        path = _source_shot_path_signature(result)
        previous = primary_winners.get(path)
        if previous is None:
            primary_winners[path] = (profile, result)
            continue
        previous_profile, previous_result = previous
        current_key = (
            result.total_score,
            -profile_order[profile],
            tuple(reversed(_proof_variant_signature(result))),
        )
        previous_key = (
            previous_result.total_score,
            -profile_order[previous_profile],
            tuple(reversed(_proof_variant_signature(previous_result))),
        )
        if current_key > previous_key:
            primary_winners[path] = (profile, result)

    primary_signatures = {
        (
            _source_shot_path_signature(result),
            _proof_variant_signature(result),
        )
        for _profile, result in primary_winners.values()
    }
    primary_pools: dict[str, list[JointTimelineResult]] = {
        profile: [] for profile, _results in profile_results
    }
    alternate_pools: dict[str, list[JointTimelineResult]] = {
        profile: [] for profile, _results in profile_results
    }
    for profile, result in primary_winners.values():
        primary_pools[profile].append(result)
    for signature, (profile, result) in variant_winners.items():
        if signature not in primary_signatures:
            alternate_pools[profile].append(result)

    def sorted_pools(
        pools: Mapping[str, Sequence[JointTimelineResult]],
    ) -> dict[str, tuple[JointTimelineResult, ...]]:
        return {
            profile: tuple(
                sorted(
                    results,
                    key=lambda item: (
                        -item.total_score,
                        _source_shot_path_signature(item),
                        _proof_variant_signature(item),
                    ),
                )
            )
            for profile, results in pools.items()
        }

    return sorted_pools(primary_pools), sorted_pools(alternate_pools)


def _assert_provider_binding(
    candidate_set: VisualTimelineOptimizerCarrier,
    provider: DecodedBoundaryOptionProvider,
    *,
    fps: float,
) -> None:
    """Reject a decoded provider configured for any other candidate lattice."""

    if not math.isclose(
        float(provider.output_fps),
        fps,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("joint provider output_fps does not match adapter fps")
    workspace = candidate_set.optimizer_workspace
    if workspace is None:  # pragma: no cover - caller checks first
        raise ValueError("joint provider binding needs an optimizer workspace")
    provider.validate_segment_count(len(workspace.candidates_by_segment))
    if tuple(Path(item) for item in provider.sources) != candidate_set.sources:
        raise ValueError("joint provider sources do not match candidate set")
    if len(provider.audio_offsets) != len(candidate_set.audio_offsets) or any(
        not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
        for left, right in zip(
            provider.audio_offsets,
            candidate_set.audio_offsets,
            strict=True,
        )
    ):
        raise ValueError("joint provider audio offsets do not match candidate set")
    if tuple(provider.source_timewarps) != candidate_set.source_timewarps:
        raise ValueError("joint provider timewarps do not match candidate set")
    if tuple(tuple(item) for item in provider.scene_cut_times) != (
        candidate_set.scene_cut_times
    ):
        raise ValueError("joint provider scene cuts do not match candidate set")
    if tuple(provider.source_ids) != tuple(
        str(index) for index in range(len(candidate_set.sources))
    ):
        raise ValueError("joint provider source identifiers are not renderer indices")


def select_joint_visual_candidates(
    candidate_set: VisualTimelineOptimizerCarrier,
    *,
    provider: DecodedBoundaryOptionProvider,
    fps: float = 29.97,
    hard_forbidden_boundary_edges: TimelineNoGoodSet = TimelineNoGoodSet(),
    minimum_unique_paths: int = MINIMUM_PRODUCTION_UNIQUE_SOURCE_SHOT_PATHS,
    maximum_candidates: int | None = None,
    progress_callback: JointTimelineProgressCallback | None = None,
) -> JointVisualSelectionSet:
    """Run the existing professional controls through the decoded joint DP.

    ``hard_forbidden_boundary_edges`` is deliberately typed as an already
    classified :class:`TimelineNoGoodSet`.  Missing or indeterminate boundary
    evidence must not be projected into this argument.
    """

    if not isinstance(
        candidate_set,
        (VisualTimelineCandidateSet, VisualTimelineOptimizerSeed),
    ):
        raise TypeError("candidate_set must carry a visual optimizer workspace")
    workspace = candidate_set.optimizer_workspace
    if workspace is None:
        raise ValueError(
            "joint visual selection requires the transient optimizer workspace"
        )
    if not isinstance(provider, DecodedBoundaryOptionProvider):
        raise TypeError("provider must be a DecodedBoundaryOptionProvider")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    if not isinstance(hard_forbidden_boundary_edges, TimelineNoGoodSet):
        raise TypeError("hard forbidden edges must be a TimelineNoGoodSet")
    if (
        isinstance(minimum_unique_paths, bool)
        or not isinstance(minimum_unique_paths, int)
        or minimum_unique_paths < MINIMUM_PRODUCTION_UNIQUE_SOURCE_SHOT_PATHS
    ):
        raise ValueError(
            "minimum_unique_paths is below the mathematical render cohort floor"
        )
    requested_maximum = (
        min(workspace.maximum_results, 3 * MAXIMUM_PROFILE_JOINT_CANDIDATES)
        if maximum_candidates is None
        else maximum_candidates
    )
    if (
        isinstance(requested_maximum, bool)
        or not isinstance(requested_maximum, int)
        or requested_maximum < max(
            minimum_unique_paths,
            MINIMUM_PRODUCTION_JOINT_CANDIDATES,
        )
        or requested_maximum > 3 * MAXIMUM_PROFILE_JOINT_CANDIDATES
    ):
        raise ValueError(
            "maximum_candidates must retain eight variants and be at most 24"
        )
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    _assert_provider_binding(candidate_set, provider, fps=fps)

    lattice: tuple[tuple[TimelineCandidate, ...], ...]
    preferred_source_ids = workspace.preferred_source_ids
    if workspace.prescribed_source_ids is None:
        lattice = workspace.candidates_by_segment
    else:
        restricted: list[tuple[TimelineCandidate, ...]] = []
        for segment, source_id in zip(
            workspace.candidates_by_segment,
            workspace.prescribed_source_ids,
            strict=True,
        ):
            exact = tuple(
                candidate for candidate in segment if candidate.source_id == source_id
            )
            if len(exact) != 1:
                raise ValueError(
                    "prescribed source is not visually readable for segment "
                    f"{segment[0].segment_index}"
                )
            restricted.append(exact)
        lattice = tuple(restricted)
        preferred_source_ids = None

    profiles = tuple(_safe_cadence_profiles(workspace.diversity))
    if len(profiles) != 3:
        raise RuntimeError(
            "joint visual selection requires exactly three cadence profiles"
        )
    total_boundary_layers = max(0, len(lattice) - 1)
    shared_option_cache: dict[BoundaryMeasurementKey, BoundaryOptionSet] = {}
    shared_bridge_evidence_cache = JointTimelineSharedBridgeEvidenceCache(
        maximum_entries=min(
            JOINT_SHARED_BRIDGE_CACHE_MAXIMUM_ENTRIES,
            sum(
                len(lattice[index]) * len(JOINT_BOUNDARY_FRAME_OFFSETS) ** 2
                for index in range(1, len(lattice) - 1)
            ),
        )
    )
    shared_option_cache_hit_count = 0
    shared_option_cache_miss_count = 0

    def shared_option_provider(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
        nonlocal shared_option_cache_hit_count
        nonlocal shared_option_cache_miss_count
        if not isinstance(key, BoundaryMeasurementKey):
            raise TypeError("shared option cache key is invalid")
        cached = shared_option_cache.get(key)
        if cached is not None:
            shared_option_cache_hit_count += 1
            return cached
        shared_option_cache_miss_count += 1
        measured = provider.selection_options(key)
        if not isinstance(measured, BoundaryOptionSet):
            raise TypeError("decoded provider returned an invalid option set")
        if measured.measurement_key != key:
            raise RuntimeError("decoded provider returned evidence for another key")
        if (
            measured.evidence_scope
            is not BoundaryOptionEvidenceScope.SELECTION_SCALAR
        ):
            raise RuntimeError(
                "decoded selection provider returned non-scalar evidence"
            )
        shared_option_cache[key] = measured
        return measured

    def emit_progress(
        *,
        phase: str,
        profile_name: str,
        profile_index: int,
        core: JointTimelineProgress | None = None,
        profile_status: str,
    ) -> None:
        if progress_callback is None:
            return
        completed = core.completed_boundary_layers if core is not None else 0
        option_keys = core.option_key_count if core is not None else 0
        bridge_calls = core.bridge_provider_call_count if core is not None else 0
        batch_calls = (
            core.bridge_disposition_batch_call_count
            if core is not None
            else 0
        )
        batch_requests = (
            core.bridge_disposition_batch_request_count
            if core is not None
            else 0
        )
        progress_callback(
            JointTimelineProgress(
                phase=phase,
                cadence_profile=profile_name,
                profile_index=profile_index,
                total_profiles=len(profiles),
                completed_boundary_layers=completed,
                total_boundary_layers=total_boundary_layers,
                shared_option_cache_hit_count=(
                    shared_option_cache_hit_count
                ),
                shared_option_cache_miss_count=(
                    shared_option_cache_miss_count
                ),
                shared_option_cache_entry_count=len(shared_option_cache),
                option_key_count=option_keys,
                bridge_provider_call_count=bridge_calls,
                bridge_disposition_batch_call_count=batch_calls,
                bridge_disposition_batch_request_count=batch_requests,
                **shared_bridge_evidence_cache._progress_counts(),
                profile_status=profile_status,
            )
        )

    profile_results: list[tuple[str, tuple[JointTimelineResult, ...]]] = []
    profile_failures: list[tuple[str, NoCompleteTimelinePath]] = []
    role_audits: list[Mapping[str, Any]] = []
    role_audit_by_profile: dict[str, Mapping[str, Any]] = {}
    bridge_disposition_batch_call_count = 0
    bridge_disposition_batch_request_count = 0
    for zero_based_profile_index, profile in enumerate(profiles):
        profile_index = zero_based_profile_index + 1
        latest_core_progress: JointTimelineProgress | None = None
        callback_no_path_failure: NoCompleteTimelinePath | None = None
        emit_progress(
            phase="profile_start",
            profile_name=profile.name,
            profile_index=profile_index,
            profile_status="in_progress",
        )

        def report_core_progress(progress: JointTimelineProgress) -> None:
            nonlocal latest_core_progress
            nonlocal callback_no_path_failure
            if not isinstance(progress, JointTimelineProgress):
                raise TypeError("joint timeline progress payload is invalid")
            latest_core_progress = progress
            if progress.phase != "boundary_layer_complete":
                return
            try:
                emit_progress(
                    phase="boundary_layer_complete",
                    profile_name=profile.name,
                    profile_index=profile_index,
                    core=progress,
                    profile_status="in_progress",
                )
            except NoCompleteTimelinePath as exc:
                callback_no_path_failure = exc
                raise

        selection_kwargs: dict[str, Any] = {}
        if workspace.professional_montage_policy is not None:
            if workspace.section_plan is None:  # pragma: no cover - invariant
                raise RuntimeError("montage policy lost its section plan")
            adjustments, switch_bonus, role_audit = (
                _role_conditioned_visual_controls(
                    lattice,
                    section_plan=workspace.section_plan,
                    phrase_times=workspace.phrase_times,
                    policy=workspace.professional_montage_policy,
                    cadence_profile=profile.name,
                    subject_metadata_by_candidate=(
                        workspace.subject_metadata_by_candidate
                    ),
                    timeline_config=profile.config,
                    forbidden_boundary_edges=hard_forbidden_boundary_edges,
                )
            )
            selection_kwargs = {
                "emission_adjustments": adjustments,
                "source_switch_bonus_by_segment": switch_bonus,
            }
            role_audit_by_profile[profile.name] = role_audit
            role_audits.append(role_audit)
        try:
            results = select_joint_timeline_candidates(
                lattice,
                option_provider=shared_option_provider,
                option_rehydrator=provider.rehydrate_selection_option,
                bridge_provider=provider.bridge_provider,
                bridge_disposition_batch_provider=(
                    provider.bridge_disposition_batch
                ),
                fps=fps,
                config=profile.config,
                preferred_source_ids=preferred_source_ids,
                maximum_results=MAXIMUM_PROFILE_JOINT_CANDIDATES,
                forbidden_boundary_edges=hard_forbidden_boundary_edges,
                progress_callback=report_core_progress,
                shared_bridge_evidence_cache=shared_bridge_evidence_cache,
                **selection_kwargs,
            )
        except NoCompleteTimelinePath as exc:
            if exc is callback_no_path_failure:
                raise
            profile_failures.append((profile.name, exc))
            audit = role_audit_by_profile.get(profile.name)
            if isinstance(audit, dict):
                audit["candidate_generation_exhaustion"] = exc.to_dict()
            results = ()
            emit_progress(
                phase="profile_complete",
                profile_name=profile.name,
                profile_index=profile_index,
                core=latest_core_progress,
                profile_status="no_complete_path",
            )
        else:
            if progress_callback is not None and (
                latest_core_progress is None
                or latest_core_progress.phase != "selection_complete"
            ):
                raise RuntimeError("joint timeline emitted no terminal progress")
            emit_progress(
                phase="profile_complete",
                profile_name=profile.name,
                profile_index=profile_index,
                core=latest_core_progress,
                profile_status="completed",
            )
        if latest_core_progress is not None:
            bridge_disposition_batch_call_count += (
                latest_core_progress.bridge_disposition_batch_call_count
            )
            bridge_disposition_batch_request_count += (
                latest_core_progress.bridge_disposition_batch_request_count
            )
        profile_results.append((profile.name, tuple(results)))

    coalesced, alternates = _coalesce_profile_results(profile_results)
    ordered: list[tuple[str, JointTimelineResult]] = []
    maximum_pool_size = max((len(pool) for pool in coalesced.values()), default=0)
    for rank in range(maximum_pool_size):
        for profile, _unused in profile_results:
            pool = coalesced[profile]
            if rank >= len(pool):
                continue
            ordered.append((profile, pool[rank]))
            if len(ordered) >= requested_maximum:
                break
        if len(ordered) >= requested_maximum:
            break
    if len(ordered) < minimum_unique_paths:
        rejection_counts = {
            "unique_source_shot_path_count": len(ordered),
            "required_unique_source_shot_path_count": minimum_unique_paths,
        }
        raise NoCompleteTimelinePath(
            "joint visual selection produced fewer than the required unique "
            f"source/shot paths ({len(ordered)}/{minimum_unique_paths})",
            reason_code="joint_visual_unique_path_floor_not_met",
            rejection_counts=rejection_counts,
            profile_failures=tuple(profile_failures),
        )

    selected_proof_variants = {
        (
            _source_shot_path_signature(result),
            _proof_variant_signature(result),
        )
        for _profile, result in ordered
    }
    selected_render_variants = {
        (
            _source_shot_path_signature(result),
            _render_variant_signature(result),
        )
        for _profile, result in ordered
    }
    supplemented_paths: set[tuple[tuple[int, str, str], ...]] = set()
    maximum_alternate_pool_size = (
        0
        if len(ordered) >= MINIMUM_PRODUCTION_JOINT_CANDIDATES
        else max(
            (len(pool) for pool in alternates.values()),
            default=0,
        )
    )
    for rank in range(maximum_alternate_pool_size):
        for profile, _unused in profile_results:
            pool = alternates[profile]
            if rank >= len(pool):
                continue
            result = pool[rank]
            path = _source_shot_path_signature(result)
            proof_signature = (path, _proof_variant_signature(result))
            render_signature = (path, _render_variant_signature(result))
            if (
                path in supplemented_paths
                or proof_signature in selected_proof_variants
                or render_signature in selected_render_variants
            ):
                continue
            ordered.append((profile, result))
            supplemented_paths.add(path)
            selected_proof_variants.add(proof_signature)
            selected_render_variants.add(render_signature)
            if len(ordered) >= MINIMUM_PRODUCTION_JOINT_CANDIDATES:
                break
        if len(ordered) >= MINIMUM_PRODUCTION_JOINT_CANDIDATES:
            break
    if len(ordered) < MINIMUM_PRODUCTION_JOINT_CANDIDATES:
        rejection_counts = {
            "unique_source_shot_path_count": len(
                {
                    _source_shot_path_signature(result)
                    for _profile, result in ordered
                }
            ),
            "required_unique_source_shot_path_count": minimum_unique_paths,
            "distinct_render_candidate_count": len(ordered),
            "required_render_candidate_count": (
                MINIMUM_PRODUCTION_JOINT_CANDIDATES
            ),
            "proof_variant_supplement_count": len(supplemented_paths),
        }
        raise NoCompleteTimelinePath(
            "joint visual selection could not retain eight distinct render "
            f"variants ({len(ordered)}/{MINIMUM_PRODUCTION_JOINT_CANDIDATES})",
            reason_code="joint_visual_candidate_cohort_floor_not_met",
            rejection_counts=rejection_counts,
            profile_failures=tuple(profile_failures),
        )

    selections = tuple(
        JointVisualSelectionRecord(
            cadence_profile=profile,
            joint_result=result,
            shifted_timeline_result=_shift_joint_timeline_result(
                result,
                fps=fps,
                candidate_set=candidate_set,
            ),
            source_shot_path=_source_shot_path_signature(result),
            proof_variant_signature=_proof_variant_signature(result),
            role_control_audit=role_audit_by_profile.get(profile),
        )
        for profile, result in ordered
    )
    reviews = tuple(
        review_timeline(
            selection.shifted_timeline_result,
            available_source_count=len(candidate_set.sources),
        )
        for selection in selections
    )
    ranked_indices = _review_ranked_candidate_indices(reviews)
    selected_candidate_index = ranked_indices[0]
    rank_by_index = {
        candidate_index: review_rank
        for review_rank, candidate_index in enumerate(ranked_indices)
    }
    records = tuple(
        VisualTimelineCandidateRecord(
            candidate_index=index,
            cadence_profile=selection.cadence_profile,
            timeline_result=selection.shifted_timeline_result,
            review=reviews[index],
            review_rank=rank_by_index[index],
            selected=index == selected_candidate_index,
        )
        for index, selection in enumerate(selections)
    )
    return JointVisualSelectionSet(
        candidate_set=candidate_set,
        selections=selections,
        candidate_records=records,
        selected_candidate_index=selected_candidate_index,
        profile_failures=tuple(profile_failures),
        role_control_audits=tuple(role_audits),
        shared_option_evidence_cache_audit={
            "contract": "joint_visual_shared_boundary_option_cache_v1",
            "entry_count": len(shared_option_cache),
            "hit_count": shared_option_cache_hit_count,
            "miss_count": shared_option_cache_miss_count,
            "evidence_scope": (
                BoundaryOptionEvidenceScope.SELECTION_SCALAR.value
            ),
            "immutable_boundary_option_sets": True,
            "raw_visual_evidence_persisted": False,
        },
        shared_bridge_evidence_cache_audit=(
            {
                **shared_bridge_evidence_cache.to_dict(),
                "bridge_disposition_batch_call_count": (
                    bridge_disposition_batch_call_count
                ),
                "bridge_disposition_batch_request_count": (
                    bridge_disposition_batch_request_count
                ),
            }
        ),
    )


def candidate_review_summaries(
    selection_set: JointVisualSelectionSet,
    *,
    selected_candidate_index: int,
) -> tuple[dict[str, Any], ...]:
    """Build the existing compact plan summaries for a selected joint path."""

    if not isinstance(selection_set, JointVisualSelectionSet):
        raise TypeError("selection_set must be a JointVisualSelectionSet")
    if not 0 <= selected_candidate_index < len(selection_set.candidate_records):
        raise IndexError("selected candidate index is outside the joint set")
    return tuple(
        _candidate_review_summary(
            candidate_index=record.candidate_index,
            cadence_profile=record.cadence_profile,
            selected=record.candidate_index == selected_candidate_index,
            candidate=record.timeline_result,
            review=record.review,
        )
        for record in selection_set.candidate_records
    )


def _editorial_track_payload(
    path: SegmentTransformPathMaterializationProof,
) -> dict[str, object]:
    track = path.editorial_transform_track
    if track is None:
        raise ValueError("segment transform proof has no renderer track")
    return {
        "reason": track.reason,
        "keyframes": [
            {
                "time_seconds": keyframe.time_seconds,
                "zoom": keyframe.zoom,
                "center_x": keyframe.center_x,
                "center_y": keyframe.center_y,
                "roll_degrees": keyframe.roll_degrees,
            }
            for keyframe in track.keyframes
        ],
    }


@dataclass(frozen=True, slots=True)
class JointVisualBoundaryProofBundle:
    """Atomic process-local proof consumed by render without another decode.

    ``selection_proof`` owns transient descriptors and face-appearance anchors,
    so it is hidden from ``repr`` and never traversed by :meth:`to_dict`.
    The serialized surface contains only exact identities, scalar completeness
    evidence, and the renderer transform tracks already authorized by the
    provider.
    """

    selection_proof: JointSelectionMaterializationProof = field(repr=False)
    frame_offsets: tuple[int, ...]
    bridge_audits: tuple[Mapping[str, object], ...] = ()
    analysis_basis: str = (
        "joint_selection_process_local_boundary_and_segment_proof_bundle_v1"
    )

    def __post_init__(self) -> None:
        proof = self.selection_proof
        if not isinstance(proof, JointSelectionMaterializationProof):
            raise TypeError(
                "selection_proof must be a JointSelectionMaterializationProof"
            )
        offsets = tuple(self.frame_offsets)
        if len(offsets) != len(proof.boundaries) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in offsets
        ):
            raise ValueError("proof-bundle frame offsets are invalid")
        if offsets != tuple(item.frame_offset for item in proof.boundaries):
            raise ValueError("proof-bundle offsets disagree with boundary proofs")
        if proof.boundary_proof_identities != tuple(
            item.proof_identity for item in proof.boundaries
        ):
            raise ValueError("boundary proof identities are inconsistent")
        if proof.segment_path_proof_identities != tuple(
            item.proof_identity for item in proof.segment_paths
        ):
            raise ValueError("segment path proof identities are inconsistent")
        if len(proof.segment_paths) != len(proof.boundaries) + 1:
            raise ValueError("proof bundle needs one path for every segment")
        expected_indices = tuple(range(len(proof.segment_paths)))
        if tuple(
            item.candidate.segment_index for item in proof.segment_paths
        ) != expected_indices:
            raise ValueError("segment path proofs are not exact and contiguous")
        for path in proof.segment_paths:
            if (
                not path.evidence_complete
                or not path.crop_compatible
                or not path.track_compatible
                or not path.endpoints_included
                or not path.full_frame_centers_included
                or path.failure_codes
                or path.editorial_transform_track is None
                or not path.samples
                or len(path.samples) != path.expected_sample_count
            ):
                raise ValueError(
                    "proof bundle contains an unauthorized segment transform path"
                )
        for boundary_index, boundary in enumerate(proof.boundaries):
            outgoing_path = proof.segment_paths[boundary_index]
            incoming_path = proof.segment_paths[boundary_index + 1]
            if (
                outgoing_path.exit_transform != boundary.outgoing_transform
                or incoming_path.entry_transform != boundary.incoming_transform
                or outgoing_path.samples[-1].transform
                != outgoing_path.exit_transform
                or incoming_path.samples[0].transform
                != incoming_path.entry_transform
            ):
                raise ValueError(
                    "boundary proof is not bound to adjacent renderer tracks"
                )
        audits = tuple(MappingProxyType(dict(item)) for item in self.bridge_audits)
        if len(audits) != max(0, len(proof.segment_paths) - 2):
            raise ValueError("proof bundle middle-bridge audit count is invalid")
        expected_audits = tuple(
            dict(item) for item in proof.selected_bridge_audits
        )
        if tuple(dict(item) for item in audits) != expected_audits:
            raise ValueError(
                "proof bundle bridge audits differ from the atomic selection proof"
            )
        if any(
            audit.get("proof_identity")
            != proof.segment_paths[index + 1].proof_identity
            for index, audit in enumerate(audits)
        ):
            raise ValueError("bridge audit identity lost its middle-segment path")
        object.__setattr__(self, "frame_offsets", offsets)
        object.__setattr__(self, "bridge_audits", audits)

    @property
    def selection_identity(self) -> str:
        return self.selection_proof.selection_identity

    @property
    def boundary_proofs(self) -> tuple[BoundaryMaterializationProof, ...]:
        return self.selection_proof.boundaries

    @property
    def segment_path_proofs(
        self,
    ) -> tuple[SegmentTransformPathMaterializationProof, ...]:
        return self.selection_proof.segment_paths

    def to_dict(self) -> dict[str, object]:
        return {
            "analysis_basis": self.analysis_basis,
            "selection_identity": self.selection_identity,
            "boundary_proof_identities": list(
                self.selection_proof.boundary_proof_identities
            ),
            "boundary_frame_offsets": list(self.frame_offsets),
            "segment_path_proof_identities": list(
                self.selection_proof.segment_path_proof_identities
            ),
            "segment_paths": [
                {
                    "proof_identity": path.proof_identity,
                    "segment_index": path.candidate.segment_index,
                    "source_index": path.source_index,
                    "start_master_seconds": path.start_master_seconds,
                    "end_master_seconds": path.end_master_seconds,
                    "duration_frames": path.duration_frames,
                    "expected_sample_count": path.expected_sample_count,
                    "observed_sample_count": len(path.samples),
                    "endpoints_included": path.endpoints_included,
                    "full_frame_centers_included": (
                        path.full_frame_centers_included
                    ),
                    "evidence_complete": path.evidence_complete,
                    "crop_compatible": path.crop_compatible,
                    "track_compatible": path.track_compatible,
                    "failure_codes": list(path.failure_codes),
                    "editorial_transform_track": _editorial_track_payload(path),
                }
                for path in self.selection_proof.segment_paths
            ],
            "middle_bridge_audits": [dict(item) for item in self.bridge_audits],
            "raw_frames_persisted": False,
            "raw_visual_descriptors_persisted": False,
            "raw_candidate_embeddings_persisted": False,
            "biometric_model_vectors_persisted": False,
            "process_local_materialization_retained": True,
        }


def _transition_annotations(
    candidates: Sequence[TimelineCandidate],
    index: int,
    candidate_set: VisualTimelineOptimizerCarrier,
) -> dict[str, object]:
    if index == 0:
        return {
            "face": None,
            "layout": None,
            "pose_basis": None,
            "pose": None,
            "silhouette": None,
            "formation": None,
        }
    previous = candidates[index - 1]
    current = candidates[index]
    current_subject_metadata = candidate_set.subject_metadata_by_candidate.get(
        current.candidate_id,
        _missing_candidate_subject_metadata(),
    )
    previous_subject_metadata = (
        candidate_set.subject_metadata_by_candidate.get(previous.candidate_id)
    )
    face_mismatch = face_geometry_minimax_mismatch(
        previous.exit_face_geometry_window,
        current.entry_face_geometry_window,
    )
    face = (
        face_mismatch
        if previous.exit_face_geometry_window
        and current.entry_face_geometry_window
        else None
    )
    previous_pose = previous.exit_pose_embedding or previous.pose_embedding
    current_pose = current.entry_pose_embedding or current.pose_embedding
    actual_pose_pair = (
        previous_subject_metadata is not None
        and previous_subject_metadata.pose_backends[2]
        != "structure_hog_fallback"
        and current_subject_metadata.pose_backends[0]
        != "structure_hog_fallback"
        and bool(previous_pose)
        and bool(current_pose)
        and len(previous_pose) == len(current_pose)
    )
    if actual_pose_pair:
        pose_basis = "opencv_zoo_mediapipe_pose"
    else:
        pose_basis = "structure_hog_fallback"
        previous_pose = (
            previous.exit_structure_embedding or previous.structure_embedding
        )
        current_pose = current.entry_structure_embedding or current.structure_embedding
    return {
        "face": face,
        "layout": cross_source_layout_mismatch(previous, current),
        "pose_basis": pose_basis,
        "pose": _embedding_rms_distance(previous_pose, current_pose),
        "silhouette": _embedding_rms_distance(
            previous.exit_silhouette_embedding or previous.silhouette_embedding,
            current.entry_silhouette_embedding or current.silhouette_embedding,
        ),
        "formation": comparable_group_formation_mismatch(
            previous.exit_formation_people,
            current.entry_formation_people,
            left_shot_type=previous.exit_shot_type or previous.shot_type,
            right_shot_type=current.entry_shot_type or current.shot_type,
        ),
    }


def _materialize_visual_plan_without_decode(
    selection_set: JointVisualSelectionSet,
    record: VisualTimelineCandidateRecord,
) -> VisualTimelinePlan:
    candidate_set = selection_set.candidate_set
    candidates = record.timeline_result.candidates
    choices: list[VisualTimelineChoice] = []
    for index, candidate in enumerate(candidates):
        source_index = int(candidate.source_id)
        face_metadata = candidate_set.face_metadata_by_candidate.get(
            candidate.candidate_id,
            _missing_candidate_face_metadata(),
        )
        subject_metadata = candidate_set.subject_metadata_by_candidate.get(
            candidate.candidate_id,
            _missing_candidate_subject_metadata(),
        )
        transition = _transition_annotations(
            candidates,
            index,
            candidate_set,
        )
        choices.append(
            VisualTimelineChoice(
                segment_index=candidate.segment_index,
                source_index=source_index,
                canonical_master_start=candidate.start_seconds,
                canonical_master_end=candidate.end_seconds,
                source_local_start=_source_clock_time(
                    candidate.start_seconds,
                    source_index=source_index,
                    audio_offsets=candidate_set.audio_offsets,
                    source_timewarps=candidate_set.source_timewarps,
                ),
                source_local_end=_source_clock_time(
                    candidate.end_seconds,
                    source_index=source_index,
                    audio_offsets=candidate_set.audio_offsets,
                    source_timewarps=candidate_set.source_timewarps,
                ),
                quality_score=candidate.quality_score,
                composition_score=candidate.composition_score,
                boundary_score=candidate.boundary_score,
                shot_scale=candidate.shot_scale,
                shot_type=candidate.shot_type,
                shot_id=candidate.shot_id,
                entry_shot_scale=candidate.entry_shot_scale,
                exit_shot_scale=candidate.exit_shot_scale,
                entry_shot_type=candidate.entry_shot_type,
                exit_shot_type=candidate.exit_shot_type,
                entry_scene_cut_near=candidate.entry_scene_cut_near,
                exit_scene_cut_near=candidate.exit_scene_cut_near,
                intentional_transition_anchor_basis=(
                    candidate.intentional_transition_anchor_basis
                ),
                internal_scene_cut_master_seconds=(
                    candidate.internal_scene_cut_master_seconds
                ),
                entry_boundary_side_stability=(
                    candidate.entry_boundary_side_stability
                ),
                exit_boundary_side_stability=(
                    candidate.exit_boundary_side_stability
                ),
                face_samples_xywh=face_metadata.boxes,
                face_match_candidate_failure=face_metadata.failure,
                face_detector_backends=face_metadata.backends,
                face_landmark_samples_xy=face_metadata.landmarks,
                face_head_roll_samples_radians=face_metadata.rolls,
                face_eye_scale_samples=face_metadata.eye_scales,
                entry_face_geometry_window=candidate.entry_face_geometry_window,
                exit_face_geometry_window=candidate.exit_face_geometry_window,
                face_transition_minimax_mismatch_from_previous=transition["face"],
                pose_backends=subject_metadata.pose_backends,
                pose_confidences=subject_metadata.pose_confidences,
                detected_person_counts=subject_metadata.person_counts,
                formation_person_counts=subject_metadata.formation_person_counts,
                formation_upper_body_counts=(
                    subject_metadata.formation_upper_body_counts
                ),
                shot_scale_bases=subject_metadata.shot_scale_bases,
                silhouette_samples_xywh=subject_metadata.silhouette_boxes,
                silhouette_centroid_samples_xy=(
                    subject_metadata.silhouette_centroids
                ),
                silhouette_backends=subject_metadata.silhouette_backends,
                silhouette_confidences=subject_metadata.silhouette_confidences,
                layout_transition_mismatch_from_previous=transition["layout"],
                pose_transition_basis_from_previous=transition["pose_basis"],
                pose_transition_mismatch_from_previous=transition["pose"],
                silhouette_transition_mismatch_from_previous=(
                    transition["silhouette"]
                ),
                formation_transition_mismatch_from_previous=(
                    transition["formation"]
                ),
            )
        )
    return VisualTimelinePlan(
        choices=tuple(choices),
        optimizer_score=record.timeline_result.total_score,
        analysis_basis=(
            "joint_decoded_boundary_offset_and_transform_materialization_v1"
        ),
        selected_candidate_index=record.candidate_index,
        candidate_generation_basis=JOINT_VISUAL_CANDIDATE_ADAPTER_BASIS,
        candidate_review_summaries=candidate_review_summaries(
            selection_set,
            selected_candidate_index=record.candidate_index,
        ),
        source_clock_basis=candidate_set.source_clock_basis,
    )


def _transform_effects(
    transform: object,
) -> tuple[bool, bool, bool]:
    crop = getattr(transform, "crop_xywh", None)
    rotation = float(getattr(transform, "rotation_degrees", 0.0))
    if crop is None:
        zoom = False
        pan = False
    else:
        x, y, width, height = (float(value) for value in crop)
        zoom = (
            not math.isclose(width, 1.0, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(height, 1.0, rel_tol=0.0, abs_tol=1e-9)
        )
        pan = (
            not math.isclose(
                x + width / 2.0,
                0.5,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                y + height / 2.0,
                0.5,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
    return (zoom, pan, not math.isclose(rotation, 0.0, abs_tol=1e-9))


def _joint_editorial_motion_plan(
    proof_bundle: JointVisualBoundaryProofBundle,
    segments: Sequence[RenderSegment],
) -> EditorialMotionPlan:
    """Expose provider tracks as the sole downstream motion authority."""

    paths = proof_bundle.segment_path_proofs
    render_segments = tuple(segments)
    if len(paths) != len(render_segments):
        raise ValueError("editorial motion paths and render segments must align")
    accepted: list[int] = []
    natural: list[int] = []
    zoom_count = 0
    pan_count = 0
    roll_count = 0
    for boundary_index, boundary in enumerate(proof_bundle.boundary_proofs):
        outgoing_effects = _transform_effects(boundary.outgoing_transform)
        incoming_effects = _transform_effects(boundary.incoming_transform)
        effects = tuple(
            left or right
            for left, right in zip(
                outgoing_effects,
                incoming_effects,
                strict=True,
            )
        )
        if any(effects):
            accepted.append(boundary_index)
            zoom_count += int(effects[0])
            pan_count += int(effects[1])
            roll_count += int(effects[2])
        else:
            natural.append(boundary_index)
    accepted_set = set(accepted)
    motion_segments: list[EditorialMotionSegment] = []
    for path, render_segment in zip(paths, render_segments, strict=True):
        track = path.editorial_transform_track
        if track is None:  # pragma: no cover - proof-bundle invariant
            raise RuntimeError("joint segment path lost its renderer track")
        if render_segment.editorial_transform is not track:
            raise ValueError("editorial motion plan lost provider track identity")
        adjacent_boundaries = tuple(
            index
            for index in (
                path.candidate.segment_index - 1,
                path.candidate.segment_index,
            )
            if index in accepted_set
        )
        reasons = ["decoded_joint_renderer_track_authority"]
        if adjacent_boundaries:
            reasons.extend(
                f"joint_boundary_{index}_transform_authorized"
                for index in adjacent_boundaries
            )
        else:
            reasons.append("joint_identity_or_natural_match_track_authority")
        motion_segments.append(
            EditorialMotionSegment(
                segment_index=path.candidate.segment_index,
                source_index=path.source_index,
                track=track,
                reasons=tuple(reasons),
            )
        )
    return EditorialMotionPlan(
        segments=tuple(motion_segments),
        boundary_match_count=len(accepted),
        zoom_event_count=zoom_count,
        pan_event_count=pan_count,
        roll_event_count=roll_count,
        blocked_dissolve_segment_count=0,
        # The provider proves an exact renderer crop path.  It does not run
        # the separate pixel-feature affine estimator, so reporting these
        # crop matches as affine matches would overstate the evidence.
        affine_boundary_match_count=0,
        interior_zoom_pulse_count=0,
        skipped_boundary_reasons=(
            (("already_naturally_matched", len(natural)),)
            if natural
            else ()
        ),
        section_directed=False,
        analysis_basis=JOINT_EDITORIAL_MOTION_AUTHORITY_BASIS,
        accepted_boundary_indices=tuple(accepted),
        natural_match_boundary_indices=tuple(natural),
    )


@dataclass(frozen=True, slots=True)
class JointMaterializedVisualCandidate:
    """Renderer-ready joint alternative and its atomic decoded proof bundle."""

    selection_set: JointVisualSelectionSet = field(repr=False)
    candidate_record: VisualTimelineCandidateRecord
    visual_plan: VisualTimelinePlan = field(repr=False)
    segments: tuple[RenderSegment, ...]
    common_master_start: float
    boundary_proof_bundle: JointVisualBoundaryProofBundle = field(repr=False)
    editorial_motion_plan: EditorialMotionPlan = field(repr=False)
    output_fps: float = 29.97

    def __post_init__(self) -> None:
        index = self.candidate_record.candidate_index
        if (
            not 0 <= index < len(self.selection_set.candidate_records)
            or self.selection_set.candidate_records[index]
            is not self.candidate_record
        ):
            raise ValueError("candidate record does not belong to joint selection")
        if self.visual_plan.selected_candidate_index != index:
            raise ValueError("visual plan selects another joint candidate")
        segments = tuple(self.segments)
        paths = self.boundary_proof_bundle.segment_path_proofs
        if not segments or len(segments) != len(paths):
            raise ValueError("render segments lost an authorized transform path")
        if len(self.visual_plan.choices) != len(segments):
            raise ValueError("visual plan and render segments do not align")
        motion_by_segment = {
            item.segment_index: item
            for item in self.editorial_motion_plan.segments
        }
        if len(motion_by_segment) != len(segments):
            raise ValueError("editorial motion authority does not cover every segment")
        expected_master_start = 0.0
        for segment, choice, path in zip(
            segments,
            self.visual_plan.choices,
            paths,
            strict=True,
        ):
            if not math.isclose(
                segment.master_start,
                expected_master_start,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("joint render segments must be contiguous")
            if segment.editorial_transform is not path.editorial_transform_track:
                raise ValueError("render segment is not bound to its decoded track")
            motion_segment = motion_by_segment.get(path.candidate.segment_index)
            if (
                motion_segment is None
                or motion_segment.source_index != path.source_index
                or motion_segment.track is not path.editorial_transform_track
            ):
                raise ValueError(
                    "editorial motion plan is not bound to provider track identity"
                )
            if (
                segment.source_index != path.source_index
                or choice.segment_index != path.candidate.segment_index
                or choice.source_index != path.source_index
                or choice.shot_id != path.candidate.shot_id
            ):
                raise ValueError("render segment path identity drifted")
            expected_master_start += segment.duration
        object.__setattr__(self, "segments", segments)
        if (
            not math.isfinite(self.common_master_start)
            or self.common_master_start < 0.0
        ):
            raise ValueError("common_master_start must be non-negative and finite")
        if not math.isfinite(self.output_fps) or self.output_fps <= 0.0:
            raise ValueError("output_fps must be positive and finite")
        if (
            self.editorial_motion_plan.analysis_basis
            != JOINT_EDITORIAL_MOTION_AUTHORITY_BASIS
        ):
            raise ValueError("editorial motion plan is not joint-track authority")

    @property
    def candidate_index(self) -> int:
        return self.candidate_record.candidate_index

    @property
    def review_rank(self) -> int:
        return self.candidate_record.review_rank

    @property
    def cadence_profile(self) -> str:
        return self.candidate_record.cadence_profile

    @property
    def candidate_set(self) -> VisualTimelineOptimizerCarrier:
        return self.selection_set.candidate_set

    @property
    def joint_result(self) -> JointTimelineResult:
        return self.selection_set.selection_for_candidate(
            self.candidate_index
        ).joint_result

    @property
    def privacy_safe_signature(self) -> str:
        selection = self.selection_set.selection_for_candidate(
            self.candidate_index
        )
        return selection.privacy_safe_signature

    @property
    def path_signature(self) -> str:
        """Privacy-safe exact source/shot/proof identity for history joins."""

        return self.privacy_safe_signature

    @property
    def candidate_history_entry(self) -> dict[str, Any]:
        return {
            **self.candidate_record.to_dict(),
            "joint_path_signature": self.privacy_safe_signature,
            "boundary_proof_bundle": self.boundary_proof_bundle.to_dict(),
            "editorial_motion_plan": self.editorial_motion_plan.to_dict(),
        }

    @property
    def reviewer_history_entry(self) -> dict[str, Any]:
        return self.candidate_record.review.to_dict()

    def to_render_plan(
        self,
        *,
        output: Path,
        reference_source: int = 0,
        aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        transitions: Sequence[RenderTransition] = (),
    ) -> RenderPlan:
        ratio = (
            aspect_ratio
            if isinstance(aspect_ratio, AspectRatio)
            else AspectRatio(aspect_ratio)
        )
        if (width is None) != (height is None):
            raise ValueError("width and height must be supplied together")
        if width is None or height is None:
            width, height = output_size(ratio)
        resolved_fps = self.output_fps if fps is None else float(fps)
        if not math.isclose(
            resolved_fps,
            self.output_fps,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("render fps cannot differ from decoded joint evidence")
        return RenderPlan(
            sources=self.selection_set.candidate_set.sources,
            segments=self.segments,
            reference_source=reference_source,
            aspect_ratio=ratio,
            output=Path(output),
            master_audio_start=self.common_master_start,
            width=width,
            height=height,
            fps=resolved_fps,
            transitions=tuple(transitions),
        )


def _proof_bundle(
    selection: JointVisualSelectionRecord,
    provider: DecodedBoundaryOptionProvider,
) -> JointVisualBoundaryProofBundle:
    result = selection.joint_result
    proof = provider.selection_materialization_proof(
        result.boundary_options,
        segment_bridge_proofs=result.segment_bridge_proofs,
    )
    return JointVisualBoundaryProofBundle(
        selection_proof=proof,
        frame_offsets=tuple(
            item.frame_offset for item in result.boundary_options
        ),
        bridge_audits=proof.selected_bridge_audits,
    )


def _materialize_joint_candidate(
    selection_set: JointVisualSelectionSet,
    record: VisualTimelineCandidateRecord,
    *,
    provider: DecodedBoundaryOptionProvider,
    fps: float,
) -> JointMaterializedVisualCandidate:
    selection = selection_set.selection_for_candidate(record.candidate_index)
    proof_bundle = _proof_bundle(selection, provider)
    visual_plan = _materialize_visual_plan_without_decode(selection_set, record)
    common_master_start = visual_plan.choices[0].canonical_master_start
    segments: list[RenderSegment] = []
    for choice, path in zip(
        visual_plan.choices,
        proof_bundle.segment_path_proofs,
        strict=True,
    ):
        if (
            path.candidate.source_id != str(choice.source_index)
            or path.candidate.shot_id != choice.shot_id
            or path.source_index != choice.source_index
            or not math.isclose(
                path.start_master_seconds,
                choice.canonical_master_start,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                path.end_master_seconds,
                choice.canonical_master_end,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                path.duration_frames,
                choice.duration_seconds * fps,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError("decoded segment path does not bind the shifted plan")
        track = path.editorial_transform_track
        if track is None:  # pragma: no cover - proof-bundle invariant
            raise RuntimeError("decoded segment path lost its renderer track")
        segment = aligned_render_segment(
            source_index=choice.source_index,
            reference_start=choice.canonical_master_start,
            common_master_start=common_master_start,
            source_offset=selection_set.candidate_set.audio_offsets[
                choice.source_index
            ],
            duration=choice.duration_seconds,
            editorial_transform=track,
            timewarp=selection_set.candidate_set.source_timewarps[
                choice.source_index
            ],
        )
        if (
            not math.isclose(
                segment.source_start,
                path.samples[0].source_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                segment.source_end,
                path.samples[-1].source_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError("renderer source clock disagrees with decoded path")
        segments.append(segment)
    return JointMaterializedVisualCandidate(
        selection_set=selection_set,
        candidate_record=record,
        visual_plan=visual_plan,
        segments=tuple(segments),
        common_master_start=common_master_start,
        boundary_proof_bundle=proof_bundle,
        editorial_motion_plan=_joint_editorial_motion_plan(
            proof_bundle,
            segments,
        ),
        output_fps=fps,
    )


def materialize_joint_visual_candidates(
    selection_set: JointVisualSelectionSet,
    *,
    provider: DecodedBoundaryOptionProvider,
    fps: float = 29.97,
    maximum_candidates: int = MINIMUM_PRODUCTION_JOINT_CANDIDATES,
) -> tuple[JointMaterializedVisualCandidate, ...]:
    """Materialize reviewer-ranked alternatives from provider RAM only."""

    if not isinstance(selection_set, JointVisualSelectionSet):
        raise TypeError("selection_set must be a JointVisualSelectionSet")
    if not isinstance(provider, DecodedBoundaryOptionProvider):
        raise TypeError("provider must be a DecodedBoundaryOptionProvider")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    _assert_provider_binding(selection_set.candidate_set, provider, fps=fps)
    if (
        isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or not MINIMUM_PRODUCTION_JOINT_CANDIDATES
        <= maximum_candidates
        <= len(selection_set.candidate_records)
    ):
        raise ValueError("maximum_candidates must retain at least exact eight")
    ranked = selection_set.ranked_records[:maximum_candidates]
    # Build in a local tuple.  If any proof is missing or stale, no partial
    # renderer candidate escapes this atomic adapter call.
    return tuple(
        _materialize_joint_candidate(
            selection_set,
            record,
            provider=provider,
            fps=fps,
        )
        for record in ranked
    )


def build_joint_visual_candidates(
    candidate_set: VisualTimelineOptimizerCarrier,
    *,
    provider: DecodedBoundaryOptionProvider,
    fps: float = 29.97,
    hard_forbidden_boundary_edges: TimelineNoGoodSet = TimelineNoGoodSet(),
) -> tuple[JointMaterializedVisualCandidate, ...]:
    """Select and atomically materialize the exact-eight production cohort."""

    selected = select_joint_visual_candidates(
        candidate_set,
        provider=provider,
        fps=fps,
        hard_forbidden_boundary_edges=hard_forbidden_boundary_edges,
        minimum_unique_paths=MINIMUM_PRODUCTION_UNIQUE_SOURCE_SHOT_PATHS,
        maximum_candidates=MINIMUM_PRODUCTION_JOINT_CANDIDATES,
    )
    return materialize_joint_visual_candidates(
        selected,
        provider=provider,
        fps=fps,
        maximum_candidates=MINIMUM_PRODUCTION_JOINT_CANDIDATES,
    )


__all__ = [
    "JOINT_EDITORIAL_MOTION_AUTHORITY_BASIS",
    "JOINT_VISUAL_CANDIDATE_ADAPTER_BASIS",
    "JointMaterializedVisualCandidate",
    "JointVisualBoundaryProofBundle",
    "JointVisualSelectionRecord",
    "JointVisualSelectionSet",
    "MINIMUM_PRODUCTION_JOINT_CANDIDATES",
    "MINIMUM_PRODUCTION_UNIQUE_SOURCE_SHOT_PATHS",
    "build_joint_visual_candidates",
    "candidate_review_summaries",
    "materialize_joint_visual_candidates",
    "select_joint_visual_candidates",
]
