"""Dense, non-identifying editorial-camera review for official member fancams.

Automatic member sources are already authenticated as official target-member
9:16 fancams by metadata and media contracts.  This layer still never names a
face from pixels: it only uses a continuously dominant face box around a
selected hard cut, then re-decodes every transformed output frame before a
bounded scale/position match is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Sequence

import cv2

from .direct_member_visual import (
    DIRECT_MEMBER_PROFESSIONAL_ATTENTION_TRIGGER_TOLERANCE,
    DirectMemberVisualSelection,
)
from .direct_member_transform import (
    DirectMemberBoundaryTransformRepair,
    DirectMemberSourceFrameGeometry,
    direct_member_source_frame_geometry,
    map_box_to_portrait_fill,
)
from .direct_member_bilateral_transform import DirectMemberBilateralTransformRepair
from .direct_member_target_face import (
    FOCAL_FACE_BASIS,
    FocalFaceCandidate,
    detect_focal_face_candidates,
    select_focal_face_matching_reference,
    select_focal_face_pair,
)
from .editorial_motion import EditorialMotionPlan, EditorialMotionSegment
from .focal_reference_consensus import (
    FocalReferenceConsensus,
    SoloFaceObservation,
    build_focal_reference_consensus,
)
from .focal_selector_geometry import (
    FOCAL_MEMBER_MINIMUM_SUBJECT_VISIBLE_FRACTION,
    FOCAL_SELECTOR_BODY_BASIS,
    FocalSelectorGeometry, focal_selector_geometry,
)
from .member_editorial_motion import (
    MemberEditorialMotionConfig,
    _State,
    _center,
    _dedupe_points,
    _frame_schedule,
    _mismatch,
    _project,
    _state_for_match,
    _track_for_states,
    _visible_fraction,
)
from .pose import SILHOUETTE_BACKEND
from .render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderSegment,
    RenderTransition,
    TransitionKind,
)
from .visual_score import _VideoSampler, _working_frame


_FACE_ATTENTION_BASIS = "unambiguous_dominant_face_bbox_geometry_only"


@dataclass(frozen=True, slots=True)
class DirectMemberFaceSample:
    bbox_xywh: tuple[float, float, float, float]
    face_count: int
    detector_backend: str
    subject_bbox_xywh: tuple[float, float, float, float] | None = None
    subject_confidence: float = 0.0
    subject_geometry_basis: str = "none"
    alignment_geometry_basis_override: str | None = None

    def __post_init__(self) -> None:
        if len(self.bbox_xywh) != 4:
            raise ValueError("direct member face box needs four values")
        x, y, width, height = tuple(float(value) for value in self.bbox_xywh)
        if any(not math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError("direct member face box must be finite")
        if not (
            0.0 <= x < 1.0
            and 0.0 <= y < 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
            and x + width <= 1.0 + 1e-7
            and y + height <= 1.0 + 1e-7
        ):
            raise ValueError("direct member face box must fit normalized bounds")
        if (
            isinstance(self.face_count, bool)
            or not isinstance(self.face_count, int)
            or self.face_count not in {0, 1}
        ):
            raise ValueError(
                "direct member geometry sample needs exactly one face or "
                "verified subject-only geometry"
            )
        backend = self.detector_backend.strip().casefold()
        subject_box = self.subject_bbox_xywh
        if subject_box is not None:
            if len(subject_box) != 4:
                raise ValueError("direct member subject box needs four values")
            sx, sy, swidth, sheight = tuple(float(value) for value in subject_box)
            if any(
                not math.isfinite(value)
                for value in (sx, sy, swidth, sheight)
            ) or not (
                0.0 <= sx < 1.0
                and 0.0 <= sy < 1.0
                and 0.0 < swidth <= 1.0
                and 0.0 < sheight <= 1.0
                and sx + swidth <= 1.0 + 1e-7
                and sy + sheight <= 1.0 + 1e-7
            ):
                raise ValueError(
                    "direct member subject box must fit normalized bounds"
                )
            subject_box = (sx, sy, swidth, sheight)
        confidence = float(self.subject_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("direct member subject confidence must be normalized")
        basis = self.subject_geometry_basis.strip()
        alignment_override = self.alignment_geometry_basis_override
        if alignment_override is not None and alignment_override not in (FOCAL_FACE_BASIS, FOCAL_SELECTOR_BODY_BASIS):
            raise ValueError("direct member alignment basis override is invalid")
        if alignment_override == FOCAL_FACE_BASIS and self.face_count != 1:
            raise ValueError("focal face basis requires one verified face")
        if alignment_override == FOCAL_SELECTOR_BODY_BASIS and (self.face_count != 0 or basis != 'source_local_focal_body_track'):
            raise ValueError('focal body basis requires a verified source-local body track')
        if subject_box is None:
            if confidence != 0.0 or basis != "none":
                raise ValueError(
                    "direct member missing subject box cannot claim geometry"
                )
        elif confidence <= 0.0 or basis not in {
            "pphumanseg_full_subject_silhouette",
            "face_associated_formation_person_box",
            "source_local_focal_body_track",
        }:
            raise ValueError("direct member subject geometry is unverified")
        if self.face_count == 1:
            if backend != "yunet":
                raise ValueError(
                    "direct member face sample needs verified YuNet geometry"
                )
        elif (
            backend != "none"
            or subject_box is None
            or any(
                abs(value - subject_value) > 1e-7
                for value, subject_value in zip(
                    (x, y, width, height),
                    subject_box,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "subject-only geometry must use its verified body box"
            )
        object.__setattr__(self, "bbox_xywh", (x, y, width, height))
        object.__setattr__(self, "detector_backend", backend)
        object.__setattr__(self, "subject_bbox_xywh", subject_box)
        object.__setattr__(self, "subject_confidence", confidence)
        object.__setattr__(self, "subject_geometry_basis", basis)
        object.__setattr__(
            self,
            "alignment_geometry_basis_override",
            alignment_override,
        )

    @property
    def alignment_geometry_basis(self) -> str:
        if self.alignment_geometry_basis_override is not None:
            return self.alignment_geometry_basis_override
        if self.face_count == 1:
            return "unambiguous_dominant_face_bbox_geometry_only"
        if self.subject_geometry_basis == "pphumanseg_full_subject_silhouette":
            return "aggregate_human_silhouette"
        return "unambiguous_dominant_person_box_geometry_only"


DirectMemberGeometryProvider = Callable[
    [int, float], DirectMemberFaceSample | None
]


def _is_face_alignment_basis(value: str | None) -> bool:
    return value in {_FACE_ATTENTION_BASIS, FOCAL_FACE_BASIS}


class DirectMemberEditorialMotionError(ValueError):
    """Structured dense-render failure that can trigger path reselection."""

    def __init__(
        self,
        message: str,
        *,
        missing_boundary_indices: Sequence[int],
        rejection_reasons: dict[int, set[str]],
        rejection_context: dict[int, str],
    ) -> None:
        super().__init__(message)
        self.missing_boundary_indices = tuple(
            int(value) for value in missing_boundary_indices
        )
        self.rejection_reasons = {
            int(boundary_index): tuple(sorted(str(value) for value in values))
            for boundary_index, values in rejection_reasons.items()
        }
        self.rejection_context = {
            int(boundary_index): str(value)
            for boundary_index, value in rejection_context.items()
        }


def _enable_dense_sequential_frame_reads(sampler: object) -> None:
    """Avoid an AV1 random seek for every adjacent dense-review frame.

    This optimization is intentionally local to the direct-member reviewer.
    It falls back to the canonical timestamp seek whenever OpenCV's reported
    frame position does not prove that the capture is exactly one frame ahead.
    """

    required_attributes = (
        "cache",
        "capture",
        "frame_at",
        "frame_decode_count",
        "frame_identity",
        "frame_rate",
    )
    if any(not hasattr(sampler, name) for name in required_attributes):
        return
    original_frame_at = sampler.frame_at  # type: ignore[attr-defined]
    capture = sampler.capture  # type: ignore[attr-defined]
    cache = sampler.cache  # type: ignore[attr-defined]
    frame_rate = float(sampler.frame_rate)  # type: ignore[attr-defined]
    if not math.isfinite(frame_rate) or frame_rate <= 0.0:
        return
    previous_decoded_identity = None
    sequential_decode_count = 0
    random_seek_decode_count = 0

    def frame_at(seconds: float):
        nonlocal previous_decoded_identity
        nonlocal sequential_decode_count
        nonlocal random_seek_decode_count
        key = sampler.frame_identity(seconds)  # type: ignore[attr-defined]
        if key in cache:
            return original_frame_at(seconds)
        expected_step = 1.0 / frame_rate
        adjacent = bool(
            previous_decoded_identity is not None
            and key.frame_index == previous_decoded_identity.frame_index + 1
            and key.fps_numerator
            == previous_decoded_identity.fps_numerator
            and key.fps_denominator
            == previous_decoded_identity.fps_denominator
            and abs(
                key.canonical_seconds
                - previous_decoded_identity.canonical_seconds
                - expected_step
            )
            <= expected_step * 0.05 + 1e-9
        )
        if adjacent:
            position_before = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if (
                math.isfinite(position_before)
                and abs(position_before - key.frame_index) <= 0.51
            ):
                success, frame = capture.read()
                sampler.frame_decode_count += 1  # type: ignore[attr-defined]
                position_after = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
                if (
                    success
                    and math.isfinite(position_after)
                    and abs(position_after - (key.frame_index + 1)) <= 0.51
                ):
                    value = _working_frame(frame)
                    cache[key] = value
                    while len(cache) > 32:
                        cache.popitem(last=False)
                    previous_decoded_identity = key
                    sequential_decode_count += 1
                    setattr(
                        sampler,
                        "dense_sequential_frame_decode_count",
                        sequential_decode_count,
                    )
                    return value
        before = int(sampler.frame_decode_count)  # type: ignore[attr-defined]
        value = original_frame_at(seconds)
        if int(sampler.frame_decode_count) > before:  # type: ignore[attr-defined]
            random_seek_decode_count += 1
            setattr(
                sampler,
                "dense_random_seek_frame_decode_count",
                random_seek_decode_count,
            )
        position_after = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
        previous_decoded_identity = (
            key
            if value is not None
            and math.isfinite(position_after)
            and abs(position_after - (key.frame_index + 1)) <= 0.51
            else None
        )
        return value

    setattr(sampler, "frame_at", frame_at)
    setattr(sampler, "dense_sequential_frame_decode_count", 0)
    setattr(sampler, "dense_random_seek_frame_decode_count", 0)


class DirectMemberDenseFaceProvider:
    """Decode selected geometry on canonical source frames, never identities."""

    def __init__(
        self,
        sources: Sequence[Path],
        *,
        feature_cache_directory: Path | None = None,
        feature_derived_cache_directory: Path | None = None,
        require_persistent_feature_cache: bool = False,
        require_focal_consensus: bool = False,
        focal_candidate_provider: Callable[[int, float], tuple[FocalFaceCandidate, ...]] | None = None,
        shared_source_frames: Sequence[object] | None = None,
    ) -> None:
        from .shared_source_frames import validate_shared_sources
        shared = validate_shared_sources(sources, shared_source_frames)
        if not isinstance(require_persistent_feature_cache, bool):
            raise TypeError("require_persistent_feature_cache must be boolean")
        if not isinstance(require_focal_consensus, bool):
            raise TypeError("require_focal_consensus must be boolean")
        if (
            require_persistent_feature_cache
            and feature_derived_cache_directory is None
        ):
            raise ValueError(
                "require_persistent_feature_cache needs an explicit "
                "feature_derived_cache_directory"
            )
        self._sources = tuple(Path(source) for source in sources)
        self._samplers: list[_VideoSampler | None] = []
        self._cache: dict[
            tuple[object, ...], DirectMemberFaceSample | None
        ] = {}
        self._focal_candidate_cache: dict[
            tuple[object, ...], tuple[FocalFaceCandidate, ...]
        ] = {}
        self._focal_references: list[list[FocalFaceCandidate]] = [
            [] for _source in self._sources
        ]
        self._require_focal_consensus = require_focal_consensus
        if focal_candidate_provider is not None and not callable(focal_candidate_provider):
            raise TypeError('focal candidate provider must be callable')
        self._external_focal_candidate_provider = focal_candidate_provider
        self._frozen_focal_banks: dict[int, FocalReferenceConsensus | None] = {}
        self._frozen_focal_queries: dict[tuple[object, ...], FocalFaceCandidate | None] = {}
        self._focal_temporal_recovery_count = 0
        self._focal_boundary_reference_capture: list[
            tuple[
                int,
                int,
                FocalFaceCandidate | None,
                FocalFaceCandidate | None,
            ]
        ] = []
        self._request_count = 0
        self._cache_hit_count = 0
        self._decode_count = 0
        self._focal_pair_request_count = 0
        self._focal_pair_match_count = 0
        self._focal_reference_match_count = 0
        self._require_persistent_feature_cache = (
            require_persistent_feature_cache
        )
        sampler_arguments: dict[str, object] = {}
        if feature_cache_directory is not None:
            sampler_arguments["feature_cache_directory"] = (
                feature_cache_directory
            )
        if feature_derived_cache_directory is not None:
            sampler_arguments["feature_derived_cache_directory"] = (
                feature_derived_cache_directory
            )
        if require_persistent_feature_cache:
            sampler_arguments["require_persistent_feature_cache"] = True
        try:
            for source_index, source in enumerate(self._sources):
                sampler: _VideoSampler | None = None
                try:
                    sampler = _VideoSampler(source, **sampler_arguments,
                        **({'shared_source_frames': shared[source_index]} if shared is not None else {}))
                    # The sampler's reader owns capture position and its pixel
                    # window, even while lazily uninitialized. A second wrapper
                    # that reads capture directly makes the reader detect an
                    # external seek and permanently disable locality reuse.
                    if not hasattr(sampler, "_seek_locality"):
                        _enable_dense_sequential_frame_reads(sampler)
                except (OSError, TypeError, ValueError, cv2.error):
                    if sampler is not None:
                        sampler.close()
                    sampler = None
                except BaseException:
                    if sampler is not None:
                        sampler.close()
                    raise
                self._samplers.append(sampler)
        except BaseException:
            for sampler in self._samplers:
                if sampler is not None:
                    sampler.close()
            self._samplers.clear()
            raise

    @staticmethod
    def _identity_key(
        source_index: int,
        sampler: _VideoSampler,
        seconds: float,
    ) -> tuple[object, ...]:
        identity = sampler.frame_identity(seconds)
        return (
            source_index,
            identity.frame_index,
            identity.fps_numerator,
            identity.fps_denominator,
        )

    def _focal_candidates(
        self,
        source_index: int,
        seconds: float,
    ) -> tuple[FocalFaceCandidate, ...]:
        sampler = self._samplers[source_index]
        if sampler is None:
            return ()
        if self._require_persistent_feature_cache:
            sampler.assert_required_persistent_cache_available()
        key = self._identity_key(source_index, sampler, seconds)
        if key in self._focal_candidate_cache:
            return self._focal_candidate_cache[key]
        before = sampler.frame_decode_count
        try:
            external = getattr(self, '_external_focal_candidate_provider', None)
            if external is not None:
                # Seed and query faces must come from the same appearance
                # basis. Do not silently mix thumbnail/native embeddings.
                candidates = external(source_index, sampler.frame_identity(seconds).canonical_seconds)
                if not isinstance(candidates, tuple) or any(not isinstance(c, FocalFaceCandidate) for c in candidates):
                    raise TypeError('external focal provider returned invalid evidence')
            else:
                frame = sampler.frame_at(seconds)
                candidates = (
                    detect_focal_face_candidates(frame)
                    if frame is not None
                    else ()
                )
        except (cv2.error, ValueError, FloatingPointError):
            candidates = ()
        self._decode_count += max(0, sampler.frame_decode_count - before)
        self._focal_candidate_cache[key] = candidates
        return candidates

    @staticmethod
    def _associated_subject(
        descriptor: object,
        candidate: FocalFaceCandidate,
    ) -> tuple[
        tuple[float, float, float, float] | None,
        float,
        str,
    ]:
        center_x, center_y = candidate.center_xy
        associated = tuple(
            person
            for person in getattr(descriptor, "formation_people", ())
            if person.detection_confidence >= 0.35
            and person.bbox_xywh[0] - 1e-6
            <= center_x
            <= person.bbox_xywh[0] + person.bbox_xywh[2] + 1e-6
            and person.bbox_xywh[1] - 1e-6
            <= center_y
            <= person.bbox_xywh[1] + person.bbox_xywh[3] + 1e-6
        )
        if not associated:
            return (None, 0.0, "none")
        selected = min(
            associated,
            key=lambda person: (
                person.bbox_xywh[2] * person.bbox_xywh[3],
                -person.detection_confidence,
            ),
        )
        return (
            selected.bbox_xywh,
            selected.detection_confidence,
            "face_associated_formation_person_box",
        )

    def _sample_for_focal_candidate(
        self,
        source_index: int,
        seconds: float,
        candidate: FocalFaceCandidate,
    ) -> DirectMemberFaceSample:
        sampler = self._samplers[source_index]
        assert sampler is not None
        before = sampler.frame_decode_count
        descriptor = sampler.feature_descriptor_at(
            seconds,
            include_subject_geometry=True,
            include_formation_pose=True,
        )
        self._decode_count += max(0, sampler.frame_decode_count - before)
        subject_box, confidence, basis = (
            self._associated_subject(descriptor, candidate)
            if descriptor is not None
            else (None, 0.0, "none")
        )
        return DirectMemberFaceSample(
            candidate.bbox_xywh,
            1,
            "yunet",
            subject_box,
            confidence,
            basis,
            FOCAL_FACE_BASIS,
        )

    def _register_focal_reference(
        self,
        source_index: int,
        candidate: FocalFaceCandidate,
    ) -> None:
        if getattr(self, "_require_focal_consensus", False):
            # Query matches must never expand the acquisition-time identity
            # bank, even when the match was recovered through a nearby frame.
            return
        references = self._focal_references[source_index]
        if candidate not in references:
            references.append(candidate)
        if len(references) > 32:
            del references[:-32]

    def _ensure_focal_consensus(self, source_index: int) -> FocalReferenceConsensus | None:
        if source_index in self._frozen_focal_banks:
            return self._frozen_focal_banks[source_index]
        sampler = self._samplers[source_index]
        if sampler is None:
            self._frozen_focal_banks[source_index] = None
            return None
        capture = sampler.capture
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        observations = []
        bank = None
        if math.isfinite(fps) and fps > 0 and math.isfinite(count) and count > 0:
            duration = count/fps
            # An initially occluded grid does not exclude a scarce source:
            # inspect the 20 interleaved frames once, with the same evidence
            # rules and a hard 41-frame per-source bootstrap bound.
            for positions in (tuple(range(21)), tuple(i+.5 for i in range(20))):
                for position in positions:
                    seconds = duration*(.03+.94*position/20)
                    candidates = self._focal_candidates(source_index, seconds)
                    if (len(candidates) == 1 and .2 <= candidates[0].center_xy[0] <= .8
                            and candidates[0].bbox_xywh[3] >= .05):
                        observations.append(SoloFaceObservation(seconds, candidates[0]))
                bank = build_focal_reference_consensus(observations)
                if bank is not None:
                    break
        self._frozen_focal_banks[source_index] = bank
        # These are immutable seeds, not pair-local discoveries. Preserve the
        # existing transient-reference interface for downstream diagnostics.
        self._focal_references[source_index] = (
            [observation.candidate for observation in bank.references] if bank else []
        )
        return bank

    def _frozen_focal_candidate(
        self, source_index: int, seconds: float, *, direction: int = 0,
    ) -> FocalFaceCandidate | None:
        sampler = self._samplers[source_index]
        if sampler is None:
            return None
        if self._require_persistent_feature_cache:
            sampler.assert_required_persistent_cache_available()
        key = (*self._identity_key(source_index, sampler, seconds), direction)
        if key in self._frozen_focal_queries:
            return self._frozen_focal_queries[key]
        bank = self._ensure_focal_consensus(source_index)
        candidates = self._focal_candidates(source_index, seconds)
        selected = bank.select(candidates) if bank else None
        if bank and candidates and selected is None:
            observations = []
            for sign in ((direction,) if direction else (-1, 1)):
                for offset in (.10, .20, .30):
                    adjacent_seconds = max(0., seconds+sign*offset)
                    adjacent = bank.select(self._focal_candidates(source_index, adjacent_seconds))
                    if adjacent is not None:
                        observations.append(SoloFaceObservation(adjacent_seconds, adjacent))
            selected = bank.recover_from_adjacent_observations(
                candidates, query_seconds=seconds, adjacent_observations=observations,
            )
            self._focal_temporal_recovery_count += int(selected is not None)
        self._frozen_focal_queries[key] = selected
        return selected

    def selector_geometry(
        self, source_index: int, seconds: float, descriptor: object | None,
    ) -> FocalSelectorGeometry:
        """Expose the same frozen target to the selector without editing raw features."""
        if not self._require_focal_consensus:
            raise ValueError('early focal selector requires the frozen reference contract')
        if (isinstance(source_index, bool) or not isinstance(source_index, int)
                or not 0 <= source_index < len(self._sources)):
            raise ValueError('focal selector source index is invalid')
        candidate = self._frozen_focal_candidate(source_index, seconds)
        return focal_selector_geometry(source_index=source_index, source_seconds=seconds,
            candidate=candidate, people=tuple(getattr(descriptor, 'formation_people', ())))

    def begin_focal_boundary_reference_capture(self) -> None:
        """Start one ordered, path-local capture before dense motion planning."""

        self._focal_boundary_reference_capture.clear()

    def focal_boundary_reference_capture(
        self,
    ) -> tuple[
        tuple[
            int,
            int,
            FocalFaceCandidate | None,
            FocalFaceCandidate | None,
        ],
        ...,
    ]:
        """Return exact transient face pairs used at this path's boundaries."""

        return tuple(self._focal_boundary_reference_capture)

    def paired_samples(
        self,
        outgoing_source_index: int,
        outgoing_seconds: float,
        incoming_source_index: int,
        incoming_seconds: float,
    ) -> tuple[DirectMemberFaceSample | None, DirectMemberFaceSample | None]:
        """Resolve one focal face pair before any aggregate fallback."""

        self._focal_pair_request_count += 1
        left_candidates = self._focal_candidates(
            outgoing_source_index,
            max(0.0, outgoing_seconds),
        )
        right_candidates = self._focal_candidates(
            incoming_source_index,
            max(0.0, incoming_seconds),
        )
        if self._require_focal_consensus:
            left = self._frozen_focal_candidate(
                outgoing_source_index, max(0., outgoing_seconds), direction=-1,
            )
            right = self._frozen_focal_candidate(
                incoming_source_index, max(0., incoming_seconds), direction=1,
            )
            left_candidates = (left,) if left else ()
            right_candidates = (right,) if right else ()
        pair = select_focal_face_pair(left_candidates, right_candidates)
        if pair is None:
            self._focal_boundary_reference_capture.append(
                (
                    outgoing_source_index,
                    incoming_source_index,
                    None,
                    None,
                )
            )
            return (
                self.sample(outgoing_source_index, outgoing_seconds),
                self.sample(incoming_source_index, incoming_seconds),
            )
        self._focal_pair_match_count += 1
        self._focal_boundary_reference_capture.append(
            (
                outgoing_source_index,
                incoming_source_index,
                pair.outgoing,
                pair.incoming,
            )
        )
        self._register_focal_reference(
            outgoing_source_index,
            pair.outgoing,
        )
        self._register_focal_reference(
            incoming_source_index,
            pair.incoming,
        )
        left_sample = self._sample_for_focal_candidate(
            outgoing_source_index,
            max(0.0, outgoing_seconds),
            pair.outgoing,
        )
        right_sample = self._sample_for_focal_candidate(
            incoming_source_index,
            max(0.0, incoming_seconds),
            pair.incoming,
        )
        left_sampler = self._samplers[outgoing_source_index]
        right_sampler = self._samplers[incoming_source_index]
        assert left_sampler is not None and right_sampler is not None
        self._cache[
            self._identity_key(
                outgoing_source_index,
                left_sampler,
                max(0.0, outgoing_seconds),
            )
        ] = left_sample
        self._cache[
            self._identity_key(
                incoming_source_index,
                right_sampler,
                max(0.0, incoming_seconds),
            )
        ] = right_sample
        return (left_sample, right_sample)

    def recover_exact_focal_pair_from_registered_references(
        self,
        outgoing_source_index: int,
        outgoing_seconds: float,
        incoming_source_index: int,
        incoming_seconds: float,
        *,
        maximum_temporal_bridge_seconds: float = 0.50,
        temporal_step_seconds: float = 0.10,
    ) -> tuple[FocalFaceCandidate, FocalFaceCandidate] | None:
        """Recover an ambiguous exact endpoint through a nearby same-source face.

        Dense planning visits every selected boundary before this method is
        called, so each source can already hold transient references from a
        different, unambiguous cut. If an exact endpoint contains several
        faces and none matches that older reference directly, a nearby frame
        can bridge appearance change back to one of the *exact-frame*
        candidates. A face absent from the exact frame is never fabricated or
        extrapolated, and neither embeddings nor pixels leave process memory.
        """

        for source_index in (
            outgoing_source_index,
            incoming_source_index,
        ):
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or not 0 <= source_index < len(self._sources)
            ):
                raise ValueError("focal recovery source index is invalid")
        for seconds in (outgoing_seconds, incoming_seconds):
            if not math.isfinite(seconds):
                raise ValueError("focal recovery time is invalid")
        if (
            not math.isfinite(maximum_temporal_bridge_seconds)
            or not 0.0 < maximum_temporal_bridge_seconds <= 0.75
            or not math.isfinite(temporal_step_seconds)
            or not 0.0 < temporal_step_seconds <= 0.25
        ):
            raise ValueError("focal recovery temporal search is invalid")
        def recover_one(
            source_index: int,
            exact_seconds: float,
            direction: float,
        ) -> FocalFaceCandidate | None:
            if getattr(self, "_require_focal_consensus", False):
                return self._frozen_focal_candidate(
                    source_index, max(0., exact_seconds), direction=int(direction),
                )
            exact_candidates = self._focal_candidates(
                source_index,
                max(0.0, exact_seconds),
            )
            references = tuple(self._focal_references[source_index])
            if not exact_candidates or not references:
                return None
            direct = select_focal_face_matching_reference(
                exact_candidates,
                references,
            )
            if direct is not None:
                return direct
            step_count = int(
                math.floor(
                    maximum_temporal_bridge_seconds
                    / temporal_step_seconds
                    + 1e-9
                )
            )
            for step in range(1, step_count + 1):
                adjacent_seconds = max(
                    0.0,
                    exact_seconds
                    + direction * step * temporal_step_seconds,
                )
                adjacent = select_focal_face_matching_reference(
                    self._focal_candidates(
                        source_index,
                        adjacent_seconds,
                    ),
                    references,
                )
                if adjacent is None:
                    continue
                bridged = select_focal_face_matching_reference(
                    exact_candidates,
                    (adjacent,),
                )
                if bridged is not None:
                    return bridged
            return None

        outgoing = recover_one(
            outgoing_source_index,
            outgoing_seconds,
            -1.0,
        )
        incoming = recover_one(
            incoming_source_index,
            incoming_seconds,
            1.0,
        )
        if outgoing is None or incoming is None:
            return None
        self._register_focal_reference(outgoing_source_index, outgoing)
        self._register_focal_reference(incoming_source_index, incoming)
        return (outgoing, incoming)

    def sample(
        self,
        source_index: int,
        source_local_seconds: float,
    ) -> DirectMemberFaceSample | None:
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not 0 <= source_index < len(self._sources)
        ):
            raise ValueError("dense member face source index is invalid")
        if not math.isfinite(source_local_seconds):
            raise ValueError("dense member face time must be finite")
        sampler = self._samplers[source_index]
        if (
            sampler is not None
            and self._require_persistent_feature_cache
        ):
            sampler.assert_required_persistent_cache_available()
        self._request_count += 1
        if sampler is None:
            return None
        normalized_seconds = max(0.0, source_local_seconds)
        key = self._identity_key(
            source_index,
            sampler,
            normalized_seconds,
        )
        if key in self._cache:
            self._cache_hit_count += 1
            return self._cache[key]
        if self._require_focal_consensus:
            selected = self._frozen_focal_candidate(source_index, normalized_seconds)
            sample = (self._sample_for_focal_candidate(source_index, normalized_seconds, selected)
                      if selected is not None else None)
            self._focal_reference_match_count += int(selected is not None)
            # A crowd silhouette or an unrelated solo face is not a verified
            # substitute when the frozen focal track is occluded.
            self._cache[key] = sample
            return sample
        references = self._focal_references[source_index]
        if references:
            selected = select_focal_face_matching_reference(
                self._focal_candidates(source_index, normalized_seconds),
                references,
            )
            if selected is not None:
                self._focal_reference_match_count += 1
                sample = self._sample_for_focal_candidate(
                    source_index,
                    normalized_seconds,
                    selected,
                )
                self._register_focal_reference(source_index, selected)
                self._cache[key] = sample
                return sample
        previous_decode_count = sampler.frame_decode_count
        try:
            descriptor = sampler.feature_descriptor_at(
                normalized_seconds,
                include_subject_geometry=True,
                include_formation_pose=True,
            )
            self._decode_count += (
                sampler.frame_decode_count - previous_decode_count
            )
            if descriptor is None:
                self._cache[key] = None
                return None
            subject_box = None
            subject_confidence = 0.0
            subject_basis = "none"
            if (
                descriptor.silhouette_bbox_xywh is not None
                and descriptor.silhouette_confidence >= 0.30
                and descriptor.silhouette_backend == SILHOUETTE_BACKEND
            ):
                subject_box = descriptor.silhouette_bbox_xywh
                subject_confidence = descriptor.silhouette_confidence
                subject_basis = "pphumanseg_full_subject_silhouette"
            elif descriptor.primary_face_bbox is not None:
                face = descriptor.primary_face_bbox
                face_center = (
                    face[0] + face[2] / 2.0,
                    face[1] + face[3] / 2.0,
                )
                associated = tuple(
                    person
                    for person in descriptor.formation_people
                    if person.bbox_xywh[0] - 1e-6
                    <= face_center[0]
                    <= person.bbox_xywh[0] + person.bbox_xywh[2] + 1e-6
                    and person.bbox_xywh[1] - 1e-6
                    <= face_center[1]
                    <= person.bbox_xywh[1] + person.bbox_xywh[3] + 1e-6
                    and person.detection_confidence >= 0.50
                )
                if len(associated) == 1:
                    subject_box = associated[0].bbox_xywh
                    subject_confidence = associated[0].detection_confidence
                    subject_basis = "face_associated_formation_person_box"
            if (
                descriptor.primary_face_bbox is not None
                and descriptor.face_count == 1
                and descriptor.face_detector_backend.casefold() == "yunet"
            ):
                sample = DirectMemberFaceSample(
                    descriptor.primary_face_bbox,
                    descriptor.face_count,
                    descriptor.face_detector_backend,
                    subject_box,
                    subject_confidence,
                    subject_basis,
                )
            elif subject_box is not None:
                sample = DirectMemberFaceSample(
                    subject_box,
                    0,
                    "none",
                    subject_box,
                    subject_confidence,
                    subject_basis,
                )
            else:
                sample = None
        except (cv2.error, ValueError, FloatingPointError):
            self._decode_count += max(
                0,
                sampler.frame_decode_count - previous_decode_count,
            )
            sample = None
        self._cache[key] = sample
        return sample

    def diagnostics(self) -> dict[str, object]:
        persistent = tuple(
            diagnostics
            for sampler in self._samplers
            if sampler is not None
            and (diagnostics := sampler.feature_cache_diagnostics()) is not None
        )
        return {
            "backend": (
                "canonical_frame_yunet_face_or_verified_full_subject_"
                "geometry_dense_selected_ramps"
            ),
            "request_count": self._request_count,
            "cache_hit_count": self._cache_hit_count,
            "decode_count": self._decode_count,
            "cache_entry_count": len(self._cache),
            "focal_pair_request_count": self._focal_pair_request_count,
            "focal_pair_match_count": self._focal_pair_match_count,
            "focal_reference_match_count": self._focal_reference_match_count,
            "focal_reference_count": sum(
                len(values) for values in self._focal_references
            ),
            "focal_face_basis": FOCAL_FACE_BASIS,
            "frozen_focal_consensus_required": self._require_focal_consensus,
            "frozen_focal_consensus": {
                str(index): bank.to_dict() if bank else None
                for index, bank in self._frozen_focal_banks.items()
            },
            "focal_temporal_recovery_count": self._focal_temporal_recovery_count,
            "persistent_feature_cache_enabled_source_count": len(persistent),
            "persistent_feature_cache_hit_count": sum(
                int(item.get("hit_count", 0)) for item in persistent
            ),
            "persistent_feature_cache_miss_count": sum(
                int(item.get("miss_count", 0)) for item in persistent
            ),
            "persistent_feature_cache_write_count": sum(
                int(item.get("write_count", 0)) for item in persistent
            ),
            "sequential_frame_decode_count": sum(
                int(
                    getattr(
                        sampler,
                        "dense_sequential_frame_decode_count",
                        0,
                    )
                )
                for sampler in self._samplers
                if sampler is not None
            ),
            "random_seek_frame_decode_count": sum(
                int(
                    getattr(
                        sampler,
                        "dense_random_seek_frame_decode_count",
                        0,
                    )
                )
                for sampler in self._samplers
                if sampler is not None
            ),
            "pixels_persisted": False,
            "face_embeddings_computed": False,
            "transient_focal_face_embeddings_computed": True,
            "focal_face_embeddings_persisted": False,
        }

    def close(self) -> None:
        for sampler in self._samplers:
            if sampler is not None:
                sampler.close()


@dataclass(frozen=True, slots=True)
class DirectMemberEditorialMotionConfig:
    ramp_seconds: float = 0.75
    maximum_boundary_zoom: float = 1.50
    # Human editors routinely accept a small, high-quality resample to make a
    # face or full-body action land cleanly across a cut.  Keep exact-pixel
    # transforms preferred, but do not make a 1080p source categorically
    # uneditable: at most fifteen percent of the requested camera zoom may be
    # supplied by Lanczos upscaling.  Containment, dense-frame continuity, and
    # residual position/scale gates remain unchanged.
    maximum_boundary_upscale_ratio: float = 1.15
    maximum_center_shift: float = 0.160
    maximum_boundary_center_distance: float = 0.30
    maximum_residual_center_distance: float = 0.08
    maximum_face_residual_position_mismatch: float = (
        0.08 + DIRECT_MEMBER_PROFESSIONAL_ATTENTION_TRIGGER_TOLERANCE
    )
    maximum_face_residual_scale_mismatch: float = (
        0.06 + DIRECT_MEMBER_PROFESSIONAL_ATTENTION_TRIGGER_TOLERANCE
    )
    maximum_subject_residual_position_mismatch: float = (
        0.12 + DIRECT_MEMBER_PROFESSIONAL_ATTENTION_TRIGGER_TOLERANCE
    )
    maximum_subject_residual_scale_mismatch: float = (
        0.10 + DIRECT_MEMBER_PROFESSIONAL_ATTENTION_TRIGGER_TOLERANCE
    )
    minimum_mismatch_improvement_fraction: float = 0.20
    minimum_face_visible_fraction: float = 0.995
    minimum_dense_face_visible_fraction: float = 0.980
    legacy_face_only_maximum_zoom: float = 1.12
    legacy_face_only_maximum_center_shift: float = 0.025
    minimum_widened_subject_confidence: float = 0.30
    # Subject-only silhouettes are conservative bounding rectangles rather
    # than anatomy masks.  Allow at most seven percent of that rectangle outside
    # the viewport during a small, lossless action-matching move, while the
    # separate vertical gate protects the head-to-feet axis. Dense center,
    # height, confidence, and residual gates still apply independently.
    minimum_widened_subject_visible_fraction: float = 0.930
    minimum_widened_subject_vertical_visible_fraction: float = 0.980
    minimum_face_associated_subject_visible_fraction: float = FOCAL_MEMBER_MINIMUM_SUBJECT_VISIBLE_FRACTION
    maximum_dense_subject_center_step: float = 0.20
    maximum_dense_subject_height_ratio: float = 1.40
    maximum_dense_center_step: float = 0.15
    maximum_dense_height_ratio: float = 1.35
    maximum_dense_face_interpolation_gap_frames: int = 24
    maximum_dense_face_edge_extrapolation_frames: int = 24
    dense_geometry_median_radius_frames: int = 3
    maximum_dense_follow_zoom_correction: float = 0.08
    maximum_dense_face_follow_zoom_correction: float = 0.03
    maximum_dense_follow_center_correction: float = 0.12
    maximum_dense_follow_zoom_step: float = 0.04
    maximum_dense_follow_center_step: float = 0.05
    maximum_endpoint_center_disagreement: float = 0.08
    maximum_endpoint_height_ratio: float = 1.25
    target_events_per_minute: float = 12.0
    minimum_event_spacing_seconds: float = 2.0
    enable_interior_zoom_pulses: bool = False
    allow_partial_required_boundary_matches: bool = False
    target_rich_motion_events_per_minute: float = 4.0
    minimum_interior_pulse_duration_seconds: float = 2.2
    interior_pulse_zoom: float = 1.05

    def __post_init__(self) -> None:
        values = tuple(float(getattr(self, name)) for name in (
            "ramp_seconds",
            "maximum_boundary_zoom",
            "maximum_boundary_upscale_ratio",
            "maximum_center_shift",
            "maximum_boundary_center_distance",
            "maximum_residual_center_distance",
            "maximum_face_residual_position_mismatch",
            "maximum_face_residual_scale_mismatch",
            "maximum_subject_residual_position_mismatch",
            "maximum_subject_residual_scale_mismatch",
            "minimum_mismatch_improvement_fraction",
            "minimum_face_visible_fraction",
            "minimum_dense_face_visible_fraction",
            "legacy_face_only_maximum_zoom",
            "legacy_face_only_maximum_center_shift",
            "minimum_widened_subject_confidence",
            "minimum_widened_subject_visible_fraction",
            "minimum_widened_subject_vertical_visible_fraction",
            "minimum_face_associated_subject_visible_fraction",
            "maximum_dense_subject_center_step",
            "maximum_dense_subject_height_ratio",
            "maximum_dense_center_step",
            "maximum_dense_height_ratio",
            "maximum_dense_follow_zoom_correction",
            "maximum_dense_face_follow_zoom_correction",
            "maximum_dense_follow_center_correction",
            "maximum_dense_follow_zoom_step",
            "maximum_dense_follow_center_step",
            "maximum_endpoint_center_disagreement",
            "maximum_endpoint_height_ratio",
            "target_events_per_minute",
            "minimum_event_spacing_seconds",
            "target_rich_motion_events_per_minute",
            "minimum_interior_pulse_duration_seconds",
            "interior_pulse_zoom",
        ))
        if any(not math.isfinite(value) for value in values):
            raise ValueError("direct member editorial configuration must be finite")
        # Reuse the confirmed-member configuration as the shared camera/schema
        # contract.  Dense-direct adds stricter continuity gates below.
        _shared_settings(self)
        if not 1.0 <= self.maximum_boundary_upscale_ratio <= 1.15:
            raise ValueError("bounded boundary upscale ratio is invalid")
        if not (
            0.0
            <= self.maximum_face_residual_position_mismatch
            <= self.maximum_subject_residual_position_mismatch
            <= 0.20
        ):
            raise ValueError("dense attention position residual gates are invalid")
        if not (
            0.0
            <= self.maximum_face_residual_scale_mismatch
            <= self.maximum_subject_residual_scale_mismatch
            <= 0.20
        ):
            raise ValueError("dense attention scale residual gates are invalid")
        if not 0.0 <= self.maximum_dense_center_step <= 0.20:
            raise ValueError("dense face center-step gate is invalid")
        if not 1.0 <= self.maximum_dense_height_ratio <= 2.0:
            raise ValueError("dense face height-ratio gate is invalid")
        for name, value, maximum in (
            (
                "maximum_dense_face_interpolation_gap_frames",
                self.maximum_dense_face_interpolation_gap_frames,
                30,
            ),
            (
                "maximum_dense_face_edge_extrapolation_frames",
                self.maximum_dense_face_edge_extrapolation_frames,
                30,
            ),
            (
                "dense_geometry_median_radius_frames",
                self.dense_geometry_median_radius_frames,
                3,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= maximum
            ):
                raise ValueError(f"{name} is invalid")
        if not 1.0 <= self.legacy_face_only_maximum_zoom <= 1.12:
            raise ValueError("legacy face-only zoom boundary is invalid")
        if not 0.0 <= self.legacy_face_only_maximum_center_shift <= 0.025:
            raise ValueError("legacy face-only pan boundary is invalid")
        if not 0.0 <= self.minimum_widened_subject_confidence <= 1.0:
            raise ValueError("widened subject confidence gate is invalid")
        if not 0.90 <= self.minimum_widened_subject_visible_fraction <= 1.0:
            raise ValueError("widened subject containment gate is invalid")
        if not (
            0.95
            <= self.minimum_widened_subject_vertical_visible_fraction
            <= 1.0
        ):
            raise ValueError("widened subject vertical containment gate is invalid")
        if not (
            0.95
            <= self.minimum_dense_face_visible_fraction
            <= self.minimum_face_visible_fraction
            <= 1.0
        ):
            raise ValueError("dense face containment gate is invalid")
        if not (
            0.60
            <= self.minimum_face_associated_subject_visible_fraction
            <= self.minimum_widened_subject_visible_fraction
        ):
            raise ValueError(
                "face-associated subject containment gate is invalid"
            )
        if not 0.0 <= self.maximum_dense_subject_center_step <= 0.20:
            raise ValueError("dense subject center-step gate is invalid")
        if not 1.0 <= self.maximum_dense_subject_height_ratio <= 2.0:
            raise ValueError("dense subject height-ratio gate is invalid")
        if not 0.0 <= self.maximum_dense_follow_zoom_correction <= 0.10:
            raise ValueError("dense follow zoom correction gate is invalid")
        if not (
            0.0
            <= self.maximum_dense_face_follow_zoom_correction
            <= self.maximum_dense_follow_zoom_correction
        ):
            raise ValueError("dense face-follow zoom correction gate is invalid")
        if not 0.0 <= self.maximum_dense_follow_center_correction <= 0.16:
            raise ValueError("dense follow center correction gate is invalid")
        if not 0.0 <= self.maximum_dense_follow_zoom_step <= 0.05:
            raise ValueError("dense follow zoom-step gate is invalid")
        if not 0.0 <= self.maximum_dense_follow_center_step <= 0.08:
            raise ValueError("dense follow center-step gate is invalid")
        if not 0.0 <= self.maximum_endpoint_center_disagreement <= 0.20:
            raise ValueError("dense endpoint center gate is invalid")
        if not 1.0 <= self.maximum_endpoint_height_ratio <= 2.0:
            raise ValueError("dense endpoint height-ratio gate is invalid")
        if not isinstance(self.enable_interior_zoom_pulses, bool):
            raise ValueError("interior zoom pulse switch must be boolean")
        if not isinstance(self.allow_partial_required_boundary_matches, bool):
            raise ValueError(
                "partial required boundary match switch must be boolean"
            )
        if not 0.0 <= self.target_rich_motion_events_per_minute <= 12.0:
            raise ValueError("rich direct-member motion density is invalid")
        if not 1.5 <= self.minimum_interior_pulse_duration_seconds <= 6.0:
            raise ValueError("interior zoom pulse duration floor is invalid")
        if not 1.02 <= self.interior_pulse_zoom <= 1.12:
            raise ValueError("interior zoom pulse must remain subtle")


def _shared_settings(
    config: DirectMemberEditorialMotionConfig,
) -> MemberEditorialMotionConfig:
    return MemberEditorialMotionConfig(
        ramp_seconds=config.ramp_seconds,
        maximum_boundary_zoom=config.maximum_boundary_zoom,
        maximum_center_shift=config.maximum_center_shift,
        maximum_boundary_center_distance=config.maximum_boundary_center_distance,
        maximum_residual_center_distance=config.maximum_residual_center_distance,
        minimum_mismatch_improvement_fraction=(
            config.minimum_mismatch_improvement_fraction
        ),
        minimum_subject_visible_fraction=config.minimum_face_visible_fraction,
        target_events_per_minute=config.target_events_per_minute,
        minimum_event_spacing_seconds=config.minimum_event_spacing_seconds,
    )


@dataclass(frozen=True, slots=True)
class _BoundaryCandidate:
    boundary_index: int
    event_time: float
    score: float
    side: str
    state: _State
    alignment_geometry_basis: str
    left_box: tuple[float, float, float, float]
    right_box: tuple[float, float, float, float]
    alternative_states: tuple[_State, ...] = ()
    paired_state: _State | None = None
    upscale_ratio: float = 1.0
    paired_upscale_ratio: float = 1.0
    required: bool = False


def _rendered_attention_mismatch(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Measure rendered center and visible subject-height continuity.

    Uniform camera zoom can match height and center, but it cannot make a
    profile face, extended arm, or changing full-body pose retain the same
    bounding-box area.  Area therefore confounds choreography with camera
    scale.  The dense planner itself optimizes subject height; persist that
    same viewer-facing scale axis for the professional gate.
    """

    position = min(
        1.0,
        math.hypot(
            _center(left)[0] - _center(right)[0],
            _center(left)[1] - _center(right)[1],
        )
        / 0.42,
    )
    scale = min(
        1.0,
        abs(math.log(left[3] / right[3])) / math.log(3.2),
    )
    return (position, scale)


def _visible_axis_fractions(
    box: tuple[float, float, float, float],
    state: _State,
) -> tuple[float, float]:
    """Return independent horizontal and vertical box containment."""

    half = 0.5 / state.zoom
    camera_left = state.center_x - half
    camera_top = state.center_y - half
    camera_right = state.center_x + half
    camera_bottom = state.center_y + half
    box_left, box_top, box_width, box_height = box
    visible_width = max(
        0.0,
        min(camera_right, box_left + box_width) - max(camera_left, box_left),
    )
    visible_height = max(
        0.0,
        min(camera_bottom, box_top + box_height) - max(camera_top, box_top),
    )
    return (
        min(1.0, visible_width / max(1e-9, box_width)),
        min(1.0, visible_height / max(1e-9, box_height)),
    )


def _attention_alignment_geometry(
    sample: object,
) -> tuple[tuple[float, float, float, float], str] | None:
    attention = getattr(sample, "attention", None)
    if attention is None:
        return None
    basis = str(getattr(attention, "basis", ""))
    descriptor = getattr(sample, "descriptor", None)
    if descriptor is None:
        return None
    if basis == FOCAL_FACE_BASIS and getattr(sample, 'focal_geometry', None) is not None:
        if sample.focal_geometry.face_bbox_xywh is None:
            return None
    elif _is_face_alignment_basis(basis):
        if getattr(descriptor, "face_count", None) != 1:
            return None
    elif basis == "aggregate_human_silhouette":
        if (
            getattr(descriptor, "silhouette_backend", None)
            != SILHOUETTE_BACKEND
            or float(getattr(descriptor, "silhouette_confidence", 0.0))
            < 0.30
        ):
            return None
    elif basis == "unambiguous_dominant_person_box_geometry_only":
        if not tuple(getattr(descriptor, "formation_people", ())):
            return None
    elif basis == FOCAL_SELECTOR_BODY_BASIS:
        focal = getattr(sample, 'focal_geometry', None)
        if focal is None or focal.person is None or (focal.face_bbox_xywh is None and focal.body_track is None):
            return None
    else:
        return None
    raw = getattr(attention, "bbox_xywh", None)
    if not isinstance(raw, tuple) or len(raw) != 4:
        return None
    box = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) for value in box):
        return None
    return (box, basis)  # type: ignore[return-value]


def _endpoint_agrees(
    dense_box: tuple[float, float, float, float],
    audited_box: tuple[float, float, float, float],
    config: DirectMemberEditorialMotionConfig,
) -> bool:
    center_distance = math.hypot(
        _center(dense_box)[0] - _center(audited_box)[0],
        _center(dense_box)[1] - _center(audited_box)[1],
    )
    height_ratio = max(dense_box[3], audited_box[3]) / max(
        1e-9, min(dense_box[3], audited_box[3])
    )
    return (
        center_distance <= config.maximum_endpoint_center_disagreement + 1e-9
        and height_ratio <= config.maximum_endpoint_height_ratio + 1e-9
    )


def _neutral(track: EditorialTransformTrack, time_seconds: float) -> bool:
    state = track.evaluate(time_seconds)
    return (
        abs(state.zoom - 1.0) <= 1e-7
        and abs(state.center_x - 0.5) <= 1e-7
        and abs(state.center_y - 0.5) <= 1e-7
        and abs(state.roll_degrees) <= 1e-7
    )


def _geometry_step_exceeds(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    *,
    maximum_center_step: float,
    maximum_height_ratio: float,
) -> bool:
    center_step = math.hypot(
        _center(left)[0] - _center(right)[0],
        _center(left)[1] - _center(right)[1],
    )
    height_ratio = max(left[3], right[3]) / max(
        1e-9, min(left[3], right[3])
    )
    return (
        center_step > maximum_center_step + 1e-9
        or height_ratio > maximum_height_ratio + 1e-9
    )


def _midpoint_box(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return tuple(
        (left_value + right_value) / 2.0
        for left_value, right_value in zip(left, right, strict=True)
    )  # type: ignore[return-value]


def _box_contains_center(
    container: tuple[float, float, float, float],
    item: tuple[float, float, float, float],
) -> bool:
    center_x, center_y = _center(item)
    return (
        container[0] - 1e-6
        <= center_x
        <= container[0] + container[2] + 1e-6
        and container[1] - 1e-6
        <= center_y
        <= container[1] + container[3] + 1e-6
    )


def _isolated_outlier_replacement(
    previous: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
    following: tuple[float, float, float, float],
    *,
    maximum_center_step: float,
    maximum_height_ratio: float,
) -> tuple[float, float, float, float] | None:
    """Repair only a one-frame detector excursion with neighbor consensus."""

    if _geometry_step_exceeds(
        previous,
        following,
        maximum_center_step=maximum_center_step,
        maximum_height_ratio=maximum_height_ratio,
    ):
        return None
    if not (
        _geometry_step_exceeds(
            previous,
            current,
            maximum_center_step=maximum_center_step,
            maximum_height_ratio=maximum_height_ratio,
        )
        or _geometry_step_exceeds(
            current,
            following,
            maximum_center_step=maximum_center_step,
            maximum_height_ratio=maximum_height_ratio,
        )
    ):
        return None
    return _midpoint_box(previous, following)


def _linear_box(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    return tuple(
        left_value + (right_value - left_value) * alpha
        for left_value, right_value in zip(left, right, strict=True)
    )  # type: ignore[return-value]


def _relative_box(
    item: tuple[float, float, float, float],
    container: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    if container[2] <= 1e-9 or container[3] <= 1e-9:
        return None
    item_center = _center(item)
    return (
        (item_center[0] - container[0]) / container[2],
        (item_center[1] - container[1]) / container[3],
        item[2] / container[2],
        item[3] / container[3],
    )


def _box_from_relative(
    relative: tuple[float, float, float, float],
    container: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    width = relative[2] * container[2]
    height = relative[3] * container[3]
    center_x = container[0] + relative[0] * container[2]
    center_y = container[1] + relative[1] * container[3]
    x = center_x - width / 2.0
    y = center_y - height / 2.0
    if (
        width <= 0.0
        or height <= 0.0
        or x < -1e-6
        or y < -1e-6
        or x + width > 1.0 + 1e-6
        or y + height > 1.0 + 1e-6
    ):
        return None
    return (
        max(0.0, x),
        max(0.0, y),
        min(width, 1.0 - max(0.0, x)),
        min(height, 1.0 - max(0.0, y)),
    )


def _median_value(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _median_smooth_geometry(
    frame_indices: Sequence[int],
    boxes: Sequence[tuple[float, float, float, float]],
    *,
    radius: int,
    preserve_run_endpoints: bool = True,
) -> list[tuple[float, float, float, float]]:
    if radius <= 0:
        return list(boxes)
    output = list(boxes)
    for index in range(len(boxes)):
        run_start = index
        while (
            run_start > 0
            and frame_indices[run_start - 1] + 1 == frame_indices[run_start]
        ):
            run_start -= 1
        run_end = index
        while (
            run_end + 1 < len(boxes)
            and frame_indices[run_end] + 1 == frame_indices[run_end + 1]
        ):
            run_end += 1
        if preserve_run_endpoints and index in {run_start, run_end}:
            continue
        low = max(run_start, index - radius)
        high = min(run_end, index + radius)
        window = boxes[low : high + 1]
        output[index] = tuple(
            _median_value(tuple(box[component] for box in window))
            for component in range(4)
        )  # type: ignore[assignment]
    return output


def _stabilize_subject_relative_face_track(
    *,
    frame_indices: Sequence[int],
    face_boxes: Sequence[tuple[float, float, float, float] | None],
    subject_boxes: Sequence[tuple[float, float, float, float] | None],
    frame_count: int,
    config: DirectMemberEditorialMotionConfig,
) -> list[tuple[float, float, float, float] | None]:
    """Keep the boundary-anchored, temporally continuous face path.

    Official portrait fancams are authenticated as target-member sources, but
    YuNet can still prefer a second member for a few frames.  The real edit
    boundary is independently decoded and audited before this function runs,
    so a transformed run touching frame zero or the final frame has a trusted
    temporal anchor.  We retain only detections connected to that anchor in
    subject-relative coordinates.  A skipped excursion is subsequently
    reconstructed only within the configured gap limit; a persistent switch
    therefore still fails closed.
    """

    output = list(face_boxes)

    def connected(left: int, right: int) -> bool:
        left_box = output[left]
        right_box = output[right]
        left_subject = subject_boxes[left]
        right_subject = subject_boxes[right]
        if (
            left_box is None
            or right_box is None
            or left_subject is None
            or right_subject is None
        ):
            return False
        left_relative = _relative_box(left_box, left_subject)
        right_relative = _relative_box(right_box, right_subject)
        if left_relative is None or right_relative is None:
            return False
        frame_gap = abs(frame_indices[right] - frame_indices[left])
        if (
            frame_gap <= 0
            or frame_gap
            > config.maximum_dense_face_interpolation_gap_frames + 1
        ):
            return False
        center_distance = math.hypot(
            left_relative[0] - right_relative[0],
            left_relative[1] - right_relative[1],
        )
        maximum_center_distance = min(
            0.30,
            0.18 + 0.010 * max(0, frame_gap - 1),
        )
        height_ratio = max(left_relative[3], right_relative[3]) / max(
            1e-9,
            min(left_relative[3], right_relative[3]),
        )
        maximum_height_ratio = min(
            1.80,
            config.maximum_dense_height_ratio
            + 0.025 * max(0, frame_gap - 1),
        )
        absolute_center_distance = math.hypot(
            _center(left_box)[0] - _center(right_box)[0],
            _center(left_box)[1] - _center(right_box)[1],
        )
        maximum_absolute_center_distance = min(
            0.32,
            0.15 + 0.010 * max(0, frame_gap - 1),
        )
        absolute_height_ratio = max(left_box[3], right_box[3]) / max(
            1e-9,
            min(left_box[3], right_box[3]),
        )
        maximum_absolute_height_ratio = min(
            1.90,
            1.50 + 0.025 * max(0, frame_gap - 1),
        )
        return (
            center_distance <= maximum_center_distance + 1e-9
            and height_ratio <= maximum_height_ratio + 1e-9
            and absolute_center_distance
            <= maximum_absolute_center_distance + 1e-9
            and absolute_height_ratio
            <= maximum_absolute_height_ratio + 1e-9
        )

    run_start = 0
    while run_start < len(frame_indices):
        run_end = run_start
        while (
            run_end + 1 < len(frame_indices)
            and frame_indices[run_end] + 1 == frame_indices[run_end + 1]
        ):
            run_end += 1
        detections = [
            index
            for index in range(run_start, run_end + 1)
            if output[index] is not None and subject_boxes[index] is not None
        ]
        if len(detections) < 3:
            run_start = run_end + 1
            continue
        if frame_indices[run_start] == 0:
            ordered = detections
            anchored = True
        elif frame_indices[run_end] == frame_count - 1:
            ordered = list(reversed(detections))
            anchored = True
        else:
            ordered = detections
            anchored = False

        scores = [float("-inf")] * len(ordered)
        lengths = [0] * len(ordered)
        predecessors: list[int | None] = [None] * len(ordered)
        for position, current in enumerate(ordered):
            if not anchored or position == 0:
                scores[position] = 1.0
                lengths[position] = 1
            for previous_position in range(position):
                if not math.isfinite(scores[previous_position]):
                    continue
                previous = ordered[previous_position]
                if not connected(previous, current):
                    continue
                frame_gap = abs(
                    frame_indices[current] - frame_indices[previous]
                )
                candidate_score = (
                    scores[previous_position]
                    + 1.0
                    - 0.02 * max(0, frame_gap - 1)
                )
                candidate_length = lengths[previous_position] + 1
                if (
                    candidate_score > scores[position] + 1e-9
                    or abs(candidate_score - scores[position]) <= 1e-9
                    and candidate_length > lengths[position]
                ):
                    scores[position] = candidate_score
                    lengths[position] = candidate_length
                    predecessors[position] = previous_position

        reachable = [
            position
            for position, score in enumerate(scores)
            if math.isfinite(score)
        ]
        if not reachable:
            run_start = run_end + 1
            continue
        if anchored:
            # Coverage is primary: if the audited face cannot be connected
            # across the run, the remaining edge gap is left for the strict
            # reconstruction limit to reject.
            endpoint = max(
                reachable,
                key=lambda position: (
                    abs(
                        frame_indices[ordered[position]]
                        - frame_indices[ordered[0]]
                    ),
                    scores[position],
                    lengths[position],
                ),
            )
        else:
            endpoint = max(
                reachable,
                key=lambda position: (
                    scores[position],
                    lengths[position],
                    -frame_indices[ordered[position]],
                ),
            )
        kept: set[int] = set()
        cursor: int | None = endpoint
        while cursor is not None:
            kept.add(ordered[cursor])
            cursor = predecessors[cursor]
        for index in detections:
            if index not in kept:
                output[index] = None
        run_start = run_end + 1
    return output


def _stabilize_face_associated_subject_track(
    *,
    frame_indices: Sequence[int],
    subject_boxes: Sequence[tuple[float, float, float, float] | None],
    face_boxes: Sequence[tuple[float, float, float, float]],
    frame_count: int,
    config: DirectMemberEditorialMotionConfig,
) -> list[tuple[float, float, float, float]] | None:
    """Smooth detector-only subject jumps from the audited boundary inward."""

    if any(subject_box is None for subject_box in subject_boxes):
        return None
    output: list[tuple[float, float, float, float] | None] = list(
        subject_boxes
    )

    def connected(left: int, right: int) -> bool:
        left_box = output[left]
        right_box = output[right]
        if left_box is None or right_box is None:
            return False
        frame_gap = abs(frame_indices[right] - frame_indices[left])
        if (
            frame_gap <= 0
            or frame_gap
            > config.maximum_dense_face_interpolation_gap_frames + 1
        ):
            return False
        center_distance = math.hypot(
            _center(left_box)[0] - _center(right_box)[0],
            _center(left_box)[1] - _center(right_box)[1],
        )
        maximum_center_distance = min(
            0.35,
            config.maximum_dense_subject_center_step
            + 0.0075 * max(0, frame_gap - 1),
        )
        height_ratio = max(left_box[3], right_box[3]) / max(
            1e-9,
            min(left_box[3], right_box[3]),
        )
        maximum_height_ratio = min(
            1.80,
            config.maximum_dense_subject_height_ratio
            + 0.025 * max(0, frame_gap - 1),
        )
        return (
            center_distance <= maximum_center_distance + 1e-9
            and height_ratio <= maximum_height_ratio + 1e-9
        )

    run_start = 0
    while run_start < len(frame_indices):
        run_end = run_start
        while (
            run_end + 1 < len(frame_indices)
            and frame_indices[run_end] + 1 == frame_indices[run_end + 1]
        ):
            run_end += 1
        detections = list(range(run_start, run_end + 1))
        if frame_indices[run_start] == 0:
            ordered = detections
            anchored = True
        elif frame_indices[run_end] == frame_count - 1:
            ordered = list(reversed(detections))
            anchored = True
        else:
            ordered = detections
            anchored = False

        scores = [float("-inf")] * len(ordered)
        lengths = [0] * len(ordered)
        predecessors: list[int | None] = [None] * len(ordered)
        for position, current in enumerate(ordered):
            if not anchored or position == 0:
                scores[position] = 1.0
                lengths[position] = 1
            for previous_position in range(position):
                if not math.isfinite(scores[previous_position]):
                    continue
                previous = ordered[previous_position]
                if not connected(previous, current):
                    continue
                frame_gap = abs(
                    frame_indices[current] - frame_indices[previous]
                )
                candidate_score = (
                    scores[previous_position]
                    + 1.0
                    - 0.02 * max(0, frame_gap - 1)
                )
                candidate_length = lengths[previous_position] + 1
                if (
                    candidate_score > scores[position] + 1e-9
                    or abs(candidate_score - scores[position]) <= 1e-9
                    and candidate_length > lengths[position]
                ):
                    scores[position] = candidate_score
                    lengths[position] = candidate_length
                    predecessors[position] = previous_position
        reachable = [
            position
            for position, score in enumerate(scores)
            if math.isfinite(score)
        ]
        if not reachable:
            return None
        if anchored:
            endpoint = max(
                reachable,
                key=lambda position: (
                    abs(
                        frame_indices[ordered[position]]
                        - frame_indices[ordered[0]]
                    ),
                    scores[position],
                    lengths[position],
                ),
            )
        else:
            endpoint = max(
                reachable,
                key=lambda position: (
                    scores[position],
                    lengths[position],
                ),
            )
        kept: set[int] = set()
        cursor: int | None = endpoint
        while cursor is not None:
            kept.add(ordered[cursor])
            cursor = predecessors[cursor]
        for index in detections:
            if index not in kept:
                output[index] = None

        anchors = [
            index
            for index in detections
            if output[index] is not None
        ]
        if not anchors:
            return None
        first_anchor = anchors[0]
        last_anchor = anchors[-1]
        if (
            first_anchor - run_start
            > config.maximum_dense_face_edge_extrapolation_frames
            or run_end - last_anchor
            > config.maximum_dense_face_edge_extrapolation_frames
        ):
            return None

        def project_anchor(anchor: int, target: int):
            anchor_subject = output[anchor]
            assert anchor_subject is not None
            relative = _relative_box(anchor_subject, face_boxes[anchor])
            if relative is None:
                return anchor_subject
            return (
                _box_from_relative(relative, face_boxes[target])
                or anchor_subject
            )

        for index in range(run_start, first_anchor):
            output[index] = project_anchor(first_anchor, index)
        for index in range(last_anchor + 1, run_end + 1):
            output[index] = project_anchor(last_anchor, index)
        for left_anchor, right_anchor in zip(anchors, anchors[1:]):
            gap = right_anchor - left_anchor - 1
            if gap <= 0:
                continue
            if gap > config.maximum_dense_face_interpolation_gap_frames:
                return None
            left_box = output[left_anchor]
            right_box = output[right_anchor]
            assert left_box is not None and right_box is not None
            for index in range(left_anchor + 1, right_anchor):
                alpha = (index - left_anchor) / (
                    right_anchor - left_anchor
                )
                output[index] = _linear_box(left_box, right_box, alpha)
        run_start = run_end + 1

    resolved = [box for box in output if box is not None]
    if len(resolved) != len(output):
        return None
    return _median_smooth_geometry(
        frame_indices,
        resolved,
        radius=config.dense_geometry_median_radius_frames,
        preserve_run_endpoints=True,
    )


def _reconstruct_dense_face_geometry(
    *,
    frame_indices: Sequence[int],
    face_boxes: Sequence[tuple[float, float, float, float] | None],
    subject_boxes: Sequence[tuple[float, float, float, float] | None],
    config: DirectMemberEditorialMotionConfig,
) -> list[tuple[float, float, float, float]] | None:
    """Bridge short detector gaps in target-subject relative coordinates."""

    output = list(face_boxes)
    run_start = 0
    while run_start < len(frame_indices):
        run_end = run_start
        while (
            run_end + 1 < len(frame_indices)
            and frame_indices[run_end] + 1 == frame_indices[run_end + 1]
        ):
            run_end += 1
        anchors = [
            index
            for index in range(run_start, run_end + 1)
            if output[index] is not None
        ]
        if not anchors:
            return None
        first_anchor = anchors[0]
        prefix = first_anchor - run_start
        if prefix > config.maximum_dense_face_edge_extrapolation_frames:
            return None
        last_anchor = anchors[-1]
        suffix = run_end - last_anchor
        if suffix > config.maximum_dense_face_edge_extrapolation_frames:
            return None

        def project_anchor(anchor: int, target: int):
            anchor_box = output[anchor]
            assert anchor_box is not None
            anchor_subject = subject_boxes[anchor]
            target_subject = subject_boxes[target]
            if anchor_subject is None or target_subject is None:
                return anchor_box
            relative = _relative_box(anchor_box, anchor_subject)
            if relative is None:
                return anchor_box
            # Keep the estimated face geometry itself unchanged.  Inflating
            # this box as an uncertainty envelope feeds the uncertainty back
            # into visible face size, which can manufacture a crop failure or
            # a two-fold geometry jump.  The downstream crop/continuity gates
            # remain the conservative uncertainty check.
            return (
                _box_from_relative(relative, target_subject)
                or anchor_box
            )

        for index in range(run_start, first_anchor):
            output[index] = project_anchor(first_anchor, index)
        for index in range(last_anchor + 1, run_end + 1):
            output[index] = project_anchor(last_anchor, index)

        for left_anchor, right_anchor in zip(anchors, anchors[1:]):
            gap = right_anchor - left_anchor - 1
            if gap <= 0:
                continue
            if gap > config.maximum_dense_face_interpolation_gap_frames:
                return None
            left_box = output[left_anchor]
            right_box = output[right_anchor]
            assert left_box is not None and right_box is not None
            for index in range(left_anchor + 1, right_anchor):
                alpha = (index - left_anchor) / (right_anchor - left_anchor)
                # Identity association is solved in subject-relative space,
                # but visible motion continuity is an output-screen contract.
                # Reconstruct a rejected detector excursion directly between
                # its verified screen-space anchors so a breathing silhouette
                # cannot recreate the very jump that was removed above.
                output[index] = _linear_box(left_box, right_box, alpha)
        if any(output[index] is None for index in range(run_start, run_end + 1)):
            return None
        run_start = run_end + 1
    resolved = [box for box in output if box is not None]
    if len(resolved) != len(output):
        return None
    if all(subject_box is not None for subject_box in subject_boxes):
        return _median_smooth_geometry(
            frame_indices,
            resolved,
            radius=config.dense_geometry_median_radius_frames,
        )
    return resolved


def _dense_tracking_camera_track(
    *,
    eligible_source_index: int,
    segment: RenderSegment,
    track: EditorialTransformTrack,
    frame_count: int,
    common_master_start: float,
    geometry_provider: DirectMemberGeometryProvider,
    fps: float,
    config: DirectMemberEditorialMotionConfig,
    source_frame_geometry: DirectMemberSourceFrameGeometry | None,
    alignment_geometry_basis: str | None,
    diagnostic_callback: Callable[[str], None] | None = None,
) -> EditorialTransformTrack:
    """Retarget an active boundary ramp to the verified moving performer.

    A straight interpolation between two safe endpoint crops can still cut an
    arm, foot, or face when the choreography crosses the frame during the
    ramp.  Human editors keyframe the virtual camera through that motion.  Do
    the same here, using only authenticated dense geometry: preserve every
    authored endpoint exactly, then add the smallest per-frame zoom/pan change
    that satisfies the same face and subject containment gates used by final
    review.  The returned track is still re-reviewed independently below.
    """

    active_frame_indices = tuple(
        frame_index
        for frame_index in range(frame_count)
        if not _neutral(track, frame_index / fps)
    )
    if not active_frame_indices:
        return track

    samples: list[DirectMemberFaceSample] = []
    face_boxes: list[tuple[float, float, float, float] | None] = []
    subject_boxes: list[tuple[float, float, float, float] | None] = []
    for frame_index in active_frame_indices:
        local_time = frame_index / fps
        canonical_time = common_master_start + segment.master_start + local_time
        from .member_dense_containment import dense_camera_sample
        sample = dense_camera_sample(geometry_provider, eligible_source_index, canonical_time,
                                     interior=0 < frame_index < frame_count - 1)
        if sample is None:
            if diagnostic_callback is not None:
                diagnostic_callback(
                    f"dense_follow_geometry_missing(frame={frame_index})"
                )
            return track
        face_box = (
            map_box_to_portrait_fill(
                sample.bbox_xywh,
                source_frame_geometry,
            )
            if source_frame_geometry is not None
            else sample.bbox_xywh
        )
        subject_box = sample.subject_bbox_xywh
        if subject_box is not None and source_frame_geometry is not None:
            subject_box = map_box_to_portrait_fill(
                subject_box,
                source_frame_geometry,
            )
        samples.append(sample)
        face_boxes.append(face_box if sample.face_count == 1 else None)
        subject_boxes.append(subject_box)

    if all(box is not None for box in subject_boxes):
        subject_boxes = list(
            _median_smooth_geometry(
                active_frame_indices,
                [box for box in subject_boxes if box is not None],
                radius=config.dense_geometry_median_radius_frames,
                preserve_run_endpoints=True,
            )
        )

    resolved_basis = alignment_geometry_basis
    if resolved_basis is None:
        resolved_basis = (
            _FACE_ATTENTION_BASIS
            if sum(sample.face_count == 1 for sample in samples)
            >= max(2, math.ceil(len(samples) / 2.0))
            else "aggregate_human_silhouette"
        )

    centered_target_box = (0.45, 0.45, 0.10, 0.10)
    shared = _shared_settings(config)

    def state_is_contained(
        *,
        state: _State,
        sample: DirectMemberFaceSample,
        face_box: tuple[float, float, float, float] | None,
        subject_box: tuple[float, float, float, float] | None,
    ) -> bool:
        if face_box is not None and (
            _visible_fraction(face_box, state) + 1e-9
            < config.minimum_dense_face_visible_fraction
        ):
            return False
        # A short YuNet gap is reconstructed from the verified continuous
        # subject track by final dense review.  Let the camera follower use
        # that same subject evidence instead of abandoning the whole ramp at
        # the first missing face frame; it still cannot certify a frame where
        # both face and subject geometry are absent.
        if (
            _is_face_alignment_basis(resolved_basis)
            and face_box is None
            and subject_box is None
        ):
            return False
        if subject_box is None:
            return not (
                state.zoom > config.legacy_face_only_maximum_zoom + 1e-9
                or abs(state.center_x - 0.5)
                > config.legacy_face_only_maximum_center_shift + 1e-9
                or abs(state.center_y - 0.5)
                > config.legacy_face_only_maximum_center_shift + 1e-9
            )
        minimum_subject_fraction = (
            config.minimum_face_associated_subject_visible_fraction
            if sample.face_count == 1
            else config.minimum_widened_subject_visible_fraction
        )
        if (
            _visible_fraction(subject_box, state) + 1e-9
            < minimum_subject_fraction
        ):
            return False
        if sample.face_count == 0:
            _horizontal, vertical = _visible_axis_fractions(subject_box, state)
            if (
                vertical + 1e-9
                < config.minimum_widened_subject_vertical_visible_fraction
            ):
                return False
        return True

    frame_points: list[tuple[float, _State]] = []
    previous_state: _State | None = None
    changed = False
    for frame_index, sample, face_box, subject_box in zip(
        active_frame_indices,
        samples,
        face_boxes,
        subject_boxes,
        strict=True,
    ):
        local_time = frame_index / fps
        base_keyframe = track.evaluate(local_time)
        base_state = _State(
            base_keyframe.zoom,
            base_keyframe.center_x,
            base_keyframe.center_y,
        )
        if state_is_contained(
            state=base_state,
            sample=sample,
            face_box=face_box,
            subject_box=subject_box,
        ):
            selected_state = base_state
        else:
            focus_boxes = tuple(
                box
                for box in (subject_box, face_box)
                if box is not None
            )
            candidates: list[_State] = []
            seen: set[tuple[float, float, float]] = set()
            zoom = base_state.zoom
            zoom_values: list[float] = [zoom]
            while zoom_values[-1] > 1.0 + 1e-9:
                next_zoom = max(1.0, zoom_values[-1] - 0.01)
                if abs(next_zoom - zoom_values[-1]) <= 1e-9:
                    break
                zoom_values.append(next_zoom)
            for candidate_zoom in zoom_values:
                focus_states: list[_State] = []
                for focus_box in focus_boxes:
                    focus_states.extend(
                        _state_for_match(
                            box=focus_box,
                            target_box=centered_target_box,
                            zoom=candidate_zoom,
                            settings=shared,
                        )
                    )
                focus_states.append(
                    _State(
                        candidate_zoom,
                        base_state.center_x,
                        base_state.center_y,
                    )
                )
                for focus_state in focus_states:
                    for alpha in (0.25, 0.50, 0.75, 1.0):
                        state = _State(
                            candidate_zoom,
                            base_state.center_x
                            + (focus_state.center_x - base_state.center_x) * alpha,
                            base_state.center_y
                            + (focus_state.center_y - base_state.center_y) * alpha,
                        )
                        key = (
                            round(state.zoom, 9),
                            round(state.center_x, 9),
                            round(state.center_y, 9),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        maximum_zoom_correction = (
                            config.maximum_dense_face_follow_zoom_correction
                            if _is_face_alignment_basis(resolved_basis)
                            else config.maximum_dense_follow_zoom_correction
                        )
                        if (
                            abs(state.zoom - base_state.zoom)
                            > maximum_zoom_correction + 1e-9
                            or math.hypot(
                                state.center_x - base_state.center_x,
                                state.center_y - base_state.center_y,
                            )
                            > config.maximum_dense_follow_center_correction + 1e-9
                        ):
                            continue
                        try:
                            EditorialTransformKeyframe(
                                local_time,
                                state.zoom,
                                state.center_x,
                                state.center_y,
                                0.0,
                            )
                        except ValueError:
                            continue
                        if state_is_contained(
                            state=state,
                            sample=sample,
                            face_box=face_box,
                            subject_box=subject_box,
                        ):
                            if previous_state is not None and (
                                abs(state.zoom - previous_state.zoom)
                                > config.maximum_dense_follow_zoom_step + 1e-9
                                or math.hypot(
                                    state.center_x - previous_state.center_x,
                                    state.center_y - previous_state.center_y,
                                )
                                > config.maximum_dense_follow_center_step + 1e-9
                            ):
                                continue
                            candidates.append(state)
            if not candidates:
                if diagnostic_callback is not None:
                    diagnostic_callback(
                        "dense_follow_no_safe_state("
                        f"frame={frame_index},basis={resolved_basis},"
                        f"base_zoom={base_state.zoom:.6f},"
                        f"base_center_x={base_state.center_x:.6f},"
                        f"base_center_y={base_state.center_y:.6f},"
                        "face_box="
                        + (
                            ",".join(f"{value:.6f}" for value in face_box)
                            if face_box is not None
                            else "none"
                        )
                        + ",subject_box="
                        + (
                            ",".join(f"{value:.6f}" for value in subject_box)
                            if subject_box is not None
                            else "none"
                        )
                        + ")"
                    )
                return track

            def candidate_cost(state: _State) -> tuple[float, float, float]:
                authored_change = (
                    2.0 * abs(state.zoom - base_state.zoom)
                    + math.hypot(
                        state.center_x - base_state.center_x,
                        state.center_y - base_state.center_y,
                    )
                )
                temporal_change = (
                    0.0
                    if previous_state is None
                    else 0.35
                    * (
                        abs(state.zoom - previous_state.zoom)
                        + math.hypot(
                            state.center_x - previous_state.center_x,
                            state.center_y - previous_state.center_y,
                        )
                    )
                )
                return (
                    authored_change + temporal_change,
                    abs(state.zoom - base_state.zoom),
                    math.hypot(
                        state.center_x - base_state.center_x,
                        state.center_y - base_state.center_y,
                    ),
                )

            selected_state = min(candidates, key=candidate_cost)
            changed = changed or selected_state != base_state
        frame_points.append((local_time, selected_state))
        previous_state = selected_state

    if not changed:
        return track
    authored_points = [
        (
            keyframe.time_seconds,
            _State(keyframe.zoom, keyframe.center_x, keyframe.center_y),
        )
        for keyframe in track.keyframes
    ]
    try:
        return EditorialTransformTrack(
            keyframes=tuple(
                EditorialTransformKeyframe(
                    time_seconds=time_value,
                    zoom=state.zoom,
                    center_x=state.center_x,
                    center_y=state.center_y,
                    roll_degrees=0.0,
                )
                for time_value, state in _dedupe_points(
                    tuple(frame_points) + tuple(authored_points)
                )
            ),
            reason=track.reason + "_dense_performer_follow",
        )
    except ValueError as exc:
        if diagnostic_callback is not None:
            diagnostic_callback(
                "dense_follow_track_invalid(" + str(exc) + ")"
            )
        return track


def _track_is_dense_face_safe(
    *,
    segment_index: int,
    eligible_source_index: int,
    segment: RenderSegment,
    track: EditorialTransformTrack,
    frame_count: int,
    common_master_start: float,
    geometry_provider: DirectMemberGeometryProvider,
    fps: float,
    config: DirectMemberEditorialMotionConfig,
    source_frame_geometry: DirectMemberSourceFrameGeometry | None,
    alignment_geometry_basis: str | None = None,
    rejection_callback: Callable[[str], None] | None = None,
    rejection_detail_callback: Callable[[str], None] | None = None,
) -> bool:
    def reject(reason: str, detail: str | None = None) -> bool:
        if rejection_callback is not None:
            rejection_callback(reason)
        if detail is not None and rejection_detail_callback is not None:
            rejection_detail_callback(f"{reason}({detail})")
        return False

    active_frames = tuple(
        not _neutral(track, frame_index / fps)
        for frame_index in range(frame_count)
    )
    # Include one neutral-frame halo on each side of every transformed run.
    # Otherwise a detector/subject jump exactly where the ramp begins or ends
    # has no adjacent dense sample and can look continuous inside the ramp.
    review_frames: set[int] = {
        frame_index
        for frame_index, active in enumerate(active_frames)
        if active
    }
    for frame_index in tuple(review_frames):
        if frame_index > 0:
            review_frames.add(frame_index - 1)
        if frame_index + 1 < frame_count:
            review_frames.add(frame_index + 1)

    widened_subject_proof_required = bool(
        any(
            keyframe.zoom
            > config.legacy_face_only_maximum_zoom + 1e-9
            or abs(keyframe.center_x - 0.5)
            > config.legacy_face_only_maximum_center_shift + 1e-9
            or abs(keyframe.center_y - 0.5)
            > config.legacy_face_only_maximum_center_shift + 1e-9
            for keyframe in track.keyframes
        )
    )

    ordered_frames = tuple(sorted(review_frames))
    samples: list[DirectMemberFaceSample] = []
    raw_boxes: list[tuple[float, float, float, float]] = []
    face_boxes: list[tuple[float, float, float, float] | None] = []
    subject_boxes: list[tuple[float, float, float, float] | None] = []
    for frame_index in ordered_frames:
        local_time = frame_index / fps
        canonical_time = common_master_start + segment.master_start + local_time
        from .member_dense_containment import dense_camera_sample
        sample = dense_camera_sample(geometry_provider, eligible_source_index, canonical_time,
                                     interior=0 < frame_index < frame_count - 1)
        if sample is None:
            return reject("dense_geometry_missing", f"source={eligible_source_index},frame={frame_index}")
        box = (
            map_box_to_portrait_fill(
                sample.bbox_xywh,
                source_frame_geometry,
            )
            if source_frame_geometry is not None
            else sample.bbox_xywh
        )
        subject_box = sample.subject_bbox_xywh
        if subject_box is not None and source_frame_geometry is not None:
            subject_box = map_box_to_portrait_fill(
                subject_box,
                source_frame_geometry,
            )
        samples.append(sample)
        raw_boxes.append(box)
        face_boxes.append(box if sample.face_count == 1 else None)
        subject_boxes.append(subject_box)

    resolved_basis = alignment_geometry_basis
    if resolved_basis is None:
        resolved_basis = (
            _FACE_ATTENTION_BASIS
            if sum(sample.face_count == 1 for sample in samples)
            >= max(2, math.ceil(len(samples) / 2.0))
            else "aggregate_human_silhouette"
        )
    if _is_face_alignment_basis(resolved_basis):
        if all(subject_box is not None for subject_box in subject_boxes):
            subject_boxes = _median_smooth_geometry(
                ordered_frames,
                [
                    subject_box
                    for subject_box in subject_boxes
                    if subject_box is not None
                ],
                radius=config.dense_geometry_median_radius_frames,
                preserve_run_endpoints=False,
            )
        for index, face_box in enumerate(face_boxes):
            subject_box = subject_boxes[index]
            if (
                face_box is not None
                and subject_box is not None
                and not _box_contains_center(subject_box, face_box)
            ):
                face_boxes[index] = None
        face_boxes = _stabilize_subject_relative_face_track(
            frame_indices=ordered_frames,
            face_boxes=face_boxes,
            subject_boxes=subject_boxes,
            frame_count=frame_count,
            config=config,
        )
        reconstructed = _reconstruct_dense_face_geometry(
            frame_indices=ordered_frames,
            face_boxes=face_boxes,
            subject_boxes=subject_boxes,
            config=config,
        )
        if reconstructed is None:
            return reject(
                "dense_face_track_gap_unresolved",
                "face_detections="
                f"{sum(face_box is not None for face_box in face_boxes)},"
                f"review_frames={len(ordered_frames)}",
            )
        boxes = reconstructed
    elif resolved_basis in {
        "aggregate_human_silhouette",
        "unambiguous_dominant_person_box_geometry_only",
        FOCAL_SELECTOR_BODY_BASIS,
    }:
        if any(subject_box is None for subject_box in subject_boxes):
            return reject("dense_subject_track_gap_unresolved")
        boxes = _median_smooth_geometry(
            ordered_frames,
            [
                subject_box
                for subject_box in subject_boxes
                if subject_box is not None
            ],
            radius=config.dense_geometry_median_radius_frames,
            preserve_run_endpoints=False,
        )
    else:
        return reject("dense_alignment_geometry_basis_invalid")

    if (
        _is_face_alignment_basis(resolved_basis)
        and all(subject_box is not None for subject_box in subject_boxes)
    ):
        stabilized_subject_boxes = _stabilize_face_associated_subject_track(
            frame_indices=ordered_frames,
            subject_boxes=subject_boxes,
            face_boxes=boxes,
            frame_count=frame_count,
            config=config,
        )
        if stabilized_subject_boxes is None:
            return reject("dense_subject_track_gap_unresolved")
        subject_boxes = list(stabilized_subject_boxes)

    if (
        not _is_face_alignment_basis(resolved_basis)
        and all(subject_box is not None for subject_box in subject_boxes)
    ):
        smoothed_subject_boxes = _median_smooth_geometry(
            ordered_frames,
            [
                subject_box
                for subject_box in subject_boxes
                if subject_box is not None
            ],
            radius=config.dense_geometry_median_radius_frames,
            preserve_run_endpoints=False,
        )
        subject_boxes = list(smoothed_subject_boxes)

    # YuNet and silhouette boxes can make a single-frame excursion while the
    # authenticated member track is otherwise continuous. Replace only that
    # isolated excursion, and only when the same verified subject contains the
    # face on all three frames. A persistent jump or native source cut remains
    # visible to the continuity gate below.
    for index in range(1, len(ordered_frames) - 1):
        if not (
            ordered_frames[index - 1] + 1 == ordered_frames[index]
            and ordered_frames[index] + 1 == ordered_frames[index + 1]
            and all(
                samples[item].face_count == 1
                for item in (index - 1, index, index + 1)
            )
            and all(
                subject_boxes[item] is not None
                for item in (index - 1, index, index + 1)
            )
        ):
            continue
        previous_subject = subject_boxes[index - 1]
        current_subject = subject_boxes[index]
        following_subject = subject_boxes[index + 1]
        assert previous_subject is not None
        assert current_subject is not None
        assert following_subject is not None
        if not all(
            _box_contains_center(subject, boxes[item])
            for item, subject in (
                (index - 1, previous_subject),
                (index, current_subject),
                (index + 1, following_subject),
            )
        ):
            continue
        replacement = _isolated_outlier_replacement(
            boxes[index - 1],
            boxes[index],
            boxes[index + 1],
            maximum_center_step=config.maximum_dense_center_step,
            maximum_height_ratio=config.maximum_dense_height_ratio,
        )
        if replacement is not None:
            boxes[index] = replacement

    # Silhouette bounds breathe with limbs. The same one-frame consensus rule
    # is safe when the independently tracked face stays associated throughout.
    for index in range(1, len(ordered_frames) - 1):
        if not (
            ordered_frames[index - 1] + 1 == ordered_frames[index]
            and ordered_frames[index] + 1 == ordered_frames[index + 1]
        ):
            continue
        previous_subject = subject_boxes[index - 1]
        current_subject = subject_boxes[index]
        following_subject = subject_boxes[index + 1]
        if (
            previous_subject is None
            or current_subject is None
            or following_subject is None
            or not all(
                _box_contains_center(subject, boxes[item])
                for item, subject in (
                    (index - 1, previous_subject),
                    (index, current_subject),
                    (index + 1, following_subject),
                )
            )
        ):
            continue
        replacement = _isolated_outlier_replacement(
            previous_subject,
            current_subject,
            following_subject,
            maximum_center_step=config.maximum_dense_subject_center_step,
            maximum_height_ratio=config.maximum_dense_subject_height_ratio,
        )
        if replacement is not None:
            subject_boxes[index] = replacement

    previous_box: tuple[float, float, float, float] | None = None
    previous_subject_box: tuple[float, float, float, float] | None = None
    previous_frame_index: int | None = None
    for frame_index, sample, raw_box, box, subject_box in zip(
        ordered_frames,
        samples,
        raw_boxes,
        boxes,
        subject_boxes,
        strict=True,
    ):
        local_time = frame_index / fps
        if (
            previous_frame_index is None
            or frame_index != previous_frame_index + 1
        ):
            previous_box = None
            previous_subject_box = None
        if previous_box is not None:
            if _geometry_step_exceeds(
                previous_box,
                box,
                maximum_center_step=config.maximum_dense_center_step,
                maximum_height_ratio=config.maximum_dense_height_ratio,
            ):
                center_step = math.hypot(
                    _center(previous_box)[0] - _center(box)[0],
                    _center(previous_box)[1] - _center(box)[1],
                )
                height_ratio = max(previous_box[3], box[3]) / max(
                    1e-9,
                    min(previous_box[3], box[3]),
                )
                return reject(
                    "dense_face_geometry_jump",
                    f"frame={frame_index},center_step={center_step:.6f},"
                    f"height_ratio={height_ratio:.6f},"
                    f"basis={resolved_basis},"
                    f"raw_box={','.join(f'{value:.6f}' for value in raw_box)},"
                    f"resolved_box={','.join(f'{value:.6f}' for value in box)}",
                )
        state_sample = track.evaluate(local_time)
        state = _State(
            state_sample.zoom,
            state_sample.center_x,
            state_sample.center_y,
        )
        if _is_face_alignment_basis(resolved_basis):
            face_visible_fraction = _visible_fraction(box, state)
            if (
                face_visible_fraction + 1e-9
                < config.minimum_dense_face_visible_fraction
            ):
                raw_face_visible_fraction = _visible_fraction(raw_box, state)
                return reject(
                    "dense_face_crop_exceeds_visible_fraction",
                    f"frame={frame_index},"
                    f"visible={face_visible_fraction:.6f},"
                    f"raw_visible={raw_face_visible_fraction:.6f},"
                    f"zoom={state.zoom:.6f},center_x={state.center_x:.6f},"
                    f"center_y={state.center_y:.6f},"
                    f"raw_box={','.join(f'{value:.6f}' for value in raw_box)},"
                    f"resolved_box="
                    f"{','.join(f'{value:.6f}' for value in box)},"
                    f"subject_basis={sample.subject_geometry_basis}",
                )
        else:
            if sample.face_count == 1:
                raw_face_visible_fraction = _visible_fraction(raw_box, state)
                if (
                    raw_face_visible_fraction + 1e-9
                    < config.minimum_dense_face_visible_fraction
                ):
                    return reject(
                        "dense_face_crop_exceeds_visible_fraction",
                        f"frame={frame_index},"
                        f"visible={raw_face_visible_fraction:.6f},"
                        f"zoom={state.zoom:.6f},"
                        f"center_x={state.center_x:.6f},"
                        f"center_y={state.center_y:.6f},"
                        "basis=face_inside_subject_alignment",
                    )
                minimum_alignment_visible_fraction = (
                    config.minimum_face_associated_subject_visible_fraction
                )
            else:
                minimum_alignment_visible_fraction = (
                    config.minimum_widened_subject_visible_fraction
                )
                horizontal_visible_fraction, vertical_visible_fraction = (
                    _visible_axis_fractions(box, state)
                )
                if (
                    vertical_visible_fraction + 1e-9
                    < config.minimum_widened_subject_vertical_visible_fraction
                ):
                    return reject(
                        "dense_subject_vertical_crop_exceeds_visible_fraction",
                        f"frame={frame_index},"
                        f"horizontal={horizontal_visible_fraction:.6f},"
                        f"vertical={vertical_visible_fraction:.6f},"
                        "minimum_vertical="
                        f"{config.minimum_widened_subject_vertical_visible_fraction:.6f},"
                        f"zoom={state.zoom:.6f}",
                    )
            alignment_visible_fraction = _visible_fraction(box, state)
            if (
                alignment_visible_fraction + 1e-9
                < minimum_alignment_visible_fraction
            ):
                return reject(
                    "dense_subject_crop_exceeds_visible_fraction",
                    f"frame={frame_index},"
                    f"visible={alignment_visible_fraction:.6f},"
                    f"minimum={minimum_alignment_visible_fraction:.6f},"
                    f"zoom={state.zoom:.6f},basis={resolved_basis}",
                )
        if widened_subject_proof_required:
            if (
                subject_box is None
                or sample.subject_confidence + 1e-9
                < config.minimum_widened_subject_confidence
                or sample.subject_geometry_basis
                not in {
                    "pphumanseg_full_subject_silhouette",
                    "face_associated_formation_person_box",
                    "source_local_focal_body_track",
                    "bracketed_observed_body_containment_only",
                }
            ):
                return reject("dense_widened_subject_evidence_missing")
            if sample.face_count == 1 and not _box_contains_center(
                subject_box,
                box,
            ):
                return reject("dense_subject_face_association_failed")
            if previous_subject_box is not None:
                if _geometry_step_exceeds(
                    previous_subject_box,
                    subject_box,
                    maximum_center_step=(
                        config.maximum_dense_subject_center_step
                    ),
                    maximum_height_ratio=(
                        config.maximum_dense_subject_height_ratio
                    ),
                ):
                    subject_center_step = math.hypot(
                        _center(previous_subject_box)[0]
                        - _center(subject_box)[0],
                        _center(previous_subject_box)[1]
                        - _center(subject_box)[1],
                    )
                    subject_height_ratio = max(
                        previous_subject_box[3],
                        subject_box[3],
                    ) / max(
                        1e-9,
                        min(previous_subject_box[3], subject_box[3]),
                    )
                    return reject(
                        "dense_subject_geometry_jump",
                        f"frame={frame_index},"
                        f"center_step={subject_center_step:.6f},"
                        f"height_ratio={subject_height_ratio:.6f}",
                    )
            minimum_subject_visible_fraction = (
                config.minimum_face_associated_subject_visible_fraction
                if sample.face_count == 1
                else config.minimum_widened_subject_visible_fraction
            )
            if sample.face_count == 0:
                horizontal_visible_fraction, vertical_visible_fraction = (
                    _visible_axis_fractions(subject_box, state)
                )
                if (
                    vertical_visible_fraction + 1e-9
                    < config.minimum_widened_subject_vertical_visible_fraction
                ):
                    return reject(
                        "dense_subject_vertical_crop_exceeds_visible_fraction",
                        f"frame={frame_index},"
                        f"horizontal={horizontal_visible_fraction:.6f},"
                        f"vertical={vertical_visible_fraction:.6f},"
                        "minimum_vertical="
                        f"{config.minimum_widened_subject_vertical_visible_fraction:.6f},"
                        f"zoom={state.zoom:.6f}",
                    )
            subject_visible_fraction = _visible_fraction(subject_box, state)
            if (
                subject_visible_fraction + 1e-9
                < minimum_subject_visible_fraction
            ):
                return reject(
                    "dense_subject_crop_exceeds_visible_fraction",
                    f"frame={frame_index},"
                    f"visible={subject_visible_fraction:.6f},"
                    f"minimum={minimum_subject_visible_fraction:.6f},"
                    f"zoom={state.zoom:.6f}",
                )
            previous_subject_box = subject_box
        previous_box = box
        previous_frame_index = frame_index
    return True


def _resolved_source_frame_geometries(
    visual_selection: DirectMemberVisualSelection,
    *,
    source_count: int,
    source_frame_sizes: Sequence[tuple[int, int]] | None,
    output_frame_size: tuple[int, int],
) -> tuple[DirectMemberSourceFrameGeometry | None, ...]:
    stored = tuple(
        getattr(visual_selection, "source_frame_geometries", ())
    )
    if stored:
        if len(stored) != source_count or any(
            item is not None
            and not isinstance(item, DirectMemberSourceFrameGeometry)
            for item in stored
        ):
            raise ValueError("direct member stored source geometry is invalid")
        return stored
    if source_frame_sizes is None:
        return (None,) * source_count
    if len(source_frame_sizes) != source_count:
        raise ValueError("direct member source frame sizes must match sources")
    return tuple(
        direct_member_source_frame_geometry(
            size,
            output_size=output_frame_size,
        )
        for size in source_frame_sizes
    )


def build_direct_member_editorial_motion_plan(
    *,
    visual_selection: DirectMemberVisualSelection,
    render_segments: Sequence[RenderSegment],
    transitions: Sequence[RenderTransition],
    common_master_start: float,
    geometry_provider: DirectMemberGeometryProvider,
    fps: float = 29.97,
    config: DirectMemberEditorialMotionConfig | None = None,
    source_frame_sizes: Sequence[tuple[int, int]] | None = None,
    output_frame_size: tuple[int, int] = (1080, 1920),
    pose_geometry_provider: Callable | None = None,
) -> EditorialMotionPlan:
    """Add dense-verified face or full-subject matches to direct 9:16 cuts."""

    renders = tuple(render_segments)
    transition_items = tuple(transitions)
    source_path = tuple(int(value) for value in visual_selection.source_indices)
    boundaries = tuple(visual_selection.boundaries)
    if not renders or len(renders) != len(source_path):
        raise ValueError("direct member motion needs one selected source per segment")
    if len(transition_items) != len(renders) - 1 or len(boundaries) != len(renders) - 1:
        raise ValueError("direct member motion needs one audited transition per boundary")
    if not math.isfinite(common_master_start) or common_master_start < 0.0:
        raise ValueError("direct member common master start is invalid")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("direct member editorial fps is invalid")
    settings = config or DirectMemberEditorialMotionConfig()
    shared = _shared_settings(settings)
    # A bounded quality proof may intentionally represent only a ranked subset
    # while retaining geometry for every analyzed source. `max(path) + 1`
    # silently truncates trailing omitted slots (for example active 0..5 out
    # of eight analyzed sources) and then rejects the correctly stored
    # eight-source geometry tuple. Resolve the complete declared universe and
    # still let `_resolved_source_frame_geometries` reject inconsistent sizes.
    stored_source_geometries = tuple(
        getattr(visual_selection, "source_frame_geometries", ())
    )
    declared_source_counts = tuple(
        getattr(visual_selection, "source_counts", ())
    )
    source_count = max(
        max(source_path) + 1,
        len(stored_source_geometries),
        len(declared_source_counts),
        len(source_frame_sizes) if source_frame_sizes is not None else 0,
    )
    source_frame_geometries = _resolved_source_frame_geometries(
        visual_selection,
        source_count=source_count,
        source_frame_sizes=source_frame_sizes,
        output_frame_size=output_frame_size,
    )
    frame_counts, last_frame_times, boundary_times = _frame_schedule(renders, fps)
    if any(segment.crop is not None or segment.pan_crop is not None for segment in renders):
        raise ValueError("direct member editorial motion requires original full frames")

    blocked: set[int] = set()
    for index, transition in enumerate(transition_items):
        if transition.kind.uses_overlap:
            blocked.update((index, index + 1))

    def boundary_side_overlaps_effect(
        boundary_index: int,
        side: str,
    ) -> bool:
        """Reserve only the segment edge actually occupied by an effect."""

        if transition_items[boundary_index].kind.uses_overlap:
            return True
        affected_index = (
            boundary_index if side == "left" else boundary_index + 1
        )
        ramp = min(settings.ramp_seconds, last_frame_times[affected_index])
        if side == "left":
            adjacent_index = affected_index - 1
            if adjacent_index < 0:
                return False
            adjacent = transition_items[adjacent_index]
            if not adjacent.kind.uses_overlap:
                return False
            occupied_until = adjacent.frame_count / fps
            ramp_starts = max(0.0, last_frame_times[affected_index] - ramp)
            return occupied_until >= ramp_starts - 1e-9
        adjacent_index = affected_index
        if adjacent_index >= len(transition_items):
            return False
        adjacent = transition_items[adjacent_index]
        if not adjacent.kind.uses_overlap:
            return False
        occupied_from = max(
            0.0,
            last_frame_times[affected_index] - adjacent.frame_count / fps,
        )
        return ramp >= occupied_from - 1e-9

    skipped: dict[str, int] = {}
    required_rejection_reasons: dict[int, set[str]] = {}
    required_rejection_context: dict[int, str] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    candidates: list[_BoundaryCandidate] = []
    required_boundary_indices: set[int] = set()
    naturally_satisfied_required_boundary_indices: set[int] = set()
    natural_match_geometry: dict[
        int,
        tuple[
            str,
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ],
    ] = {}
    for boundary_index, diagnostic in enumerate(boundaries):
        if getattr(diagnostic, "boundary_index", None) != boundary_index:
            raise ValueError("direct member boundary indices must be contiguous")
        left_source = source_path[boundary_index]
        right_source = source_path[boundary_index + 1]
        if left_source == right_source:
            continue
        mismatch = getattr(diagnostic, "mismatch", None)
        repair = getattr(mismatch, "transform_repair", None)
        if repair is not None and not isinstance(
            repair,
            (DirectMemberBoundaryTransformRepair, DirectMemberBilateralTransformRepair),
        ):
            raise ValueError("direct member boundary transform repair is invalid")
        required = bool(
            (
                repair is not None
                or bool(
                    getattr(
                        mismatch,
                        "dense_attention_validation_required",
                        False,
                    )
                )
            )
            and not transition_items[boundary_index].kind.uses_overlap
        )
        if required:
            required_boundary_indices.add(boundary_index)

        def reject_boundary(reason: str) -> None:
            skip(reason)
            if required:
                required_rejection_reasons.setdefault(
                    boundary_index,
                    set(),
                ).add(reason)

        if (
            getattr(diagnostic, "outgoing_eligible_source_index", None)
            != left_source
            or getattr(diagnostic, "incoming_eligible_source_index", None)
            != right_source
        ):
            raise ValueError("direct member boundary source evidence disagrees with path")
        audited_left_geometry = _attention_alignment_geometry(
            diagnostic.outgoing_sample
        )
        audited_right_geometry = _attention_alignment_geometry(
            diagnostic.incoming_sample
        )
        left_geometry = source_frame_geometries[left_source]
        right_geometry = source_frame_geometries[right_source]
        if isinstance(repair, DirectMemberBilateralTransformRepair):
            if (repair.outgoing.source_index != left_source
                    or repair.incoming.source_index != right_source):
                raise ValueError("paired member transform sources disagree with path")
            if (repair.outgoing.source_geometry != left_geometry
                    or repair.incoming.source_geometry != right_geometry):
                raise ValueError("paired member transform source capacity disagrees with render")
        left_time = (
            common_master_start
            + renders[boundary_index].master_start
            + last_frame_times[boundary_index]
        )
        right_time = common_master_start + renders[boundary_index + 1].master_start
        paired_provider = getattr(geometry_provider, "paired_samples", None)
        if callable(paired_provider):
            left_sample, right_sample = paired_provider(
                left_source,
                left_time,
                right_source,
                right_time,
            )
        else:
            left_sample = geometry_provider(left_source, left_time)
            right_sample = geometry_provider(right_source, right_time)
        if left_sample is None or right_sample is None:
            reject_boundary("dense_boundary_face_missing")
            continue
        from .member_dense_containment import MemberDenseContainmentSample
        if isinstance(left_sample, MemberDenseContainmentSample) or isinstance(right_sample, MemberDenseContainmentSample):
            reject_boundary("containment_only_endpoint_forbidden")
            continue
        focal_face_pair = bool(
            left_sample.alignment_geometry_basis == FOCAL_FACE_BASIS
            and right_sample.alignment_geometry_basis == FOCAL_FACE_BASIS
        )
        dense_common_basis = (
            left_sample.alignment_geometry_basis
            if left_sample.alignment_geometry_basis
            == right_sample.alignment_geometry_basis
            else None
        )
        dense_common_subject_basis = None
        if (
            left_sample.subject_bbox_xywh is not None
            and right_sample.subject_bbox_xywh is not None
            and left_sample.subject_geometry_basis
            == right_sample.subject_geometry_basis
            and left_sample.subject_confidence + 1e-9
            >= settings.minimum_widened_subject_confidence
            and right_sample.subject_confidence + 1e-9
            >= settings.minimum_widened_subject_confidence
        ):
            dense_common_subject_basis = (
                "aggregate_human_silhouette"
                if left_sample.subject_geometry_basis
                == "pphumanseg_full_subject_silhouette"
                else "unambiguous_dominant_person_box_geometry_only"
            )
        dense_only_audit = False
        if focal_face_pair:
            # Metadata already binds both sources to the requested official
            # member.  The transient SFace pair proves which two visible
            # faces are the same focal performer, so the aggregate coarse
            # silhouette must not override that more specific evidence.
            audited_left = left_sample.bbox_xywh
            audited_right = right_sample.bbox_xywh
            audited_basis = FOCAL_FACE_BASIS
            dense_only_audit = True
            repair = None
            if not transition_items[boundary_index].kind.uses_overlap:
                required = True
                required_boundary_indices.add(boundary_index)
        elif (
            audited_left_geometry is None
            or audited_right_geometry is None
            or audited_left_geometry[1] != audited_right_geometry[1]
        ):
            # A provisional coarse edge is deliberately allowed to ask the
            # dense reviewer for evidence.  Requiring the same ambiguous
            # coarse pair again defeats that contract.  When no precomputed
            # repair is being replayed, two endpoint samples with one common
            # verified basis can establish the audit directly.
            if required and repair is None and (
                dense_common_basis is not None
                or dense_common_subject_basis is not None
            ):
                if dense_common_basis is not None:
                    audited_left = left_sample.bbox_xywh
                    audited_right = right_sample.bbox_xywh
                    audited_basis = dense_common_basis
                else:
                    assert left_sample.subject_bbox_xywh is not None
                    assert right_sample.subject_bbox_xywh is not None
                    assert dense_common_subject_basis is not None
                    audited_left = left_sample.subject_bbox_xywh
                    audited_right = right_sample.subject_bbox_xywh
                    audited_basis = dense_common_subject_basis
                dense_only_audit = True
            else:
                reject_boundary("no_unambiguous_face_pair")
                continue
        else:
            audited_left, audited_basis = audited_left_geometry
            audited_right, _right_basis = audited_right_geometry
        audited_left = (
            map_box_to_portrait_fill(audited_left, left_geometry)
            if left_geometry is not None
            else audited_left
        )
        audited_right = (
            map_box_to_portrait_fill(audited_right, right_geometry)
            if right_geometry is not None
            else audited_right
        )
        if not dense_only_audit and (
            left_sample.alignment_geometry_basis != audited_basis
            or right_sample.alignment_geometry_basis != audited_basis
        ):
            reject_boundary("dense_boundary_face_disagrees_with_audit")
            continue
        left_box = (
            audited_left
            if dense_only_audit
            else map_box_to_portrait_fill(left_sample.bbox_xywh, left_geometry)
            if left_geometry is not None
            else left_sample.bbox_xywh
        )
        right_box = (
            audited_right
            if dense_only_audit
            else map_box_to_portrait_fill(right_sample.bbox_xywh, right_geometry)
            if right_geometry is not None
            else right_sample.bbox_xywh
        )
        if (
            not dense_only_audit
            and (
                not _endpoint_agrees(left_box, audited_left, settings)
                or not _endpoint_agrees(right_box, audited_right, settings)
            )
        ):
            reject_boundary("dense_boundary_face_disagrees_with_audit")
            continue
        center_distance = math.hypot(
            _center(left_box)[0] - _center(right_box)[0],
            _center(left_box)[1] - _center(right_box)[1],
        )
        if center_distance > settings.maximum_boundary_center_distance + 1e-9:
            reject_boundary("boundary_position_exceeds_transform_bound")
            continue
        ratio = max(left_box[3], right_box[3]) / max(
            1e-9, min(left_box[3], right_box[3])
        )
        if ratio <= 1.015 and center_distance <= 0.025:
            if required:
                naturally_satisfied_required_boundary_indices.add(
                    boundary_index
                )
                natural_match_geometry[boundary_index] = (
                    audited_basis,
                    left_box,
                    right_box,
                )
                skip("required_dense_natural_match")
            else:
                skip("already_scale_position_matched")
            continue
        baseline = _mismatch(left_box, right_box)
        if required and repair is not None:
            if repair.alignment_geometry_basis != audited_basis:
                raise ValueError(
                    "required transform geometry basis disagrees with dense audit"
                )
            if isinstance(repair, DirectMemberBilateralTransformRepair):
                preferred_side = "left"
            else:
                preferred_side = (
                    "left" if repair.transformed_side == "outgoing" else "right"
                )
                expected_source = (
                    left_source if preferred_side == "left" else right_source
                )
                if repair.transformed_source_index != expected_source:
                    raise ValueError(
                        "direct member required transform source disagrees with path"
                    )
        elif left_box[3] + 1e-9 < right_box[3]:
            preferred_side = "left"
        else:
            preferred_side = "right"
        required_improvement_fraction = (
            min(settings.minimum_mismatch_improvement_fraction, 0.05)
            if required
            and left_sample.subject_bbox_xywh is not None
            and right_sample.subject_bbox_xywh is not None
            else settings.minimum_mismatch_improvement_fraction
        )
        professional_position_limit = (
            settings.maximum_face_residual_position_mismatch
            if _is_face_alignment_basis(audited_basis)
            else settings.maximum_subject_residual_position_mismatch
        )
        professional_scale_limit = (
            settings.maximum_face_residual_scale_mismatch
            if _is_face_alignment_basis(audited_basis)
            else settings.maximum_subject_residual_scale_mismatch
        )
        natural_position, natural_scale = _rendered_attention_mismatch(
            left_box,
            right_box,
        )
        if (
            required
            and natural_position <= professional_position_limit + 1e-9
            and natural_scale <= professional_scale_limit + 1e-9
        ):
            naturally_satisfied_required_boundary_indices.add(boundary_index)
            natural_match_geometry[boundary_index] = (
                audited_basis,
                left_box,
                right_box,
            )
            skip("required_dense_natural_match")
            continue

        if required and isinstance(repair, DirectMemberBilateralTransformRepair):
            # Reproduce the selector's TWO source-bound endpoint states, then
            # subject both full ramps to the existing atomic dense publication.
            # Never reinterpret the second camera as a one-sided certificate.
            if any(boundary_side_overlaps_effect(boundary_index, side) for side in ("left", "right")):
                reject_boundary("bilateral_dissolve_adjacent")
                continue
            if min(last_frame_times[boundary_index:boundary_index + 2]) < .55 - 1e-9:
                reject_boundary("bilateral_window_shorter_than_reference_ramp")
                continue
            if any(camera.zoom > min(settings.maximum_boundary_zoom,
                       camera.source_geometry.lossless_zoom_capacity * settings.maximum_boundary_upscale_ratio) + 1e-9
                   or camera.center_shift > settings.maximum_center_shift + 1e-9 for camera in repair.cameras):
                reject_boundary("bilateral_camera_exceeds_render_bounds")
                continue
            states = tuple(_State(zoom=c.zoom, center_x=c.center_x, center_y=c.center_y) for c in repair.cameras)
            projected = tuple(_project(box, state) for box, state in zip((left_box, right_box), states))
            position, scale = _rendered_attention_mismatch(*projected)
            transformed = _mismatch(*projected)
            minimum_visible = (settings.minimum_dense_face_visible_fraction if _is_face_alignment_basis(audited_basis)
                               else settings.minimum_widened_subject_visible_fraction)
            if (position > professional_position_limit + 1e-9 or scale > professional_scale_limit + 1e-9
                    or math.dist(_center(projected[0]), _center(projected[1])) > settings.maximum_residual_center_distance + 1e-9
                    or transformed > baseline * (1.0 - required_improvement_fraction) + 1e-9
                    or min(_visible_fraction(box, state) for box, state in zip((left_box, right_box), states)) + 1e-9 < minimum_visible):
                reject_boundary("bilateral_selector_camera_disagrees_with_dense_geometry")
                continue
            candidates.append(_BoundaryCandidate(
                boundary_index=boundary_index, event_time=boundary_times[boundary_index],
                score=baseline - transformed, side="both", state=states[0], paired_state=states[1],
                upscale_ratio=repair.outgoing.upscale_ratio, paired_upscale_ratio=repair.incoming.upscale_ratio,
                alignment_geometry_basis=audited_basis, left_box=left_box, right_box=right_box, required=True,
            ))
            continue

        def qualifying_for_side(
            candidate_side: str,
        ) -> tuple[
            list[tuple[float, _State]],
            float | None,
            str | None,
        ]:
            affected_index = (
                boundary_index
                if candidate_side == "left"
                else boundary_index + 1
            )
            if boundary_side_overlaps_effect(
                boundary_index,
                candidate_side,
            ):
                return ([], None, "dissolve_adjacent")
            if last_frame_times[affected_index] < 0.55 - 1e-9:
                return ([], None, "window_shorter_than_reference_ramp")
            box, target_box = (
                (left_box, right_box)
                if candidate_side == "left"
                else (right_box, left_box)
            )
            affected_source = source_path[affected_index]
            affected_geometry = source_frame_geometries[affected_source]
            if required and affected_geometry is None:
                return (
                    [],
                    None,
                    "required_transform_source_resolution_missing",
                )
            lossless_zoom_capacity = (
                affected_geometry.lossless_zoom_capacity
                if affected_geometry is not None
                else settings.maximum_boundary_zoom
            )
            render_quality_zoom_limit = (
                min(
                    settings.maximum_boundary_zoom,
                    lossless_zoom_capacity
                    * settings.maximum_boundary_upscale_ratio,
                )
                if affected_geometry is not None
                else settings.maximum_boundary_zoom
            )
            if render_quality_zoom_limit <= 1.0 + 1e-9:
                return (
                    [],
                    render_quality_zoom_limit,
                    "source_resolution_has_no_bounded_upscale_headroom",
                )
            desired_center_shift = min(
                settings.maximum_center_shift,
                math.hypot(
                    _center(box)[0] - _center(target_box)[0],
                    _center(box)[1] - _center(target_box)[1],
                ),
            )
            pan_headroom_zoom = 1.0 / max(
                1e-9,
                1.0 - 2.0 * desired_center_shift,
            )
            requested_zoom = max(
                1.02,
                target_box[3] / box[3],
                pan_headroom_zoom,
            )
            maximum_candidate_zoom = min(
                render_quality_zoom_limit,
                requested_zoom,
            )
            if (
                required
                and repair is not None
                and candidate_side == preferred_side
            ):
                maximum_candidate_zoom = min(
                    render_quality_zoom_limit,
                    max(maximum_candidate_zoom, repair.zoom),
                )
            zoom_values = [maximum_candidate_zoom]
            while zoom_values[-1] > 1.02 + 1e-9:
                next_zoom = max(1.02, zoom_values[-1] - 0.02)
                if abs(next_zoom - zoom_values[-1]) <= 1e-9:
                    break
                zoom_values.append(next_zoom)
            qualifying: list[tuple[float, _State]] = []
            seen_states: set[tuple[float, float, float]] = set()
            for zoom in zoom_values:
                candidate_states = list(
                    _state_for_match(
                        box=box,
                        target_box=target_box,
                        zoom=zoom,
                        settings=shared,
                    )
                )
                half_viewport = 0.5 / zoom
                camera_axis_values = (
                    half_viewport,
                    0.5,
                    1.0 - half_viewport,
                )
                for center_x in camera_axis_values:
                    for center_y in camera_axis_values:
                        if (
                            math.hypot(center_x - 0.5, center_y - 0.5)
                            > settings.maximum_center_shift + 1e-9
                        ):
                            continue
                        candidate_states.append(
                            _State(
                                zoom=zoom,
                                center_x=center_x,
                                center_y=center_y,
                            )
                        )
                for state in candidate_states:
                    state_key = (
                        round(state.zoom, 9),
                        round(state.center_x, 9),
                        round(state.center_y, 9),
                    )
                    if state_key in seen_states:
                        continue
                    seen_states.add(state_key)
                    minimum_alignment_visible_fraction = (
                        settings.minimum_dense_face_visible_fraction
                        if _is_face_alignment_basis(audited_basis)
                        else settings.minimum_widened_subject_visible_fraction
                    )
                    if (
                        _visible_fraction(box, state) + 1e-9
                        < minimum_alignment_visible_fraction
                    ):
                        continue
                    projected_left = (
                        _project(left_box, state)
                        if candidate_side == "left"
                        else left_box
                    )
                    projected_right = (
                        _project(right_box, state)
                        if candidate_side == "right"
                        else right_box
                    )
                    transformed = _mismatch(
                        projected_left,
                        projected_right,
                    )
                    residual_center = math.hypot(
                        _center(projected_left)[0]
                        - _center(projected_right)[0],
                        _center(projected_left)[1]
                        - _center(projected_right)[1],
                    )
                    if (
                        residual_center
                        > settings.maximum_residual_center_distance + 1e-9
                    ):
                        continue
                    residual_position, residual_scale = (
                        _rendered_attention_mismatch(
                            projected_left,
                            projected_right,
                        )
                    )
                    if (
                        residual_position > professional_position_limit + 1e-9
                        or residual_scale > professional_scale_limit + 1e-9
                    ):
                        continue
                    if transformed > baseline * (
                        1.0 - required_improvement_fraction
                    ) + 1e-9:
                        continue
                    if transformed + 1e-4 < baseline:
                        qualifying.append((transformed, state))
            # Exact-pixel states are a separate first tier.  Only use bounded
            # upscaling when no lossless state satisfies the same professional
            # residual and containment gates.
            lossless_qualifying = [
                item
                for item in qualifying
                if item[1].zoom <= lossless_zoom_capacity + 1e-9
            ]
            if lossless_qualifying:
                qualifying = lossless_qualifying
            qualifying.sort(
                key=lambda item: (
                    item[0],
                    max(1.0, item[1].zoom / max(1e-9, lossless_zoom_capacity)),
                    item[1].zoom,
                )
            )
            return (qualifying, maximum_candidate_zoom, None)

        side_results: list[
            tuple[str, list[tuple[float, _State]], float | None, str | None]
        ] = []
        for candidate_side in (
            preferred_side,
            "right" if preferred_side == "left" else "left",
        ):
            qualifying, maximum_zoom, failure = qualifying_for_side(
                candidate_side
            )
            side_results.append(
                (candidate_side, qualifying, maximum_zoom, failure)
            )

        bilateral_match: tuple[float, _State, _State, float] | None = None
        bilateral_failure: str | None = None
        if required and not any(
            qualifying for _side, qualifying, _zoom, _failure in side_results
        ):
            if boundary_side_overlaps_effect(
                boundary_index,
                "left",
            ) or boundary_side_overlaps_effect(boundary_index, "right"):
                bilateral_failure = "bilateral_dissolve_adjacent"
            elif (
                last_frame_times[boundary_index] < 0.55 - 1e-9
                or last_frame_times[boundary_index + 1] < 0.55 - 1e-9
            ):
                bilateral_failure = "bilateral_window_shorter_than_reference_ramp"
            elif left_geometry is None or right_geometry is None:
                bilateral_failure = "bilateral_source_resolution_missing"
            else:
                left_lossless_capacity = left_geometry.lossless_zoom_capacity
                right_lossless_capacity = right_geometry.lossless_zoom_capacity
                left_zoom_limit = min(
                    settings.maximum_boundary_zoom,
                    left_lossless_capacity
                    * settings.maximum_boundary_upscale_ratio,
                )
                right_zoom_limit = min(
                    settings.maximum_boundary_zoom,
                    right_lossless_capacity
                    * settings.maximum_boundary_upscale_ratio,
                )
                minimum_target_height = max(left_box[3], right_box[3])
                maximum_target_height = min(
                    left_box[3] * left_zoom_limit,
                    right_box[3] * right_zoom_limit,
                )
                if maximum_target_height + 1e-9 < minimum_target_height:
                    bilateral_failure = "bilateral_bounded_upscale_exceeded"
                else:
                    # A fixed 3x3 target grid misses valid shared-camera
                    # solutions when two edge-framed fancams must split the
                    # pan between them.  Search the complete bounded camera
                    # interval at four-percent steps.  Every resulting state
                    # is still checked by the exact pan, containment, dense
                    # continuity, and residual gates below.
                    target_axis_values = tuple(
                        0.50 + 0.04 * offset
                        for offset in range(-4, 5)
                    )
                    target_centers = tuple(
                        (target_x, target_y)
                        for target_x in target_axis_values
                        for target_y in target_axis_values
                    )
                    minimum_alignment_visible_fraction = (
                        settings.minimum_dense_face_visible_fraction
                        if _is_face_alignment_basis(audited_basis)
                        else settings.minimum_widened_subject_visible_fraction
                    )
                    options: list[
                        tuple[bool, float, float, float, _State, _State]
                    ] = []
                    height_span = maximum_target_height - minimum_target_height
                    for step in range(33):
                        target_height = (
                            minimum_target_height
                            if height_span <= 1e-12
                            else minimum_target_height
                            + height_span * step / 32.0
                        )
                        left_zoom = max(1.0, target_height / left_box[3])
                        right_zoom = max(1.0, target_height / right_box[3])
                        for target_x, target_y in target_centers:
                            target_box = (
                                target_x - 0.05,
                                target_y - target_height / 2.0,
                                0.10,
                                target_height,
                            )
                            left_states = _state_for_match(
                                box=left_box,
                                target_box=target_box,
                                zoom=left_zoom,
                                settings=shared,
                            )
                            right_states = _state_for_match(
                                box=right_box,
                                target_box=target_box,
                                zoom=right_zoom,
                                settings=shared,
                            )
                            for left_state in left_states:
                                if (
                                    _visible_fraction(left_box, left_state) + 1e-9
                                    < minimum_alignment_visible_fraction
                                ):
                                    continue
                                for right_state in right_states:
                                    if (
                                        _visible_fraction(right_box, right_state)
                                        + 1e-9
                                        < minimum_alignment_visible_fraction
                                    ):
                                        continue
                                    projected_left = _project(
                                        left_box,
                                        left_state,
                                    )
                                    projected_right = _project(
                                        right_box,
                                        right_state,
                                    )
                                    residual_position, residual_scale = (
                                        _rendered_attention_mismatch(
                                            projected_left,
                                            projected_right,
                                        )
                                    )
                                    residual_center = math.hypot(
                                        _center(projected_left)[0]
                                        - _center(projected_right)[0],
                                        _center(projected_left)[1]
                                        - _center(projected_right)[1],
                                    )
                                    transformed = _mismatch(
                                        projected_left,
                                        projected_right,
                                    )
                                    if (
                                        residual_center
                                        > settings.maximum_residual_center_distance
                                        + 1e-9
                                        or residual_position
                                        > professional_position_limit + 1e-9
                                        or residual_scale
                                        > professional_scale_limit + 1e-9
                                        or transformed
                                        > baseline
                                        * (1.0 - required_improvement_fraction)
                                        + 1e-9
                                    ):
                                        continue
                                    camera_effort = (
                                        left_state.zoom
                                        + right_state.zoom
                                        - 2.0
                                        + math.hypot(
                                            left_state.center_x - 0.5,
                                            left_state.center_y - 0.5,
                                        )
                                        + math.hypot(
                                            right_state.center_x - 0.5,
                                            right_state.center_y - 0.5,
                                        )
                                    )
                                    upscale_ratio = max(
                                        1.0,
                                        left_state.zoom
                                        / max(1e-9, left_lossless_capacity),
                                        right_state.zoom
                                        / max(1e-9, right_lossless_capacity),
                                    )
                                    options.append(
                                        (
                                            upscale_ratio > 1.0 + 1e-9,
                                            transformed,
                                            camera_effort,
                                            upscale_ratio,
                                            left_state,
                                            right_state,
                                        )
                                    )
                    if options:
                        (
                            _uses_upscale,
                            transformed,
                            _effort,
                            bilateral_upscale_ratio,
                            left_state,
                            right_state,
                        ) = min(
                            options,
                            key=lambda item: (
                                item[0],
                                item[1],
                                item[2],
                                item[3],
                            ),
                        )
                        bilateral_match = (
                            transformed,
                            left_state,
                            right_state,
                            bilateral_upscale_ratio,
                        )
                    else:
                        bilateral_failure = (
                            "bilateral_residual_or_improvement_failed"
                        )

        if bilateral_match is not None:
            (
                bilateral_score,
                left_state,
                right_state,
                bilateral_upscale_ratio,
            ) = bilateral_match
            candidates.append(
                _BoundaryCandidate(
                    boundary_index=boundary_index,
                    event_time=boundary_times[boundary_index],
                    score=baseline - bilateral_score,
                    side="both",
                    state=left_state,
                    paired_state=right_state,
                    upscale_ratio=max(
                        1.0,
                        left_state.zoom
                        / max(1e-9, left_geometry.lossless_zoom_capacity),
                    ),
                    paired_upscale_ratio=max(
                        1.0,
                        right_state.zoom
                        / max(1e-9, right_geometry.lossless_zoom_capacity),
                    ),
                    alignment_geometry_basis=audited_basis,
                    left_box=left_box,
                    right_box=right_box,
                    required=required,
                )
            )
            continue

        if not any(qualifying for _side, qualifying, _zoom, _failure in side_results):
            failures = {
                failure or "no_improving_contained_transform"
                for _side, _qualifying, _zoom, failure in side_results
            }
            if bilateral_failure is not None:
                failures.add(bilateral_failure)
            for failure in sorted(failures):
                reject_boundary(failure)
            if required:
                required_rejection_context[boundary_index] = (
                    f"baseline={baseline:.6f},center={center_distance:.6f},"
                    f"height_ratio={ratio:.6f},sides="
                    + ",".join(
                        f"{candidate_side}:"
                        + (
                            f"{maximum_zoom:.6f}"
                            if maximum_zoom is not None
                            else "none"
                        )
                        for candidate_side, _states, maximum_zoom, _failure
                        in side_results
                    )
                )
            continue
        for candidate_side, qualifying, _maximum_zoom, _failure in side_results:
            if not qualifying:
                continue
            best_score, best_state = qualifying[0]
            affected_source = (
                left_source if candidate_side == "left" else right_source
            )
            affected_geometry = source_frame_geometries[affected_source]
            upscale_ratio = (
                max(
                    1.0,
                    best_state.zoom
                    / max(1e-9, affected_geometry.lossless_zoom_capacity),
                )
                if affected_geometry is not None
                else 1.0
            )
            candidates.append(
                _BoundaryCandidate(
                    boundary_index=boundary_index,
                    event_time=boundary_times[boundary_index],
                    score=baseline - best_score,
                    side=candidate_side,
                    state=best_state,
                    upscale_ratio=upscale_ratio,
                    alignment_geometry_basis=audited_basis,
                    left_box=left_box,
                    right_box=right_box,
                    alternative_states=tuple(
                        state for _score, state in qualifying[1:]
                    ),
                    required=required,
                )
            )

    quantized_duration = sum(frame_counts) / fps
    maximum_events = int(math.floor(
        quantized_duration * settings.target_events_per_minute / 60.0 + 1e-12
    ))
    entry_states: list[_State | None] = [None] * len(renders)
    exit_states: list[_State | None] = [None] * len(renders)
    tracks: list[EditorialTransformTrack | None] = [None] * len(renders)
    segment_alignment_bases: list[set[str]] = [
        set() for _segment in renders
    ]
    accepted: list[_BoundaryCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            not item.required,
            -item.score,
            item.event_time,
        ),
    ):
        if any(
            item.boundary_index == candidate.boundary_index
            for item in accepted
        ):
            skip("alternate_side_boundary_already_satisfied")
            continue
        if not candidate.required and len(accepted) >= maximum_events:
            skip("event_density_cap")
            continue
        if not candidate.required and any(
            abs(candidate.event_time - item.event_time)
            < settings.minimum_event_spacing_seconds - 1e-9
            for item in accepted
        ):
            skip("event_spacing_cap")
            continue
        if candidate.side == "both":
            if candidate.paired_state is None:
                raise ValueError("bilateral member camera is missing its paired state")
            left_index = candidate.boundary_index
            right_index = candidate.boundary_index + 1
            previous_left_exit = exit_states[left_index]
            previous_right_entry = entry_states[right_index]
            previous_left_track = tracks[left_index]
            previous_right_track = tracks[right_index]
            previous_left_bases = set(segment_alignment_bases[left_index])
            previous_right_bases = set(segment_alignment_bases[right_index])
            exit_states[left_index] = candidate.state
            entry_states[right_index] = candidate.paired_state
            left_bases = set(previous_left_bases)
            right_bases = set(previous_right_bases)
            left_bases.add(candidate.alignment_geometry_basis)
            right_bases.add(candidate.alignment_geometry_basis)
            dense_rejection_reasons: set[str] = set()
            dense_rejection_details: set[str] = set()

            def bilateral_track(
                index: int,
                bases: set[str],
            ) -> EditorialTransformTrack | None:
                try:
                    track = _track_for_states(
                        entry=entry_states[index],
                        exit=exit_states[index],
                        last_frame_time=last_frame_times[index],
                        settings=shared,
                    )
                except ValueError:
                    return None
                review_basis = (
                    next(iter(bases)) if len(bases) == 1 else None
                )
                if track is None:
                    return None

                def reviewed(proposal):
                    followed = _dense_tracking_camera_track(
                        eligible_source_index=source_path[index], segment=renders[index],
                        track=proposal, frame_count=frame_counts[index], common_master_start=common_master_start,
                        geometry_provider=geometry_provider, fps=fps, config=settings,
                        source_frame_geometry=source_frame_geometries[source_path[index]],
                        alignment_geometry_basis=review_basis, diagnostic_callback=dense_rejection_details.add,
                    )
                    safe = _track_is_dense_face_safe(
                        segment_index=index, eligible_source_index=source_path[index], segment=renders[index],
                        track=followed, frame_count=frame_counts[index], common_master_start=common_master_start,
                        geometry_provider=geometry_provider, fps=fps, config=settings,
                        source_frame_geometry=source_frame_geometries[source_path[index]],
                        alignment_geometry_basis=review_basis, rejection_callback=dense_rejection_reasons.add,
                        rejection_detail_callback=dense_rejection_details.add,
                    )
                    return followed if safe else None

                verified = reviewed(track)
                if verified is not None:
                    return verified
                # Use the same single bounded alternative as one-sided hard
                # cuts. Preserve the exact cut camera and .55 s/velocity floor;
                # an unobserved tail remains neutral, not fabricated geometry.
                from .member_dense_containment import dense_camera_sample
                from .visible_hard_ramp import shorten_hard_ramp_to_observations
                shortened = shorten_hard_ramp_to_observations(track, fps=fps,
                    frame_count=frame_counts[index], frame_available=lambda frame:
                    dense_camera_sample(geometry_provider, source_path[index],
                        common_master_start + renders[index].master_start + frame/fps,
                        interior=0 < frame < frame_counts[index]-1) is not None)
                return reviewed(shortened) if shortened is not None else None

            left_track = bilateral_track(left_index, left_bases)
            right_track = bilateral_track(right_index, right_bases)
            if left_track is None or right_track is None:
                exit_states[left_index] = previous_left_exit
                entry_states[right_index] = previous_right_entry
                tracks[left_index] = previous_left_track
                tracks[right_index] = previous_right_track
                segment_alignment_bases[left_index] = previous_left_bases
                segment_alignment_bases[right_index] = previous_right_bases
                skip("dense_bilateral_containment_or_continuity_failed")
                if candidate.required:
                    required_rejection_reasons.setdefault(
                        candidate.boundary_index,
                        set(),
                    ).update(
                        dense_rejection_reasons
                        or {"dense_bilateral_containment_or_continuity_failed"}
                    )
                    if dense_rejection_details:
                        required_rejection_context[
                            candidate.boundary_index
                        ] = ";".join(sorted(dense_rejection_details))
                continue
            tracks[left_index] = left_track
            tracks[right_index] = right_track
            segment_alignment_bases[left_index] = left_bases
            segment_alignment_bases[right_index] = right_bases
            accepted.append(candidate)
            continue
        affected_index = (
            candidate.boundary_index
            if candidate.side == "left"
            else candidate.boundary_index + 1
        )
        previous_entry = entry_states[affected_index]
        previous_exit = exit_states[affected_index]
        accepted_state: _State | None = None
        accepted_track: EditorialTransformTrack | None = None
        dense_rejection_reasons: set[str] = set()
        dense_rejection_details: set[str] = set()
        candidate_alignment_bases = set(
            segment_alignment_bases[affected_index]
        )
        candidate_alignment_bases.add(candidate.alignment_geometry_basis)
        review_alignment_basis = (
            next(iter(candidate_alignment_bases))
            if len(candidate_alignment_bases) == 1
            else None
        )
        for state in (candidate.state,) + candidate.alternative_states:
            if candidate.side == "left":
                exit_states[affected_index] = state
            else:
                entry_states[affected_index] = state
            try:
                track = _track_for_states(
                    entry=entry_states[affected_index],
                    exit=exit_states[affected_index],
                    last_frame_time=last_frame_times[affected_index],
                    settings=shared,
                )
            except ValueError:
                track = None
            nominal_track = track
            if track is not None:
                track = _dense_tracking_camera_track(
                    eligible_source_index=source_path[affected_index],
                    segment=renders[affected_index],
                    track=track,
                    frame_count=frame_counts[affected_index],
                    common_master_start=common_master_start,
                    geometry_provider=geometry_provider,
                    fps=fps,
                    config=settings,
                    source_frame_geometry=source_frame_geometries[
                        source_path[affected_index]
                    ],
                    alignment_geometry_basis=review_alignment_basis,
                    diagnostic_callback=dense_rejection_details.add,
                )
            dense_safe = track is not None and _track_is_dense_face_safe(
                segment_index=affected_index,
                eligible_source_index=source_path[affected_index],
                segment=renders[affected_index],
                track=track,
                frame_count=frame_counts[affected_index],
                common_master_start=common_master_start,
                geometry_provider=geometry_provider,
                fps=fps,
                config=settings,
                source_frame_geometry=source_frame_geometries[
                    source_path[affected_index]
                ],
                alignment_geometry_basis=review_alignment_basis,
                rejection_callback=dense_rejection_reasons.add,
                rejection_detail_callback=dense_rejection_details.add,
            )
            if not dense_safe and nominal_track is not None:
                from .visible_hard_ramp import shorten_hard_ramp_to_observations

                shortened = shorten_hard_ramp_to_observations(
                    nominal_track,
                    fps=fps,
                    frame_count=frame_counts[affected_index],
                    frame_available=lambda frame: geometry_provider(
                        source_path[affected_index],
                        common_master_start + renders[affected_index].master_start
                        + frame / fps,
                    ) is not None,
                )
                if shortened is not None:
                    track = _dense_tracking_camera_track(
                        eligible_source_index=source_path[affected_index],
                        segment=renders[affected_index],
                        track=shortened,
                        frame_count=frame_counts[affected_index],
                        common_master_start=common_master_start,
                        geometry_provider=geometry_provider,
                        fps=fps,
                        config=settings,
                        source_frame_geometry=source_frame_geometries[source_path[affected_index]],
                        alignment_geometry_basis=review_alignment_basis,
                        diagnostic_callback=dense_rejection_details.add,
                    )
                    dense_safe = track is not None and _track_is_dense_face_safe(
                        segment_index=affected_index,
                        eligible_source_index=source_path[affected_index],
                        segment=renders[affected_index],
                        track=track,
                        frame_count=frame_counts[affected_index],
                        common_master_start=common_master_start,
                        geometry_provider=geometry_provider,
                        fps=fps,
                        config=settings,
                        source_frame_geometry=source_frame_geometries[source_path[affected_index]],
                        alignment_geometry_basis=review_alignment_basis,
                        rejection_callback=dense_rejection_reasons.add,
                        rejection_detail_callback=dense_rejection_details.add,
                    )
            if dense_safe:
                accepted_state = state
                accepted_track = track
                break
        if accepted_state is None or accepted_track is None:
            entry_states[affected_index] = previous_entry
            exit_states[affected_index] = previous_exit
            skip("dense_face_containment_or_continuity_failed")
            if candidate.required:
                required_rejection_reasons.setdefault(
                    candidate.boundary_index,
                    set(),
                ).update(
                    dense_rejection_reasons
                    or {"dense_face_containment_or_continuity_failed"}
                )
                if dense_rejection_details:
                    required_rejection_context[
                        candidate.boundary_index
                    ] = ";".join(sorted(dense_rejection_details))
            continue
        tracks[affected_index] = accepted_track
        segment_alignment_bases[affected_index] = candidate_alignment_bases
        accepted.append(
            _BoundaryCandidate(
                boundary_index=candidate.boundary_index,
                event_time=candidate.event_time,
                score=candidate.score,
                side=candidate.side,
                state=accepted_state,
                upscale_ratio=max(
                    1.0,
                    accepted_state.zoom
                    / max(
                        1e-9,
                        source_frame_geometries[
                            source_path[affected_index]
                        ].lossless_zoom_capacity,
                    ),
                )
                if source_frame_geometries[source_path[affected_index]]
                is not None
                else 1.0,
                alignment_geometry_basis=(
                    candidate.alignment_geometry_basis
                ),
                left_box=candidate.left_box,
                right_box=candidate.right_box,
                required=candidate.required,
            )
        )

    interior_pulse_segments: set[int] = set()
    if settings.enable_interior_zoom_pulses:
        target_rich_motion_count = min(
            maximum_events,
            math.ceil(
                quantized_duration
                * settings.target_rich_motion_events_per_minute
                / 60.0
                - 1e-12
            ),
        )
        accepted_event_times = [item.event_time for item in accepted]

        def interior_activity_score(segment_index: int) -> float:
            values: list[float] = []
            for boundary_index in (segment_index - 1, segment_index):
                if not 0 <= boundary_index < len(boundaries):
                    continue
                motion = getattr(
                    getattr(boundaries[boundary_index], "mismatch", None),
                    "motion_connection",
                    None,
                )
                if motion is None:
                    continue
                for name in (
                    "outgoing_motion_magnitude",
                    "incoming_motion_magnitude",
                ):
                    value = getattr(motion, name, None)
                    if value is not None and math.isfinite(float(value)):
                        values.append(float(value))
            return max(values, default=0.0)

        pulse_candidates = sorted(
            range(len(renders)),
            key=lambda index: (
                -interior_activity_score(index),
                -last_frame_times[index],
                renders[index].master_start,
                index,
            ),
        )
        for index in pulse_candidates:
            if len(accepted) + len(interior_pulse_segments) >= target_rich_motion_count:
                break
            if tracks[index] is not None or index in blocked:
                continue
            if (
                last_frame_times[index]
                < settings.minimum_interior_pulse_duration_seconds - 1e-9
            ):
                skip("interior_pulse_segment_too_short")
                continue
            event_time = renders[index].master_start + last_frame_times[index] / 2.0
            if any(
                abs(event_time - other_time)
                < settings.minimum_event_spacing_seconds - 1e-9
                for other_time in accepted_event_times
            ):
                skip("interior_pulse_spacing_cap")
                continue
            source_geometry = source_frame_geometries[source_path[index]]
            if source_geometry is None:
                skip("interior_pulse_source_resolution_missing")
                continue
            zoom = min(
                settings.interior_pulse_zoom,
                settings.maximum_boundary_zoom,
                source_geometry.lossless_zoom_capacity,
            )
            if zoom < 1.02 - 1e-9:
                skip("interior_pulse_no_lossless_zoom_headroom")
                continue
            track = EditorialTransformTrack(
                keyframes=(
                    EditorialTransformKeyframe(0.0, 1.0),
                    EditorialTransformKeyframe(
                        last_frame_times[index] / 2.0,
                        zoom,
                    ),
                    EditorialTransformKeyframe(
                        last_frame_times[index],
                        1.0,
                    ),
                ),
                reason=(
                    "official_fancam_dense_subject_lossless_interior_zoom_pulse"
                ),
            )
            track = _dense_tracking_camera_track(
                eligible_source_index=source_path[index],
                segment=renders[index],
                track=track,
                frame_count=frame_counts[index],
                common_master_start=common_master_start,
                geometry_provider=geometry_provider,
                fps=fps,
                config=settings,
                source_frame_geometry=source_geometry,
                alignment_geometry_basis=None,
            )
            if not _track_is_dense_face_safe(
                segment_index=index,
                eligible_source_index=source_path[index],
                segment=renders[index],
                track=track,
                frame_count=frame_counts[index],
                common_master_start=common_master_start,
                geometry_provider=geometry_provider,
                fps=fps,
                config=settings,
                source_frame_geometry=source_geometry,
            ):
                skip("interior_pulse_dense_containment_or_continuity_failed")
                continue
            tracks[index] = track
            interior_pulse_segments.add(index)
            accepted_event_times.append(event_time)

    for index in range(len(renders)):
        if entry_states[index] is None and exit_states[index] is None:
            continue
        # Review the accepted camera, including its observed ramp timing.
        # Rebuilding from endpoints discards that timing and dense following.
        # Every accepted entry/exit update already replaces tracks[index].
        track = tracks[index]
        review_alignment_basis = (
            next(iter(segment_alignment_bases[index]))
            if len(segment_alignment_bases[index]) == 1
            else None
        )
        if track is None or not _track_is_dense_face_safe(
            segment_index=index,
            eligible_source_index=source_path[index],
            segment=renders[index],
            track=track,
            frame_count=frame_counts[index],
            common_master_start=common_master_start,
            geometry_provider=geometry_provider,
            fps=fps,
            config=settings,
            source_frame_geometry=source_frame_geometries[source_path[index]],
            alignment_geometry_basis=review_alignment_basis,
        ):
            raise ValueError("accepted direct member camera failed final dense review")
        tracks[index] = track

    def final_track_state(
        segment_index: int,
        local_time: float,
    ) -> _State:
        track = tracks[segment_index]
        if track is None:
            return _State()
        keyframe = track.evaluate(local_time)
        return _State(
            keyframe.zoom,
            keyframe.center_x,
            keyframe.center_y,
        )

    def final_boundary_residual(
        boundary_index: int,
        left_box: tuple[float, float, float, float],
        right_box: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        projected_left = _project(
            left_box,
            final_track_state(
                boundary_index,
                last_frame_times[boundary_index],
            ),
        )
        projected_right = _project(
            right_box,
            final_track_state(boundary_index + 1, 0.0),
        )
        return _rendered_attention_mismatch(
            projected_left,
            projected_right,
        )

    final_boundary_residuals: dict[int, tuple[float, float]] = {}
    final_residual_geometry_bases: dict[int, str] = {}
    for item in accepted:
        final_boundary_residuals[item.boundary_index] = (
            final_boundary_residual(
                item.boundary_index,
                item.left_box,
                item.right_box,
            )
        )
        final_residual_geometry_bases[item.boundary_index] = (
            item.alignment_geometry_basis
        )
    for boundary_index, (
        geometry_basis,
        left_box,
        right_box,
    ) in natural_match_geometry.items():
        final_boundary_residuals[boundary_index] = final_boundary_residual(
            boundary_index,
            left_box,
            right_box,
        )
        final_residual_geometry_bases[boundary_index] = geometry_basis

    final_residual_failure_indices: set[int] = set()
    for boundary_index, (position, scale) in final_boundary_residuals.items():
        geometry_basis = final_residual_geometry_bases[boundary_index]
        position_limit = (
            settings.maximum_face_residual_position_mismatch
            if _is_face_alignment_basis(geometry_basis)
            else settings.maximum_subject_residual_position_mismatch
        )
        scale_limit = (
            settings.maximum_face_residual_scale_mismatch
            if _is_face_alignment_basis(geometry_basis)
            else settings.maximum_subject_residual_scale_mismatch
        )
        if (
            position <= position_limit + 1e-9
            and scale <= scale_limit + 1e-9
        ):
            continue
        final_residual_failure_indices.add(boundary_index)
        required_rejection_reasons.setdefault(boundary_index, set()).add(
            "final_rendered_attention_residual_exceeds_professional_limit"
        )
        required_rejection_context[boundary_index] = (
            f"basis={geometry_basis},position={position:.6f},"
            f"position_limit={position_limit:.6f},scale={scale:.6f},"
            f"scale_limit={scale_limit:.6f}"
        )

    accepted_segments: set[int] = set()
    bounded_upscale_segments: set[int] = set()
    for item in accepted:
        if item.side in {"left", "both"}:
            accepted_segments.add(item.boundary_index)
            if item.upscale_ratio > 1.0 + 1e-9:
                bounded_upscale_segments.add(item.boundary_index)
        if item.side in {"right", "both"}:
            accepted_segments.add(item.boundary_index + 1)
            if item.paired_upscale_ratio > 1.0 + 1e-9 or (
                item.side == "right" and item.upscale_ratio > 1.0 + 1e-9
            ):
                bounded_upscale_segments.add(item.boundary_index + 1)
    accepted_boundary_indices = tuple(
        sorted(item.boundary_index for item in accepted)
    )
    missing_required = tuple(
        sorted(
            required_boundary_indices.difference(
                accepted_boundary_indices
            ).difference(naturally_satisfied_required_boundary_indices)
            .union(final_residual_failure_indices)
        )
    )
    if missing_required and not config.allow_partial_required_boundary_matches:
        rejection_counts: dict[str, int] = {}
        for boundary_index in missing_required:
            reasons = required_rejection_reasons.get(
                boundary_index,
                {"required_candidate_missing_without_reason"},
            )
            for reason in reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        rejection_summary = ",".join(
            f"{reason}={count}"
            for reason, count in sorted(rejection_counts.items())
        )
        rejection_details = "|".join(
            f"{boundary_index}:"
            + ",".join(
                sorted(
                    required_rejection_reasons.get(
                        boundary_index,
                        {"required_candidate_missing_without_reason"},
                    )
                )
            )
            + (
                f"[{required_rejection_context[boundary_index]}]"
                if boundary_index in required_rejection_context
                else ""
            )
            for boundary_index in missing_required
        )
        message = (
            "required direct member transform failed dense review at boundaries "
            + ",".join(str(index) for index in missing_required)
            + "; required rejection counts "
            + rejection_summary
            + "; required rejection details "
            + rejection_details
        )
        raise DirectMemberEditorialMotionError(
            message,
            missing_boundary_indices=missing_required,
            rejection_reasons=required_rejection_reasons,
            rejection_context=required_rejection_context,
        )
    if missing_required:
        skipped["required_dense_partial_unresolved"] = len(missing_required)

    no_roll_plan = EditorialMotionPlan(
        segments=tuple(
            EditorialMotionSegment(
                segment_index=index,
                source_index=source_path[index],
                track=tracks[index],
                reasons=(
                    (
                        "official_fancam_dense_subject_bounded_"
                        "perceptual_upscale_match",
                    )
                    if index in bounded_upscale_segments
                    else (
                        "official_fancam_dense_face_subject_"
                        "scale_position_match",
                    )
                    if index in accepted_segments
                    else (
                        "official_fancam_dense_subject_"
                        "lossless_interior_zoom_pulse",
                    )
                    if index in interior_pulse_segments
                    else ("no_safe_dense_face_editorial_motion",)
                ),
            )
            for index in range(len(renders))
        ),
        boundary_match_count=len(accepted),
        zoom_event_count=sum(
            2 if item.side == "both" else 1 for item in accepted
        )
        + len(interior_pulse_segments),
        pan_event_count=sum(
            int(
                abs(item.state.center_x - 0.5) > 1e-6
                or abs(item.state.center_y - 0.5) > 1e-6
            )
            + int(
                item.paired_state is not None
                and (
                    abs(item.paired_state.center_x - 0.5) > 1e-6
                    or abs(item.paired_state.center_y - 0.5) > 1e-6
                )
            )
            for item in accepted
        ),
        roll_event_count=0,
        interior_zoom_pulse_count=len(interior_pulse_segments),
        blocked_dissolve_segment_count=len(blocked),
        skipped_boundary_reasons=tuple(sorted(skipped.items())),
        analysis_basis=(
            "official_target_fancam_face_or_verified_full_subject_dense_"
            "output_frame_boundary_anchored_subject_relative_face_track_"
            "adaptive_review_v10"
        ),
        accepted_boundary_indices=accepted_boundary_indices,
        natural_match_boundary_indices=tuple(
            sorted(naturally_satisfied_required_boundary_indices)
        ),
        boundary_residual_attention_mismatches=tuple(
            (
                boundary_index,
                *final_boundary_residuals[boundary_index],
            )
            for boundary_index in sorted(final_boundary_residuals)
        ),
        required_boundary_failures=tuple(
            (index, tuple(sorted(required_rejection_reasons.get(index,
                {"required_candidate_missing_without_reason"}))), required_rejection_context.get(index))
            for index in missing_required
        ),
        member_containment_trial_audit=(
            geometry_provider.interior_containment_audit()
            if callable(getattr(geometry_provider, 'interior_containment_audit', None)) else ()
        ),
    )
    from .direct_member_roll import apply_direct_member_roll
    return apply_direct_member_roll(no_roll_plan,visual_selection=visual_selection,
        render_segments=renders,transitions=transition_items,source_frame_geometries=source_frame_geometries,
        pose_geometry_provider=pose_geometry_provider,common_master_start=common_master_start,
        fps=fps,config=settings)


__all__ = [
    "DirectMemberDenseFaceProvider",
    "DirectMemberEditorialMotionConfig",
    "DirectMemberFaceSample",
    "build_direct_member_editorial_motion_plan",
]
