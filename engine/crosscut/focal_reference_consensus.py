"""Frozen same-source solo-face consensus; no names or persistent embeddings.

The official member dense-camera provider uses this bank. Pair-local
correspondence cannot become target membership when performers overlap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from .direct_member_target_face import FocalFaceCandidate
from .face_identity import cosine_similarity


def _finite_number(value: object) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _similarity(left: FocalFaceCandidate, right: FocalFaceCandidate) -> float:
    value = cosine_similarity(left.embedding, right.embedding)
    return -1.0 if value is None or not math.isfinite(value) else value


@dataclass(frozen=True, slots=True)
class SoloFaceObservation:
    source_seconds: float
    candidate: FocalFaceCandidate = field(repr=False)

    def __post_init__(self) -> None:
        if (not _finite_number(self.source_seconds)
                or self.source_seconds < 0):
            raise ValueError('solo observation time must be finite and nonnegative')
        if not isinstance(self.candidate, FocalFaceCandidate):
            raise TypeError('solo observation requires actual face evidence')


@dataclass(frozen=True, slots=True)
class FocalReferenceConsensus:
    references: tuple[SoloFaceObservation, ...] = field(repr=False)
    observation_count: int
    alternate_support_count: int
    minimum_query_similarity: float = .50
    minimum_secondary_similarity: float = .42
    minimum_query_margin: float = .06

    def __post_init__(self) -> None:
        if (not isinstance(self.references, tuple)
                or any(not isinstance(r, SoloFaceObservation) for r in self.references)):
            raise TypeError('consensus references must be frozen solo observations')
        if len(self.references) < 3 or len(self.references) > 32:
            raise ValueError('consensus needs three through 32 frozen observations')
        if len({r.source_seconds for r in self.references}) != len(self.references):
            raise ValueError('repeated frame evidence cannot strengthen a consensus')
        if (any(isinstance(v, bool) or not isinstance(v, int)
                for v in (self.observation_count, self.alternate_support_count))
                or not len(self.references) <= self.observation_count <= 64
                or not 0 <= self.alternate_support_count <= self.observation_count):
            raise ValueError('consensus support counts are invalid')
        if not all(_finite_number(v) and 0 <= v <= 1 for v in (
            self.minimum_query_similarity, self.minimum_secondary_similarity,
            self.minimum_query_margin,
        )):
            raise ValueError('consensus query thresholds are invalid')

    def select(self, candidates: Sequence[FocalFaceCandidate]) -> FocalFaceCandidate | None:
        ranked = []
        for candidate in candidates:
            if not isinstance(candidate, FocalFaceCandidate):
                raise TypeError('query requires actual face candidates')
            scores = sorted((_similarity(candidate, ref.candidate)
                             for ref in self.references), reverse=True)
            ranked.append(((scores[0]+scores[1])/2, candidate, scores))
        ranked.sort(key=lambda row: row[0], reverse=True)
        if not ranked or (len(ranked) > 1 and ranked[0][0]-ranked[1][0] < self.minimum_query_margin):
            return None
        if (ranked[0][2][0] < self.minimum_query_similarity
                or ranked[0][2][1] < self.minimum_secondary_similarity):
            return None
        return ranked[0][1]

    def recover_from_adjacent_observations(
        self, candidates: Sequence[FocalFaceCandidate], *,
        query_seconds: float,
        adjacent_observations: Sequence[SoloFaceObservation],
        maximum_temporal_distance_seconds: float = .30,
    ) -> FocalFaceCandidate | None:
        """Bridge to an observed exact face, never extrapolate a missing face.

        Every adjacent observation must independently match the frozen bank;
        it need not be a solo frame. Two distinct nearby frames, agreement,
        appearance and head-relative geometry are required. Recovered queries
        never become new seeds, so there is no transitive identity drift.
        """
        if (not _finite_number(query_seconds) or query_seconds < 0
                or not _finite_number(maximum_temporal_distance_seconds)
                or not 0 < maximum_temporal_distance_seconds <= .50):
            raise ValueError('focal temporal recovery bounds are invalid')
        if len(adjacent_observations) > 12:
            raise ValueError('focal temporal recovery must be bounded to 12 frames')
        direct = self.select(candidates)
        if direct is not None:
            return direct
        support: dict[FocalFaceCandidate, set[float]] = {}
        for observation in adjacent_observations:
            if not isinstance(observation, SoloFaceObservation):
                raise TypeError('temporal recovery needs observed face evidence')
            distance = abs(observation.source_seconds-query_seconds)
            if (distance < .02 or distance > maximum_temporal_distance_seconds+1e-9
                    or self.select((observation.candidate,)) is None):
                continue
            ranked = []
            reference = observation.candidate
            for candidate in candidates:
                seed_scores = sorted((_similarity(candidate, ref.candidate)
                                      for ref in self.references), reverse=True)
                if seed_scores[0] < .30 or seed_scores[1] < .25:
                    continue
                score = _similarity(candidate, reference)
                center_distance = math.dist(candidate.center_xy, reference.center_xy)
                maximum_height = max(candidate.bbox_xywh[3], reference.bbox_xywh[3])
                scale_distance = abs(math.log(candidate.bbox_xywh[3]/reference.bbox_xywh[3]))
                if score >= .50 and center_distance <= 1.5*maximum_height and scale_distance <= .5:
                    ranked.append((score, candidate))
            ranked.sort(key=lambda row: row[0], reverse=True)
            if not ranked or (len(ranked)>1 and ranked[0][0]-ranked[1][0] < .06):
                continue
            support.setdefault(ranked[0][1], set()).add(observation.source_seconds)
        # Any disagreement remains unresolved, even if one candidate wins a
        # small majority. Adjacent detections must describe the same track.
        if len(support) != 1:
            return None
        candidate, times = next(iter(support.items()))
        return candidate if len(times) >= 2 and max(times)-min(times) >= .02 else None

    def to_dict(self) -> dict:
        return {'contract':'frozen_same_source_solo_consensus_v1',
                'observation_count':self.observation_count,
                'support_count':len(self.references),
                'alternate_support_count':self.alternate_support_count,
                'source_seconds':[r.source_seconds for r in self.references],
                'minimum_query_similarity':self.minimum_query_similarity,
                'minimum_secondary_similarity':self.minimum_secondary_similarity,
                'minimum_query_margin':self.minimum_query_margin,
                'identity_name_inferred':False,'embeddings_persisted':False,
                'pair_match_may_update_frozen_seeds':False}


def build_focal_reference_consensus(
    observations: Sequence[SoloFaceObservation], *,
    minimum_seed_similarity: float = .50,
    minimum_temporal_separation_seconds: float = 1.,
) -> FocalReferenceConsensus | None:
    """Require a dominant star-consistent bank, never a transitive drift chain.

    Callers must supply frames with exactly one detected, central face from an
    authenticated focal-camera source. This function cannot establish that
    acquisition contract or assign the performer a name.
    """
    if (not _finite_number(minimum_seed_similarity) or not 0 <= minimum_seed_similarity <= 1
            or not _finite_number(minimum_temporal_separation_seconds)
            or minimum_temporal_separation_seconds <= 0):
        raise ValueError('consensus seed thresholds are invalid')
    if len(observations) > 64:
        raise ValueError('source reference bootstrap must be bounded to 64 frames')
    if any(not isinstance(item, SoloFaceObservation) for item in observations):
        raise TypeError('consensus seed must be a solo observation')
    unique = []
    for observation in sorted(observations, key=lambda item:item.source_seconds):
        if not unique or observation.source_seconds-unique[-1].source_seconds >= minimum_temporal_separation_seconds:
            unique.append(observation)
    if len(unique) < 3:
        return None
    if len(unique) > 64:
        raise ValueError('source reference bootstrap must be bounded to 64 frames')
    candidates = []
    supports = []
    for i, observation in enumerate(unique):
        support = frozenset(j for j, other in enumerate(unique)
            if _similarity(observation.candidate, other.candidate) >= minimum_seed_similarity)
        supports.append(support)
        if len(support) >= 3:
            candidates.append((len(support),i,support))
    if not candidates:
        return None
    candidates.sort(key=lambda row:(-row[0],row[1]))
    count, _anchor, selected = candidates[0]
    # Near-equal disjoint performer clusters are ambiguous, not an invitation
    # to let processing order or the most recent cross-cut pair choose one.
    alternate = max((len(support-selected) for support in supports),default=0)
    if alternate >= count-1:
        return None
    references = tuple(unique[i] for i in sorted(selected))[:32]
    return FocalReferenceConsensus(references,len(unique),alternate)
