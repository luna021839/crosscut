from __future__ import annotations

from dataclasses import replace
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import crosscut.stage_edit_source_pair_contextual_v2_production as contextual_production

from crosscut.stage_edit_source_pair_contextual_diagnostic import (
    StageEditSourcePairContextualCandidate,
    StageEditSourcePairContextualGroup,
    StageEditSourcePairContextualResidualModel,
    _contextual_feature_items,
)
from crosscut.stage_edit_source_pair_contextual_v2 import (
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_AUTHORITY,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RAW_MAXIMUM,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RAW_MINIMUM,
    StageEditContextualSourcePairScorerV2,
    StageEditContextualSourcePairV2Error,
    StageEditContextualTrainingConfigV2,
    materialize_stage_edit_contextual_pair_features_v2,
    materialize_stage_edit_contextual_runtime_frame_v2,
    materialize_stage_edit_contextual_training_group_v2,
    new_stage_edit_contextual_training_corpus_v2,
    stage_edit_contextual_raw_to_decoder_score_v2,
)
from crosscut.stage_edit_source_pair_contextual_v2_production import (
    StageEditContextualBeamQACaseV2,
    StageEditContextualBeamQAEvidenceV2,
    StageEditContextualSourcePairV2ArtifactError,
    StageEditContextualSourcePairProductionReportV2,
    cross_validate_stage_edit_contextual_source_pair_v2,
    inspect_stage_edit_contextual_training_corpus_v2,
    inspect_stage_edit_contextual_source_pair_checkpoint_v2,
    inspect_stage_edit_contextual_source_pair_report_v2,
    load_stage_edit_contextual_source_pair_evaluation_bundle_v2,
    load_stage_edit_contextual_training_corpus_v2,
    load_stage_edit_contextual_source_pair_production_bundle_v2,
    new_stage_edit_contextual_beam_qa_evidence_v2,
    new_stage_edit_contextual_source_pair_checkpoint_v2,
    new_stage_edit_contextual_source_pair_production_report_v2,
    save_stage_edit_contextual_source_pair_checkpoint_v2,
    save_stage_edit_contextual_source_pair_report_v2,
    save_stage_edit_contextual_training_corpus_v2,
    stage_edit_contextual_group_balance_weights_v2,
    train_stage_edit_contextual_source_pair_v2,
)
from crosscut.stage_edit_source_pair_scorer import (
    StageEditSourcePairCandidate,
    StageEditSourcePairTrainingGroup,
)
from crosscut.stage_edit_source_pair_training import (
    stage_edit_source_pair_observation_from_candidate,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditSequence,
    StageEditToken,
    mask_stage_edit_recent_history_for_semimarkov,
)


_SCHEMA = DEFAULT_STAGE_EDIT_SCHEMA
_FAMILIES = {"a": 0, "b": 1, "c": 2, "d": 3}


def _source(source_id: str, base: float) -> SourceCandidateToken:
    values: list[float] = []
    for index, name in enumerate(_SCHEMA.source_feature_names):
        if name.startswith("pose_model_embedding_"):
            value = min(0.95, base + (index % 7) * 0.003)
        elif name.startswith("formation_layout_embedding_"):
            value = min(0.95, base * 0.8 + (index % 11) * 0.002)
        elif name.startswith("formation_pose_person_"):
            value = min(0.95, base * 0.7 + (index % 5) * 0.004)
        elif name in {"face_center_x", "face_center_y", "body_center_x", "body_center_y"}:
            value = min(0.95, 0.35 + base * 0.4 + (index % 2) * 0.01)
        elif name == "camera_roll_degrees_normalized":
            value = min(0.95, base * 0.25)
        else:
            value = min(0.95, base + (index % 3) * 0.01)
        values.append(value)
    missing = [False] * len(values)
    missing[_SCHEMA.source_feature_names.index("formation_layout_embedding_010")] = True
    values[_SCHEMA.source_feature_names.index("formation_layout_embedding_010")] = 0.73
    return SourceCandidateToken(
        source_id=source_id,
        features=tuple(values),
        missing_mask=tuple(missing),
        available=True,
        hard_gate_passed=True,
    )


def _token(
    *,
    recent_value: float = 0.25,
    context_offset: float = 0.0,
    candidate_b_base: float = 0.21,
) -> StageEditToken:
    music = tuple(
        min(0.95, 0.05 * (index + 1) + context_offset)
        for index in range(len(_SCHEMA.music_feature_names))
    )
    section = tuple(
        min(0.95, 0.07 * (index + 1) + context_offset)
        for index in range(len(_SCHEMA.section_feature_names))
    )
    style = tuple(
        min(0.95, 0.06 * (index + 1) + context_offset)
        for index in range(len(_SCHEMA.style_feature_names))
    )
    return StageEditToken(
        master_time_seconds=12.5,
        music_features=music,
        section_features=section,
        source_candidates=(
            _source("a", 0.20),
            _source("b", candidate_b_base),
            _source("c", 0.82),
            _source("d", 0.64),
        ),
        recent_edit_features=(recent_value,) * len(_SCHEMA.recent_edit_feature_names),
        style_features=style,
        missing_mask=(False,) * _SCHEMA.raw_context_dimension,
    )


def _masked_token(**kwargs: float) -> StageEditToken:
    sequence = StageEditSequence("synthetic", (_token(**kwargs),))
    return mask_stage_edit_recent_history_for_semimarkov(sequence, _SCHEMA).tokens[0]


def _frame(**kwargs: float):
    return materialize_stage_edit_contextual_runtime_frame_v2(
        _masked_token(**kwargs),
        canonical_source_ids=("a", "b", "c", "d"),
        derivative_family_id_by_source=_FAMILIES,
        schema=_SCHEMA,
    )


def _pair_group(token: StageEditToken) -> StageEditSourcePairTrainingGroup:
    candidates = {value.source_id: value for value in token.source_candidates}
    outgoing = stage_edit_source_pair_observation_from_candidate(
        candidates["a"], schema=_SCHEMA
    )
    return StageEditSourcePairTrainingGroup(
        group_id="reference:source-pair:000000",
        reference_id="reference",
        outgoing_source_id="a",
        outgoing=outgoing,
        candidates=tuple(
            StageEditSourcePairCandidate(
                source_id,
                stage_edit_source_pair_observation_from_candidate(
                    candidates[source_id], schema=_SCHEMA
                ),
            )
            for source_id in ("b", "c", "d")
        ),
        chosen_incoming_source_id="b",
    )


def _diagnostic_group(token: StageEditToken) -> StageEditSourcePairContextualGroup:
    pair_group = _pair_group(token)
    token_candidates = {value.source_id: value for value in token.source_candidates}
    contextual_candidates: list[StageEditSourcePairContextualCandidate] = []
    feature_names: tuple[str, ...] | None = None
    for pair_candidate in pair_group.candidates:
        items = _contextual_feature_items(
            pair_group,
            pair_candidate,
            token,
            token_candidates["a"],
            token_candidates[pair_candidate.source_id],
            schema=_SCHEMA,
        )
        names = tuple(name for name, _value in items)
        if feature_names is None:
            feature_names = names
        else:
            assert names == feature_names
        contextual_candidates.append(
            StageEditSourcePairContextualCandidate(
                source_id=pair_candidate.source_id,
                features=tuple(value for _name, value in items),
            )
        )
    assert feature_names is not None
    return StageEditSourcePairContextualGroup(
        pair_group=pair_group,
        song_id="song",
        token_index=0,
        feature_names=feature_names,
        candidates=tuple(contextual_candidates),
    )


def _training_group(
    *,
    song_id: str,
    reference_id: str,
    group_index: int,
    context_offset: float = 0.0,
):
    frame = _frame(context_offset=context_offset)
    return materialize_stage_edit_contextual_training_group_v2(
        group_id=f"{reference_id}:source-pair:{group_index:06d}",
        song_id=song_id,
        reference_id=reference_id,
        token_index=group_index,
        frame=frame,
        outgoing_source_id="a",
        incoming_source_ids=("b", "c", "d"),
        chosen_incoming_source_id="b",
        feature_snapshot_sha256=("a" if song_id == "song-a" else "b") * 64,
    )


def _corpus():
    groups = (
        _training_group(
            song_id="song-a",
            reference_id="reference-a",
            group_index=0,
            context_offset=0.0,
        ),
        _training_group(
            song_id="song-b",
            reference_id="reference-b",
            group_index=0,
            context_offset=0.03,
        ),
    )
    return new_stage_edit_contextual_training_corpus_v2(
        groups,
        token_schema_fingerprint=_SCHEMA.fingerprint,
        source_data_id="synthetic-v2",
    )


def _qa_case(
    song_id: str,
    *,
    request_digit: str,
    report_digit: str,
    **overrides: object,
) -> StageEditContextualBeamQACaseV2:
    values: dict[str, object] = {
        "case_id": f"{song_id.lower()}-full-decode",
        "song_id": song_id,
        "request_sha256": request_digit * 64,
        "decode_report_sha256": report_digit * 64,
        "full_decode": True,
        "duration_seconds": 180.0,
        "switch_count": 45,
        "eligible_switch_opportunity_count": 100,
        "visual_segment_median_seconds": 1.8,
        "source_run_median_seconds": 4.0,
        "longest_source_run_seconds": 6.5,
        "distinct_source_count": 9,
        "effective_source_count": 8.4,
        "dominant_source_ratio": 0.25,
        "quality_score_p10": 0.42,
        "quality_score_median_span": 0.18,
        "boundary_match_score_p10": 62.0,
        "weighted_proxy_score": 65.0,
        "highlight_window_count": 3,
        "highlight_switch_opportunity_count": 6,
        "highlight_switch_count": 3,
        "verified_highlight_switch_count": 3,
        "closeup_motion_opportunity_count": 4,
        "closeup_motion_switch_count": 2,
        "verified_closeup_motion_switch_count": 2,
        "geometry_issue_count": 0,
        "sync_issue_count": 0,
    }
    values.update(overrides)
    return StageEditContextualBeamQACaseV2(**values)


def _qa_cases(**second_overrides: object) -> tuple[StageEditContextualBeamQACaseV2, ...]:
    return (
        _qa_case("BANG", request_digit="1", report_digit="a"),
        _qa_case(
            "BLACKHOLE",
            request_digit="2",
            report_digit="b",
            **second_overrides,
        ),
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _reseal_validation_payload(payload: dict[str, object]) -> dict[str, object]:
    payload["validation_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "validation_sha256"
        }
    )
    return payload


class StageEditContextualSourcePairV2Test(unittest.TestCase):
    def test_shared_id3_vector_is_exactly_equal_to_diagnostic(self) -> None:
        token = _masked_token()
        frame = materialize_stage_edit_contextual_runtime_frame_v2(
            token,
            canonical_source_ids=("a", "b", "c", "d"),
            derivative_family_id_by_source=_FAMILIES,
            schema=_SCHEMA,
        )
        diagnostic = _diagnostic_group(token)
        self.assertEqual(729, len(STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES))
        self.assertEqual(
            diagnostic.feature_names,
            STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES,
        )
        self.assertEqual(64, len(STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_SCHEMA_SHA256))
        for diagnostic_candidate in diagnostic.candidates:
            runtime = materialize_stage_edit_contextual_pair_features_v2(
                frame,
                "a",
                diagnostic_candidate.source_id,
            )
            self.assertEqual(diagnostic_candidate.features, runtime.values)
        self.assertFalse(
            any(
                "recent" in name or "logit" in name or "reference_match" in name
                for name in STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES
            )
        )

    def test_out_of_range_inputs_preserve_diagnostic_pair_math_then_bound_vector(self) -> None:
        token = _masked_token()
        candidates = []
        zoom_index = _SCHEMA.source_feature_names.index("camera_zoom_velocity")
        for candidate in token.source_candidates:
            values = list(candidate.features)
            if candidate.source_id == "a":
                values[zoom_index] = 1.2
            elif candidate.source_id == "b":
                values[zoom_index] = 1.1
            candidates.append(replace(candidate, features=tuple(values)))
        token = replace(token, source_candidates=tuple(candidates))
        music = list(token.music_features)
        music[_SCHEMA.music_feature_names.index("downbeat_probability")] = 1.4
        token = replace(token, music_features=tuple(music))
        frame = materialize_stage_edit_contextual_runtime_frame_v2(
            token,
            canonical_source_ids=("a", "b", "c", "d"),
            derivative_family_id_by_source=_FAMILIES,
            schema=_SCHEMA,
        )
        diagnostic = _diagnostic_group(token)
        candidate = next(
            value for value in diagnostic.candidates if value.source_id == "b"
        )
        runtime = materialize_stage_edit_contextual_pair_features_v2(
            frame, "a", "b"
        )
        bounded_diagnostic = tuple(
            min(1.0, max(-1.0, value)) for value in candidate.features
        )
        self.assertEqual(bounded_diagnostic, runtime.values)
        zoom_delta_index = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES.index(
            "pair.camera_zoom_velocity_delta"
        )
        self.assertEqual(
            candidate.features[zoom_delta_index], runtime.values[zoom_delta_index]
        )
        self.assertGreater(frame.clipped_observed_value_count, 0)

    def test_recent_history_mutation_is_invariant_after_required_mask(self) -> None:
        left = _frame(recent_value=-0.91)
        right = _frame(recent_value=0.87)
        self.assertEqual(left, right)
        self.assertEqual(
            materialize_stage_edit_contextual_pair_features_v2(left, "a", "b"),
            materialize_stage_edit_contextual_pair_features_v2(right, "a", "b"),
        )
        with self.assertRaisesRegex(
            StageEditContextualSourcePairV2Error, "recent history"
        ):
            materialize_stage_edit_contextual_runtime_frame_v2(
                _token(recent_value=0.4),
                canonical_source_ids=("a", "b", "c", "d"),
                derivative_family_id_by_source=_FAMILIES,
                schema=_SCHEMA,
            )

    def test_runtime_frame_and_score_are_source_order_invariant(self) -> None:
        token = _masked_token()
        permuted = replace(
            token,
            source_candidates=tuple(reversed(token.source_candidates)),
        )
        left = materialize_stage_edit_contextual_runtime_frame_v2(
            token,
            canonical_source_ids=("a", "b", "c", "d"),
            derivative_family_id_by_source=_FAMILIES,
            schema=_SCHEMA,
        )
        right = materialize_stage_edit_contextual_runtime_frame_v2(
            permuted,
            canonical_source_ids=("d", "c", "b", "a"),
            derivative_family_id_by_source=_FAMILIES,
            schema=_SCHEMA,
        )
        self.assertEqual(left, right)
        scorer = StageEditContextualSourcePairScorerV2(
            token_schema_fingerprint=_SCHEMA.fingerprint,
            weights=(0.0,) * len(STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES),
        )
        self.assertEqual(
            scorer.score_boundary(left, "a", "b"),
            scorer.score_boundary(right, "a", "b"),
        )

    def test_current_context_and_candidate_embedding_change_features(self) -> None:
        base = _frame(context_offset=0.0, candidate_b_base=0.21)
        context_changed = _frame(context_offset=0.05, candidate_b_base=0.21)
        candidate_changed = _frame(context_offset=0.0, candidate_b_base=0.39)
        base_values = materialize_stage_edit_contextual_pair_features_v2(
            base, "a", "b"
        )
        self.assertNotEqual(
            base_values.values,
            materialize_stage_edit_contextual_pair_features_v2(
                context_changed, "a", "b"
            ).values,
        )
        self.assertNotEqual(
            base_values.values,
            materialize_stage_edit_contextual_pair_features_v2(
                candidate_changed, "a", "b"
            ).values,
        )

    def test_clipping_audit_reaches_training_and_checkpoint_lineage(self) -> None:
        token = _masked_token()
        candidates = list(token.source_candidates)
        b_index = next(
            index for index, candidate in enumerate(candidates) if candidate.source_id == "b"
        )
        b_values = list(candidates[b_index].features)
        b_values[0] = 1.4
        candidates[b_index] = replace(candidates[b_index], features=tuple(b_values))
        music = list(token.music_features)
        music[_SCHEMA.music_feature_names.index("downbeat_probability")] = -1.3
        token = replace(
            token,
            music_features=tuple(music),
            source_candidates=tuple(candidates),
        )
        frame = materialize_stage_edit_contextual_runtime_frame_v2(
            token,
            canonical_source_ids=("a", "b", "c", "d"),
            derivative_family_id_by_source=_FAMILIES,
            schema=_SCHEMA,
        )
        self.assertEqual(2, frame.clipped_observed_value_count)
        group = materialize_stage_edit_contextual_training_group_v2(
            group_id="clip-reference:source-pair:000000",
            song_id="clip-song",
            reference_id="clip-reference",
            token_index=0,
            frame=frame,
            outgoing_source_id="a",
            incoming_source_ids=("b", "c", "d"),
            chosen_incoming_source_id="b",
            feature_snapshot_sha256="e" * 64,
        )
        corpus = new_stage_edit_contextual_training_corpus_v2(
            (group,),
            token_schema_fingerprint=_SCHEMA.fingerprint,
            source_data_id="clip-corpus",
        )
        training = train_stage_edit_contextual_source_pair_v2(corpus)
        checkpoint = new_stage_edit_contextual_source_pair_checkpoint_v2(
            "clip-checkpoint",
            training,
            created_at="2026-08-25T12:00:00+09:00",
        )
        self.assertEqual(
            2,
            checkpoint.clipped_observed_value_count,
        )
        self.assertEqual(
            corpus.feature_snapshot_bindings_sha256,
            checkpoint.feature_snapshot_bindings_sha256,
        )
    def test_t6_affine_endpoints_and_strict_ranking_equivalence(self) -> None:
        self.assertEqual(
            -1.0,
            stage_edit_contextual_raw_to_decoder_score_v2(
                STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RAW_MINIMUM
            ),
        )
        self.assertEqual(
            0.0,
            stage_edit_contextual_raw_to_decoder_score_v2(
                STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_RAW_MAXIMUM
            ),
        )
        raw = (-2.5, -1.7, -0.4, 0.0, 1.5)
        bounded = tuple(stage_edit_contextual_raw_to_decoder_score_v2(value) for value in raw)
        self.assertEqual(tuple(range(len(raw))), tuple(np.argsort(raw)))
        self.assertEqual(tuple(range(len(raw))), tuple(np.argsort(bounded)))
        with self.assertRaisesRegex(StageEditContextualSourcePairV2Error, "bounds"):
            stage_edit_contextual_raw_to_decoder_score_v2(1.5 + 0.5e-12)
        with self.assertRaisesRegex(StageEditContextualSourcePairV2Error, "bounds"):
            stage_edit_contextual_raw_to_decoder_score_v2(1.5 + 2.0e-12)
        oversized = [0.0] * len(STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES)
        oversized[0] = 1.5 + 5.0e-10
        with self.assertRaisesRegex(StageEditContextualSourcePairV2Error, "L1"):
            StageEditContextualSourcePairScorerV2(
                token_schema_fingerprint=_SCHEMA.fingerprint,
                weights=tuple(oversized),
            )

    def test_raw_score_and_rank_match_existing_diagnostic_exactly(self) -> None:
        token = _masked_token()
        frame = materialize_stage_edit_contextual_runtime_frame_v2(
            token,
            canonical_source_ids=("a", "b", "c", "d"),
            derivative_family_id_by_source=_FAMILIES,
            schema=_SCHEMA,
        )
        diagnostic_group = _diagnostic_group(token)
        weights = [0.0] * len(STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES)
        weights[0] = 0.45
        weights[1] = -0.35
        weights[100] = 0.20
        diagnostic_model = StageEditSourcePairContextualResidualModel(
            feature_names=STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES,
            weights=tuple(weights),
            handcrafted_baseline_fraction=0.30,
            l1_radius=1.5,
        )
        scorer = StageEditContextualSourcePairScorerV2(
            token_schema_fingerprint=_SCHEMA.fingerprint,
            weights=tuple(weights),
        )
        diagnostic_scores: dict[str, float] = {}
        runtime_scores: dict[str, float] = {}
        bounded_scores: dict[str, float] = {}
        for candidate in diagnostic_group.candidates:
            diagnostic_scores[candidate.source_id] = diagnostic_model.score(
                diagnostic_group, candidate
            )
            scored = scorer.score_boundary(frame, "a", candidate.source_id)
            runtime_scores[candidate.source_id] = scored.raw_logit
            bounded_scores[candidate.source_id] = scored.decoder_score
        for source_id in diagnostic_scores:
            self.assertAlmostEqual(
                diagnostic_scores[source_id], runtime_scores[source_id], places=15
            )
        expected_rank = tuple(
            sorted(diagnostic_scores, key=lambda value: (-diagnostic_scores[value], value))
        )
        self.assertEqual(
            expected_rank,
            tuple(sorted(runtime_scores, key=lambda value: (-runtime_scores[value], value))),
        )
        self.assertEqual(
            expected_rank,
            tuple(sorted(bounded_scores, key=lambda value: (-bounded_scores[value], value))),
        )

    def test_same_derivative_family_is_exact_neutral_bypass(self) -> None:
        families = dict(_FAMILIES)
        families["b"] = families["a"]
        frame = materialize_stage_edit_contextual_runtime_frame_v2(
            _masked_token(),
            canonical_source_ids=("a", "b", "c", "d"),
            derivative_family_id_by_source=families,
            schema=_SCHEMA,
        )
        scorer = StageEditContextualSourcePairScorerV2(
            token_schema_fingerprint=_SCHEMA.fingerprint,
            weights=(0.0,) * len(STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES),
        )
        result = scorer.score_boundary(frame, "a", "b")
        self.assertFalse(result.applied)
        self.assertEqual("same_derivative_family", result.neutral_reason)
        self.assertEqual((0.0, 0.0), (result.raw_logit, result.decoder_score))
        audit_identity = frame.audit_identity_dict()
        serialized_identity = json.dumps(audit_identity, sort_keys=True)
        self.assertNotIn("context_values", serialized_identity)
        self.assertNotIn("embedding", serialized_identity)
        self.assertNotIn('"values"', serialized_identity)
        self.assertEqual(frame.runtime_evidence_sha256, audit_identity["runtime_evidence_sha256"])
        score_lineage = result.to_dict()
        self.assertEqual(result.scorer_fingerprint, score_lineage["scorer_fingerprint"])
        self.assertFalse(any("embedding" in key for key in score_lineage))

    def test_exact_frozen_config_and_hierarchical_group_weights(self) -> None:
        config = StageEditContextualTrainingConfigV2()
        self.assertEqual(
            ("A", "id3", 160, 0.5, 24, 1.0, 0.0, 0.30, 1.5, 0.05),
            (
                config.architecture,
                config.feature_design,
                config.steps,
                config.learning_rate,
                config.backtrack_attempts,
                config.temperature,
                config.margin,
                config.handcrafted_baseline_fraction,
                config.signed_l1_radius,
                config.l2_penalty,
            ),
        )

    def test_high_dimensional_l1_projection_stays_inside_hard_radius(self) -> None:
        values = np.linspace(
            -4.25,
            3.75,
            len(STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES),
            dtype=np.float64,
        )
        projected = contextual_production._project_signed_l1(values, 1.5)
        self.assertLessEqual(
            math.fsum(abs(float(value)) for value in projected),
            1.5,
        )
        scorer = StageEditContextualSourcePairScorerV2(
            token_schema_fingerprint=_SCHEMA.fingerprint,
            weights=tuple(float(value) for value in projected),
        )
        self.assertEqual(tuple(projected), scorer.weights)
        with self.assertRaisesRegex(StageEditContextualSourcePairV2Error, "frozen"):
            StageEditContextualTrainingConfigV2(steps=159)
        first = _training_group(
            song_id="song-a", reference_id="reference-a", group_index=0
        )
        second = replace(first, group_id="reference-a:source-pair:000001", token_index=1)
        third = _training_group(
            song_id="song-a", reference_id="reference-b", group_index=0
        )
        fourth = _training_group(
            song_id="song-b", reference_id="reference-c", group_index=0
        )
        self.assertEqual(
            (
                (first.group_id, 0.125),
                (second.group_id, 0.125),
                (third.group_id, 0.25),
                (fourth.group_id, 0.5),
            ),
            stage_edit_contextual_group_balance_weights_v2(
                (first, second, third, fourth)
            ),
        )

    def test_loso_fold_separation_and_full_fit_runtime_equivalence(self) -> None:
        corpus = _corpus()
        validation = cross_validate_stage_edit_contextual_source_pair_v2(corpus)
        self.assertEqual(corpus.song_ids, tuple(value.heldout_song_id for value in validation.folds))
        self.assertTrue(validation.authorized)
        heldout_groups: set[str] = set()
        for fold in validation.folds:
            self.assertNotIn(fold.heldout_song_id, fold.training_song_ids)
            self.assertFalse(
                set(fold.training_reference_ids).intersection(fold.heldout_reference_ids)
            )
            self.assertFalse(
                set(fold.training_group_ids).intersection(fold.heldout_group_ids)
            )
            heldout_groups.update(fold.heldout_group_ids)
        self.assertEqual(set(corpus.group_ids), heldout_groups)
        training = train_stage_edit_contextual_source_pair_v2(corpus)
        self.assertLessEqual(training.final_loss, training.initial_loss + 1.0e-12)
        self.assertLessEqual(
            sum(abs(value) for value in training.scorer.weights), 1.5
        )
        self.assertEqual(64, len(training.runtime_equivalence_sha256))

        fold = validation.folds[0]
        with self.assertRaisesRegex(
            StageEditContextualSourcePairV2Error, "embedded training songs"
        ):
            replace(
                fold,
                training=replace(
                    fold.training,
                    training_song_ids=corpus.song_ids,
                    training_reference_ids=corpus.reference_ids,
                    training_group_ids=tuple(sorted(corpus.group_ids)),
                ),
            )
        with self.assertRaisesRegex(
            StageEditContextualSourcePairV2Error, "canonical subset"
        ):
            replace(
                fold,
                training=replace(
                    fold.training,
                    corpus_sha256="f" * 64,
                    feature_snapshot_bindings_sha256="e" * 64,
                ),
            )
        with self.assertRaisesRegex(
            StageEditContextualSourcePairV2Error, "exact universe partition"
        ):
            replace(
                validation,
                reference_ids=(*validation.reference_ids, "rogue-reference"),
            )
        with self.assertRaisesRegex(
            StageEditContextualSourcePairV2Error, "exact universe partition"
        ):
            replace(
                validation,
                group_ids=(*validation.group_ids, "rogue-group"),
            )

    def test_snapshot_digest_reuse_is_song_scoped(self) -> None:
        first = _training_group(
            song_id="song-a", reference_id="reference-a", group_index=0
        )
        same_song = _training_group(
            song_id="song-a", reference_id="reference-b", group_index=0
        )
        allowed = new_stage_edit_contextual_training_corpus_v2(
            (first, same_song),
            token_schema_fingerprint=_SCHEMA.fingerprint,
            source_data_id="same-song-snapshot",
        )
        self.assertEqual(("song-a",), allowed.song_ids)
        other_song = _training_group(
            song_id="song-b", reference_id="reference-c", group_index=0
        )
        reused = replace(
            other_song, feature_snapshot_sha256=first.feature_snapshot_sha256
        )
        with self.assertRaisesRegex(
            StageEditContextualSourcePairV2Error, "reused across songs"
        ):
            new_stage_edit_contextual_training_corpus_v2(
                (first, reused),
                token_schema_fingerprint=_SCHEMA.fingerprint,
                source_data_id="cross-song-snapshot",
            )
        with self.assertRaisesRegex(
            StageEditContextualSourcePairV2Error, "token schema"
        ):
            new_stage_edit_contextual_training_corpus_v2(
                (first,),
                token_schema_fingerprint="f" * 64,
                source_data_id="mismatched-runtime-schema",
            )

    def test_corpus_deep_roundtrip_rejects_nested_schema_and_audit_tamper(self) -> None:
        corpus = _corpus()
        self.assertEqual(corpus, type(corpus).from_dict(corpus.to_dict()))

        nested_extra = corpus.to_dict()
        nested_extra["groups"][0]["runtime_frame"]["sources"][0][
            "invented"
        ] = True
        with self.assertRaisesRegex(
            StageEditContextualSourcePairV2Error, "keys are not exact"
        ):
            type(corpus).from_dict(nested_extra)

        clipped = corpus.to_dict()
        runtime = clipped["groups"][0]["runtime_frame"]
        runtime["clipped_observed_value_count"] += 1
        runtime_unsigned = {
            key: value
            for key, value in runtime.items()
            if key != "runtime_evidence_sha256"
        }
        runtime["runtime_evidence_sha256"] = _canonical_sha256(runtime_unsigned)
        group = clipped["groups"][0]
        group["runtime_evidence_sha256"] = runtime["runtime_evidence_sha256"]
        group["clipped_observed_value_count"] += 1
        clipped["clipped_observed_value_count"] += 1
        clipped["content_sha256"] = _canonical_sha256(
            {
                key: value
                for key, value in clipped.items()
                if key != "content_sha256"
            }
        )
        with self.assertRaisesRegex(
            StageEditContextualSourcePairV2Error, "clipping audit"
        ):
            type(corpus).from_dict(clipped)

    def test_report_replay_rejects_song_ownership_swap_and_full_fit_fold_leak(self) -> None:
        corpus = _corpus()
        validation = cross_validate_stage_edit_contextual_source_pair_v2(corpus)
        final_training = train_stage_edit_contextual_source_pair_v2(corpus)
        checkpoint = new_stage_edit_contextual_source_pair_checkpoint_v2(
            "attack-replay-checkpoint",
            final_training,
            created_at="2026-08-25T12:00:00+09:00",
        )

        swapped_payload = validation.to_dict()
        first, second = swapped_payload["folds"]
        for field in (
            "training_reference_ids",
            "heldout_reference_ids",
            "training_group_ids",
            "heldout_group_ids",
        ):
            first[field], second[field] = second[field], first[field]
        for field in ("training_reference_ids", "training_group_ids"):
            first["training"][field], second["training"][field] = (
                second["training"][field],
                first["training"][field],
            )
        swapped = type(validation).from_dict(
            _reseal_validation_payload(swapped_payload)
        )
        self.assertTrue(swapped.authorized)
        self.assertEqual(
            ("reference-b",), swapped.folds[0].heldout_reference_ids
        )
        with self.assertRaisesRegex(
            StageEditContextualSourcePairV2Error, "deterministic full-corpus replay"
        ):
            new_stage_edit_contextual_source_pair_production_report_v2(
                report_id="swapped-fold-ownership",
                corpus=corpus,
                cross_validation=swapped,
                final_training=final_training,
                checkpoint=checkpoint,
                beam_qa_evaluation_corpus_sha256="c" * 64,
                verified_beam_qa_cases=(),
                created_at="2026-08-25T12:01:00+09:00",
            )

        leaked_payload = validation.to_dict()
        full_scorer = final_training.scorer.to_dict()
        for fold in leaked_payload["folds"]:
            fold["training"]["scorer"] = full_scorer
            fold["training"]["scorer_fingerprint"] = (
                final_training.scorer.fingerprint
            )
        leaked = type(validation).from_dict(
            _reseal_validation_payload(leaked_payload)
        )
        self.assertEqual(
            {final_training.scorer.fingerprint},
            {fold.training.scorer.fingerprint for fold in leaked.folds},
        )
        with self.assertRaisesRegex(
            StageEditContextualSourcePairV2Error, "deterministic full-corpus replay"
        ):
            new_stage_edit_contextual_source_pair_production_report_v2(
                report_id="full-fit-fold-leak",
                corpus=corpus,
                cross_validation=leaked,
                final_training=final_training,
                checkpoint=checkpoint,
                beam_qa_evaluation_corpus_sha256="c" * 64,
                verified_beam_qa_cases=(),
                created_at="2026-08-25T12:01:00+09:00",
            )

    def test_report_free_evaluation_bundle_replays_exact_checkpoint(self) -> None:
        corpus = _corpus()
        training = train_stage_edit_contextual_source_pair_v2(corpus)
        checkpoint = new_stage_edit_contextual_source_pair_checkpoint_v2(
            "report-free-evaluation",
            training,
            created_at="2026-08-25T12:00:00+09:00",
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            directory = Path(temporary).resolve()
            corpus_path = save_stage_edit_contextual_training_corpus_v2(
                directory, corpus
            )
            checkpoint_path = save_stage_edit_contextual_source_pair_checkpoint_v2(
                directory, checkpoint
            )
            corpus_file_sha256 = hashlib.sha256(
                corpus_path.read_bytes()
            ).hexdigest()
            checkpoint_file_sha256 = hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest()
            bundle = load_stage_edit_contextual_source_pair_evaluation_bundle_v2(
                directory,
                expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
                expected_checkpoint_file_sha256=checkpoint_file_sha256,
                expected_corpus_sha256=corpus.content_sha256,
                expected_corpus_file_sha256=corpus_file_sha256,
                expected_token_schema_fingerprint=corpus.token_schema_fingerprint,
                expected_scorer_fingerprint=checkpoint.scorer_fingerprint,
                expected_runtime_equivalence_sha256=(
                    checkpoint.runtime_equivalence_sha256
                ),
            )
            self.assertEqual(checkpoint, bundle.checkpoint)
            self.assertEqual(corpus, bundle.corpus)
            self.assertEqual(training.scorer.to_dict(), bundle.scorer.to_dict())
            self.assertFalse(
                (directory / "source-pair-contextual-production-report-v2.json").exists()
            )
            with self.assertRaisesRegex(
                StageEditContextualSourcePairV2ArtifactError,
                "runtime-equivalence pin changed",
            ):
                load_stage_edit_contextual_source_pair_evaluation_bundle_v2(
                    directory,
                    expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
                    expected_checkpoint_file_sha256=checkpoint_file_sha256,
                    expected_corpus_sha256=corpus.content_sha256,
                    expected_corpus_file_sha256=corpus_file_sha256,
                    expected_token_schema_fingerprint=(
                        corpus.token_schema_fingerprint
                    ),
                    expected_scorer_fingerprint=checkpoint.scorer_fingerprint,
                    expected_runtime_equivalence_sha256="f" * 64,
                )

    def test_checkpoint_and_report_reload_tamper_and_no_replace(self) -> None:
        corpus = _corpus()
        validation = cross_validate_stage_edit_contextual_source_pair_v2(corpus)
        self.assertTrue(validation.authorized)
        training = train_stage_edit_contextual_source_pair_v2(corpus)
        checkpoint = new_stage_edit_contextual_source_pair_checkpoint_v2(
            "synthetic-contextual-v2",
            training,
            created_at="2026-08-25T12:00:00+09:00",
        )
        self.assertEqual(STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_AUTHORITY, checkpoint.authority)
        qa = new_stage_edit_contextual_beam_qa_evidence_v2(
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            evaluation_corpus_sha256="c" * 64,
            cases=_qa_cases(),
        )
        verified_envelopes = (object(), object())
        with patch(
            "crosscut.stage_edit_contextual_beam_qa_adapter_v2."
            "require_verified_stage_edit_contextual_beam_qa_case_v2",
            side_effect=_qa_cases(),
        ):
            report = new_stage_edit_contextual_source_pair_production_report_v2(
                report_id="synthetic-contextual-v2-report",
                corpus=corpus,
                cross_validation=validation,
                final_training=training,
                checkpoint=checkpoint,
                beam_qa_evaluation_corpus_sha256="c" * 64,
                verified_beam_qa_cases=verified_envelopes,
                created_at="2026-08-25T12:01:00+09:00",
            )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            directory = Path(temporary)
            corpus_path = save_stage_edit_contextual_training_corpus_v2(
                directory, corpus
            )
            checkpoint_path = save_stage_edit_contextual_source_pair_checkpoint_v2(
                directory, checkpoint
            )
            report_path = save_stage_edit_contextual_source_pair_report_v2(
                directory, report
            )
            corpus_file_sha256 = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
            checkpoint_file_sha256 = hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest()
            report_file_sha256 = hashlib.sha256(
                report_path.read_bytes()
            ).hexdigest()
            loaded_corpus = inspect_stage_edit_contextual_training_corpus_v2(
                directory,
                expected_corpus_sha256=corpus.content_sha256,
                expected_corpus_file_sha256=corpus_file_sha256,
                expected_token_schema_fingerprint=_SCHEMA.fingerprint,
            )
            loaded_corpus_with_sha, observed_corpus_file_sha256 = (
                load_stage_edit_contextual_training_corpus_v2(
                    directory,
                    expected_corpus_sha256=corpus.content_sha256,
                    expected_corpus_file_sha256=corpus_file_sha256,
                    expected_token_schema_fingerprint=_SCHEMA.fingerprint,
                )
            )
            loaded_checkpoint = inspect_stage_edit_contextual_source_pair_checkpoint_v2(
                directory,
                expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
                expected_corpus_sha256=corpus.content_sha256,
                expected_token_schema_fingerprint=_SCHEMA.fingerprint,
            )
            loaded_report = inspect_stage_edit_contextual_source_pair_report_v2(
                directory,
                expected_report_sha256=report.report_sha256,
                expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
            )
            with patch(
                "crosscut.stage_edit_contextual_beam_qa_adapter_v2."
                "require_verified_stage_edit_contextual_beam_qa_case_v2",
                side_effect=_qa_cases(),
            ):
                bundle = load_stage_edit_contextual_source_pair_production_bundle_v2(
                    directory,
                    expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
                    expected_checkpoint_file_sha256=checkpoint_file_sha256,
                    expected_report_sha256=report.report_sha256,
                    expected_report_file_sha256=report_file_sha256,
                    expected_beam_qa_report_sha256=qa.qa_report_sha256,
                    expected_corpus_sha256=corpus.content_sha256,
                    expected_corpus_file_sha256=corpus_file_sha256,
                    expected_token_schema_fingerprint=_SCHEMA.fingerprint,
                    verified_beam_qa_cases=verified_envelopes,
                )
            self.assertEqual(corpus, loaded_corpus)
            self.assertEqual(corpus, loaded_corpus_with_sha)
            self.assertEqual(
                corpus_file_sha256, observed_corpus_file_sha256
            )
            self.assertEqual(checkpoint, loaded_checkpoint)
            self.assertEqual(report, loaded_report)
            self.assertEqual(checkpoint, bundle.checkpoint)
            self.assertEqual(report, bundle.report)
            self.assertEqual(corpus, bundle.corpus)
            self.assertEqual(checkpoint.scorer_fingerprint, bundle.scorer.fingerprint)
            self.assertEqual(
                checkpoint_file_sha256, bundle.checkpoint_file_sha256
            )
            self.assertEqual(corpus_file_sha256, bundle.corpus_file_sha256)
            self.assertEqual(report_file_sha256, bundle.report_file_sha256)
            with self.assertRaisesRegex(
                StageEditContextualSourcePairV2ArtifactError,
                "activation loader",
            ):
                replace(bundle, _activation_seal=object())
            with self.assertRaisesRegex(
                StageEditContextualSourcePairV2ArtifactError,
                "Beam-QA reverification failed",
            ):
                load_stage_edit_contextual_source_pair_production_bundle_v2(
                    directory,
                    expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
                    expected_checkpoint_file_sha256=checkpoint_file_sha256,
                    expected_report_sha256=report.report_sha256,
                    expected_report_file_sha256=report_file_sha256,
                    expected_beam_qa_report_sha256=qa.qa_report_sha256,
                    expected_corpus_sha256=corpus.content_sha256,
                    expected_corpus_file_sha256=corpus_file_sha256,
                    expected_token_schema_fingerprint=_SCHEMA.fingerprint,
                    verified_beam_qa_cases=_qa_cases(),
                )
            linked_parent = directory / "linked-parent"
            linked_parent.symlink_to(directory, target_is_directory=True)
            with self.assertRaisesRegex(
                StageEditContextualSourcePairV2ArtifactError, "symlink"
            ):
                inspect_stage_edit_contextual_source_pair_report_v2(linked_parent)
            with self.assertRaisesRegex(
                StageEditContextualSourcePairV2ArtifactError, "symlink"
            ):
                save_stage_edit_contextual_source_pair_checkpoint_v2(
                    linked_parent / "nested", checkpoint
                )
            loop_a = directory / "loop-a"
            loop_b = directory / "loop-b"
            loop_a.symlink_to("loop-b", target_is_directory=True)
            loop_b.symlink_to("loop-a", target_is_directory=True)
            with self.assertRaisesRegex(
                StageEditContextualSourcePairV2ArtifactError, "symlink"
            ):
                save_stage_edit_contextual_source_pair_report_v2(
                    loop_a / "nested", report
                )
            with self.assertRaisesRegex(
                StageEditContextualSourcePairV2ArtifactError, "absolute lexical"
            ):
                inspect_stage_edit_contextual_source_pair_checkpoint_v2(
                    Path("relative-artifact-directory")
                )
            checkpoint_bytes = checkpoint_path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_stage_edit_contextual_source_pair_checkpoint_v2(
                    directory, checkpoint
                )
            self.assertEqual(checkpoint_bytes, checkpoint_path.read_bytes())
            self.assertFalse(
                any(".tmp-" in value.name for value in directory.iterdir())
            )
            with self.assertRaisesRegex(
                StageEditContextualSourcePairV2ArtifactError, "file-byte"
            ):
                load_stage_edit_contextual_source_pair_production_bundle_v2(
                    directory,
                    expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
                    expected_checkpoint_file_sha256="0" * 64,
                    expected_report_sha256=report.report_sha256,
                    expected_report_file_sha256=report_file_sha256,
                    expected_beam_qa_report_sha256=qa.qa_report_sha256,
                    expected_corpus_sha256=corpus.content_sha256,
                    expected_corpus_file_sha256=corpus_file_sha256,
                    expected_token_schema_fingerprint=_SCHEMA.fingerprint,
                    verified_beam_qa_cases=verified_envelopes,
                )
            with self.assertRaisesRegex(
                StageEditContextualSourcePairV2ArtifactError, "corpus file-byte"
            ):
                load_stage_edit_contextual_source_pair_production_bundle_v2(
                    directory,
                    expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
                    expected_checkpoint_file_sha256=checkpoint_file_sha256,
                    expected_report_sha256=report.report_sha256,
                    expected_report_file_sha256=report_file_sha256,
                    expected_beam_qa_report_sha256=qa.qa_report_sha256,
                    expected_corpus_sha256=corpus.content_sha256,
                    expected_corpus_file_sha256="0" * 64,
                    expected_token_schema_fingerprint=_SCHEMA.fingerprint,
                    verified_beam_qa_cases=verified_envelopes,
                )
            report_payload = report.to_dict()
            report_payload["final_full_corpus_training"]["config"]["steps"] = 159
            with self.assertRaisesRegex(
                StageEditContextualSourcePairV2Error, "frozen config"
            ):
                StageEditContextualSourcePairProductionReportV2.from_dict(
                    report_payload
                )
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            payload["weights"][0] += 0.01
            checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(StageEditContextualSourcePairV2ArtifactError):
                inspect_stage_edit_contextual_source_pair_checkpoint_v2(directory)
            # The separately sealed report remains valid and carries its own
            # complete embedded checkpoint rather than trusting the mutated file.
            self.assertEqual(
                report,
                inspect_stage_edit_contextual_source_pair_report_v2(directory),
            )
            with self.assertRaises(StageEditContextualSourcePairV2ArtifactError):
                load_stage_edit_contextual_source_pair_production_bundle_v2(
                    directory,
                    expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
                    expected_checkpoint_file_sha256=checkpoint_file_sha256,
                    expected_report_sha256=report.report_sha256,
                    expected_report_file_sha256=report_file_sha256,
                    expected_beam_qa_report_sha256=qa.qa_report_sha256,
                    expected_corpus_sha256=corpus.content_sha256,
                    expected_corpus_file_sha256=corpus_file_sha256,
                    expected_token_schema_fingerprint=_SCHEMA.fingerprint,
                    verified_beam_qa_cases=verified_envelopes,
                )
            self.assertTrue(report_path.is_file())

    def test_exfat_fallback_is_transactional_and_journal_recovery_is_owned(self) -> None:
        corpus = _corpus()
        unsupported = OSError(errno.ENOTSUP, "hard links unsupported")
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            directory = Path(temporary)
            with patch.object(os, "link", side_effect=unsupported):
                path = save_stage_edit_contextual_training_corpus_v2(
                    directory, corpus
                )
            file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(
                corpus,
                inspect_stage_edit_contextual_training_corpus_v2(
                    directory,
                    expected_corpus_sha256=corpus.content_sha256,
                    expected_corpus_file_sha256=file_sha256,
                ),
            )
            self.assertEqual(
                [STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME],
                sorted(value.name for value in directory.iterdir()),
            )
            recovery_temporary_name = (
                f".{path.name}.tmp-complete-final-recovery"
            )
            recovery_temporary = directory / recovery_temporary_name
            recovery_temporary.write_bytes(path.read_bytes())
            recovery_journal = contextual_production._journal_payload(
                target_name=path.name,
                expected_file_sha256=file_sha256,
                temporary_name=recovery_temporary_name,
                temporary_status=recovery_temporary.stat(),
                phase="prepared",
                final_status=None,
            )
            recovery_journal_path = directory / contextual_production._publish_journal_name(
                path.name
            )
            recovery_journal_path.write_bytes(
                contextual_production._canonical_json_bytes(recovery_journal)
            )
            self.assertEqual(
                path,
                save_stage_edit_contextual_training_corpus_v2(directory, corpus),
            )
            self.assertFalse(recovery_temporary.exists())
            self.assertFalse(recovery_journal_path.exists())

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            directory = Path(temporary)
            with patch.object(os, "link", side_effect=unsupported), patch.object(
                contextual_production,
                "_copy_temporary_to_final",
                side_effect=OSError(errno.EIO, "injected copy failure"),
            ):
                with self.assertRaises(OSError):
                    save_stage_edit_contextual_training_corpus_v2(
                        directory, corpus
                    )
            self.assertEqual([], list(directory.iterdir()))

        expected_bytes = contextual_production._canonical_json_bytes(
            corpus.to_dict(), pretty=True
        )
        expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()
        target_name = STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME
        journal_name = contextual_production._publish_journal_name(target_name)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            directory = Path(temporary)
            temporary_name = f".{target_name}.tmp-owned-recovery"
            temporary_path = directory / temporary_name
            temporary_path.write_bytes(expected_bytes)
            partial_path = directory / target_name
            partial_path.write_bytes(expected_bytes[:37])
            journal = contextual_production._journal_payload(
                target_name=target_name,
                expected_file_sha256=expected_sha256,
                temporary_name=temporary_name,
                temporary_status=temporary_path.stat(),
                phase="copying",
                final_status=partial_path.stat(),
            )
            (directory / journal_name).write_bytes(
                contextual_production._canonical_json_bytes(journal)
            )
            with self.assertRaisesRegex(
                StageEditContextualSourcePairV2ArtifactError,
                "active publish journal",
            ):
                inspect_stage_edit_contextual_training_corpus_v2(directory)
            with patch.object(os, "link", side_effect=unsupported):
                recovered = save_stage_edit_contextual_training_corpus_v2(
                    directory, corpus
                )
            self.assertEqual(expected_bytes, recovered.read_bytes())
            self.assertFalse((directory / journal_name).exists())
            self.assertFalse(temporary_path.exists())

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            directory = Path(temporary)
            temporary_name = f".{target_name}.tmp-unrelated-final"
            temporary_path = directory / temporary_name
            temporary_path.write_bytes(expected_bytes)
            unrelated = b"unrelated-existing-final"
            final_path = directory / target_name
            final_path.write_bytes(unrelated)
            journal = contextual_production._journal_payload(
                target_name=target_name,
                expected_file_sha256=expected_sha256,
                temporary_name=temporary_name,
                temporary_status=temporary_path.stat(),
                phase="prepared",
                final_status=None,
            )
            (directory / journal_name).write_bytes(
                contextual_production._canonical_json_bytes(journal)
            )
            with self.assertRaisesRegex(
                StageEditContextualSourcePairV2ArtifactError,
                "not the journal-owned partial",
            ):
                save_stage_edit_contextual_training_corpus_v2(directory, corpus)
            self.assertEqual(unrelated, final_path.read_bytes())

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            directory = Path(temporary)
            temporary_name = f".{target_name}.tmp-cleanup-interrupted"
            temporary_path = directory / temporary_name
            temporary_path.write_bytes(expected_bytes)
            journal = contextual_production._journal_payload(
                target_name=target_name,
                expected_file_sha256=expected_sha256,
                temporary_name=temporary_name,
                temporary_status=temporary_path.stat(),
                phase="prepared",
                final_status=None,
            )
            (directory / journal_name).write_bytes(
                contextual_production._canonical_json_bytes(journal)
            )
            temporary_path.unlink()
            with patch.object(os, "link", side_effect=unsupported):
                recovered = save_stage_edit_contextual_training_corpus_v2(
                    directory, corpus
                )
            self.assertEqual(expected_bytes, recovered.read_bytes())
            self.assertFalse((directory / journal_name).exists())

    def test_bundle_rejects_resealed_full_fit_scorer_in_loso_folds(self) -> None:
        corpus = _corpus()
        validation = cross_validate_stage_edit_contextual_source_pair_v2(corpus)
        training = train_stage_edit_contextual_source_pair_v2(corpus)
        checkpoint = new_stage_edit_contextual_source_pair_checkpoint_v2(
            "bundle-fold-leak",
            training,
            created_at="2026-08-25T12:00:00+09:00",
        )
        cases = _qa_cases()
        qa = new_stage_edit_contextual_beam_qa_evidence_v2(
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            evaluation_corpus_sha256="c" * 64,
            cases=cases,
        )
        envelopes = (object(), object())
        with patch(
            "crosscut.stage_edit_contextual_beam_qa_adapter_v2."
            "require_verified_stage_edit_contextual_beam_qa_case_v2",
            side_effect=cases,
        ):
            valid_report = new_stage_edit_contextual_source_pair_production_report_v2(
                report_id="bundle-fold-leak",
                corpus=corpus,
                cross_validation=validation,
                final_training=training,
                checkpoint=checkpoint,
                beam_qa_evaluation_corpus_sha256="c" * 64,
                verified_beam_qa_cases=envelopes,
                created_at="2026-08-25T12:01:00+09:00",
            )
        forged_payload = valid_report.to_dict()
        cross_validation_payload = forged_payload["cross_validation"]
        for fold in cross_validation_payload["folds"]:
            fold["training"]["scorer"] = training.scorer.to_dict()
            fold["training"]["scorer_fingerprint"] = training.scorer.fingerprint
        _reseal_validation_payload(cross_validation_payload)
        forged_payload["report_sha256"] = _canonical_sha256(
            {
                key: value
                for key, value in forged_payload.items()
                if key != "report_sha256"
            }
        )
        forged_report = StageEditContextualSourcePairProductionReportV2.from_dict(
            forged_payload
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            directory = Path(temporary)
            corpus_path = save_stage_edit_contextual_training_corpus_v2(
                directory, corpus
            )
            checkpoint_path = save_stage_edit_contextual_source_pair_checkpoint_v2(
                directory, checkpoint
            )
            report_path = save_stage_edit_contextual_source_pair_report_v2(
                directory, forged_report
            )
            with self.assertRaisesRegex(
                StageEditContextualSourcePairV2ArtifactError,
                "LOSO differs from trusted corpus replay",
            ):
                load_stage_edit_contextual_source_pair_production_bundle_v2(
                    directory,
                    expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
                    expected_checkpoint_file_sha256=hashlib.sha256(
                        checkpoint_path.read_bytes()
                    ).hexdigest(),
                    expected_report_sha256=forged_report.report_sha256,
                    expected_report_file_sha256=hashlib.sha256(
                        report_path.read_bytes()
                    ).hexdigest(),
                    expected_beam_qa_report_sha256=qa.qa_report_sha256,
                    expected_corpus_sha256=corpus.content_sha256,
                    expected_corpus_file_sha256=hashlib.sha256(
                        corpus_path.read_bytes()
                    ).hexdigest(),
                    expected_token_schema_fingerprint=_SCHEMA.fingerprint,
                    verified_beam_qa_cases=envelopes,
                )

    def test_real_external_exfat_publish_when_scratch_is_available(self) -> None:
        scratch = Path("/Volumes/CrossCutMedia/CrossCut-Scratch")
        if not scratch.is_dir():
            self.skipTest("external ExFAT scratch directory is not mounted")
        corpus = _corpus()
        with tempfile.TemporaryDirectory(
            prefix="contextual-v2-exfat-", dir=scratch
        ) as temporary:
            directory = Path(temporary)
            path = save_stage_edit_contextual_training_corpus_v2(
                directory, corpus
            )
            file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(
                corpus,
                inspect_stage_edit_contextual_training_corpus_v2(
                    directory,
                    expected_corpus_sha256=corpus.content_sha256,
                    expected_corpus_file_sha256=file_sha256,
                    expected_token_schema_fingerprint=_SCHEMA.fingerprint,
                ),
            )
            self.assertFalse(
                any(
                    ".tmp-" in value.name or "publish-journal" in value.name
                    for value in directory.iterdir()
                )
            )

    def test_beam_qa_derives_target_authorization_from_measured_cases(self) -> None:
        target = _qa_case("BANG", request_digit="1", report_digit="a")
        self.assertTrue(target.hard_safe)
        self.assertTrue(target.authorized)

        low_target_cadence = _qa_case(
            "BLACKHOLE",
            request_digit="2",
            report_digit="b",
            duration_seconds=300.0,
            switch_count=63,
            eligible_switch_opportunity_count=140,
        )
        self.assertAlmostEqual(12.6, low_target_cadence.switches_per_minute)
        self.assertTrue(low_target_cadence.hard_safe)
        self.assertFalse(low_target_cadence.authorized)

        dominant_hard_only = _qa_case(
            "BLACKHOLE",
            request_digit="2",
            report_digit="b",
            dominant_source_ratio=0.40,
        )
        self.assertTrue(dominant_hard_only.hard_safe)
        self.assertFalse(dominant_hard_only.authorized)

        evidence = new_stage_edit_contextual_beam_qa_evidence_v2(
            checkpoint_sha256="c" * 64,
            evaluation_corpus_sha256="d" * 64,
            cases=(target, low_target_cadence),
        )
        self.assertFalse(evidence.authorized)
        self.assertFalse(evidence.all_target_profiles_met)
        case_payload = target.to_dict()
        case_payload["quality_score_median_span"] = 0.23
        with self.assertRaisesRegex(StageEditContextualSourcePairV2Error, "digest"):
            StageEditContextualBeamQACaseV2.from_dict(case_payload)
        evidence_payload = evidence.to_dict()
        evidence_payload["qa_report_sha256"] = "f" * 64
        with self.assertRaisesRegex(StageEditContextualSourcePairV2Error, "digest"):
            StageEditContextualBeamQAEvidenceV2.from_dict(evidence_payload)

    def test_report_factory_rejects_raw_beam_qa_dtos(self) -> None:
        corpus = _corpus()
        validation = cross_validate_stage_edit_contextual_source_pair_v2(corpus)
        training = train_stage_edit_contextual_source_pair_v2(corpus)
        checkpoint = new_stage_edit_contextual_source_pair_checkpoint_v2(
            "raw-qa-rejected",
            training,
            created_at="2026-08-25T12:00:00+09:00",
        )
        with self.assertRaisesRegex(RuntimeError, "verified Beam-QA case"):
            new_stage_edit_contextual_source_pair_production_report_v2(
                report_id="raw-qa-rejected",
                corpus=corpus,
                cross_validation=validation,
                final_training=training,
                checkpoint=checkpoint,
                beam_qa_evaluation_corpus_sha256="c" * 64,
                verified_beam_qa_cases=_qa_cases(),
                created_at="2026-08-25T12:01:00+09:00",
            )

    def test_report_factory_explicitly_binds_beam_qa_decode_scorer(self) -> None:
        corpus = _corpus()
        validation = cross_validate_stage_edit_contextual_source_pair_v2(corpus)
        training = train_stage_edit_contextual_source_pair_v2(corpus)
        checkpoint = new_stage_edit_contextual_source_pair_checkpoint_v2(
            "beam-qa-scorer-binding",
            training,
            created_at="2026-08-25T12:00:00+09:00",
        )
        expected = {
            "expected_checkpoint_sha256": checkpoint.checkpoint_sha256,
            "expected_corpus_sha256": checkpoint.corpus_sha256,
            "expected_scorer_fingerprint": checkpoint.scorer_fingerprint,
            "expected_runtime_equivalence_sha256": (
                checkpoint.runtime_equivalence_sha256
            ),
            "expected_token_schema_fingerprint": (
                checkpoint.token_schema_fingerprint
            ),
        }
        with patch(
            "crosscut.stage_edit_contextual_beam_qa_adapter_v2."
            "require_verified_stage_edit_contextual_beam_qa_case_v2",
            side_effect=_qa_cases(),
        ) as require_verified:
            new_stage_edit_contextual_source_pair_production_report_v2(
                report_id="beam-qa-scorer-binding",
                corpus=corpus,
                cross_validation=validation,
                final_training=training,
                checkpoint=checkpoint,
                beam_qa_evaluation_corpus_sha256="c" * 64,
                verified_beam_qa_cases=(object(), object()),
                created_at="2026-08-25T12:01:00+09:00",
            )
        self.assertEqual(2, require_verified.call_count)
        for invocation in require_verified.call_args_list:
            self.assertEqual(expected, invocation.kwargs)

    def test_report_rejects_missing_beam_level_highlight_gate(self) -> None:
        corpus = _corpus()
        validation = cross_validate_stage_edit_contextual_source_pair_v2(corpus)
        training = train_stage_edit_contextual_source_pair_v2(corpus)
        checkpoint = new_stage_edit_contextual_source_pair_checkpoint_v2(
            "synthetic-contextual-v2",
            training,
            created_at="2026-08-25T12:00:00+09:00",
        )
        qa = new_stage_edit_contextual_beam_qa_evidence_v2(
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            evaluation_corpus_sha256="c" * 64,
            cases=_qa_cases(verified_highlight_switch_count=0),
        )
        with self.assertRaisesRegex(StageEditContextualSourcePairV2Error, "beam QA"):
            with patch(
                "crosscut.stage_edit_contextual_beam_qa_adapter_v2."
                "require_verified_stage_edit_contextual_beam_qa_case_v2",
                side_effect=qa.cases,
            ):
                new_stage_edit_contextual_source_pair_production_report_v2(
                    report_id="denied",
                    corpus=corpus,
                    cross_validation=validation,
                    final_training=training,
                    checkpoint=checkpoint,
                    beam_qa_evaluation_corpus_sha256="c" * 64,
                    verified_beam_qa_cases=(object(), object()),
                    created_at="2026-08-25T12:01:00+09:00",
                )


if __name__ == "__main__":
    unittest.main()
