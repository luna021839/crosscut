from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crosscut.ive import AspectRatio
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.source_derivative_diversity import SourceDerivativeFamilyMetadata
from crosscut.stage_edit_continuity_authority import (
    CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2,
    PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
)
from crosscut.stage_edit_semimarkov_beam import (
    StageEditBoundaryContinuityResolution,
    StageEditDecodeInputError,
    StageEditHeadFrame,
    StageEditSemiMarkovConfig,
    StageEditSourceContinuity,
    StageEditSourceMetadata,
    decode_stage_edit_semimarkov_beam,
    decode_stage_edit_semimarkov_beam_contextual_evaluation_v2,
    resolve_stage_edit_boundary_continuity,
    stage_edit_boundary_continuity_score,
)
from crosscut.stage_edit_source_pair_contextual_v2_production import (
    load_stage_edit_contextual_source_pair_evaluation_bundle_v2,
    new_stage_edit_contextual_source_pair_checkpoint_v2,
    save_stage_edit_contextual_source_pair_checkpoint_v2,
    save_stage_edit_contextual_training_corpus_v2,
    train_stage_edit_contextual_source_pair_v2,
)
from crosscut.stage_edit_source_pair_contextual_v2 import (
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT,
    StageEditContextualSourcePairScorerV2,
    materialize_stage_edit_contextual_runtime_frame_v2,
)
from crosscut.stage_edit_source_pair_scorer import StageEditSourcePairScorer
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditReadinessReport,
    StageEditToken,
)


SOURCE_IDS = tuple(f"source-{index}" for index in range(4))
FAMILIES = {source_id: index for index, source_id in enumerate(SOURCE_IDS)}


def _ready_report() -> StageEditReadinessReport:
    return StageEditReadinessReport(
        architecture_available=True,
        learned_inference_ready=True,
        status="production_checkpoint_ready",
        reason="contextual integration test",
        mlx_version="test",
        checkpoint_path="/tmp/stageedit-production",
        training_steps=100,
    )


def _token(
    master_time_seconds: float,
    *,
    source_ids: tuple[str, ...] = SOURCE_IDS,
) -> StageEditToken:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    music = tuple(0.1 for _ in schema.music_feature_names)
    section = tuple(0.1 for _ in schema.section_feature_names)
    recent = tuple(0.0 for _ in schema.recent_edit_feature_names)
    style = tuple(0.1 for _ in schema.style_feature_names)
    missing_mask = (
        (False,) * (len(music) + len(section))
        + (True,) * len(recent)
        + (False,) * len(style)
    )
    candidates = []
    for source_index, source_id in enumerate(source_ids):
        values = tuple(
            min(0.8, 0.05 + 0.02 * source_index + 0.001 * feature_index)
            for feature_index, _name in enumerate(schema.source_feature_names)
        )
        candidates.append(
            SourceCandidateToken(
                source_id=source_id,
                features=values,
                missing_mask=(False,) * len(values),
            )
        )
    return StageEditToken(
        master_time_seconds=master_time_seconds,
        music_features=music,
        section_features=section,
        source_candidates=tuple(candidates),
        recent_edit_features=recent,
        style_features=style,
        missing_mask=missing_mask,
    )


def _runtime(
    master_time_seconds: float,
    *,
    families: dict[str, int] | None = None,
    source_ids: tuple[str, ...] = SOURCE_IDS,
):
    selected_families = families or {
        source_id: index for index, source_id in enumerate(source_ids)
    }
    return materialize_stage_edit_contextual_runtime_frame_v2(
        _token(master_time_seconds, source_ids=source_ids),
        canonical_source_ids=source_ids,
        derivative_family_id_by_source=selected_families,
        schema=DEFAULT_STAGE_EDIT_SCHEMA,
    )


def _scorer() -> StageEditContextualSourcePairScorerV2:
    return StageEditContextualSourcePairScorerV2(
        token_schema_fingerprint=DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
        weights=(0.0,) * len(
            STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES
        ),
    )


def _head_frame(
    master_time_seconds: float,
    *,
    runtime=None,
) -> StageEditHeadFrame:
    source_count = len(SOURCE_IDS)
    preferred = int(master_time_seconds) % source_count
    duration_bucket = DEFAULT_STAGE_EDIT_SCHEMA.duration_bucket(1.0)
    return StageEditHeadFrame(
        master_time_seconds=master_time_seconds,
        cut_logit=2.0 if master_time_seconds else 0.0,
        source_logits=tuple(
            2.5 if index == preferred else 0.2 - 0.01 * index
            for index in range(source_count)
        ),
        duration_logits=tuple(
            2.0 if index == duration_bucket else -0.2
            for index in range(DEFAULT_STAGE_EDIT_SCHEMA.duration_bucket_count)
        ),
        transition_logits=(2.0, 0.8, 0.2, -0.2),
        crop=(1.0, 0.5, 0.5, 0.0),
        available_source_mask=(True,) * source_count,
        hard_source_gate_mask=(True,) * source_count,
        source_shot_sizes=("wide", "medium", "closeup", "full"),
        source_continuity=tuple(
            StageEditSourceContinuity(
                face_scale=0.2 + 0.02 * index,
                face_center_x=0.3 + 0.04 * index,
                face_center_y=0.4,
                pose_motion=0.3 + 0.02 * index,
                camera_motion_x=0.1 * index,
                camera_motion_y=-0.1 * index,
                camera_zoom_velocity=0.02 * index,
                sharpness=0.8,
            )
            for index in range(source_count)
        ),
        candidate_id=f"contextual-{master_time_seconds:.3f}",
        section_progress=master_time_seconds / 5.0,
        phrase_progress=(master_time_seconds % 2.0) / 2.0,
        section_role="chorus_like",
        contextual_source_pair_runtime_v2=(
            runtime if runtime is not None else _runtime(master_time_seconds)
        ),
    )


def _derivative_metadata(
    families: tuple[int, ...] = (0, 1, 2, 3),
) -> SourceDerivativeFamilyMetadata:
    return SourceDerivativeFamilyMetadata(
        source_count=4,
        family_ids=families,
        audit_sha256="d" * 64,
        production_ready=True,
        uncertain_pair_count=0,
        clock_contract_complete=True,
    )


def _config(scorer: StageEditContextualSourcePairScorerV2):
    return StageEditSemiMarkovConfig(
        beam_width=16,
        timeline_count=8,
        minimum_unique_sources=3,
        maximum_global_switches_per_minute=120.0,
        section_cut_density_multiplier=4.0,
        enforce_montage_grammar=False,
        continuity_authority=PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
        continuity_scorer_contract=(
            STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT
        ),
        continuity_scorer_fingerprint=scorer.fingerprint,
    )


class StageEditContextualDecoderIntegrationTests(unittest.TestCase):
    def test_legacy_head_serialization_hash_is_unchanged(self) -> None:
        frame = StageEditHeadFrame(
            master_time_seconds=0.0,
            cut_logit=0.0,
            source_logits=(1.0, 0.0),
            duration_logits=(0.0,),
            transition_logits=(0.0,),
            crop=(1.0, 0.5, 0.5, 0.0),
            available_source_mask=(True, True),
            hard_source_gate_mask=(True, True),
        )
        payload = frame.to_dict()
        self.assertNotIn("contextual_source_pair_runtime_v2", payload)
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            "3ab2e8cf483527c225fd5f60343683c2a9cbdb54548cb76ce290ab5d6add813f",
            digest,
        )

    def test_head_serializes_only_redacted_contextual_identity(self) -> None:
        payload = _head_frame(0.0).to_dict()
        identity = payload["contextual_source_pair_runtime_v2"]
        self.assertEqual(64, len(identity["runtime_evidence_sha256"]))
        encoded = json.dumps(identity, sort_keys=True)
        self.assertNotIn("context_values", encoded)
        self.assertNotIn("values", encoded)
        self.assertNotIn("embedding", encoded)

    def test_head_rejects_contextual_runtime_time_mismatch(self) -> None:
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "runtime time differs",
        ):
            _head_frame(1.0, runtime=_runtime(0.0))

    def test_head_rejects_untyped_motion_continuity_evidence(self) -> None:
        frame = _head_frame(0.0)
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "must contain StageEditSourceContinuity",
        ):
            replace(
                frame,
                source_continuity=(object(),) * len(SOURCE_IDS),
            )

    def test_contextual_resolver_applies_exact_math_and_scalar_lineage(self) -> None:
        scorer = _scorer()
        frame = _head_frame(1.0)
        resolution = resolve_stage_edit_boundary_continuity(
            frame,
            0,
            1,
            SOURCE_IDS,
            (0, 1, 2, 3),
            continuity_authority=PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
            source_pair_scorer=None,
            contextual_source_pair_scorer_v2=scorer,
        )
        expected = scorer.score_boundary(
            frame.contextual_source_pair_runtime_v2,
            SOURCE_IDS[0],
            SOURCE_IDS[1],
        )
        self.assertTrue(resolution.applied)
        self.assertEqual(expected.decoder_score, resolution.decoder_score)
        lineage = resolution.to_dict()["contextual_score_v2"]
        self.assertEqual(expected.raw_logit, lineage["raw_logit"])
        self.assertEqual(expected.blended_baseline, lineage["blended_baseline"])
        self.assertEqual(expected.signed_residual, lineage["signed_residual"])
        encoded = json.dumps(lineage, sort_keys=True)
        self.assertNotIn("feature_names", encoded)
        self.assertNotIn("weights", encoded)

    def test_contextual_resolver_opening_and_same_family_are_exact_neutral(self) -> None:
        scorer = _scorer()
        families = (0, 0, 1, 2)
        family_map = dict(zip(SOURCE_IDS, families, strict=True))
        frame = _head_frame(1.0, runtime=_runtime(1.0, families=family_map))
        opening = resolve_stage_edit_boundary_continuity(
            frame,
            None,
            0,
            SOURCE_IDS,
            families,
            continuity_authority=PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
            source_pair_scorer=None,
            contextual_source_pair_scorer_v2=scorer,
        )
        same_family = resolve_stage_edit_boundary_continuity(
            frame,
            0,
            1,
            SOURCE_IDS,
            families,
            continuity_authority=PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
            source_pair_scorer=None,
            contextual_source_pair_scorer_v2=scorer,
        )
        self.assertEqual((0.0, "opening_boundary"), (
            opening.decoder_score,
            opening.neutral_reason,
        ))
        self.assertEqual((0.0, "same_derivative_family"), (
            same_family.decoder_score,
            same_family.neutral_reason,
        ))
        self.assertFalse(same_family.applied)
        self.assertIsNotNone(same_family.contextual_score_v2)

    def test_contextual_resolver_rejects_missing_wrong_scorer_sidecar_and_family(self) -> None:
        scorer = _scorer()
        frame = _head_frame(1.0)
        kwargs = {
            "continuity_authority": (
                PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
            ),
            "source_pair_scorer": None,
            "contextual_source_pair_scorer_v2": scorer,
        }
        with self.assertRaisesRegex(StageEditDecodeInputError, "typed scorer"):
            resolve_stage_edit_boundary_continuity(
                frame,
                0,
                1,
                SOURCE_IDS,
                (0, 1, 2, 3),
                **{**kwargs, "contextual_source_pair_scorer_v2": None},
            )
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "only its typed scorer",
        ):
            resolve_stage_edit_boundary_continuity(
                frame,
                0,
                1,
                SOURCE_IDS,
                (0, 1, 2, 3),
                **{
                    **kwargs,
                    "source_pair_scorer": StageEditSourcePairScorer.uniform(),
                },
            )
        with self.assertRaisesRegex(StageEditDecodeInputError, "runtime sidecar"):
            resolve_stage_edit_boundary_continuity(
                replace(frame, contextual_source_pair_runtime_v2=None),
                0,
                1,
                SOURCE_IDS,
                (0, 1, 2, 3),
                **kwargs,
            )
        with self.assertRaisesRegex(StageEditDecodeInputError, "families differ"):
            resolve_stage_edit_boundary_continuity(
                frame,
                0,
                1,
                SOURCE_IDS,
                (0, 0, 1, 2),
                **kwargs,
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_report_free_evaluation_decoder_is_separate_from_production(
        self, _mocked
    ) -> None:
        from tests.test_stage_edit_source_pair_contextual_v2 import _corpus

        corpus = _corpus()
        training = train_stage_edit_contextual_source_pair_v2(corpus)
        checkpoint = new_stage_edit_contextual_source_pair_checkpoint_v2(
            "decoder-evaluation-bootstrap",
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
            evaluation_bundle = (
                load_stage_edit_contextual_source_pair_evaluation_bundle_v2(
                    directory,
                    expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
                    expected_checkpoint_file_sha256=hashlib.sha256(
                        checkpoint_path.read_bytes()
                    ).hexdigest(),
                    expected_corpus_sha256=corpus.content_sha256,
                    expected_corpus_file_sha256=hashlib.sha256(
                        corpus_path.read_bytes()
                    ).hexdigest(),
                    expected_token_schema_fingerprint=(
                        corpus.token_schema_fingerprint
                    ),
                    expected_scorer_fingerprint=checkpoint.scorer_fingerprint,
                    expected_runtime_equivalence_sha256=(
                        checkpoint.runtime_equivalence_sha256
                    ),
                )
            )

            frames = tuple(_head_frame(float(index)) for index in range(6))
            metadata = tuple(
                StageEditSourceMetadata(
                    source_index=index,
                    source_id=source_id,
                    source_duration_seconds=6.0,
                    maximum_occupancy_fraction=0.60,
                    quality_log_prior=-0.01 * index,
                    default_shot_size=(
                        "wide",
                        "medium",
                        "closeup",
                        "full",
                    )[index],
                )
                for index, source_id in enumerate(SOURCE_IDS)
            )
            plan = SectionPlan(
                (
                    SectionDirective(
                        section_index=0,
                        role=SectionRole.CHORUS_LIKE,
                        start_seconds=0.0,
                        end_seconds=6.0,
                        energy_score=0.8,
                        confidence=0.9,
                        phrase_count=1,
                        preferred_source_hold_beats=2,
                        target_source_switches_per_minute=20.0,
                        camera_motion_budget=0.7,
                        target_roll_events_per_minute=0.2,
                        establish_with_wide_shot=False,
                        protect_last_beat_from_extra_cut=False,
                    ),
                )
            )
            config = replace(
                _config(evaluation_bundle.scorer),
                continuity_authority=(
                    CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2
                ),
            )
            kwargs = {
                "checkpoint_directory": Path("/tmp/stageedit-production"),
                "sources": tuple(
                    Path(f"/tmp/{source_id}.mp4") for source_id in SOURCE_IDS
                ),
                "source_metadata": metadata,
                "head_frames": frames,
                "section_plan": plan,
                "output": Path("/tmp/contextual-stageedit-evaluation.mp4"),
                "config": config,
                "aspect_ratio": AspectRatio.LANDSCAPE,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "source_derivative_metadata": _derivative_metadata(),
            }
            result = decode_stage_edit_semimarkov_beam_contextual_evaluation_v2(
                evaluation_bundle_v2=evaluation_bundle,
                **kwargs,
            )
            self.assertEqual(
                CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2,
                result.audit.config["continuity_authority"],
            )
            self.assertEqual(
                evaluation_bundle.scorer.fingerprint,
                result.audit.config["continuity_scorer_fingerprint"],
            )
            with self.assertRaisesRegex(
                StageEditDecodeInputError,
                "restricted to its dedicated decoder",
            ):
                decode_stage_edit_semimarkov_beam(
                    contextual_source_pair_scorer_v2=(
                        evaluation_bundle.scorer
                    ),
                    **kwargs,
                )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_full_decoder_contextual_path_and_fail_closed_mutations(
        self, _mocked
    ) -> None:
        scorer = _scorer()
        frames = tuple(_head_frame(float(index)) for index in range(6))
        metadata = tuple(
            StageEditSourceMetadata(
                source_index=index,
                source_id=source_id,
                source_duration_seconds=6.0,
                maximum_occupancy_fraction=0.60,
                quality_log_prior=-0.01 * index,
                default_shot_size=("wide", "medium", "closeup", "full")[index],
            )
            for index, source_id in enumerate(SOURCE_IDS)
        )
        plan = SectionPlan(
            (
                SectionDirective(
                    section_index=0,
                    role=SectionRole.CHORUS_LIKE,
                    start_seconds=0.0,
                    end_seconds=6.0,
                    energy_score=0.8,
                    confidence=0.9,
                    phrase_count=1,
                    preferred_source_hold_beats=2,
                    target_source_switches_per_minute=20.0,
                    camera_motion_budget=0.7,
                    target_roll_events_per_minute=0.2,
                    establish_with_wide_shot=False,
                    protect_last_beat_from_extra_cut=False,
                ),
            )
        )

        def run(candidate_frames, *, derivative=None):
            return decode_stage_edit_semimarkov_beam(
                checkpoint_directory=Path("/tmp/stageedit-production"),
                sources=tuple(
                    Path(f"/tmp/{source_id}.mp4") for source_id in SOURCE_IDS
                ),
                source_metadata=metadata,
                head_frames=candidate_frames,
                section_plan=plan,
                output=Path("/tmp/contextual-stageedit.mp4"),
                config=_config(scorer),
                aspect_ratio=AspectRatio.LANDSCAPE,
                width=1920,
                height=1080,
                fps=30.0,
                source_derivative_metadata=(
                    derivative or _derivative_metadata()
                ),
                contextual_source_pair_scorer_v2=scorer,
            )

        result = run(frames)
        # Exercise the public audit serializer; its scalar-payload guard must
        # accept contextual lineage without raw token/source vectors.
        result.to_dict()
        provenance = result.audit.input_summary[
            "boundary_continuity_score_provenance"
        ]
        self.assertEqual(
            PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
            provenance["authority"],
        )
        self.assertEqual(scorer.fingerprint, provenance["scorer_fingerprint"])
        decisions = result.timelines[0].decisions
        self.assertTrue(decisions)
        self.assertTrue(
            all(
                decision.boundary_continuity_score_lineage is not None
                for decision in decisions
            )
        )
        self.assertEqual(
            "opening_boundary",
            decisions[0].boundary_continuity_score_lineage.neutral_reason,
        )
        self.assertIsInstance(
            decisions[0].boundary_continuity_score_lineage,
            StageEditBoundaryContinuityResolution,
        )
        primary_differs_from_handcrafted = False
        for previous, decision in zip(decisions, decisions[1:]):
            exact_frame = frames[decision.start_candidate_index]
            contextual = scorer.score_boundary(
                exact_frame.contextual_source_pair_runtime_v2,
                previous.source_id,
                decision.source_id,
            )
            handcrafted = stage_edit_boundary_continuity_score(
                exact_frame.source_continuity[previous.source_index],
                exact_frame.source_continuity[decision.source_index],
            )
            self.assertEqual(
                contextual.decoder_score,
                decision.boundary_continuity_score,
            )
            self.assertEqual(
                handcrafted,
                decision.boundary_handcrafted_motion_continuity_qualifier_score,
            )
            primary_differs_from_handcrafted |= (
                contextual.decoder_score != handcrafted
            )
        self.assertTrue(primary_differs_from_handcrafted)

        missing = list(frames)
        missing[0] = replace(
            missing[0], contextual_source_pair_runtime_v2=None
        )
        with self.assertRaisesRegex(StageEditDecodeInputError, "runtime sidecar"):
            run(tuple(missing))

        missing_motion_observations = list(frames)
        missing_motion_observations[0] = replace(
            missing_motion_observations[0],
            source_continuity=(),
        )
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "independent handcrafted motion qualifier",
        ):
            run(tuple(missing_motion_observations))

        misaligned_motion_observations = list(frames)
        misaligned_motion_observations[0] = replace(
            misaligned_motion_observations[0],
            source_continuity=(
                misaligned_motion_observations[0].source_continuity[:-1]
            ),
        )
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "continuity count does not match sources",
        ):
            run(tuple(misaligned_motion_observations))

        wrong_family_map = {**FAMILIES, SOURCE_IDS[1]: 0}
        wrong_family = list(frames)
        wrong_family[0] = replace(
            wrong_family[0],
            contextual_source_pair_runtime_v2=_runtime(
                0.0, families=wrong_family_map
            ),
        )
        with self.assertRaisesRegex(StageEditDecodeInputError, "family mismatch"):
            run(tuple(wrong_family))

        wrong_source_ids = (*SOURCE_IDS[:3], "source-X")
        wrong_source = list(frames)
        wrong_source[0] = replace(
            wrong_source[0],
            contextual_source_pair_runtime_v2=_runtime(
                0.0,
                source_ids=wrong_source_ids,
            ),
        )
        with self.assertRaisesRegex(StageEditDecodeInputError, "source identity"):
            run(tuple(wrong_source))

        tampered_schema = list(frames)
        object.__setattr__(
            tampered_schema[0].contextual_source_pair_runtime_v2,
            "token_schema_fingerprint",
            "0" * 64,
        )
        with self.assertRaisesRegex(StageEditDecodeInputError, "schema mismatch"):
            run(tuple(tampered_schema))

        # Rebuild clean frames because the prior frozen object was deliberately
        # attacked through object.__setattr__.
        tampered_digest = list(
            tuple(_head_frame(float(index)) for index in range(6))
        )
        object.__setattr__(
            tampered_digest[0].contextual_source_pair_runtime_v2,
            "runtime_evidence_sha256",
            "0" * 64,
        )
        with self.assertRaisesRegex(StageEditDecodeInputError, "digest mismatch"):
            run(tuple(tampered_digest))

        tampered_time = list(
            tuple(_head_frame(float(index)) for index in range(6))
        )
        object.__setattr__(
            tampered_time[0].contextual_source_pair_runtime_v2,
            "master_time_seconds",
            0.25,
        )
        with self.assertRaisesRegex(StageEditDecodeInputError, "time mismatch"):
            run(tuple(tampered_time))


if __name__ == "__main__":
    unittest.main()
