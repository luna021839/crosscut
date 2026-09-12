"""Synthetic control-flow proofs, not a claim of rendered artistic quality."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from crosscut.stage_edit_release_policy import (
    STAGE_EDIT_SINGLE_DELIVERY_VISUAL_CONTRACT,
    stage_edit_release_policy_payload, stage_edit_release_binding_valid,
    select_stage_edit_delivery_cohort,
)
from crosscut.stage_edit_pipeline_bridge import (
    StageEditPipelineBridgePreflight, StageEditPipelineBridgeError,
    run_stage_edit_pipeline_bridge, _exact_clock_candidate,
)
from crosscut.stage_edit_semimarkov_beam import (
    StageEditSemiMarkovConfig, HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
    HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
)
from crosscut.stage_edit_continuity_authority import HANDCRAFTED_CONTINUITY_AUTHORITY
from crosscut.stage_edit_visual_candidate_adapter import (
    StageEditVisualCandidateBatch, StageEditVisualCandidateRecord,
    StageEditMaterializedVisualCandidate, StageEditVisualCandidateError,
    materialize_stage_edit_visual_candidates,
)
from crosscut.visual_candidate_materialization import materialize_visual_timeline_candidates
from crosscut.post_render_qa import evaluate_post_render_qa, PostRenderQAVerdict
from crosscut.rendered_candidate_selection import RenderedCandidate, RenderedCandidateReview
from engine.tests.test_rendered_candidate_selection import _plan, _payload
from engine.tests.test_stage_edit_pipeline_bridge import (
    _snapshot, _snapshot_identity, _ready, _typed_direct_bridge_result,
    _typed_timelines, _typed_decode_result, _typed_head_result,
    _typed_section_director, _section_plan,
)
from engine.tests import test_visual_candidate_materialization as visual_fixture
from engine.tests.test_post_render_qa import (
    _persist_fixture, _FakeRunner, _perceptual_pass_payload,
    _activate_registered_cut_local_sync, _registered_qa_config,
)


def _preflight(version=2):
    snapshot = _snapshot()
    checkpoint = Path("/tmp/stageedit-bridge-ready").resolve()
    path = Path("/tmp/stageedit-feature.json").resolve()
    return StageEditPipelineBridgePreflight(
        snapshot_path=path, checkpoint_directory=checkpoint,
        source_ids=("source-a", "source-b"), snapshot=snapshot,
        readiness=_ready(checkpoint, dynamic_crop_version=version),
        dynamic_crop_version=version, snapshot_identity=_snapshot_identity(snapshot, path),
        edit_mode="group", target_member=None,
    )


def _learned_delivery(analysis, _plan, selection):
    policy = stage_edit_release_policy_payload()
    selection["release_policy"] = deepcopy(policy)
    selection["candidate_pool"] = "stage_edit_generation_only"
    analysis["stage_edit_generation"] = {
        "activation": {"activation": "explicit_request_bundle"}
    }
    selection["stage_edit_render_promotion"] = {
        "activation": "explicit_stage_edit_generation_request_only",
        "candidate_pool": "stage_edit_generation_only",
    }
    analysis["render"].update(
        release_policy=deepcopy(policy), candidate_origin="explicit_stage_edit_generation",
        stage_edit_promotion_contract=STAGE_EDIT_SINGLE_DELIVERY_VISUAL_CONTRACT,
    )
    selection["stage_edit_visual_candidate_batch"] = dict(
        contract=STAGE_EDIT_SINGLE_DELIVERY_VISUAL_CONTRACT,
        release_policy=deepcopy(policy), candidate_pool="stage_edit_generation_only",
        candidate_count=1, learned_candidate_count=1, hybrid_visual_floor_candidate_count=0,
        required_candidate_count=1, all_prescribed_paths_passed_visual_hard_gates=True,
        silent_baseline_fallback_allowed=False, dynamic_crop_version=3,
        candidates=[dict(candidate_index=0, candidate_origin="stage_edit_semimarkov")],
    )
    for field in ("requested_review_count", "reviewed_candidate_count",
                  "minimum_required_candidate_count", "successful_review_count",
                  "passing_candidate_count", "complete_candidate_count"):
        selection[field] = 1
    selection["candidates"] = selection["candidates"][:1]
    selection["complete_candidates"] = selection["complete_candidates"][:1]


class StageEditDeliveryQATests(unittest.TestCase):
    def evaluate(self, mutation=None, *, registered=False):
        def mutate(a, p, s):
            _learned_delivery(a, p, s)
            if mutation:
                mutation(a, p, s)
        with tempfile.TemporaryDirectory() as directory:
            artifacts, _, _, _ = _persist_fixture(Path(directory), mutate=mutate)
            with patch("crosscut.post_render_qa.evaluate_post_render_perceptual_quality",
                       return_value=_perceptual_pass_payload()):
                return evaluate_post_render_qa(artifacts, runner=_FakeRunner(),
                    config=_registered_qa_config() if registered else None)

    def test_single_delivery_crosses_independent_qa_with_source_floor_intact(self):
        report = self.evaluate()
        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS,
                         [(i.code, i.message) for i in report.issues])
        self.assertEqual(report.policy.required_complete_candidate_count, 1)
        self.assertEqual(report.policy.minimum_distinct_sources_when_required, 8)

    def test_failed_or_absent_decoded_review_stays_failed(self):
        for mutation in (
            lambda a,p,s: s["final_selected_review"].update(passed=False),
            lambda a,p,s: s.update(candidates=[]),
            lambda a,p,s: s.update(complete_candidates=[]),
        ):
            with self.subTest(mutation=mutation):
                self.assertEqual(self.evaluate(mutation).verdict, PostRenderQAVerdict.FAIL)

    def test_malformed_cross_artifact_policy_or_zero_learned_stays_failed(self):
        for mutation in (
            lambda a,p,s: a["render"].pop("release_policy"),
            lambda a,p,s: a["render"].update(candidate_origin="joint_decoded_visual_selector"),
            lambda a,p,s: s["release_policy"].update(minimum_passing_candidate_count=0),
            lambda a,p,s: s["stage_edit_visual_candidate_batch"].update(learned_candidate_count=0),
            lambda a,p,s: s["stage_edit_visual_candidate_batch"].update(candidate_count=True),
            lambda a,p,s: s["stage_edit_visual_candidate_batch"].update(candidates=[]),
            lambda a,p,s: s["stage_edit_visual_candidate_batch"].update(dynamic_crop_version=99),
            lambda a,p,s: s["stage_edit_visual_candidate_batch"].pop("release_policy"),
            lambda a,p,s: s.update(candidate_pool="ordinary_visual_baseline"),
        ):
            with self.subTest(mutation=mutation):
                report = self.evaluate(mutation)
                self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
                self.assertIn("candidate_release_policy_mismatch", {i.code for i in report.issues})

    def test_registered_delivery_still_requires_cut_local_and_montage_evidence(self):
        def registered(a, p, s):
            _activate_registered_cut_local_sync(a, p, s)
            a["render"]["stage_edit_promotion_contract"] = STAGE_EDIT_SINGLE_DELIVERY_VISUAL_CONTRACT
        report = self.evaluate(registered, registered=True)
        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS,
                         [(i.code, i.message) for i in report.issues])
        self.assertTrue(report.metrics["production_sync"]["cut_local_sync"]["required"])
        def remove(a,p,s):
            registered(a,p,s)
            a["render"].pop("cut_local_sync", None)
        failed = self.evaluate(remove, registered=True)
        self.assertEqual(failed.verdict, PostRenderQAVerdict.FAIL)


class StageEditDeliveryBridgeTests(unittest.TestCase):
    def run_bridge(self, count, *, enabled=True, audit_changes=None,
                   reject_clock_ranks=(), clock_error=StageEditPipelineBridgeError):
        preflight = _preflight()
        sources = (Path("/tmp/a.mp4"), Path("/tmp/b.mp4"))
        config = StageEditSemiMarkovConfig(minimum_unique_sources=1)
        timelines = _typed_timelines(sources, count=count,
            authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
            scorer_contract=HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
            scorer_fingerprint=HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT)
        decoded = _typed_decode_result(timelines, config=config,
            scorer_contract=HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
            scorer_fingerprint=HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
            request_fulfilled=count == 8)
        if audit_changes:
            decoded = replace(decoded, audit=replace(decoded.audit, **audit_changes))
        def project(*, timeline, **kwargs):
            if timeline.rank in reject_clock_ranks:
                raise clock_error("synthetic clock rejection")
            return _exact_clock_candidate(timeline=timeline, **kwargs)
        with (
            patch("crosscut.stage_edit_pipeline_bridge._exact_clock_candidate", side_effect=project),
            patch("crosscut.stage_edit_pipeline_bridge.materialize_stage_edit_head_frames_v2",
                  return_value=_typed_head_result(preflight.checkpoint_directory)),
            patch("crosscut.stage_edit_pipeline_bridge.decode_stage_edit_semimarkov_beam",
                  return_value=decoded),
            patch("crosscut.stage_edit_pipeline_bridge.build_learned_section_plan_from_checkpoint",
                  return_value=_typed_section_director()),
        ):
            return run_stage_edit_pipeline_bridge(preflight=preflight, sources=sources,
                source_durations_seconds=(10.,10.), source_start_offsets_seconds=(0.,0.),
                internal_cut_times_seconds=((),()), source_timewarps=(None,None),
                source_quality_log_priors=(0.,0.), section_plan=_section_plan(),
                output=Path("/tmp/out.mp4"), reference_source=0, config=config,
                quality_gated_single_delivery=enabled)

    def test_partial_nonempty_decode_proceeds_without_faking_search_completion(self):
        for count in (1, 2, 7, 8):
            with self.subTest(count=count):
                result = self.run_bridge(count)
                self.assertEqual(len(result.retained_candidates), count)
                self.assertEqual(result.semimarkov.audit.request_fulfilled, count == 8)
                self.assertFalse(result.selection_policy["selected_for_render"])
                self.assertFalse(result.selection_policy["visual_baseline_replaced"])

    def test_zero_decode_and_legacy_shortfall_fail(self):
        for count, enabled in ((0, True), (1, False), (7, False)):
            with self.subTest(count=count, enabled=enabled):
                with self.assertRaises(StageEditPipelineBridgeError):
                    self.run_bridge(count, enabled=enabled)

    def test_comparison_audit_cannot_claim_missing_paths_were_returned(self):
        for changes in ({"request_fulfilled": True}, {"returned_timeline_count": 8},
                        {"requested_timeline_count": 1}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(StageEditPipelineBridgeError, "audit disagrees"):
                    self.run_bridge(1, audit_changes=changes)

    def test_individual_clock_rejections_do_not_discard_remaining_exact_path(self):
        result = self.run_bridge(8, reject_clock_ranks=range(1,8))
        self.assertEqual([v.rank for v in result.retained_candidates], [8])
        self.assertEqual(len(result.selection_policy["exact_clock_candidate_rejections"]), 7)
        self.assertTrue(result.semimarkov.audit.request_fulfilled)
        with self.assertRaises(StageEditPipelineBridgeError):
            self.run_bridge(8, reject_clock_ranks=range(1,9))
        with self.assertRaises(StageEditPipelineBridgeError):
            self.run_bridge(8, enabled=False, reject_clock_ranks=(1,))
        with self.assertRaises(OSError):
            self.run_bridge(8, reject_clock_ranks=(1,), clock_error=OSError)


class StageEditDeliveryAdapterTests(unittest.TestCase):
    def setUp(self):
        self.bridge = _typed_direct_bridge_result(_preflight())
        # Geometry gates are separately exercised; this fixture controls their
        # pass/fail verdict to isolate candidate-cohort and lineage handling.
        candidate_set = visual_fixture.VisualCandidateMaterializationTests.candidate_set()
        record = candidate_set.candidates[0]
        record = replace(record, review=replace(record.review, hard_failures=()),
                         candidate_index=0, review_rank=0, selected=True)
        candidate_set = replace(candidate_set, candidates=(record,), selected_candidate_index=0)
        values = materialize_visual_timeline_candidates(
            candidate_set, maximum_candidates=1,
            passed_only=False)
        self.base = values[0]

    def materialized(self, retained, *, candidate_index, review_rank, **kwargs):
        record = StageEditVisualCandidateRecord(candidate_index=candidate_index,
            review_rank=review_rank, cadence_profile="balanced",
            retained_candidate=retained, base_record=self.base.candidate_record)
        return StageEditMaterializedVisualCandidate(record, self.base,
                                                   self.base.visual_plan, retained)

    def adapt(self, passes, *, single=True, error_type=StageEditVisualCandidateError):
        calls = []
        def gate(retained, **kwargs):
            calls.append(retained.rank)
            if retained.rank not in passes:
                raise error_type("synthetic gate rejection")
            return self.materialized(retained, **kwargs)
        with patch("crosscut.stage_edit_visual_candidate_adapter.materialize_stage_edit_visual_candidate",
                   side_effect=gate):
            batch = materialize_stage_edit_visual_candidates(self.bridge,
                sources=self.bridge.retained_candidates[0].edit.sources,
                audio_offsets=(0.,0.), sync_scores=(1.,1.), scene_cut_times=((),()),
                source_timewarps=(None,None), quality_gated_single_delivery=single)
        return batch, calls

    def test_one_two_seven_eight_survivors_keep_typed_lineage_and_audit(self):
        for count in (1,2,7,8):
            passes = set(range(9-count, 9))
            with self.subTest(count=count):
                batch, calls = self.adapt(passes)
                self.assertEqual(len(calls), 8)
                self.assertEqual(len(batch.candidates), count)
                self.assertEqual([c.candidate_index for c in batch.candidates], list(range(count)))
                self.assertEqual([c.retained_candidate.rank for c in batch.candidates], sorted(passes))
                self.assertEqual(len(batch.learned_candidate_rejections), 8-count)
                self.assertEqual(batch.minimum_passing_candidate_count, 1)
                self.assertEqual(batch.to_dict()["contract"], STAGE_EDIT_SINGLE_DELIVERY_VISUAL_CONTRACT)
                self.assertTrue(all(c.lineage[-1]["adapter"] == batch.contract
                                    for c in batch.candidates))

    def test_zero_survivors_fail_and_legacy_does_not_silently_change(self):
        with self.assertRaises(StageEditVisualCandidateError): self.adapt(set())
        with self.assertRaises(StageEditVisualCandidateError): self.adapt({8}, single=False)
        batch, _ = self.adapt(set(range(1,9)), single=False)
        self.assertEqual(batch.minimum_passing_candidate_count, 8)
        self.assertIsNone(batch.to_dict()["release_policy"])

    def test_runtime_storage_error_is_not_a_candidate_rejection(self):
        with self.assertRaises(OSError): self.adapt(set(), error_type=OSError)

    def test_hybrid_keeps_one_learned_plus_floor_but_never_floor_only(self):
        bridge = _typed_direct_bridge_result(_preflight(3))
        floor = replace(bridge.retained_candidates[0], rank=9,
                        candidate_origin="hybrid_visual_floor_dynamic_crop_v3")
        def gate(retained, **kwargs):
            if retained.rank != 8:
                raise StageEditVisualCandidateError("synthetic individual rejection")
            return self.materialized(retained, **kwargs)
        arguments = dict(sources=bridge.retained_candidates[0].edit.sources,
            audio_offsets=(0.,0.), sync_scores=(1.,1.), scene_cut_times=((),()),
            source_timewarps=(None,None), hybrid_visual_floor_candidate=self.base,
            quality_gated_single_delivery=True)
        with (
            patch("crosscut.stage_edit_visual_candidate_adapter.materialize_stage_edit_visual_candidate",
                  side_effect=gate),
            patch("crosscut.stage_edit_visual_candidate_adapter._build_hybrid_visual_floor_retained_candidate",
                  return_value=floor) as convert,
        ):
            batch = materialize_stage_edit_visual_candidates(bridge, **arguments)
        self.assertEqual([c.retained_candidate.rank for c in batch.candidates], [8,9])
        self.assertEqual(batch.hybrid_visual_floor_candidate_count, 1)
        self.assertEqual(len(batch.learned_candidate_rejections), 7)
        policy = stage_edit_release_policy_payload()
        self.assertTrue(stage_edit_release_binding_valid(
            dict(release_policy=policy, candidate_pool=batch.candidate_pool,
                 stage_edit_visual_candidate_batch=batch.to_dict()),
            dict(release_policy=policy, candidate_origin="explicit_stage_edit_generation",
                 stage_edit_promotion_contract=batch.contract)))
        convert.assert_called_once()
        with self.assertRaisesRegex(ValueError, "all learned"):
            StageEditVisualCandidateBatch((replace(batch.candidates[1],
                candidate_record=replace(batch.candidates[1].candidate_record,
                                         candidate_index=0, review_rank=0)),),
                quality_gated_single_delivery=True)
        with (
            patch("crosscut.stage_edit_visual_candidate_adapter.materialize_stage_edit_visual_candidate",
                  side_effect=StageEditVisualCandidateError("all failed")),
            patch("crosscut.stage_edit_visual_candidate_adapter._build_hybrid_visual_floor_retained_candidate") as convert,
        ):
            with self.assertRaises(StageEditVisualCandidateError):
                materialize_stage_edit_visual_candidates(bridge, **arguments)
            convert.assert_not_called()


class StageEditDecodedComparisonTests(unittest.TestCase):
    def items(self, scores):
        return tuple(RenderedCandidateReview(RenderedCandidate(i, _plan(i), 70.),
                     _payload(passed=True, score=score, issues=0., p10=65.))
                     for i, score in enumerate(scores))

    def test_best_available_not_first_pass_and_no_extra_render(self):
        reviews = self.items((65., 84., 90.))
        bundles = tuple(SimpleNamespace(candidate_index=i) for i in range(3))
        cohort, audit = select_stage_edit_delivery_cohort(bundles, reviews)
        self.assertEqual(cohort, ((bundles[2], reviews[2]),))
        self.assertEqual(audit["comparison_candidate_count"], 3)
        self.assertEqual(len(audit["comparison"]["candidates"]), 3)
        self.assertEqual(audit["required_delivery_count"], 1)

    def test_zero_mismatched_or_failed_review_cannot_become_delivery(self):
        reviews = self.items((90.,))
        for bundles, values in (
            ((), ()), ((SimpleNamespace(candidate_index=2),), reviews),
            ((SimpleNamespace(candidate_index=0),),
             (replace(reviews[0], review_payload=_payload(passed=False,score=99.,issues=0.,p10=99.)),)),
        ):
            with self.subTest(bundles=bundles):
                with self.assertRaises(ValueError):
                    select_stage_edit_delivery_cohort(bundles, values)


if __name__ == "__main__":
    unittest.main()
