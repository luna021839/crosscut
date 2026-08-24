from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.render import EditorialTransformKeyframe
from crosscut.stage_edit_dynamic_crop_capacity_probe import (
    PROBE_TRAINING_SCHEDULE,
    DeviceMemoryBudget,
    default_dynamic_crop_v2_probe_candidates,
)
from crosscut.stage_edit_dynamic_crop_head_v2 import (
    DynamicCropTrajectoryTargetV2,
    StageEditDynamicCropTargetSequenceV2,
)
from crosscut.stage_edit_dynamic_crop_mlx_smoke_cli import (
    StageEditMlxSmokeRunnerError,
    _build_isolated_worker_environment,
    _compute_distribution_record_provenance,
    _execution_from_process,
    _lane_assignments,
    _pack_fixed_lane_training_arrays,
    _parse_manifest_map,
    _runtime_provenance,
    _runtime_module_pointer,
    _safe_mlx_memory,
    _worker_failure_kind,
)
from crosscut.stage_edit_reference_adapter import ReferenceTrainingExample
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditSequence,
    StageEditTarget,
    StageEditTargetSequence,
    StageEditToken,
)


BUDGET = DeviceMemoryBudget(
    device_name="Apple M4 Pro",
    physical_memory_bytes=25_769_803_776,
)


def _token(value: float) -> StageEditToken:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    sources = tuple(
        SourceCandidateToken(
            source_id=f"source-{index}",
            features=(value + index,) * schema.raw_source_dimension,
            missing_mask=(False,) * schema.raw_source_dimension,
        )
        for index in range(2)
    )
    return StageEditToken(
        master_time_seconds=value,
        music_features=(value,) * len(schema.music_feature_names),
        section_features=(value,) * len(schema.section_feature_names),
        source_candidates=sources,
        recent_edit_features=(0.0,) * len(schema.recent_edit_feature_names),
        style_features=(value,) * len(schema.style_feature_names),
        missing_mask=(False,) * schema.raw_context_dimension,
    )


def _example(reference: str, values: tuple[float, ...]) -> ReferenceTrainingExample:
    sequence_id = f"sequence:{reference}"
    tokens = tuple(_token(value) for value in values)
    targets = tuple(
        StageEditTarget(
            cut=float(index % 2),
            selected_source_id="source-0",
            rejected_source_ids=("source-1",),
            duration_bucket=index,
            transition_label="hard_cut",
            crop=(1.0 + value, 0.5, 0.5, value),
        )
        for index, value in enumerate(values)
    )
    return ReferenceTrainingExample(
        reference_id=reference,
        song_id="IVE-BANG-BANG",
        style_id="unit",
        sequence=StageEditSequence(sequence_id, tokens),
        targets=StageEditTargetSequence(sequence_id, targets),
        statistics={},
        feature_snapshot_sha256="a" * 64,
    )


def _dynamic(example: ReferenceTrainingExample) -> StageEditDynamicCropTargetSequenceV2:
    targets = []
    for index, token in enumerate(example.sequence.tokens):
        value = token.music_features[0]
        targets.append(
            DynamicCropTrajectoryTargetV2(
                shot_index=index,
                source_id="source-0",
                shot_duration_seconds=1.0,
                keyframes=(
                    EditorialTransformKeyframe(0.0, 1.0 + value, 0.5, 0.5, 0.0),
                    EditorialTransformKeyframe(1.0, 1.1 + value, 0.5, 0.5, value),
                ),
                motion_label="dynamic",
                motion_label_loss_masked=False,
                artifact_content_sha256=f"{index + 1:064x}",
                build_fingerprint_sha256=f"{index + 11:064x}",
            )
        )
    return StageEditDynamicCropTargetSequenceV2(
        sequence_id=example.sequence.sequence_id,
        targets=tuple(targets),
    )


def _memory_stage(name: str, elapsed: float) -> dict[str, object]:
    return {
        "stage": name,
        "synchronized": True,
        "active_memory_bytes": 1024,
        "cache_memory_bytes": 0,
        "peak_memory_bytes": 1024**3,
        "elapsed_seconds": elapsed,
    }


class StageEditDynamicCropMlxSmokeCliTests(unittest.TestCase):
    def test_empty_message_memory_error_is_classified_as_oom(self) -> None:
        self.assertEqual("out_of_memory", _worker_failure_kind(MemoryError()))
        self.assertEqual("runtime_error", _worker_failure_kind(RuntimeError()))

    def test_safe_memory_getter_preserves_failure_peak(self) -> None:
        fake_mx = SimpleNamespace(get_peak_memory=lambda: 123456)
        with patch(
            "crosscut.stage_edit_dynamic_crop_mlx_smoke_cli.mx", fake_mx
        ):
            self.assertEqual(123456, _safe_mlx_memory("get_peak_memory"))

    def test_verified_user_site_roots_import_in_real_isolated_child(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        runtime = _runtime_provenance(BUDGET, repo)
        run_nonce = "d" * 64
        environment = _build_isolated_worker_environment(
            repo=repo,
            runtime_provenance=runtime,
            run_nonce=run_nonce,
        )
        result = subprocess.run(
            (
                sys.executable,
                "-P",
                "-S",
                "-c",
                "import json,sys,numpy,mlx.core,mlx.nn,mlx.optimizers;"
                "import crosscut.stage_edit_dynamic_crop_mlx_smoke_cli;"
                "print(json.dumps({'sitecustomize': 'sitecustomize' in sys.modules,"
                "'numpy': numpy.__file__, 'mlx': mlx.core.__file__,"
                "'sys_path': sys.path}, sort_keys=True))",
            ),
            cwd=repo,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        observed = json.loads(result.stdout)
        self.assertFalse(observed["sitecustomize"])
        self.assertIn("site-packages/numpy/", observed["numpy"])
        self.assertIn("site-packages/mlx/", observed["mlx"])
        self.assertNotIn("", observed["sys_path"])
        self.assertNotIn(str(repo), observed["sys_path"])
        self.assertEqual("1", environment["PYTHONSAFEPATH"])
        self.assertEqual(
            runtime["worker_pythonpath_entries"],
            observed["sys_path"][: len(runtime["worker_pythonpath_entries"])],
        )
        expected_nonce_sha256 = hashlib.sha256(
            run_nonce.encode("ascii")
        ).hexdigest()
        self.assertIn(expected_nonce_sha256, environment["PYTHONPYCACHEPREFIX"])
        self.assertFalse(Path(environment["PYTHONPYCACHEPREFIX"]).exists())

    def test_distribution_record_byte_tamper_is_rejected(self) -> None:
        with patch(
            "crosscut.stage_edit_dynamic_crop_mlx_smoke_cli._file_sha256",
            return_value="0" * 64,
        ):
            with self.assertRaisesRegex(
                StageEditMlxSmokeRunnerError, "differs from RECORD"
            ):
                _compute_distribution_record_provenance("mlx")

    def test_shadow_module_outside_corresponding_distribution_is_rejected(self) -> None:
        fake_mlx = SimpleNamespace(__name__="mlx.core", __file__=np.__file__)
        with self.assertRaisesRegex(
            StageEditMlxSmokeRunnerError, "not uniquely present.*mlx RECORD"
        ):
            _runtime_module_pointer(
                fake_mlx,
                version=BUDGET.mlx_version,
                distribution_name="mlx",
            )

    def test_quarantined_manifest_text_is_rejected_before_open(self) -> None:
        with self.assertRaisesRegex(
            StageEditMlxSmokeRunnerError, "quarantined"
        ):
            _parse_manifest_map(
                [
                    "IVE-BANG-BANG=/tmp/bang.json",
                    "IVE-BLACKHOLE=/tmp/black.json",
                    "IVE-REBEL-HEART=/tmp/XOXZ-hidden.json",
                ]
            )

    def test_lane_assignment_contains_all_fifteen_once(self) -> None:
        references = {
            song: tuple(f"{song}-ref-{index}" for index in range(5))
            for song in (
                "IVE-BANG-BANG",
                "IVE-BLACKHOLE",
                "IVE-REBEL-HEART",
            )
        }
        lanes = _lane_assignments(references)
        flattened = (*lanes["A"], *lanes["B"])
        self.assertEqual(15, len(flattened))
        self.assertEqual(15, len(set(flattened)))

    def test_actual_feature_base_and_dynamic_targets_tile_together(self) -> None:
        candidate = default_dynamic_crop_v2_probe_candidates()[0]
        first = _example("reference-a", (0.00, 0.02))
        second = _example("reference-b", (0.04, 0.06))
        inputs, base, dynamic, provenance = _pack_fixed_lane_training_arrays(
            examples_by_reference={
                first.reference_id: first,
                second.reference_id: second,
            },
            dynamic_by_reference={
                first.reference_id: _dynamic(first),
                second.reference_id: _dynamic(second),
            },
            lane_assignments={"A": (first.reference_id,), "B": (second.reference_id,)},
            candidate=candidate,
            sequence_length=4,
            max_sources=2,
        )
        self.assertTrue(np.all(inputs["sequence_mask"]))
        np.testing.assert_allclose(
            inputs["context_features"][0, :, 0], [0.00, 0.02, 0.00, 0.02]
        )
        np.testing.assert_allclose(
            inputs["context_features"][1, :, 0], [0.04, 0.06, 0.04, 0.06]
        )
        np.testing.assert_array_equal(base["cut_targets"][0], [0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(base["cut_targets"][1], [0.0, 1.0, 0.0, 1.0])
        self.assertTrue(np.all(base["cut_mask"]))
        np.testing.assert_allclose(
            base["crop_targets"][0, :, 0], [1.00, 1.02, 1.00, 1.02]
        )
        np.testing.assert_allclose(
            dynamic["values"][0, :, 0, 0], [1.00, 1.02, 1.00, 1.02]
        )
        np.testing.assert_allclose(
            dynamic["values"][1, :, 0, 0], [1.04, 1.06, 1.04, 1.06]
        )
        self.assertTrue(np.all(dynamic["track_mask"]))
        np.testing.assert_array_equal(dynamic["keyframe_count_targets"], 2)
        self.assertTrue(np.all(dynamic["presence_targets"][:, :, :2]))
        self.assertFalse(np.any(dynamic["presence_targets"][:, :, 2:]))
        self.assertEqual({"A": 2, "B": 2}, provenance["lane_repeat_counts"])
        self.assertEqual({"A": 0, "B": 0}, provenance["lane_padding_counts"])
        self.assertGreater(provenance["base_label_mask_counts"]["cut"], 0)
        self.assertGreater(provenance["dynamic_label_mask_counts"]["tracks"], 0)

    def test_dirty_development_run_preserves_runtime_pass_but_not_release_pass(self) -> None:
        candidate = default_dynamic_crop_v2_probe_candidates()[0]
        code = {"release_trust_authorized": False}
        provenance = {
            "candidate_contract_sha256": "1" * 64,
            "dataset_provenance_sha256": "2" * 64,
            "code_provenance_sha256": "3" * 64,
            "runtime_provenance_sha256": "4" * 64,
            "hardware_provenance_sha256": "5" * 64,
            "git_commit": "a" * 40,
        }
        events = [
            {
                "event": "provenance_verified",
                "candidate_id": candidate.candidate_id,
                "candidate_contract_sha256": provenance[
                    "candidate_contract_sha256"
                ],
                "expected_dataset_provenance_sha256": provenance[
                    "dataset_provenance_sha256"
                ],
                "observed_dataset_provenance_sha256": provenance[
                    "dataset_provenance_sha256"
                ],
                "expected_code_provenance_sha256": provenance[
                    "code_provenance_sha256"
                ],
                "observed_code_provenance_sha256": provenance[
                    "code_provenance_sha256"
                ],
                "expected_runtime_provenance_sha256": provenance[
                    "runtime_provenance_sha256"
                ],
                "observed_runtime_provenance_sha256": provenance[
                    "runtime_provenance_sha256"
                ],
                "expected_hardware_provenance_sha256": provenance[
                    "hardware_provenance_sha256"
                ],
                "observed_hardware_provenance_sha256": provenance[
                    "hardware_provenance_sha256"
                ],
                "expected_git_commit": provenance["git_commit"],
                "observed_git_commit": provenance["git_commit"],
                "matched": True,
            },
            {
                "event": "input_and_limits_ready",
                "tensor_provenance": {"real_input_materialized": True},
                "previous_memory_limit_bytes": BUDGET.physical_memory_bytes,
                "applied_memory_limit_bytes": BUDGET.allowed_training_bytes,
                "previous_cache_limit_bytes": 0,
                "applied_cache_limit_bytes": 512 * 1024**2,
                "memory_limit_contract": "mlx_limit_is_guidance_measured_peak_must_independently_fit_budget_v1",
                "pre_reset_active_memory_bytes": 0,
                "pre_reset_cache_memory_bytes": 0,
            },
            {"event": "input_materialized", "memory_stage": _memory_stage("input_materialized", 1.0)},
            {
                "event": "model_allocated",
                "memory_stage": _memory_stage("model_allocated", 2.0),
                "parameter_tree": {"leaf_count": 1},
            },
            {
                "event": "forward_backward_completed",
                "loss_value_for_execution_only": 1.0,
                "gradient_tree": {"leaf_count": 1},
                "memory_stage": _memory_stage("forward_backward_completed", 3.0),
            },
            {
                "event": "gradient_clip_completed",
                "gradient_global_norm_before_clip": 1.0,
                "memory_stage": _memory_stage("gradient_clip_completed", 4.0),
            },
            {
                "event": "optimizer_step_completed",
                "gradients_finite": True,
                "optimizer_state_materialized": True,
                "post_update_parameter_tree": {"leaf_count": 1},
                "updated_parameters_finite": True,
                "optimizer_state_tree": {"leaf_count": 3},
                "optimizer_state_finite": True,
                "memory_stage": _memory_stage("optimizer_step_completed", 5.0),
            },
        ]
        _, execution = _execution_from_process(
            candidate=candidate,
            budget=BUDGET,
            code_provenance=code,
            provenance_expectations=provenance,
            tensor_template={"real_input_materialized": False},
            events=events,
            returncode=0,
        )
        self.assertTrue(execution["runtime_execution_passed"])
        self.assertFalse(execution["passed"])
        self.assertEqual(
            "provenance_not_release_trusted", execution["failure_kind"]
        )
        self.assertEqual(
            PROBE_TRAINING_SCHEDULE["gradient_clip_norm"],
            execution["gradient_clip_norm"],
        )

        nonfinite_events = [dict(event) for event in events]
        nonfinite_events[-1] = {
            **nonfinite_events[-1],
            "post_update_parameter_tree": {
                "leaf_count": 1,
                "nonfinite_count": 1,
            },
            "updated_parameters_finite": False,
            "optimizer_state_finite": True,
        }
        _, nonfinite_execution = _execution_from_process(
            candidate=candidate,
            budget=BUDGET,
            code_provenance={"release_trust_authorized": True},
            provenance_expectations=provenance,
            tensor_template={"real_input_materialized": False},
            events=nonfinite_events,
            returncode=71,
        )
        self.assertFalse(nonfinite_execution["runtime_execution_passed"])
        self.assertFalse(nonfinite_execution["passed"])
        self.assertEqual(
            "non_finite_loss_or_gradients",
            nonfinite_execution["failure_kind"],
        )


if __name__ == "__main__":
    unittest.main()
