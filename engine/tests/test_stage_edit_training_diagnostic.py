"""Small diagnostic-instrument tests, not a substitute for the real model run."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as Item
import unittest
from unittest import mock


MODULE_PATH=Path(__file__).parents[2]/"tools/diagnose_stage_edit_training.py"
spec=importlib.util.spec_from_file_location("training_diagnostic_tool",MODULE_PATH)
diagnostic=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def tracks():
    return [Item(song_id=song,token_index=index,source_id=str(index),
        values=((1.02,.5,.5,0.),)*count, motion_label=motion,
        motion_label_loss_masked=False)
        for song in diagnostic.TRAIN_REFERENCES
        for index,count,motion in ((0,1,"static"),(1,2,"dynamic"),(2,3,"dynamic"),
            (3,2,"static"),(4,4,"dynamic"),(5,2,"static"),(6,1,"static"))]


class TrainingDiagnosticInstrumentTests(unittest.TestCase):
    def test_fixture_is_fixed_label_stratified_and_deterministic(self):
        values=tracks()
        for value in values:
            if value.motion_label=="dynamic":
                value.values=tuple((1.02+i*.01,.5,.5,0.) for i in range(len(value.values)))
        expected=diagnostic.select_fixture(values)
        actual=diagnostic.select_fixture(tuple(reversed(values)))
        self.assertEqual(expected,actual)
        self.assertEqual(len(actual),12)
        self.assertEqual({v.song_id for v in actual},set(diagnostic.TRAIN_REFERENCES))
        self.assertEqual({len(v.values) for v in actual},{1,2,3,4})

    def test_no_motion_range_fails_instead_of_moving_split(self):
        with self.assertRaisesRegex(ValueError,"no_target_trajectory_range"):
            diagnostic.select_fixture(tracks())

    def test_insufficient_real_song_data_fails(self):
        with self.assertRaisesRegex(ValueError,"insufficient real train"):
            diagnostic.select_fixture(tracks()[:4])

    def test_immutable_json_digest_is_not_optional_when_supplied(self):
        with self.assertRaisesRegex(ValueError,"digest mismatch"):
            diagnostic.read_bound_json(MODULE_PATH,"0"*64)

    def test_historical_input_requires_both_exact_path_and_expected_old_digest(self):
        binding={"direct_inputs":[dict(kind="snapshot",reference_id="A",path=str(MODULE_PATH),sha256="0"*64)]}
        with self.assertRaisesRegex(ValueError,"digest mismatch"):
            diagnostic.read_historical_input(MODULE_PATH,kind="snapshot",reference_id="A",binding=binding)
        with mock.patch.object(diagnostic,"read_bound_json") as read:
            with self.assertRaisesRegex(ValueError,"path/reference binding mismatch"):
                diagnostic.read_historical_input(MODULE_PATH.with_suffix(".wrong"),kind="snapshot",reference_id="A",binding=binding)
            read.assert_not_called()

    def test_historical_fixture_rejects_changed_targets_partition_and_config_instead_of_refitting(self):
        import copy
        binding=json.loads(diagnostic.HISTORICAL_BINDING.read_text())
        # A deliberately tiny test fixture uses the same verifier, but never
        # updates the committed real-data authority or reads external sources.
        evidence={key:copy.deepcopy(binding[key]) for key in (
            "plan_sha256","config_report_sha256","train_reference_ids","data_id")}
        evidence["tracks"]=[dict(count=2,motion="dynamic",values=[[1.02,.5,.5,0.]])]
        binding["selected_tracks_sha256"]=diagnostic.hashlib.sha256(json.dumps(
            evidence["tracks"],sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        diagnostic.require_historical_fixture_identity(evidence,binding)
        for key in ("plan_sha256","config_report_sha256","train_reference_ids","data_id","tracks"):
            changed=copy.deepcopy(evidence)
            if key=="tracks":changed[key][0]["count"]=3
            elif key=="train_reference_ids":changed[key]=["validation-reference"]
            else:changed[key]="changed"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,"historical diagnostic fixture"):
                diagnostic.require_historical_fixture_identity(changed,binding)

    def test_budget_is_bounded_without_starting_watchdog(self):
        import mlx.core as mx
        budget=diagnostic.DiagnosticBudget(Path("/unwritten"))
        budget.start-=budget.seconds+1
        with self.assertRaisesRegex(RuntimeError,"budget exceeded"):
            budget.check()

    def test_budget_includes_process_start_and_does_not_import_mlx_to_watch_preparation(self):
        process=Item(create_time=lambda:970.,memory_info=lambda:Item(rss=123))
        with mock.patch("psutil.Process",return_value=process), \
                mock.patch.object(diagnostic.time,"time",return_value=1000.), \
                mock.patch.object(diagnostic.time,"monotonic",return_value=500.), \
                mock.patch.dict(diagnostic.sys.modules,{"mlx.core":None}):
            budget=diagnostic.DiagnosticBudget(Path("/unwritten"))
            self.assertEqual(budget.start,470.)
            self.assertEqual(budget.check(),dict(elapsed_seconds=30.,rss_bytes=123,mlx_peak_bytes=None))

    def test_mlx_peak_is_included_when_runtime_becomes_available(self):
        budget=diagnostic.DiagnosticBudget(Path("/unwritten"))
        runtime=Item(get_peak_memory=lambda:budget.mlx_bytes+1)
        with mock.patch.dict(diagnostic.sys.modules,{"mlx.core":runtime}):
            with self.assertRaisesRegex(RuntimeError,"budget exceeded"):
                budget.check()

    def test_watchdog_starts_before_prepare_and_same_budget_reaches_run_or_prepare_only(self):
        fixture=(None,None,None,None,Item(tracks=(1,2)),{})
        for prepare_only in (False,True):
            with self.subTest(prepare_only=prepare_only), tempfile.TemporaryDirectory() as directory:
                out=Path(directory);budget=mock.Mock(start=diagnostic.time.monotonic())
                budget.seconds=600;budget.rss_bytes=5*2**30;budget.mlx_bytes=int(4.5*2**30)
                budget.check.return_value={"elapsed_seconds":.1,"rss_bytes":12,"mlx_peak_bytes":None}
                watcher=mock.Mock()
                def prepare():
                    watcher.start.assert_called_once()
                    self.assertGreater(budget.check.call_count,0)
                    return fixture
                with mock.patch.object(diagnostic,"DiagnosticBudget",return_value=budget), \
                        mock.patch.object(diagnostic.threading,"Thread",return_value=watcher), \
                        mock.patch.object(diagnostic,"prepare_fixture",side_effect=prepare), \
                        mock.patch.object(diagnostic,"run_diagnostic") as run:
                    diagnostic.execute_diagnostic(out,prepare_only=prepare_only)
                if prepare_only:run.assert_not_called()
                else:run.assert_called_once_with(fixture,out,budget=budget)
                budget.done.set.assert_called_once()
                watcher.join.assert_called_once_with(timeout=1.)
                self.assertEqual(json.loads((out/"fixture.json").read_text()),{})

    def test_prepare_failure_stops_watchdog_and_preserves_failure_without_starting_model(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);budget=mock.Mock(start=diagnostic.time.monotonic())
            watcher=mock.Mock()
            with mock.patch.object(diagnostic,"DiagnosticBudget",return_value=budget), \
                    mock.patch.object(diagnostic.threading,"Thread",return_value=watcher), \
                    mock.patch.object(diagnostic,"prepare_fixture",side_effect=ValueError("fixture mismatch")), \
                    mock.patch.object(diagnostic,"run_diagnostic") as run:
                with self.assertRaisesRegex(ValueError,"fixture mismatch"):
                    diagnostic.execute_diagnostic(out)
            run.assert_not_called()
            budget.done.set.assert_called_once()
            watcher.join.assert_called_once_with(timeout=1.)
            self.assertIn("fixture mismatch",json.loads((out/"failure.json").read_text())["error"])

    def test_missing_authority_does_not_hide_original_failure_or_accept_current_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);budget=mock.Mock(start=diagnostic.time.monotonic())
            with mock.patch.object(diagnostic,"DiagnosticBudget",return_value=budget), \
                    mock.patch.object(diagnostic.threading,"Thread"), \
                    mock.patch.object(diagnostic,"HISTORICAL_BINDING",out/"missing.json"), \
                    mock.patch.object(diagnostic,"prepare_fixture",side_effect=FileNotFoundError("missing authority")), \
                    mock.patch.object(diagnostic,"run_diagnostic") as run:
                with self.assertRaisesRegex(FileNotFoundError,"missing authority"):
                    diagnostic.execute_diagnostic(out)
            failure=json.loads((out/"failure.json").read_text())
            self.assertIsNone(failure["historical_binding_sha256"])
            self.assertIn("missing authority",failure["error"])
            run.assert_not_called()
            budget.done.set.assert_called_once()

    def test_zero_tree_has_finite_norm(self):
        self.assertEqual(diagnostic.tree_norm({}),0.)


if __name__=="__main__":unittest.main()
