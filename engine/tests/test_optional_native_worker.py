import os
from pathlib import Path
import tempfile
import subprocess
import sys
from threading import Event
from time import monotonic
import unittest
from unittest.mock import patch

from crosscut.optional_native_worker import run_isolated_python_call, optional_native_scope, current_optional_deadline
from crosscut.review_first_runtime import ReviewFirstSearchCutoff
from tests.process_fixture import assert_group_member_stopped


@unittest.skipUnless(os.name == "posix", "native workers require POSIX")
class OptionalNativeWorkerTests(unittest.TestCase):
    def test_script_style_parent_without_pythonpath_or_project_cwd(self):
        engine_root = Path(__file__).resolve().parents[1]
        code = (
            "import sys,time; "
            f"sys.path.insert(0,{str(engine_root)!r}); "
            "from crosscut.optional_native_worker import run_isolated_python_call; "
            "print(run_isolated_python_call('builtins','str',{'object':'bootstrap-ok'},"
            "deadline=time.monotonic()+8.))"
        )
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-c', code], cwd=directory,
                                    env=environment, capture_output=True, text=True, timeout=10.)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'bootstrap-ok')

    def test_native_failure_keeps_bounded_stderr_without_corrupting_ipc(self):
        with self.assertRaisesRegex(RuntimeError, 'native startup diagnostic retained') as caught:
            run_isolated_python_call('tests.optional_worker_fixture', 'native_exit_with_stderr',
                                     {}, deadline=monotonic()+8.)
        self.assertIn('exit code 7', str(caught.exception))
        self.assertLess(len(str(caught.exception)), 8500)

    def test_actual_member_camera_entry_owns_provider_outside_analysis(self):
        from types import SimpleNamespace
        from crosscut.member_camera_worker import propose_member_camera_in_worker
        from crosscut.direct_member_visual import DirectMemberVisualConfig
        from crosscut.ive import AspectRatio
        from crosscut.render import RenderPlan, RenderSegment
        from tests.optional_worker_fixture import SyntheticMemberCameraSpec
        for mode in ('normal', 'error', 'block'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'camera-pid'
                plan = RenderPlan((Path('/synthetic/source'),), (RenderSegment(0, 0., 1., 0.),), 0,
                    AspectRatio.PORTRAIT, Path('/synthetic/out'), width=1080, height=1920, member_pose_dissolve_requested=True,
                    member_pose_dissolve_effective=False, member_pose_dissolve_count=0)
                arguments = dict(direct_visual_selection=SimpleNamespace(config=DirectMemberVisualConfig(),
                    source_indices=(0,), boundaries=(), selected_boundary_master_seconds=(), source_frame_geometries=(None,)),
                    segments=plan.segments, member_transitions=(), member_common_start=0., eligible_indices=(0,),
                    source_timewarps=(None,), offsets=(0.,), direct_full_frame_ratios=(9/16,), primitive_plan=plan)
                started = monotonic()
                with optional_native_scope(started+4.):
                    if mode == 'normal':
                        result = propose_member_camera_in_worker(provider_spec=SyntheticMemberCameraSpec(str(path), mode), arguments=arguments)
                        self.assertTrue(result[0].member_pose_dissolve_requested)
                        self.assertEqual(result[0].segments, plan.segments)
                    elif mode == 'error':
                        with self.assertRaisesRegex(RuntimeError, 'synthetic member camera failure'):
                            propose_member_camera_in_worker(provider_spec=SyntheticMemberCameraSpec(str(path), mode), arguments=arguments)
                    else:
                        with self.assertRaises(ReviewFirstSearchCutoff):
                            propose_member_camera_in_worker(provider_spec=SyntheticMemberCameraSpec(str(path), mode), arguments=arguments)
                pid, group = map(int, path.read_text().split())
                self.assertNotEqual(pid, os.getpid())
                assert_group_member_stopped(self, pid, group)
                self.assertLess(monotonic()-started, 5.5)

    def test_returned_worker_cannot_leave_inherited_pipe_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'child'
            started = monotonic()
            result = run_isolated_python_call('tests.optional_worker_fixture', 'returned_with_inherited_pipe',
                {'pid_file': str(path)}, deadline=started+5.)
            self.assertEqual(result, 'leader-returned')
            self.assertLess(monotonic()-started, 4.)
            assert_group_member_stopped(self, *map(int, path.read_text().split()))

    def test_real_worker_round_trips_typed_result_and_exceptions(self):
        value = {"result": (1, 2, Path("/synthetic/checkpoint"))}
        self.assertEqual(run_isolated_python_call("tests.optional_worker_fixture", "echo",
            {"value": value}, deadline=monotonic()+10.), value)
        large = b'private-in-memory-only' * 100000
        with patch('tempfile.TemporaryDirectory', side_effect=AssertionError('IPC must not persist descriptors')):
            self.assertEqual(run_isolated_python_call("tests.optional_worker_fixture", "noisy_echo",
                {"value": large}, deadline=monotonic()+10.), large)
        with self.assertRaisesRegex(RuntimeError, "injected model failure"):
            run_isolated_python_call("tests.optional_worker_fixture", "fail", {}, deadline=monotonic()+10.)

    def test_real_immutable_visual_results_cross_both_ipc_directions(self):
        from dataclasses import replace
        from types import MappingProxyType
        from crosscut.visual_boundary_repair import VisualBoundaryRepairPlan
        from crosscut.visual_candidate_materialization import MaterializedVisualTimelineCandidate
        for name, expected_type in (('synthetic_boundary_repair', VisualBoundaryRepairPlan),
                                    ('synthetic_visual_materialized', MaterializedVisualTimelineCandidate)):
            with self.subTest(name=name):
                value = run_isolated_python_call('tests.optional_worker_fixture', name, {},
                    deadline=monotonic()+15.)
                self.assertIsInstance(value, expected_type)
                clone = run_isolated_python_call('tests.optional_worker_fixture', 'echo', {'value': value},
                    deadline=monotonic()+15.)
                # Re-run the production constructors after the worker crossing.
                self.assertEqual(replace(clone), value)
                if isinstance(clone, VisualBoundaryRepairPlan):
                    audit = clone.repairs[0].result_edge_context
                    self.assertIsInstance(audit, MappingProxyType)
                    self.assertIsInstance(audit['outgoing'], MappingProxyType)
                else:
                    audit = clone.candidate_set.face_metadata_by_candidate
                    self.assertIsInstance(audit, MappingProxyType)
                    self.assertIs(clone.candidate_record, clone.candidate_set.candidates[0])
                with self.assertRaises(TypeError):
                    audit['mutation'] = True

    def test_decorated_repair_keeps_explicit_provider_and_readonly_audit(self):
        import pickle
        from types import MappingProxyType
        from crosscut.visual_score import VisualTimelinePlan
        from crosscut.visual_boundary_repair import repair_visual_timeline_boundaries
        from tests.test_visual_boundary_repair import _choice
        from tests.optional_worker_fixture import synthetic_boundary_descriptor
        kwargs = dict(visual_timeline=VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 1.),
            sources=(Path('/synthetic/a'), Path('/synthetic/b')),
            audio_offsets=(0., 0.), scene_cut_times=((), ()),
            descriptor_provider=synthetic_boundary_descriptor)
        strict = repair_visual_timeline_boundaries(**kwargs)
        with optional_native_scope(monotonic()+15.):
            isolated = repair_visual_timeline_boundaries(**kwargs)
        self.assertEqual(strict, isolated)
        self.assertEqual(isolated.reviewed_cross_source_count, 1)
        self.assertIsInstance(isolated.repairs[0].result_edge_context, MappingProxyType)
        # IPC support must not change global pickle policy or erase a provider.
        with self.assertRaises(TypeError):
            pickle.dumps(MappingProxyType({'private': 'memory-only'}))
        kwargs['descriptor_provider'] = lambda *_args: None
        with optional_native_scope(monotonic()+15.), self.assertRaises((AttributeError, TypeError)):
            repair_visual_timeline_boundaries(**kwargs)
        with self.assertRaisesRegex(RuntimeError, 'optional native call failed:.*pickle'):
            run_isolated_python_call('tests.optional_worker_fixture', 'unserializable_result', {},
                deadline=monotonic()+15.)

    def test_native_block_and_descendant_are_killed_at_optional_deadline(self):
        for name in ("native_block", "native_descendant"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "pid"
                started = monotonic()
                with self.assertRaises(ReviewFirstSearchCutoff):
                    run_isolated_python_call("tests.optional_worker_fixture", name,
                        {"pid_file": str(path)}, deadline=started+3.)
                self.assertTrue(path.is_file(), "fixture startup did not reach native blocking code")
                pid, group = map(int, path.read_text().split())
                assert_group_member_stopped(self, pid, group)
                self.assertLess(monotonic()-started, 4.5)

    def test_materializers_use_isolation_only_inside_optional_scope(self):
        from crosscut.stage_edit_head_materializer import materialize_stage_edit_head_frames_v2, materialize_stage_edit_head_frames_v3
        self.assertIsNone(current_optional_deadline())
        for function in (materialize_stage_edit_head_frames_v2, materialize_stage_edit_head_frames_v3):
            with self.subTest(function=function.__name__), optional_native_scope(123.), patch(
                    'crosscut.optional_native_worker.run_isolated_python_call', return_value="typed-result") as run:
                self.assertEqual(function(checkpoint_directory=Path('/synthetic/checkpoint'), sequence=None,
                                          canonical_source_ids=('a',)), "typed-result")
                self.assertEqual(run.call_args.args[:2], ('crosscut.stage_edit_head_materializer', function.__name__))
                self.assertEqual(run.call_args.kwargs['deadline'], 123.)
        self.assertIsNone(current_optional_deadline())

    def test_guard_terminates_native_work_when_owning_analysis_disappears(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native-pid"
            code = ("from crosscut.optional_native_worker import run_isolated_python_call; "
                    "from time import monotonic; "
                    f"run_isolated_python_call('tests.optional_worker_fixture','native_block',"
                    f"{{'pid_file':{str(path)!r}}}, deadline=monotonic()+15.)")
            parent = subprocess.Popen([sys.executable, '-c', code])
            try:
                until = monotonic()+10.
                while (not path.is_file() or len(path.read_text().split()) != 2) and monotonic() < until:
                    Event().wait(.01)
                self.assertTrue(path.is_file(), 'native fixture never started')
                pid, group = map(int, path.read_text().split())
                parent.kill()
                parent.wait(timeout=2.)
                assert_group_member_stopped(self, pid, group)
            finally:
                if parent.poll() is None:
                    parent.kill()
                parent.wait(timeout=2.)

    def test_actual_tiny_mlx_head_result_crosses_process_boundary(self):
        from crosscut.stage_edit_transformer import mlx_runtime_version
        from crosscut.stage_edit_head_materializer import StageEditHeadMaterializationResult
        if mlx_runtime_version() is None:
            self.skipTest('MLX unavailable')
        result = run_isolated_python_call('tests.optional_worker_fixture', 'synthetic_mlx_head', {}, deadline=monotonic()+20.)
        self.assertIsInstance(result, StageEditHeadMaterializationResult)
        self.assertEqual(result.dynamic_crop_version, 3)
        self.assertEqual(result.canonical_source_ids, ('A', 'B'))
        self.assertEqual(result.to_dict()['frame_count'], 2)

    def test_controller_alarm_and_native_guard_preserve_render_reserve(self):
        from crosscut.first_result_controller import FirstResultController
        from crosscut.review_first_runtime import ReviewFirstRuntime
        from crosscut.render import RenderPlan, RenderSegment
        from crosscut.ive import AspectRatio
        from tests.optional_worker_fixture import decorated_native_block
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'blocked-pid'
            runtime = ReviewFirstRuntime(enabled=True, total_budget_seconds=8.,
                first_render_budget_seconds=3., refinement_stop_seconds=4.)
            plan = RenderPlan(sources=(Path('/synthetic/a'),),
                segments=(RenderSegment(source_index=0, source_start=0., master_start=0., duration=1.),),
                reference_source=0, aspect_ratio=AspectRatio.LANDSCAPE, output=Path('/synthetic/out'))
            controller = FirstResultController(runtime, sources=plan.sources, source_durations=(2.,),
                master_start=0., duration=1.)
            controller.install_baseline(plan, plan=plan, authority='synthetic-baseline')
            result = controller.run_optional('native_camera', lambda: decorated_native_block(pid_file=str(path)))
            self.assertIsNone(result)
            self.assertIs(controller.incumbent, plan)
            self.assertTrue(path.is_file(), 'test must enter native blocking code')
            pid, group = map(int, path.read_text().split())
            assert_group_member_stopped(self, pid, group)
            self.assertGreater(runtime.total_budget_seconds - runtime.elapsed_seconds, 3.)
            self.assertEqual(controller.to_dict()['attempts'][0]['status'], 'search_cutoff')

    def test_all_opted_in_native_boundaries_dispatch_before_execution(self):
        from crosscut.boundary_affine_plan import build_boundary_affine_plan
        from crosscut.boundary_subject_alignment import build_boundary_subject_alignment_plan
        from crosscut.visual_boundary_repair import repair_visual_timeline_boundaries
        from crosscut.ive_causal_sensing_pipeline_inputs import materialize_ive_causal_sensing_for_stage_edit_pipeline
        from crosscut.stage_edit_inference_snapshot import materialize_stage_edit_inference_snapshot
        from crosscut.stage_edit_visual_candidate_adapter import materialize_stage_edit_visual_candidate
        from crosscut.visual_score import rehydrate_visual_timeline_dissolve_evidence
        from crosscut.source_look_match import analyze_source_look_match
        import inspect
        for function in (build_boundary_affine_plan, build_boundary_subject_alignment_plan,
                         repair_visual_timeline_boundaries, materialize_ive_causal_sensing_for_stage_edit_pipeline,
                         materialize_stage_edit_inference_snapshot, materialize_stage_edit_visual_candidate,
                         rehydrate_visual_timeline_dissolve_evidence, analyze_source_look_match):
            with self.subTest(function=function.__name__), optional_native_scope(123.), patch(
                    'crosscut.optional_native_worker.run_isolated_python_call', return_value='isolated') as run:
                kwargs = {name: None for name, parameter in inspect.signature(function).parameters.items()
                          if parameter.default is inspect.Parameter.empty}
                self.assertEqual(function(**kwargs), 'isolated')
                self.assertEqual(run.call_args.args[:2], (function.__module__, function.__name__))

    def test_expired_or_invalid_deadline_cannot_launch(self):
        with self.assertRaises(ReviewFirstSearchCutoff):
            run_isolated_python_call("no-module", "no-call", {}, deadline=monotonic()-1)
        for value in (True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                run_isolated_python_call("no-module", "no-call", {}, deadline=value)
