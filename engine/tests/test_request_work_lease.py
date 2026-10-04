from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import monotonic
import unittest
from unittest.mock import patch

from crosscut.first_result_controller import FirstResultController
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.optional_native_worker import run_isolated_python_call
from crosscut.render import RenderPlan, RenderSegment
from crosscut.request_work_lease import PHASES, RESOURCES, PhaseAllocation, RequestWorkExhausted
from crosscut.review_first_runtime import ReviewFirstRuntime


class RequestWorkLeaseTests(unittest.TestCase):
    def make(self, root, *, model_calls=2, model_loads=2, new_decodes=4, clock=None, journal_bytes=128*1024,
             payload_bytes=64*1024, source_path=None):
        runtime = ReviewFirstRuntime(enabled=True, **({} if clock is None else {"_clock":lambda:clock[0]}))
        plan = RenderPlan((root/"synthetic-source.mp4" if source_path is None else source_path,),
                          (RenderSegment(0,0.,1.,0.),),0,
                          "16:9",root/"not-rendered.mp4")
        controller = FirstResultController(runtime,sources=plan.sources,source_durations=(2.,),
            master_start=0.,duration=1.,journal_path=root/"owner.json")
        controller.install_baseline(plan,plan=plan,authority="synthetic-baseline")
        caps=dict(states=24,edges=128,dense_paths=4,new_decodes=new_decodes,model_calls=model_calls,model_loads=model_loads,
            pose_comparisons=10,pose_evidence_bytes=8192,region_flow_feature_updates=64,
            association_nodes=16,association_edges=32,workers=4,ipc_bytes=2*1024*1024,
            identity_reads=128,model_hash_bytes=1024*1024,source_probes=4,decoder_opens=4,
            journal_bytes=journal_bytes)
        allocations={p:PhaseAllocation(caps,10.,payload_bytes) for p in PHASES}
        totals={k:sum(a.resources[k] for a in allocations.values()) for k in RESOURCES}
        lease=controller.reserve_request_work(total_limits=totals,allocations=allocations,
                                              journal_path=root/"work-prefix.jsonl")
        self.addCleanup(lease.close)
        return controller,lease,plan

    def rows(self,lease):
        data=lease.journal_path.read_bytes().splitlines(keepends=True)
        prefix="0"*64
        rows=[]
        for index,raw in enumerate(data):
            row=json.loads(raw)
            self.assertEqual(row["sequence"],index)
            self.assertEqual(row["previous_sha256"],prefix)
            prefix=sha256(raw).hexdigest()
            rows.append(row)
        self.assertEqual(lease.to_dict()["journal_event_count"],len(rows))
        self.assertEqual(lease.to_dict()["journal_prefix_sha256"],prefix)
        return rows

    def test_live_parent_aggregates_new_local_sessions_and_keeps_future_reserves(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "crosscut.first_result_controller.enforce_review_first_deadline",return_value=nullcontext()):
            clock=[0.]
            controller,lease,plan=self.make(Path(directory),clock=clock)
            # Baseline control demonstrates the original local-only reset.
            for _ in range(3):
                budget=JointGroupWorkBudget(absolute_deadline=100.,maximum_model_calls=1,clock=lambda:0.)
                budget.spend("model_calls",model="control")
                self.assertEqual(budget.counts["model_calls"],1)
            entered=[]
            def operation():
                for index in range(3):
                    budget=JointGroupWorkBudget(absolute_deadline=100.,clock=lambda:0.)
                    budget.spend("model_calls",model="measured-test-admission")
                    entered.append(index)
            with self.assertRaises(RequestWorkExhausted) as caught:
                controller.run_work_phase("discovery",operation)
            self.assertTrue(caught.exception.partial)
            self.assertEqual(entered,[0,1])
            self.assertEqual(lease.ports["discovery"].charged["model_calls"],2)
            self.assertEqual(lease.states["parent"],"not_requested")
            self.assertEqual(lease.allocations["parent"].resources["model_calls"],2)
            self.assertIs(controller.incumbent,plan)
            with self.assertRaisesRegex(ValueError,"already consumed"):
                controller.run_work_phase("discovery",lambda:None)
            controller.run_work_phase("parent",lambda:JointGroupWorkBudget(
                absolute_deadline=100.,clock=lambda:0.).spend("model_calls",model="fresh-parent"))
            self.assertEqual(lease.ports["parent"].charged["model_calls"],1)
            self.rows(lease)

    def test_parent_authority_survives_real_worker_counter_resets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            controller,lease,plan=self.make(root)
            with self.assertRaises(RequestWorkExhausted) as caught:
                controller.run_work_phase("discovery",lambda:run_isolated_python_call(
                    "tests.optional_worker_fixture","phase_budget_calls",
                    dict(calls=3,event_file=str(root/"callbacks")),deadline=monotonic()+9.))
            self.assertTrue(caught.exception.partial)
            self.assertEqual((root/"callbacks").read_text().splitlines(),["0","1"])
            self.assertEqual(lease.ports["discovery"].model_calls,{"synthetic-phase-observer":2})
            self.assertEqual(lease.ports["discovery"].live_workers,{})
            rows=self.rows(lease)
            terminal=[r["payload"] for r in rows if r["kind"]=="worker_terminal"][-1]
            self.assertEqual(terminal["status"],"cutoff")
            self.assertTrue(terminal["computation_result_unknown"])
            self.assertFalse(terminal["budget_refunded"])
            self.assertIs(controller.incumbent,plan)

    def test_dead_worker_keeps_charges_prefix_and_cannot_mint_a_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            controller,lease,_=self.make(root)
            with self.assertRaisesRegex(RuntimeError,"exit code 7"):
                controller.run_work_phase("discovery",lambda:run_isolated_python_call(
                    "tests.optional_worker_fixture","phase_budget_calls",
                    dict(calls=2,event_file=str(root/"callbacks"),exit_after=True),deadline=monotonic()+9.))
            self.assertEqual(lease.ports["discovery"].charged["model_calls"],2)
            self.assertEqual(lease.ports["discovery"].charged["workers"],1)
            self.assertEqual(lease.ports["discovery"].live_workers,{})
            self.assertEqual(lease.states["discovery"],"failed")
            with self.assertRaisesRegex(ValueError,"already consumed"):
                controller.run_work_phase("discovery",lambda:None)
            terminal=[r["payload"] for r in self.rows(lease) if r["kind"]=="worker_terminal"][-1]
            self.assertTrue(terminal["function_entered"])
            self.assertTrue(terminal["computation_result_unknown"])

    def test_native_primitive_work_and_terminal_settlement_are_parent_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            controller,lease,_=self.make(Path(directory))
            value=controller.run_work_phase("selected_completion",lambda:run_isolated_python_call(
                "tests.optional_worker_fixture","phase_numeric_primitive",{},deadline=monotonic()+9.))
            self.assertEqual(value["value"],(1,2.))
            from crosscut.group_joint_primitive_journal import validate_journal
            validate_journal(value["local_journal"])
            port=lease.ports["selected_completion"]
            self.assertEqual(port.charged["pose_comparisons"],2)
            self.assertGreater(port.charged["pose_evidence_bytes"],0)
            self.assertEqual(port.reserved["pose_evidence_bytes"],0)
            self.assertEqual(port._reservations,{})
            self.assertTrue(any(r["kind"]=="primitive_terminal" for r in self.rows(lease)))
            self.assertFalse(lease.to_dict()["all_raw_results_journaled"])

    def test_ipc_serialization_failure_is_terminal_without_executing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            controller,lease,_=self.make(Path(directory))
            with self.assertRaises(Exception):
                controller.run_work_phase("heads",lambda:run_isolated_python_call(
                    "tests.optional_worker_fixture","echo",{"value":lambda:None},deadline=monotonic()+9.))
            port=lease.ports["heads"]
            self.assertEqual(port.charged["workers"],1)
            self.assertFalse(port.function_entered)
            terminal=[r["payload"] for r in self.rows(lease) if r["kind"]=="worker_terminal"][-1]
            self.assertFalse(terminal["function_entered"])
            self.assertFalse(terminal["budget_refunded"])

    def test_result_byte_limit_rejects_before_unbounded_parent_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            controller,lease,_=self.make(Path(directory))
            with self.assertRaisesRegex(RuntimeError,"pre-reserved byte bound"):
                controller.run_work_phase("heads",lambda:run_isolated_python_call(
                    "tests.optional_worker_fixture","phase_oversized_result",{},deadline=monotonic()+9.))
            self.assertEqual(lease.ports["heads"].live_workers,{})
            self.rows(lease)

    def test_missing_phase_and_failed_initial_admission_launch_zero_optional_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            controller,lease,_=self.make(root)
            calls=[]
            self.assertIsNone(controller.run_optional("old-rank",lambda:calls.append(True)))
            self.assertEqual(calls,[])
            self.assertEqual(controller.attempts[-1]["status"],"budget_not_started")
            with self.assertRaisesRegex(ValueError,"admitted once"):
                controller.reserve_request_work(total_limits={},allocations={},journal_path=root/"replacement.jsonl")

    def test_future_phase_wall_reserves_are_checked_before_start(self):
        with tempfile.TemporaryDirectory() as directory:
            clock=[0.]
            controller,lease,_=self.make(Path(directory),clock=clock)
            clock[0]=520.
            calls=[]
            with self.assertRaises(RequestWorkExhausted) as caught:
                controller.run_work_phase("discovery",lambda:calls.append(True))
            self.assertEqual(caught.exception.dimension,"protected_future_phase_wall")
            self.assertFalse(caught.exception.partial)
            self.assertEqual(calls,[])
            self.assertEqual(lease.states["discovery"],"not_requested")

    def test_whole_request_shortage_is_not_a_new_lease_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            old,lease,plan=self.make(root)
            controller=FirstResultController(old.runtime,sources=plan.sources,source_durations=(2.,),
                                            master_start=0.,duration=1.)
            controller.install_baseline(plan,plan=plan,authority="complete-original")
            totals=dict(lease.total_limits)
            totals["model_calls"]-=1
            with self.assertRaises(RequestWorkExhausted):
                controller.reserve_request_work(total_limits=totals,allocations=dict(lease.allocations),
                                                journal_path=root/"never-created.jsonl")
            self.assertFalse((root/"never-created.jsonl").exists())
            calls=[]
            self.assertIsNone(controller.run_optional("retry",lambda:calls.append(True)))
            self.assertEqual(calls,[])
            with self.assertRaisesRegex(ValueError,"admitted once"):
                controller.reserve_request_work(total_limits=dict(lease.total_limits),
                    allocations=dict(lease.allocations),journal_path=root/"new-budget.jsonl")

    def test_parent_journal_failure_denies_callback_and_retains_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            controller,lease,plan=self.make(root)
            actual=lease._append
            def failed(kind,payload,port=None,terminal_key=None):
                if kind=="work_admitted" and payload.get("amounts")=={"model_calls":1}:
                    lease.journal_failed=True
                    raise OSError("injected parent journal failure")
                return actual(kind,payload,port,terminal_key)
            with patch.object(lease,"_append",side_effect=failed):
                with self.assertRaisesRegex(OSError,"journal failed|journal failure"):
                    controller.run_work_phase("discovery",lambda:run_isolated_python_call(
                        "tests.optional_worker_fixture","phase_budget_calls",
                        dict(calls=1,event_file=str(root/"callbacks")),deadline=monotonic()+9.))
            self.assertFalse((root/"callbacks").exists())
            self.assertEqual(lease.ports["discovery"].charged["model_calls"],0)
            self.assertEqual(lease.states["discovery"],"journal_failed")
            self.assertIs(controller.incumbent,plan)

    def test_exact_scalar_resource_types_and_reserved_terminal_records(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "crosscut.first_result_controller.enforce_review_first_deadline",return_value=nullcontext()):
            controller,lease,_=self.make(Path(directory),clock=[0.])
            def controls():
                port=lease.active
                for value in (True,1.,-1):
                    with self.assertRaises(ValueError):
                        port.charge("model_calls",value,model="model")
                with self.assertRaisesRegex(ValueError,"forged"):
                    port.start("copied-ticket")
                for values in ({"model_calls":1,"pose_evidence_bytes":128},
                               {"ipc_bytes":1,"pose_evidence_bytes":128},
                               {"pose_comparisons":1}):
                    with self.assertRaisesRegex(ValueError,"native primitive"):
                        port.reserve(values)
                self.assertGreaterEqual(port.reserved["journal_bytes"],4096)
            controller.run_work_phase("discovery",controls)
            self.assertEqual(lease.ports["discovery"].charged["model_calls"],0)
            self.rows(lease)

    def test_short_journal_writes_are_completed_before_work_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            controller,lease,_=self.make(Path(directory))
            stream=lease._stream
            class ShortWriter:
                def write(self,data):return stream.write(data[:17])
                def fileno(self):return stream.fileno()
                def close(self):return stream.close()
            lease._stream=ShortWriter()
            controller.run_work_phase("discovery",lambda:lease.active.charge("model_calls",model="ack-after-durability"))
            self.assertEqual(lease.ports["discovery"].charged["model_calls"],1)
            self.rows(lease)

    def test_initial_header_bytes_are_accounted_and_preserved_when_terminal_phase_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            controller,lease,_=self.make(Path(directory))
            size=lease.journal_path.stat().st_size
            self.assertEqual(lease.ports['terminal'].charged['journal_bytes'],size)
            controller.run_work_phase('terminal',lambda:None)
            self.assertGreater(lease.ports['terminal'].charged['journal_bytes'],size)
            self.assertEqual(sum(p.charged['journal_bytes'] for p in lease.ports.values()),lease.journal_path.stat().st_size)
            self.rows(lease)

    def test_actual_tiny_v2_forward_and_checkpoint_load_are_admitted_by_parent(self):
        from crosscut.stage_edit_transformer import mlx_runtime_version
        if mlx_runtime_version() is None:self.skipTest('MLX unavailable')
        with tempfile.TemporaryDirectory() as directory:
            controller,lease,_=self.make(Path(directory))
            result=controller.run_work_phase('heads',lambda:run_isolated_python_call(
                'tests.optional_worker_fixture','synthetic_mlx_head',{'version':2},deadline=monotonic()+9.))
            self.assertEqual(result.dynamic_crop_version,2)
            self.assertEqual(len(result.frames),2)
            port=lease.ports['heads']
            self.assertEqual(port.charged['model_loads'],1)
            self.assertEqual(port.model_calls,{'stage_edit_dynamic_crop_v2':1})
            self.assertGreater(port.charged['model_hash_bytes'],0)
            self.rows(lease)

    def test_tiny_v2_no_forward_reserve_rejects_before_actual_model_call(self):
        from crosscut.stage_edit_transformer import mlx_runtime_version
        if mlx_runtime_version() is None:self.skipTest('MLX unavailable')
        with tempfile.TemporaryDirectory() as directory:
            controller,lease,_=self.make(Path(directory),model_calls=0)
            with self.assertRaises(RequestWorkExhausted) as caught:
                controller.run_work_phase('heads',lambda:run_isolated_python_call(
                    'tests.optional_worker_fixture','synthetic_mlx_head',{'version':2},deadline=monotonic()+9.))
            self.assertEqual(caught.exception.dimension,'model_calls')
            port=lease.ports['heads']
            self.assertEqual(port.charged['model_loads'],1)
            self.assertEqual(port.charged['model_calls'],0)
            self.assertFalse(any(r['payload'].get('amounts')=={'model_calls':1} for r in self.rows(lease)))

    def test_zero_journal_write_cannot_advance_prefix_or_execute_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            controller,lease,_=self.make(Path(directory));before=lease.sequence
            stream=lease._stream
            class ZeroWriter:
                def write(self,data):return 0
                def fileno(self):return stream.fileno()
                def close(self):return stream.close()
            lease._stream=ZeroWriter();calls=[]
            with self.assertRaisesRegex(OSError,"complete record"):
                controller.run_work_phase("discovery",lambda:calls.append(True))
            self.assertEqual(lease.sequence,before)
            self.assertTrue(lease.journal_failed)
            self.assertEqual(calls,[])

    def test_optional_cannot_swallow_phase_budget_and_record_success(self):
        with tempfile.TemporaryDirectory() as directory:
            controller,lease,_=self.make(Path(directory))
            def operation():
                return controller.run_optional("one-owner",lambda:run_isolated_python_call(
                    "tests.optional_worker_fixture","phase_budget_calls",
                    dict(calls=3,event_file=str(Path(directory)/"calls")),deadline=monotonic()+9.))
            with self.assertRaises(RequestWorkExhausted):controller.run_work_phase("discovery",operation)
            self.assertEqual(controller.attempts[-1]['status'],'budget_partial')
            self.assertEqual(lease.states['discovery'],'budget_partial')
            self.rows(lease)

    def test_phase_deadline_reaps_blocked_native_group_and_preserves_parent_final(self):
        from tests.process_fixture import assert_group_member_stopped
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);controller,lease,plan=self.make(root)
            started=monotonic()
            with self.assertRaises(RequestWorkExhausted) as caught:
                controller.run_work_phase('discovery',lambda:run_isolated_python_call(
                    'tests.optional_worker_fixture','native_descendant',
                    {'pid_file':str(root/'native-pid')},deadline=started+2.))
            self.assertTrue(caught.exception.partial)
            assert_group_member_stopped(self,*map(int,(root/'native-pid').read_text().split()))
            self.assertLess(monotonic()-started,3.5)
            self.assertEqual(lease.ports['discovery'].live_workers,{})
            self.assertEqual(lease.states['discovery'],'budget_partial')
            self.assertEqual(lease.states['parent'],'not_requested')
            self.assertEqual(lease.states['final'],'not_requested')
            self.assertIs(controller.incumbent,plan)
            self.rows(lease)

    def test_native_crash_keeps_unreturned_primitive_charges_and_durable_started_prefix(self):
        from tests.process_fixture import assert_group_member_stopped
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);controller,lease,_=self.make(root);started=monotonic()
            with self.assertRaises(RequestWorkExhausted):
                controller.run_work_phase('discovery',lambda:run_isolated_python_call(
                    'tests.optional_worker_fixture','phase_blocked_primitive',
                    {'pid_file':str(root/'native-pid')},deadline=started+2.))
            assert_group_member_stopped(self,*map(int,(root/'native-pid').read_text().split()))
            port=lease.ports['discovery']
            self.assertEqual(port.charged['pose_comparisons'],2)
            self.assertGreaterEqual(port.reserved['pose_evidence_bytes'],128)
            self.assertEqual(port.reserved['pose_evidence_bytes'],sum(
                row['amounts']['pose_evidence_bytes'] for row in port._reservations.values()))
            self.assertEqual(len(port._reservations),1)
            self.assertEqual(port.live_workers,{})
            rows=self.rows(lease)
            self.assertTrue(any(r['kind']=='primitive_started' for r in rows))
            self.assertFalse(any(r['kind']=='primitive_terminal' for r in rows))
            self.assertEqual(lease.states['discovery'],'budget_partial')

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"FFmpeg required")
    def test_real_group_decoder_and_observer_sessions_consume_one_parent_phase(self):
        from crosscut.group_source_domain import GroupSourceDomain
        from crosscut.render_frame_schedule import probe_render_frame_clock
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            video=root/"chart.mp4"
            subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","testsrc2=s=96x54:r=30",
                "-frames:v","90","-c:v","libx264","-crf","0","-pix_fmt","yuv420p",str(video)],
                capture_output=True,check=True,timeout=10.)
            clock=probe_render_frame_clock(video)
            domain=GroupSourceDomain((video,),(0,),(0.,),(None,),((),),(clock,))
            model=root/"nonhuman-fixture-model.bin"
            model.write_bytes(b"explicitly NOT actual weights")
            controller,lease,_=self.make(root,source_path=video)
            with self.assertRaises(RequestWorkExhausted) as caught:
                controller.run_work_phase("discovery",lambda:run_isolated_python_call(
                    "tests.optional_worker_fixture","phase_chart_observations",
                    dict(domain=domain,model_path=model,event_file=str(root/"observed"),frames=(0,1,2)),
                    deadline=monotonic()+9.))
            self.assertEqual(caught.exception.dimension,"model_calls")
            self.assertEqual((root/"observed").read_text().splitlines(),["0","1"])
            port=lease.ports["discovery"]
            self.assertEqual(port.charged["new_decodes"],3)
            self.assertEqual(port.charged["source_probes"],3)
            self.assertEqual(port.charged["decoder_opens"],3)
            self.assertEqual(port.model_calls,{"nonhuman-chart-mean":2})
            self.assertEqual(port.charged["model_hash_bytes"],3*model.stat().st_size)
            self.assertGreater(port.charged["identity_reads"],0)
            self.rows(lease)
