"""Synthetic importable targets for the real optional native process boundary."""
from crosscut.optional_native_worker import isolated_optional

@isolated_optional()
def decorated_native_block(*, pid_file):
    return native_block(pid_file=pid_file)

def echo(*, value):
    return value


def returned_with_inherited_pipe(*, pid_file):
    import os
    import signal
    import time
    from pathlib import Path
    child = os.fork()
    if child == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(12.)
        os._exit(0)
    Path(pid_file).write_text(f'{child} {os.getpgrp()}')
    return 'leader-returned'


def noisy_echo(*, value):
    import os
    print('synthetic Python diagnostic', flush=True)
    os.write(1, b'synthetic native stdout diagnostic\n')
    return value


def fail():
    raise ValueError("injected model failure")


def phase_budget_calls(*, calls, event_file, exit_after=False):
    """Counter-reset control; the callback is NOT an actual neural model."""
    import os
    from pathlib import Path
    from time import monotonic
    from crosscut.group_joint_source_camera import JointGroupWorkBudget
    for index in range(calls):
        budget = JointGroupWorkBudget(absolute_deadline=monotonic()+30.)
        budget.spend("model_calls", model="synthetic-phase-observer")
        with Path(event_file).open("a") as stream:
            stream.write(str(index)+"\n")
    if exit_after:
        os._exit(7)
    return dict(callback_count=calls, actual_neural_model=False)


def phase_numeric_primitive():
    from time import monotonic
    from crosscut.group_joint_source_camera import JointGroupWorkBudget
    budget = JointGroupWorkBudget(absolute_deadline=monotonic()+30.)
    value = budget.measure_primitive(label="synthetic-numeric-return", input_binding={"source":0},
        work={"pose_comparisons":2}, maximum_result_bytes=128, producer=lambda:(1,2.))
    return dict(value=value, local_journal=budget.primitive_journal.to_dict())


def phase_blocked_primitive(*,pid_file):
    from time import monotonic
    from crosscut.group_joint_source_camera import JointGroupWorkBudget
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+30.)
    return budget.measure_primitive(label='explicit-nonhuman-blocked-native',input_binding={'source':0},
        work={'pose_comparisons':2},maximum_result_bytes=128,producer=lambda:native_block(pid_file=pid_file))


def phase_oversized_result():
    return b"x"*80000


def phase_chart_observations(*, domain, model_path, event_file, frames):
    """Real native decoder/session; explicitly non-human pixel-mean observer."""
    from pathlib import Path
    from time import monotonic
    from crosscut.group_joint_observations import GroupJointObservationSession
    from crosscut.group_joint_source_camera import JointGroupWorkBudget
    reports=[]
    for native in frames:
        # Deliberately reopen sessions. The local counts all restart at zero;
        # parent-owned admission must nevertheless retain prior work.
        budget=JointGroupWorkBudget(absolute_deadline=monotonic()+30.)
        with GroupJointObservationSession(domain=domain,budget=budget) as session:
            def observer(pixels):
                with Path(event_file).open("a") as stream:
                    stream.write(str(native)+"\n")
                return float(pixels.mean())
            session.register_model("nonhuman-chart-mean",model_path,observer)
            value=session.observe_native(0,native,model="nonhuman-chart-mean")
            reports.append(dict(value=value,diagnostics=session.diagnostics()))
    return dict(reports=reports,actual_neural_model=False,human_quality=False)


def native_exit_with_stderr():
    import os
    os.write(2, b'x' * 200_000)
    os.write(2, b'\nnative startup diagnostic retained\n')
    os._exit(7)


def unserializable_result():
    return lambda: 'unsupported callback result'


class SyntheticMemberLook:
    def apply(self, segments): return tuple(segments)


class SyntheticMemberCameraSpec:
    """Child-local model injection; production camera/provider plumbing remains real."""
    def __init__(self, pid_file, mode):
        self.pid_file, self.mode = pid_file, mode
    def validate_files(self): pass
    def open(self):
        import crosscut.pipeline as pipeline
        import crosscut.member_look_preflight as look
        from crosscut.editorial_motion import EditorialMotionPlan, EditorialMotionSegment
        def camera(**kwargs):
            kwargs['geometry_provider'](0, 0.)
            return EditorialMotionPlan(segments=(EditorialMotionSegment(0, 0, None),),
                boundary_match_count=0, zoom_event_count=0, pan_event_count=0,
                roll_event_count=0, blocked_dissolve_segment_count=0)
        pipeline.build_direct_member_editorial_motion_plan = camera
        look.build_member_look_proxy_proposal = lambda **kw: SyntheticMemberLook()
        return SyntheticMemberCameraProvider(self.pid_file, self.mode)


class SyntheticMemberCameraProvider:
    def __init__(self, pid_file, mode): self.pid_file, self.mode = pid_file, mode
    def sample(self, *_args):
        import os
        from pathlib import Path
        Path(self.pid_file).write_text(f'{os.getpid()} {os.getpgrp()}')
        if self.mode == 'block': return native_block(pid_file=self.pid_file)
        if self.mode == 'error': raise ValueError('synthetic member camera failure')
        return None
    def paired_samples(self, *args): return self.sample(), self.sample()
    def begin_focal_boundary_reference_capture(self): pass
    def focal_boundary_reference_capture(self): return ()
    def close(self): pass


def synthetic_boundary_descriptor(*_args):
    from tests.test_visual_boundary_repair import _descriptor
    return _descriptor(.5)


def synthetic_boundary_repair():
    """Real repair result, including its recursively immutable audit maps."""
    from pathlib import Path
    from crosscut.visual_score import VisualTimelinePlan
    from crosscut.visual_boundary_repair import repair_visual_timeline_boundaries
    from tests.test_visual_boundary_repair import _choice
    return repair_visual_timeline_boundaries(
        visual_timeline=VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 1.),
        sources=(Path('/synthetic/a'), Path('/synthetic/b')),
        audio_offsets=(0., 0.), scene_cut_times=((), ()),
        descriptor_provider=synthetic_boundary_descriptor)


def synthetic_visual_materialized():
    """Real visual DP/materialization; only decoded descriptors are synthetic."""
    from pathlib import Path
    from unittest.mock import patch
    from crosscut.timeline import ShotType
    from crosscut.visual_score import FrameVisualDescriptor, build_prescribed_visual_timeline_candidate_set
    from crosscut.visual_candidate_materialization import materialize_visual_timeline_candidates
    descriptor = FrameVisualDescriptor(
        quality_score=.9, composition_score=.9, shot_scale=.35,
        shot_type=ShotType.GROUP_WIDE, face_count=0,
        structure_embedding=(.1, .2, .3), motion=(0., 0.),
        color_embedding=(.2, .3, .4), detected_person_count=6,
        person_layout_embedding=(.1, .2, .3, .4), silhouette_embedding=(.1, .2),
        silhouette_bbox_xywh=(.2, .1, .6, .8), silhouette_centroid_xy=(.5, .5),
        silhouette_area_ratio=.4, silhouette_confidence=.9, silhouette_backend='synthetic')

    class Sampler:
        frame_rate = 30.
        def __init__(self, *_args, **_kwargs):
            pass
        def feature_descriptor_at(self, *_args, **_kwargs):
            return descriptor
        descriptor_at = feature_descriptor_at
        def sface_embedding_at(self, *_args, **_kwargs):
            return ()
        def close(self):
            pass

    with patch('crosscut.visual_score._VideoSampler', Sampler):
        candidate_set = build_prescribed_visual_timeline_candidate_set(
            sources=tuple(Path(f'/synthetic/source-{index}.mp4') for index in range(8)),
            canonical_spans=tuple((float(index), float(index+1)) for index in range(60)),
            prescribed_source_indices=tuple((index//4) % 8 for index in range(60)),
            audio_offsets=(0.,)*8, sync_scores=(.9,)*8,
            scene_cut_times=((),)*8, source_timewarps=(None,)*8)
        return materialize_visual_timeline_candidates(candidate_set,
            maximum_candidates=1, passed_only=True)[0]


def native_block(*, pid_file):
    import ctypes
    import os
    from pathlib import Path
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(pid_file).write_text(f"{os.getpid()} {os.getpgrp()}")
    ctypes.CDLL(None).sleep(30)


def native_descendant(*, pid_file):
    import subprocess
    import sys
    import time
    from pathlib import Path
    child = ("from tests.optional_worker_fixture import native_block; "
             f"native_block(pid_file={str(pid_file)!r})")
    subprocess.Popen([sys.executable, "-c", child])
    while not Path(pid_file).is_file():
        time.sleep(.005)
    time.sleep(30)


def synthetic_mlx_head(version=3):
    """Real tiny CPU forward with explicitly synthetic readiness evidence."""
    from pathlib import Path
    import tempfile
    from unittest.mock import patch
    import mlx.core as mx
    from tests.test_stage_edit_head_materializer import StageEditHeadMaterializerTests
    from crosscut.stage_edit_dynamic_crop_head_v3 import StageEditTransformerDynamicCropV3, save_stage_edit_dynamic_crop_checkpoint_v3
    from crosscut.stage_edit_dynamic_crop_head_v2 import StageEditTransformerDynamicCropV2, save_stage_edit_dynamic_crop_checkpoint_v2
    from crosscut.stage_edit_head_materializer import materialize_stage_edit_head_frames_v3,materialize_stage_edit_head_frames_v2
    from crosscut.stage_edit_transformer import StageEditReadinessReport, mlx_runtime_version
    fixture = StageEditHeadMaterializerTests()
    fixture.setUp()
    with tempfile.TemporaryDirectory() as directory, mx.stream(mx.cpu):
        path = Path(directory)
        if version not in (2,3):raise ValueError('explicit tiny v2/v3 fixture only')
        saved = (save_stage_edit_dynamic_crop_checkpoint_v3(path,
            StageEditTransformerDynamicCropV3(fixture.config), fixture.metadata_v3) if version==3 else
            save_stage_edit_dynamic_crop_checkpoint_v2(path,StageEditTransformerDynamicCropV2(fixture.config),fixture.metadata))
        readiness = StageEditReadinessReport(architecture_available=True, learned_inference_ready=True,
            status="production_checkpoint_ready", reason="synthetic worker serialization test only",
            mlx_version=mlx_runtime_version(), checkpoint_path=str(path.resolve()), training_steps=saved.training_steps,
            checkpoint_format=f"dynamic_crop_v{version}", dynamic_crop_version=version,
            checkpoint_metadata_filename=f"dynamic-crop-v{version}-metadata.json")
        with patch('crosscut.stage_edit_head_materializer.stage_edit_readiness_report', return_value=readiness):
            if version==3:return materialize_stage_edit_head_frames_v3(checkpoint_directory=path, sequence=fixture.sequence,
                canonical_source_ids=('A','B'), derivative_family_id_by_source={'A':0,'B':1})
            return materialize_stage_edit_head_frames_v2(checkpoint_directory=path,sequence=fixture.sequence,canonical_source_ids=('A','B'))
