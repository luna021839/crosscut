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


def synthetic_mlx_head():
    """Real tiny CPU forward with explicitly synthetic readiness evidence."""
    from pathlib import Path
    import tempfile
    from unittest.mock import patch
    import mlx.core as mx
    from tests.test_stage_edit_head_materializer import StageEditHeadMaterializerTests
    from crosscut.stage_edit_dynamic_crop_head_v3 import StageEditTransformerDynamicCropV3, save_stage_edit_dynamic_crop_checkpoint_v3
    from crosscut.stage_edit_head_materializer import materialize_stage_edit_head_frames_v3
    from crosscut.stage_edit_transformer import StageEditReadinessReport, mlx_runtime_version
    fixture = StageEditHeadMaterializerTests()
    fixture.setUp()
    with tempfile.TemporaryDirectory() as directory, mx.stream(mx.cpu):
        path = Path(directory)
        saved = save_stage_edit_dynamic_crop_checkpoint_v3(path,
            StageEditTransformerDynamicCropV3(fixture.config), fixture.metadata_v3)
        readiness = StageEditReadinessReport(architecture_available=True, learned_inference_ready=True,
            status="production_checkpoint_ready", reason="synthetic worker serialization test only",
            mlx_version=mlx_runtime_version(), checkpoint_path=str(path.resolve()), training_steps=saved.training_steps,
            checkpoint_format="dynamic_crop_v3", dynamic_crop_version=3,
            checkpoint_metadata_filename="dynamic-crop-v3-metadata.json")
        with patch('crosscut.stage_edit_head_materializer.stage_edit_readiness_report', return_value=readiness):
            return materialize_stage_edit_head_frames_v3(checkpoint_directory=path, sequence=fixture.sequence,
                canonical_source_ids=('A','B'), derivative_family_id_by_source={'A':0,'B':1})
