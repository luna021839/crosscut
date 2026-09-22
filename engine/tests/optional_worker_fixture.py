"""Synthetic importable targets for the real optional native process boundary."""
from crosscut.optional_native_worker import isolated_optional

@isolated_optional()
def decorated_native_block(*, pid_file):
    return native_block(pid_file=pid_file)

def echo(*, value):
    return value


def noisy_echo(*, value):
    import os
    print('synthetic Python diagnostic', flush=True)
    os.write(1, b'synthetic native stdout diagnostic\n')
    return value


def fail():
    raise ValueError("injected model failure")


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
