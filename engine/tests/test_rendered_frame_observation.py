"""Decoded PTS/raster contracts; synthetic pixels are not AV quality labels."""
from fractions import Fraction
import json
import shutil
import subprocess
import numpy as np
import pytest
from crosscut.rendered_frame_observation import _parsed_pts,observe_rendered_frames


def test_exact_integer_ticks_not_rounded_seconds_and_invalid_census():
    text='config in time_base: 1/30000, frame_rate: 30000/1001\n'
    text+='\n'.join(f'n: {i} pts: {90000+i*1001} pts_time: 0' for i in range(3))
    ticks,base=_parsed_pts(text,3)
    assert ticks.tolist()==[90000,91001,92002] and base==Fraction(1,30000)
    assert not ticks.flags.writeable
    for bad in (text.replace('n: 1','n: 0'),text.replace('91001','90000'),
                text+'\nconfig in time_base: 1/1000',text.replace('1/30000','1/0')):
        with pytest.raises(ValueError):_parsed_pts(bad,3)


@pytest.mark.parametrize('options',[{'expected_frames':True},{'expected_frames':20001},
    {'expected_frames':100,'stride':0},{'expected_frames':100,'max_rasters':1},
    {'expected_frames':100,'timeout_seconds':float('inf')},
    {'expected_frames':100,'width':100000000}])
def test_resource_contract_rejects_before_launch(tmp_path,monkeypatch,options):
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('must not launch'))
    with pytest.raises(ValueError):observe_rendered_frames(tmp_path/'absent.mp4',**options)


def test_one_pass_matches_independent_tiny_ffprobe_and_nonzero_origin(tmp_path):
    ffmpeg,ffprobe=shutil.which('ffmpeg'),shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:pytest.skip('optional local codec tools unavailable')
    path=tmp_path/'synthetic-nonzero.mkv'
    subprocess.run([ffmpeg,'-nostdin','-v','error','-n','-f','lavfi','-i','testsrc2=size=64x36:rate=24',
        '-vf','setpts=PTS+2/TB','-frames:v','24','-c:v','ffv1','-fps_mode','passthrough',str(path)],
        check=True,capture_output=True,timeout=10)
    expected=json.loads(subprocess.run([ffprobe,'-v','error','-select_streams','v:0','-show_entries',
        'frame=best_effort_timestamp_time','-of','json',str(path)],check=True,capture_output=True,timeout=10).stdout)
    observed=observe_rendered_frames(path,expected_frames=24,stride=8,width=32,height=18,ffmpeg=ffmpeg,timeout_seconds=10)
    np.testing.assert_allclose(observed.pts_ticks*float(observed.time_base),
        [float(f['best_effort_timestamp_time']) for f in expected['frames']],atol=1e-9,rtol=0)
    assert observed.selected_indexes.tolist()==[0,8,16]
    assert observed.selected_seconds[0]>=2. and observed.grayscale.shape==(3,18,32)
    assert not observed.grayscale.flags.writeable and not observed.selected_indexes.flags.writeable
    with pytest.raises(ValueError,match='census'):
        observe_rendered_frames(path,expected_frames=23,stride=8,width=32,height=18,ffmpeg=ffmpeg,timeout_seconds=10)
