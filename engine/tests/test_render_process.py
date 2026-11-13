import io
from pathlib import Path
import tempfile
import shutil
import os
import subprocess
import threading
from time import monotonic
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.render_process import run_observed_render,render_log_callback
from tests.process_fixture import wait_for_pid_file


class FakeProcess:
    pid=123
    def __init__(self,code=0):
        self.stdout=io.StringIO('frame=15\nout_time_us=500000\nspeed=1.2x\nprogress=continue\nframe=30\nout_time_us=1000000\nprogress=end\n')
        self.code=code;self.terminated=False
    def wait(self,timeout=None):return self.code
    def poll(self):return None
    def terminate(self):self.terminated=True


class RenderProcessTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg unavailable')
    def test_real_ffmpeg_output_reports_actual_encoded_frames(self):
        events=[]
        with tempfile.TemporaryDirectory() as tmp:
            result=run_observed_render([shutil.which('ffmpeg'),'-nostdin','-v','error','-f','lavfi',
                '-i','color=c=blue:s=32x32:r=10:d=0.5','-c:v','mpeg4',str(Path(tmp)/'actual.mp4')],
                stderr_path=Path(tmp)/'stderr',progress_callback=events.append)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertGreater((Path(tmp)/'actual.mp4').stat().st_size,0)
        self.assertEqual(events[-1]['phase'],'end')
        self.assertEqual(events[-1]['frame'],5)

    def test_actual_progress_is_streamed_but_nonzero_exit_is_preserved(self):
        for code in (0,1):
            fake=FakeProcess(code);events=[]
            with tempfile.TemporaryDirectory() as tmp,patch('crosscut.render_process.subprocess.Popen',return_value=fake) as popen:
                result=run_observed_render(['ffmpeg','-i','source.mp4','result.mp4'],stderr_path=Path(tmp)/'stderr',progress_callback=events.append)
            self.assertEqual(result.returncode,code)
            self.assertEqual([e['frame'] for e in events],[15,30])
            self.assertEqual(events[0]['out_time_us'],500000)
            self.assertEqual(popen.call_args.args[0][-1],'result.mp4')
            self.assertIn('pipe:1',popen.call_args.args[0])

    def test_callback_cancellation_terminates_child(self):
        fake=FakeProcess()
        def cancel(event):raise RuntimeError('cancelled')
        with tempfile.TemporaryDirectory() as tmp,patch('crosscut.render_process.subprocess.Popen',return_value=fake):
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                run_observed_render(['ffmpeg'],stderr_path=Path(tmp)/'stderr',progress_callback=cancel)
        self.assertTrue(fake.terminated)
        self.assertTrue(fake.stdout.closed)

    def test_stalled_callback_cannot_defeat_total_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "progress-child"
            ready_file = root / "ready"
            release_file = root / "release"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import os, time\nfrom pathlib import Path\n"
                f"Path({str(ready_file)!r}).write_text(str(os.getpid()))\n"
                f"while not Path({str(release_file)!r}).exists(): time.sleep(.005)\n"
                "print('frame=1', flush=True)\n"
                "print('progress=continue', flush=True)\n"
                "time.sleep(10)\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            blocker = threading.Event()
            callback_started = threading.Event()
            started = monotonic()
            offset = [0.]
            ready_at = [None]
            real_popen = subprocess.Popen
            def ready_popen(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                wait_for_pid_file(process, ready_file)
                ready_at[0] = monotonic()
                offset[0] = ready_at[0] - started
                release_file.touch()
                return process
            def block_callback(_event):
                callback_started.set()
                blocker.wait()
            try:
                with patch('crosscut.render_process.subprocess.Popen', side_effect=ready_popen), patch(
                    'crosscut.render_process.monotonic', side_effect=lambda: monotonic() - offset[0]
                ), self.assertRaises(subprocess.TimeoutExpired):
                    run_observed_render([str(executable)], stderr_path=root / "stderr",
                                        progress_callback=block_callback, timeout_seconds=.2)
            finally:
                blocker.set()
            self.assertTrue(callback_started.is_set(), 'startup timeout is not a stalled callback test')
            self.assertLess(monotonic() - ready_at[0], 3.0)

    @unittest.skipUnless(os.name == "posix", "POSIX process groups required")
    def test_exited_parent_is_not_held_by_grandchild_progress_pipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "inherited-pipe-parent"
            ready_file = root / "ready"
            release_file = root / "release"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import os, subprocess, sys, time\nfrom pathlib import Path\n"
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])\n"
                f"Path({str(ready_file)!r}).write_text(str(os.getpid()))\n"
                f"while not Path({str(release_file)!r}).exists(): time.sleep(.005)\n"
                "print('frame=1', flush=True)\n"
                "print('progress=end', flush=True)\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            started = monotonic()
            offset = [0.]
            ready_at = [None]
            real_popen = subprocess.Popen
            def ready_popen(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                wait_for_pid_file(process, ready_file)
                ready_at[0] = monotonic()
                offset[0] = ready_at[0] - started
                release_file.touch()
                return process
            with patch('crosscut.render_process.subprocess.Popen', side_effect=ready_popen), patch(
                'crosscut.render_process.monotonic', side_effect=lambda: monotonic() - offset[0]
            ):
                result = run_observed_render([str(executable)], stderr_path=root / "stderr",
                    progress_callback=lambda _event: None, timeout_seconds=2.)
            self.assertEqual(result.returncode, 0)
            self.assertLess(monotonic() - ready_at[0], 1.0)

    def test_app_log_uses_encoder_counts_not_quality_approval(self):
        logs=[];callback=render_log_callback(SimpleNamespace(log=logs.append),total_frames=100)
        for frame in (0,1,4,5,6,100):callback(dict(pid=1,frame=frame,phase='continue'))
        self.assertEqual(len(logs),3)
        self.assertTrue(all('품질 QA 별도' in line for line in logs))
        callback(dict(pid=2,frame=0,phase='continue'))
        self.assertEqual(len(logs),4)

    @unittest.skipUnless(os.name == 'posix', 'POSIX process supervision required')
    def test_unterminated_progress_line_cannot_block_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / 'partial-progress'
            ready_file = root / 'ready'
            executable.write_text('#!/usr/bin/env python3\nimport os,sys,time\nfrom pathlib import Path\n'
                'sys.stdout.write("frame=1");sys.stdout.flush()\n'
                f'Path({str(ready_file)!r}).write_text(str(os.getpid()))\ntime.sleep(20)\n')
            executable.chmod(0o755)
            started = monotonic()
            offset = [0.]
            ready_at = [None]
            real_popen = subprocess.Popen
            def ready_popen(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                wait_for_pid_file(process, ready_file)
                ready_at[0] = monotonic()
                offset[0] = ready_at[0] - started
                return process
            with patch('crosscut.render_process.subprocess.Popen', side_effect=ready_popen), patch(
                'crosscut.render_process.monotonic', side_effect=lambda: monotonic() - offset[0]
            ), self.assertRaises(subprocess.TimeoutExpired):
                run_observed_render([str(executable)], stderr_path=root / 'stderr',
                    progress_callback=lambda event: None, timeout_seconds=.2)
            self.assertIsNotNone(ready_at[0])
            self.assertLess(monotonic() - ready_at[0], 1.5)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(ready_file.read_text()), 0)


if __name__=='__main__':unittest.main()
