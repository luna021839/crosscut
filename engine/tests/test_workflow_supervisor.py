import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from time import monotonic
from threading import Event
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from crosscut.workflow_supervisor import run_supervised_workflow


@unittest.skipUnless(os.name == "posix", "app requires POSIX supervision")
class WorkflowSupervisorTests(unittest.TestCase):
    def test_cli_wraps_default_and_strict_modes_in_the_same_absolute_deadline(self):
        import crosscut_cli
        for review_first in (True, False):
            with self.subTest(review_first=review_first), patch.object(
                crosscut_cli.AnalysisRequest, "from_json_file", return_value=SimpleNamespace(review_first_enabled=review_first)
            ), patch.object(crosscut_cli, "_prepare_analysis_process_group"), patch(
                "crosscut.workflow_supervisor.run_supervised_workflow", return_value=7
            ) as supervise, patch.object(crosscut_cli, "monotonic", return_value=100.):
                self.assertEqual(crosscut_cli.supervised_main(["analyze", "--request", "fixture.json"]), 7)
                command = supervise.call_args.args[0]
                self.assertEqual(command[-2:], ["--workflow-worker-deadline", "700.0"])
                self.assertEqual(supervise.call_args.kwargs["absolute_deadline_monotonic_seconds"], 700.)

    def test_native_block_and_ignored_term_are_killed_by_independent_supervisor(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "pid"
            code = (
                "import ctypes, os, signal; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"Path({str(pid_file)!r}).write_text(str(os.getpid())); "
                "ctypes.CDLL(None).sleep(20)"
            )
            started = monotonic()
            offset = [0.]
            ready_at = [None]
            real_popen = subprocess.Popen
            def ready_popen(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                until = monotonic() + 3.
                while not pid_file.exists() and monotonic() < until:
                    Event().wait(.005)
                if not pid_file.exists():
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    self.fail('native-block fixture never became ready')
                ready_at[0] = monotonic()
                # Separate interpreter startup from the 250ms native-block
                # fixture budget. Production clocks and limits are unchanged.
                offset[0] = ready_at[0] - started
                return process
            with patch('crosscut.workflow_supervisor.subprocess.Popen', side_effect=ready_popen), patch(
                'crosscut.workflow_supervisor.monotonic', side_effect=lambda: monotonic() - offset[0]
            ), self.assertRaises(subprocess.TimeoutExpired):
                run_supervised_workflow([sys.executable, "-c", code],
                                        absolute_deadline_monotonic_seconds=started + .25)
            self.assertIsNotNone(ready_at[0])
            self.assertLess(monotonic() - ready_at[0], 1.8)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid_file.read_text()), 0)

    def test_exited_parent_does_not_leave_term_ignoring_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "descendant"
            child = ("import os,signal,time; from pathlib import Path; "
                     "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                     f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(20)")
            parent = ("import subprocess,sys,time; from pathlib import Path; "
                      f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
                      f"p=Path({str(pid_file)!r})\nwhile not p.exists(): time.sleep(.01)")
            self.assertEqual(run_supervised_workflow([sys.executable, "-c", parent],
                absolute_deadline_monotonic_seconds=monotonic() + 2), 0)
            pid = int(pid_file.read_text())
            result = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True)
            self.assertTrue(not result.stdout.strip() or result.stdout.strip().startswith("Z"), result.stdout)

    def test_successful_workflow_preserves_exit_code_and_supervised_environment(self):
        result = run_supervised_workflow(
            [sys.executable, "-c", "import os,sys; sys.exit(7 if os.environ.get('IVE_CROSSCUT_SUPERVISED_WORKFLOW')=='1' else 9)"],
            absolute_deadline_monotonic_seconds=monotonic() + 2,
        )
        self.assertEqual(result, 7)

    def test_invalid_or_expired_deadline_never_starts_child(self):
        for value in (True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                run_supervised_workflow(["not-a-program"], absolute_deadline_monotonic_seconds=value)
        with self.assertRaises(subprocess.TimeoutExpired):
            run_supervised_workflow(["not-a-program"], absolute_deadline_monotonic_seconds=monotonic() - 1)
