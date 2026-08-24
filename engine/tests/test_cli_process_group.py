from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import unittest


@unittest.skipUnless(os.name == "posix", "POSIX process-group contract")
class CLIProcessGroupTests(unittest.TestCase):
    def test_group_term_reaches_media_child_and_unwinds_cli(self) -> None:
        engine_root = Path(__file__).resolve().parents[1]
        program = textwrap.dedent(
            """
            import json
            import os
            import subprocess
            import sys

            from crosscut_cli import _prepare_analysis_process_group

            assert _prepare_analysis_process_group()
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
            )
            print(json.dumps({"pid": os.getpid(), "child": child.pid}), flush=True)
            try:
                child.wait()
            except KeyboardInterrupt:
                child.wait(timeout=5)
                raise SystemExit(130)
            """
        )
        process = subprocess.Popen(
            [sys.executable, "-B", "-c", program],
            cwd=engine_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(lambda: process.poll() is None and process.kill())
        assert process.stdout is not None
        payload = json.loads(process.stdout.readline())
        self.assertEqual(payload["pid"], process.pid)
        self.assertEqual(os.getpgid(process.pid), process.pid)

        os.killpg(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=8)

        self.assertEqual(process.returncode, 130, stderr or stdout)
        with self.assertRaises(ProcessLookupError):
            os.kill(int(payload["child"]), 0)


if __name__ == "__main__":
    unittest.main()
