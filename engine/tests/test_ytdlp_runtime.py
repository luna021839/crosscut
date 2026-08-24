from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from crosscut.ytdlp_runtime import discover_ytdlp_frontend


class YTDLPRuntimeTests(unittest.TestCase):
    @patch("crosscut.ytdlp_runtime.importlib.util.find_spec", return_value=object())
    def test_newer_python_module_beats_older_path_executable(self, _find_spec) -> None:
        def runner(command, **_kwargs):
            version = "2026.08.18.122307" if "-m" in command else "2026.03.17"
            return subprocess.CompletedProcess(command, 0, stdout=version + "\n", stderr="")

        frontend = discover_ytdlp_frontend(
            runner=runner,
            python_executable="/python3",
            path_executable="/yt-dlp",
        )
        assert frontend is not None
        self.assertEqual(frontend.command, ("/python3", "-m", "yt_dlp"))
        self.assertEqual(frontend.kind, "python_module")

    @patch("crosscut.ytdlp_runtime.importlib.util.find_spec", return_value=None)
    def test_path_executable_remains_supported_without_module(self, _find_spec) -> None:
        def runner(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, stdout="2026.06.09\n", stderr="")

        frontend = discover_ytdlp_frontend(
            runner=runner,
            python_executable="/python3",
            path_executable="/yt-dlp",
        )
        assert frontend is not None
        self.assertEqual(frontend.command, ("/yt-dlp",))
        self.assertEqual(frontend.version, "2026.06.09")


if __name__ == "__main__":
    unittest.main()
