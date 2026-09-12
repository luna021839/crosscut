import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from crosscut.decoded_proxy_diagnostic import FailedProxyDiagnosticStore
from crosscut.rendered_candidate_selection import RenderedCandidate, render_one_candidate_review
from crosscut.rendered_proxy_review import RenderedProxyReview
from engine.tests.test_rendered_candidate_selection import _plan, _payload


class FailedProxyDiagnosticTests(unittest.TestCase):
    def test_failed_proxy_plan_and_review_survive_working_proxy_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sink = FailedProxyDiagnosticStore(root / 'diagnostics')
            working = []
            payload = _payload(passed=False, score=50., issues=.5, p10=40.)

            def render(plan, output):
                output.write_bytes(b'completed-render-fixture')
                working.append(output)
                return plan, RenderedProxyReview(payload)

            result = render_one_candidate_review(
                RenderedCandidate(0, _plan(0), 90.), scratch_directory=root,
                render_review=render, diagnostic_sink=sink)
            audit = result.review_payload['diagnostic_retention']
            self.assertFalse(result.review_payload['passed'])
            self.assertFalse(audit['quality_accepted'])
            self.assertTrue(audit['retained'])
            self.assertFalse(working[0].exists())
            video = Path(audit['path'])
            self.assertEqual(video.read_bytes(), b'completed-render-fixture')
            self.assertEqual(audit['video_sha256'], hashlib.sha256(video.read_bytes()).hexdigest())
            self.assertEqual(json.loads(Path(audit['review_path']).read_text()), payload)
            self.assertEqual(json.loads(Path(audit['render_plan_path']).read_text()),
                             json.loads(json.dumps(_plan(0).to_json())))
            second = render_one_candidate_review(
                RenderedCandidate(1, _plan(1), 90.), scratch_directory=root,
                render_review=render, diagnostic_sink=sink)
            self.assertFalse(second.review_payload['diagnostic_retention']['retained'])
            self.assertEqual(len(list((root / 'diagnostics').iterdir())), 1)
            self.assertFalse(working[1].exists())

    def test_passing_review_does_not_consume_slot_or_create_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'diagnostics'
            sink = FailedProxyDiagnosticStore(root)
            self.assertFalse(sink(Path('absent.mp4'), _plan(0), {'passed': True})['retained'])
            self.assertFalse(root.exists())
            self.assertFalse(sink._attempted)

    def test_size_limit_no_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'proxy.mp4'
            source.write_bytes(b'large')
            sink = FailedProxyDiagnosticStore(root / 'diagnostics', maximum_video_bytes=1)
            self.assertFalse(sink(source, _plan(0), {'passed': False})['retained'])
            self.assertFalse((root / 'diagnostics').exists())

    def test_retention_io_error_preserves_review_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            working = []
            def render(plan, output):
                output.write_bytes(b'complete')
                working.append(output)
                return plan, RenderedProxyReview({'passed': False, 'issue_reasons': ['color_shock']})
            def fail(*_args):
                raise OSError('SSD disconnected')
            result = render_one_candidate_review(
                RenderedCandidate(0, _plan(0), 90.), scratch_directory=Path(directory),
                render_review=render, diagnostic_sink=fail)
            self.assertEqual(result.review_payload['issue_reasons'], ['color_shock'])
            self.assertFalse(result.review_payload['passed'])
            self.assertFalse(result.review_payload['diagnostic_retention']['retained'])
            self.assertIsNone(result.error)
            self.assertFalse(working[0].exists())


if __name__ == '__main__':
    unittest.main()
