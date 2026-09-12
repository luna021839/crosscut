from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

from crosscut.ive_causal_sensing_media_job import IVECausalSensingMediaRequest
from crosscut.stage_edit_reference_causal_sensing import (
    materialize_reference_causal_sensing, reference_sensing_sources,
)
from crosscut.stage_edit_feature_snapshot_job import (
    _reference_scaffold, build_stage_edit_feature_snapshots,
)
from tests.test_stage_edit_feature_snapshot_job import _shifted_media_payload

MODULE = 'crosscut.stage_edit_reference_causal_sensing.'


def source(offset=0.):
    return SimpleNamespace(source_id='A', path=Path('/unused/A.mp4'),
        content_sha256='a'*64, frame_rate=10., declared_frame_count=20,
        clock={'basis': 'test-clock', 'global_offset_seconds': offset})


class ReferenceSensingTests(unittest.TestCase):
    def test_exact_clock_and_native_frame_edge_mask(self):
        times = (0., .1, 1.8, 1.9, 2.)
        typed, = reference_sensing_sources([source(-.04)], times, {'A': (.5,)})
        self.assertEqual(typed.unavailable_master_times, (0., 1.9, 2.))
        self.assertEqual(typed.native_cut_source_seconds, (.5,))
        self.assertAlmostEqual(typed.clock.master_to_source(.1), .06)
        request = IVECausalSensingMediaRequest(song='BANG BANG', edit_mode='group',
            sources=(typed,), master_decision_times=times,
            gallery_artifact_path=Path('/unused/gallery.json'), gallery_content_sha256='b'*64,
            artifact_directory=Path('/unused/output'), created_at_utc='2026-09-13T00:00:00Z')
        self.assertEqual(request.sources[0], typed)
        with self.assertRaisesRegex(ValueError, 'unavailable mask|availability'):
            replace(request, sources=(replace(typed, unavailable_master_times=()),))

    def test_piecewise_mapping_not_replaced_by_scalar_offset(self):
        item = source()
        item.clock = {'basis': 'test', 'control_points': [
            {'master_seconds': 0., 'source_seconds': .1},
            {'master_seconds': 2., 'source_seconds': 1.5}]}
        typed, = reference_sensing_sources([item], (.1, .5, 1.), {'A': ()})
        self.assertAlmostEqual(typed.clock.master_to_source(1.), .8)

    def test_missing_native_cut_evidence_is_not_filled_with_empty_list(self):
        with self.assertRaisesRegex(ValueError, 'cover every'):
            reference_sensing_sources([source()], (.1, .5), {})

    def test_materializer_receives_only_source_grid_not_reference_labels(self):
        installation = SimpleNamespace(gallery_artifact_path=Path('/unused/gallery.json'),
                                       gallery_content_sha256='b'*64)
        with patch(MODULE+'discover_ive_causal_sensing_installation', return_value=installation), \
             patch(MODULE+'detect_scene_changes_in_video_cached',
                   return_value=((SimpleNamespace(frame_index=5),), {})), \
             patch(MODULE+'materialize_ive_causal_sensing_from_media', return_value='result') as run:
            result = materialize_reference_causal_sensing(song='BANG BANG', sources=[source()],
                master_times=(.1, .5, 1.), artifact_directory=Path('/unused/output'))
        self.assertEqual(result, 'result')
        request = run.call_args.args[0]
        self.assertEqual(request.master_decision_times, (.1, .5, 1.))
        self.assertEqual(request.sources[0].native_cut_source_seconds, (.5,))
        self.assertFalse(hasattr(request, 'shot_labels'))

    def test_quarantine_precedes_gallery_and_source_access(self):
        with patch(MODULE+'discover_ive_causal_sensing_installation') as gallery, \
             patch(MODULE+'detect_scene_changes_in_video_cached') as scan:
            with self.assertRaisesRegex(ValueError, 'quarantined'):
                materialize_reference_causal_sensing(song='ATTITUDE', sources=[source()],
                    master_times=(.1,), artifact_directory=Path('/unused/output'))
            gallery.assert_not_called()
            scan.assert_not_called()

    def test_feature_job_passes_its_exact_scaffold_grid_to_auto_sensing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = (root/'A.mp4', root/'B.mp4')
            for path in paths:
                path.write_bytes(b'test source')
            payload = _shifted_media_payload('ref-auto', paths)
            supervision = root/'supervision.json'
            supervision.write_text(json.dumps(payload))
            scaffold = _reference_scaffold(payload, song_id='BANG BANG',
                negative_samples_per_hold=1, boundary_guard_seconds=.2, max_sources=8)
            with patch(MODULE+'materialize_reference_causal_sensing',
                       side_effect=RuntimeError('capture exact request')) as materialize:
                with self.assertRaisesRegex(RuntimeError, 'capture exact request'):
                    build_stage_edit_feature_snapshots([supervision], song_id='BANG BANG',
                        output_directory=root/'output', negative_samples_per_hold=1,
                        boundary_guard_seconds=.2, max_sources=8, auto_causal_sensing=True)
            self.assertEqual(materialize.call_args.kwargs['master_times'], tuple(sorted(set(scaffold.times))))
            self.assertEqual(len(materialize.call_args.kwargs['sources']), 2)


if __name__ == '__main__':
    unittest.main()
