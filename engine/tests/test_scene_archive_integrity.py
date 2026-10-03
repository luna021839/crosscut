"""Re-signed referenced-archive omissions; real NON-HUMAN parent replay.

Explicit chart detector outputs are not human inference or quality approval.
The existing decode, completion and independent replay are not substituted.
Internal reference closure is NOT a parent-owned pre-receipt history journal.
"""
from copy import deepcopy
from dataclasses import replace
from time import monotonic
import shutil
import unittest
from unittest.mock import patch

from crosscut import group_performer_family_owner as family
from crosscut import group_source_changing_owner as source
from crosscut.optional_native_worker import optional_native_scope
from engine.tests import test_scene_source_owner as charts


def resign_archive(candidate, change):
    receipt = deepcopy(family.plain(candidate.performer_family_review))
    dense = receipt['source_camera_refresh']
    archive = dense['scene_discovery']
    if change == 'empty':
        dense['scene_discovery'] = {}
    elif change == 'selected_anchor':
        missing = dense['scene_hypotheses'][dense['hypothesis_keys'][0]]['raw_observation_sha256'][0]
        archive['native_rows'] = [row for row in archive['native_rows'] if family.digest(row) != missing]
    elif change == 'baseline_hypothesis':
        missing = next(key for group in receipt['selected_scene_program']['baseline_hypothesis_keys']
            for key in group)
        del archive['hypotheses'][missing]
    elif change == 'variant_inventory':
        variant = next(variant for query in reversed(archive['discoveries'])
            for variant in query['variants'] if variant['heads_by_anchor'][0])
        variant['heads_by_anchor'][0].pop()
    elif change == 'authority':
        archive['quality_approved'] = True
    else:
        raise AssertionError(change)
    program = receipt['selected_scene_program']
    if 'discovery_archive_binding' in program:
        from crosscut.group_scene_archive import archive_binding
        program['discovery_archive_binding'] = archive_binding(dense['scene_discovery'])
    program.pop('program_sha256')
    program['program_sha256'] = family.digest(program)
    candidate = replace(candidate, editorial_motion_plan=replace(candidate.editorial_motion_plan,
        source_camera_refresh=dense), performer_family_review=None)
    receipt['metadata_sha256'] = family.digest(family.metadata_payload(candidate))
    return replace(candidate, performer_family_review=receipt)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'local tools required')
class SceneArchiveIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        charts.SceneSourceIntegrationTests.setUpClass.__func__(cls)
        if isinstance(cls.result, family.PerformerFamilyRejection):
            errors = [row.get('error') for row in cls.result.evidence.get('candidate_alternatives', ())]
            cls.temp.cleanup()
            raise AssertionError((cls.result.error, errors))

    tearDownClass = classmethod(charts.SceneSourceIntegrationTests.tearDownClass.__func__)

    def test_normal_binding_rejects_consistently_resigned_referenced_omissions(self):
        self.assertNotIsInstance(self.result, family.PerformerFamilyRejection)
        family.validate_bundle_binding(self.result)
        for change in ('empty', 'selected_anchor', 'baseline_hypothesis', 'variant_inventory', 'authority'):
            with self.subTest(change=change), self.assertRaises(ValueError):
                family.validate_bundle_binding(resign_archive(self.result, change))

    def test_independent_parent_does_not_certify_deleted_archive(self):
        candidate = resign_archive(self.result, 'empty')
        with optional_native_scope(monotonic()+90), \
                patch.object(charts, 'scene_chart_selected_replay', new=charts.scene_chart_selected_replay), \
                patch('crosscut.group_scene_selected_replay.replay_selected_scene',
                    new=charts.scene_chart_selected_replay):
            with self.assertRaises(ValueError):
                source.validate_parent(candidate, self.context)

    def test_parent_reports_only_internal_closure_not_pre_receipt_history(self):
        with optional_native_scope(monotonic()+90):
            proof = charts.scene_chart_selected_replay(self.context, self.result)
        self.assertTrue(proof['fresh_complete_family_equal'])
        self.assertTrue(proof.get('referenced_discovery_archive_preserved'))
        self.assertFalse(proof.get('pre_receipt_history_completeness_certified', True))
        self.assertFalse(proof['quality_approved'])
