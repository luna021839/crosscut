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
from crosscut import group_scene_observations as scene
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

    def test_known_rejected_geometry_cannot_be_made_admissible_by_edge_flags(self):
        original=family.plain(self.result.performer_family_review)['source_camera_refresh']
        for scope in ('mutable','incident_row','incident_edge'):
            with self.subTest(scope=scope):
                dense=deepcopy(original)
                if scope=='mutable':row=dense['mutable_shots'][0]['rows'][0]
                elif scope=='incident_row':row=dense['incident_projected_observations'][0]['rows'][0]
                else:row=dense['incident_projected_observations'][0]
                row['frame_admission_violations']=['EXPLICIT_NEGATIVE_REQUIRED_FRAME_NOT_HUMAN']
                row['role_geometry_constraints_satisfied']=False
                if scope!='mutable':
                    dense['incident_projected_observations'][0].update(available=True,
                        required_semantics_available=True)
                with self.assertRaisesRegex(ValueError,'role/visibility rejection'):
                    scene.validate_scene_dense(dense)

    def test_normal_owner_binds_original_duties_in_complete_dense_and_incident_consumers(self):
        receipt=family.plain(self.result.performer_family_review)
        dense=receipt['source_camera_refresh'];program=dense['original_duty_program']
        self.assertEqual(receipt['selected_scene_program']['original_duty_program'],program)
        self.assertTrue(receipt['search']['original_duties_frozen_before_donor_discovery'])
        self.assertFalse(program['donor_observations_define_duties'])
        self.assertFalse(program['coherent_cross_camera_actor_correspondence_certified'])
        for span in dense['mutable_shots']:
            for row in span['rows']:
                self.assertEqual(row['original_duty_reference']['program_sha256'],program['program_sha256'])
        for edge in dense['incident_projected_observations']:
            for endpoint in edge['required_endpoint_evidence']:
                self.assertEqual(endpoint['original_duty_program_sha256'],program['program_sha256'])

    def test_received_original_duties_cannot_be_removed_from_any_consumer_scope(self):
        original=family.plain(self.result.performer_family_review)['source_camera_refresh']
        for scope in ('programme','mutable_ref','incident_ref','incident_parts','endpoint_ref'):
            dense=deepcopy(original)
            if scope=='programme':dense.pop('original_duty_program')
            elif scope=='mutable_ref':dense['mutable_shots'][0]['rows'][0]['original_duty_reference']=None
            elif scope=='incident_ref':dense['incident_projected_observations'][0]['rows'][0]['original_duty_reference']=None
            elif scope=='incident_parts':dense['incident_projected_observations'][0]['rows'][0]['role_obligations']['required_parts']=[(99,'left_arm')]
            else:dense['incident_projected_observations'][0]['required_endpoint_evidence'][0]['original_duty_program_sha256']=None
            with self.subTest(scope=scope),self.assertRaises(ValueError):scene.validate_scene_dense(dense)

    def test_resigned_selected_role_or_native_programme_versions_cannot_disappear(self):
        from crosscut.group_scene_archive import archive_binding,validate_archive
        receipt=family.plain(self.result.performer_family_review)
        for change in ('selected_role_ref','role_version','native_version','active_version','history_authority'):
            with self.subTest(change=change):
                dense=deepcopy(receipt['source_camera_refresh'])
                program=deepcopy(receipt['selected_scene_program'])
                archive=dense['scene_discovery'];records=archive['original_programme_records']
                if change=='selected_role_ref':program['original_role_programme_sha256'].pop()
                elif change=='role_version':records['original_role_programmes'].pop(program['original_role_programme_sha256'][0])
                elif change=='native_version':records['native_duty_programmes'].pop(program['original_duty_program']['program_sha256'])
                elif change=='active_version':records['active_native_duty_programme_sha256']='e'*64
                else:records['pre_receipt_history_completeness_certified']=True
                program['discovery_archive_binding']=archive_binding(archive)
                program.pop('program_sha256');program['program_sha256']=family.digest(program)
                with self.assertRaises(ValueError):validate_archive(program,dense)
