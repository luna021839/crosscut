"""Full GROUP analyze path with explicit headless geometric-chart upstream.

Inherits the original normal-path assertions, not a simulated renderer or QA.
Only chart authoring/native model substitutions differ from the original test.
The body-only producer/LK/search/refresh/isolated parent/CAS/replay/encode and
ORIGINAL failing GROUP QA are real. No MEMBER or human quality is tested.
"""
from contextlib import ExitStack
from unittest.mock import patch
from crosscut.group_joint_source_camera import GroupOutputRole
from crosscut import group_performer_family_owner as family
from engine.tests import test_scene_normal_analyze as normal
from engine.tests.test_wide_composition_owner import (
    build_wide_fixture, encode_wide_chart, wide_chart_propose,wide_chart_final,wide_chart_sensors)


class WideCompositionNormalAnalyzeTests(normal.SceneSourceNormalAnalyzeTests):
    def run_normal(self,*,corrupt_parent=False):
        with ExitStack() as stack:
            for name,value in dict(build_scene_fixture=build_wide_fixture,
                    encode_scene_chart=encode_wide_chart,scene_chart_propose=wide_chart_propose,
                    scene_chart_final=wide_chart_final,scene_chart_sensors=wide_chart_sensors).items():
                stack.enter_context(patch.object(normal,name,new=value))
            result=super().run_normal(corrupt_parent=corrupt_parent)
        baseline,manifest,reports,rendered,calls,failure=result
        if not corrupt_parent:
            self.assertTrue(calls)
            self.assertNotIsInstance(calls[0],family.PerformerFamilyRejection,getattr(calls[0],'error',None))
            chosen=manifest['rendered_candidate_selection']['complete_candidates'][0]
            dense=chosen['performer_family_review']['source_camera_refresh']
            self.assertTrue(all(r==GroupOutputRole.COMPOSITION.value for r in dense['roles']))
            rows=[r for s in dense['mutable_shots'] for r in s['rows']]
            self.assertGreaterEqual(len(rows),60)
            self.assertTrue(all(not r['scene_heads'] and not r['projected_observed_arms'] for r in rows))
            self.assertTrue(all(len(r['scene_people'])==2 for r in rows))
            self.assertTrue(all(r['native_observation']['heads_state']=='not_requested' for r in rows))
            # Distant approved-looking structures remain exact, not neutralized.
            self.assertEqual(rendered[-1].segments[4:],baseline.render_plan.segments[4:])
        return result
