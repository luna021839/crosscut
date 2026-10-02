"""Normal analyze -> typed review artifact event; original full QA still FAIL.

The original fixture authoring/alignment/derivative classifier is synthetic.
No replacement of the new trial producer, real renderer, technical QA or parent
in the positive. Negative mutates one child raw hash only, before real parent.
"""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut import group_scene_review_proposal as trial
from crosscut import group_performer_family_owner as family
from engine.tests import test_scene_normal_analyze as normal
from engine.tests.test_wide_composition_owner import (
    build_wide_fixture,encode_wide_chart,wide_chart_propose,wide_chart_final,wide_chart_sensors)


class SceneReviewNormalTests(unittest.TestCase):
    def run_normal_trial(self,corrupt=False):
        events=[];artifacts=[]
        class CaptureReporter(normal.Reporter):
            def emit(self,*args,**kwargs):
                super().emit(*args,**kwargs)
                if args and args[0]=='scene_review_proposal':
                    item=kwargs['scene_review_proposal'];events.append(item)
                    artifacts.append(json.loads(Path(item['manifest_path']).read_text()))
                    assert Path(item['video_path']).is_file()
                    from crosscut.render_clock_artifact import render_clock_path
                    clock=render_clock_path(Path(item['video_path']))
                    artifact=artifacts[-1]
                    assert clock.is_file()
                    assert artifact['render_clock_path']==str(clock)
                    assert artifact['render_clock_sha256']==family._file_hash(clock)
                    bound=json.loads(clock.read_text())
                    assert bound['video_content_sha256']==item['video_sha256']
                    assert bound['plan']==artifact['proxy_plan']
                    assert not render_clock_path(Path(item['video_path']).with_name(
                        Path(item['video_path']).stem+'.unpublished.mp4')).exists()
        with ExitStack() as stack:
            destination=os.environ.get('SCENE_TEST_ARTIFACT_DIR')
            if destination:
                stack.enter_context(patch.dict(os.environ, {'SCENE_TEST_ARTIFACT_DIR':
                    str(Path(destination)/('new-normal-negative' if corrupt else 'new-normal-positive'))}))
            for name,value in dict(build_scene_fixture=build_wide_fixture,
                encode_scene_chart=encode_wide_chart,scene_chart_propose=wide_chart_propose,
                scene_chart_final=wide_chart_final,scene_chart_sensors=wide_chart_sensors,
                Reporter=CaptureReporter).items():
                stack.enter_context(patch.object(normal,name,new=value))
            if corrupt:
                original=trial.propose
                def bad_child(context):
                    result,reason=original(context)
                    if result:
                        plan,receipt=result;receipt=deepcopy(receipt)
                        receipt['dense'][0]['after']['raw_pixels_sha256']='0'*64
                        return (plan,receipt),reason
                    return result,reason
                stack.enter_context(patch.object(trial,'propose',side_effect=bad_child))
            # Existing test scaffolding runs analyze(), not the new wrapper in
            # isolation. Its fixture renderer and ORIGINAL final QA execute.
            helper=normal.SceneSourceNormalAnalyzeTests(methodName='runTest')
            result=helper.run_normal(corrupt_parent=False)
        return events,artifacts,result

    def test_normal_positive_exposes_only_separate_unapproved_review(self):
        events,artifacts,result=self.run_normal_trial()
        baseline,manifest,reports,rendered,calls,failure=result
        self.assertEqual(len(events),1,(failure,manifest.get('review_first_visual_selection')))
        event=events[0];artifact=artifacts[0]
        self.assertFalse(event['quality_approved'])
        self.assertFalse(event['delivery_completion_message_allowed'])
        self.assertEqual(event['artifact_role'],'scene_review_only')
        self.assertEqual(artifact['receipt']['work']['counts']['model_calls'],0)
        self.assertEqual(artifact['receipt']['work']['counts']['dense_paths'],2)
        self.assertTrue(all(artifact['technical_qa']['final_media']['checks'].values()))
        self.assertEqual(artifact['original_plan']['sources'],artifact['candidate_plan']['sources'])
        self.assertNotEqual(artifact['receipt']['original_plan_sha256'],artifact['receipt']['candidate_plan_sha256'])
        self.assertFalse(artifact['technical_qa']['professional_qa_passed'])
        self.assertTrue(reports)
        self.assertFalse(reports[-1][1]['passed'])
        self.assertFalse(manifest['app_result_exposure']['delivery_completion_message_allowed'])
        self.assertEqual(len(calls),2)  # Strict owner still gets independent parent.
        self.assertTrue(any(i['code']==family.QUALITY_CODE for i in reports[-1][1]['issues']))
        print('NORMAL_SCENE_REVIEW_EVENT',json.dumps(event),'ORIGINAL_FINAL_QA',failure)

    def test_normal_bad_child_does_not_publish_or_poison_strict_result(self):
        events,artifacts,result=self.run_normal_trial(corrupt=True)
        baseline,manifest,reports,rendered,calls,failure=result
        self.assertEqual(events,[])
        self.assertEqual(artifacts,[])
        self.assertEqual(len(calls),2)
        self.assertTrue(reports)
        self.assertFalse(reports[-1][1]['passed'])
        self.assertFalse(manifest['app_result_exposure']['delivery_completion_message_allowed'])
        print('NORMAL_SCENE_REVIEW_NEGATIVE',failure)
