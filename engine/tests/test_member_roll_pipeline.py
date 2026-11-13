"""Real child IPC and production camera construction with synthetic observations."""
from dataclasses import replace
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
import unittest

from crosscut.first_result_controller import FirstResultController
from crosscut.member_camera_worker import validate_member_camera_clock
from crosscut.optional_native_worker import run_isolated_python_call
from crosscut.review_first_runtime import ReviewFirstRuntime
from crosscut.render import TransitionKind


class MemberRollPipelineTests(unittest.TestCase):
    def test_child_candidate_is_promoted_with_roll_for_dissolve_on_and_off(self):
        with TemporaryDirectory() as tmp:
            sources=tuple(Path(tmp)/f'{i}.mp4' for i in range(4))
            for source in sources:source.write_bytes(b'synthetic identity only')
            for dissolve in (False,True):
                with self.subTest(dissolve=dissolve):
                    pid,base,proposal=run_isolated_python_call('tests.member_roll_fixture','worker_proposal',
                        dict(sources=sources,output=Path(tmp)/'result.mp4',dissolve=dissolve),deadline=monotonic()+20.)
                    self.assertNotEqual(pid,os.getpid())
                    validate_member_camera_clock(proposal,base)
                    controller=FirstResultController(ReviewFirstRuntime(enabled=True),sources=sources,
                        source_durations=(10.,)*4,master_start=0.,duration=base.duration)
                    original=(base,None,None,{}, {})
                    controller.install_baseline(original,plan=base,authority='verified_member_source_primitive')
                    accepted=controller.run_optional('member_camera_and_dissolve',lambda:proposal,
                        validate=lambda value:validate_member_camera_clock(value,base))
                    self.assertIsNotNone(accepted)
                    controller.promote(accepted,plan=accepted[0],authority='registered_member_camera_proposal')
                    plan=controller.incumbent[0]
                    self.assertTrue(any(abs(p.roll_degrees)>.5 for s in plan.segments
                        if s.editorial_transform for p in s.editorial_transform.keyframes),proposal[3])
                    self.assertEqual(plan.member_pose_dissolve_requested,dissolve)
                    self.assertEqual(plan.member_pose_dissolve_effective,dissolve)
                    self.assertEqual(sum(t.kind is TransitionKind.FACE_DISSOLVE for t in plan.transitions),2 if dissolve else 0)
                    if dissolve:
                        self.assertTrue(all(row['registration']['roll']['applied'] for row in proposal[3]['boundaries']))
                        self.assertTrue(any(s.outgoing_overlap_transform for s in plan.segments))
                    else:
                        self.assertEqual(proposal[1].roll_event_count,4)
                        self.assertTrue(all(row['accepted'] for row in proposal[1].member_roll_audit[::2]))
                    bad=(replace(plan,segments=(replace(plan.segments[0],source_start=.1),*plan.segments[1:])),*proposal[1:])
                    rejected=controller.run_optional('stale_roll',lambda:bad,validate=lambda value:validate_member_camera_clock(value,base))
                    self.assertIsNone(rejected)
                    self.assertIs(controller.incumbent,accepted)
