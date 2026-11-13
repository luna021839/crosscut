"""Renderer-derived dependency scopes, EXPLICIT nonmedia numeric fixtures.

The real numeric source/state assembler and renderer grouping are exercised.
No decoder/model/descriptor/quality result is certified by these pure tests.
"""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from time import monotonic
import shutil
import unittest

import pytest

from crosscut import group_source_changing_owner as source
from crosscut.group_joint_source_camera import GroupOutputRole, joint_state_from_render_plan
from crosscut.group_joint_state_factory import candidate_joint_group_state
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.render import RenderPlan, RenderSegment, RenderTransition, segment_frame_counts
from crosscut.render_frame_schedule import VerifiedRenderFrameClock


def numeric_long_path(middle_cells=30):
    paths = tuple(Path(f'EXPLICIT_NONMEDIA_FIXTURE_SOURCE_{i}.mp4') for i in range(4))
    clocks = (VerifiedRenderFrameClock(Fraction(30),1000,'a'*64),)*4
    domain = GroupSourceDomain(paths,(0,1,2,3),(0.,)*4,(None,)*4,((),)*4,clocks)
    indices = (0, *(1 for _ in range(middle_cells)), 2)
    plan = RenderPlan(sources=paths, segments=tuple(RenderSegment(index,n/3,1/3,n/3)
        for n,index in enumerate(indices)), reference_source=0, aspect_ratio='16:9',
        output=Path('EXPLICIT_NEVER_RENDERED.mp4'),fps=30.,width=320,height=180,
        transitions=tuple(RenderTransition(tier='D' if a == b else 'C')
            for a,b in zip(indices,indices[1:])))
    intervals = ((0,10),(10,10*(middle_cells+1)),(10*(middle_cells+1),10*(middle_cells+2)))
    baseline = tuple(joint_state_from_render_plan(plan=plan,domain=domain,slot=i,
        first_output_frame=first,end_output_frame=end,role=GroupOutputRole.FOCAL_HEAD)
        for i,(first,end) in enumerate(intervals))
    middle = candidate_joint_group_state(plan=plan,domain=domain,
        incumbent_state=baseline[1],eligible_source_index=3)
    return plan,domain,(baseline[0],middle,baseline[2])


def test_oversized_path_fails_before_any_descriptor_materialization():
    plan,domain,path = numeric_long_path()
    proposed,changed = source._assembler(path)(plan=plan,domain=domain,path=path)
    assert len(changed) == 30
    assert segment_frame_counts(proposed) == segment_frame_counts(plan)
    calls = []
    def unneeded_expensive_materializer(**kwargs):
        calls.append(kwargs)
        return NS(plan=proposed,changed_segment_indices=changed,
            refreshed_segment_indices=tuple(range(32)))
    context = NS(incumbent=NS(render_plan=plan,visual_timeline=NS(),
        boundary_look_match=NS(decisions=())),
        domain=domain,source_derivative_binding=None,sync_scores=(1.,)*4)
    with patch('crosscut.group_joint_render_observations.refresh_joint_group_render_observations',
            new=NS(__wrapped__=unneeded_expensive_materializer)):
        with pytest.raises(ValueError,match='refresh cell budget'):
            source.complete_source_candidate(context,path,None,None,None,None,None,None)
    assert calls == [], 'the doomed oversized path already did expensive descriptor work'


def separated_plan(count=12, *, fps=30000/1001):
    # Numeric clocks only. Every cell initially uses a different source so
    # renderer clock closure must not fill the entire min..max output hull.
    paths = tuple(Path(f'EXPLICIT_NONMEDIA_{i}.mp4') for i in range(count+1))
    plan = RenderPlan(sources=paths,segments=tuple(RenderSegment(i,float(i),1.,float(i))
        for i in range(count)),reference_source=0,aspect_ratio='16:9',
        output=Path('EXPLICIT_NEVER_RENDERED.mp4'),fps=fps,width=320,height=180,
        transitions=(RenderTransition(tier='C'),)*(count-1))
    after = replace(plan,segments=tuple(replace(segment,source_index=count)
        if i in (2,9) else segment for i,segment in enumerate(plan.segments)))
    return plan,after


def test_two_distant_changes_have_dependency_union_not_between_shot_hull():
    from crosscut.group_scene_dependency_support import derive_support
    before,after = separated_plan()
    support = derive_support(before,after,retained_applied_look_edges=())
    assert support['numeric_write_cells'] == [2,9]
    assert support['descriptor_refresh_cells'] == [1,2,3,8,9,10]
    assert support['read_cells'] == [1,2,3,8,9,10]
    assert support['descriptor_refresh_intervals'] == [[1,4],[8,11]]
    assert support['dependency_cells_are_not_minmax_hull']
    assert support['quality_approved'] is False
    assert not support['multiple_repairs_selected_or_completed']


def test_candidate_and_original_clock_units_both_extend_read_dependency():
    from crosscut.group_scene_dependency_support import derive_support
    before,after = separated_plan()
    before = replace(before,segments=tuple(replace(segment,source_index=0)
        if i in (0,1) else segment for i,segment in enumerate(before.segments)))
    after = replace(after,segments=tuple(replace(segment,source_index=0)
        if i in (0,1) else segment for i,segment in enumerate(after.segments)))
    # Original (0,1) unit extends a read at 1 to 0; candidate (9,10,11)
    # unit extends a read at 10 to 11. Different per-cell geometry does not
    # prevent the renderer from sharing a native sampling clock.
    after = replace(after,segments=tuple(replace(segment,source_index=12)
        if i in (10,11) else segment for i,segment in enumerate(after.segments)))
    support = derive_support(before,after,retained_applied_look_edges=())
    assert 0 in support['read_cells'] and 11 in support['read_cells']
    assert support['renderer_clock_closure_from_both_plans']
    assert support['original_clock_units'][0]['cells'] == [0,1]
    assert support['candidate_clock_units'][-1]['cells'] == [9,10,11]


def test_retained_look_proof_has_read_only_scope_distinct_from_new_look_writes():
    from crosscut.group_scene_dependency_support import derive_support
    before,after = separated_plan()
    support = derive_support(before,after,retained_applied_look_edges=(5,))
    assert support['incident_look_edges'] == [1,2,8,9]
    assert support['retained_look_proof_edges'] == [5]
    assert support['retained_look_read_cells'] == [5,6]
    assert all(i in support['read_cells'] for i in (5,6))
    assert all(i not in support['write_cells'] for i in (5,6))


@pytest.mark.parametrize('changed', ['clock','size','overlap','outside_transition'])
def test_unowned_plan_clock_or_transition_change_is_not_a_dependency_scope(changed):
    from crosscut.group_scene_dependency_support import derive_support
    before,after = separated_plan()
    if changed == 'clock': after = replace(after,fps=30.)
    elif changed == 'size': after = replace(after,width=640,height=360)
    elif changed == 'overlap':
        from crosscut.render import TransitionKind
        after = replace(after,transitions=(replace(after.transitions[0],
            kind=TransitionKind.FACE_DISSOLVE,frame_count=3),*after.transitions[1:]))
    else: after = replace(after,transitions=(replace(after.transitions[0],reason='unowned'),
        *after.transitions[1:]))
    with pytest.raises(ValueError): derive_support(before,after,retained_applied_look_edges=())


@pytest.mark.parametrize('edges', [None,[],(True,),(-1,),(11,),(5,5),(6,5)])
def test_retained_look_edge_domain_is_explicit_and_canonical(edges):
    from crosscut.group_scene_dependency_support import derive_support
    before,after = separated_plan()
    with pytest.raises(ValueError): derive_support(before,after,retained_applied_look_edges=edges)


@pytest.mark.parametrize('change', ['read','write','unit','hash','original','authority'])
def test_re_signed_support_mutation_is_independently_rederived(change):
    from crosscut.group_scene_dependency_support import derive_support,validate_support
    before,after = separated_plan()
    support = deepcopy(derive_support(before,after,retained_applied_look_edges=(5,)))
    if change == 'read': support['read_cells'].remove(5)
    elif change == 'write': support['write_cells'].append(5)
    elif change == 'unit': support['original_clock_units'][0]['cells'].append(1)
    elif change == 'hash': support['candidate_plan_sha256'] = '0'*64
    elif change == 'original': support['incumbent_plan_sha256'] = '0'*64
    else: support['quality_approved'] = True
    from crosscut.group_performer_family_owner import digest
    support.pop('support_sha256');support['support_sha256'] = digest(support)
    with pytest.raises(ValueError):
        validate_support(support,before,after,retained_applied_look_edges=(5,))


def test_read_read_is_not_conflict_but_every_write_read_and_write_write_is():
    from crosscut.group_scene_dependency_support import dependency_conflicts
    a = dict(read_cells=[0,1,2],write_cells=[1])
    b = dict(read_cells=[0,3,4],write_cells=[3])
    assert dependency_conflicts(a,b) == []
    assert dependency_conflicts(a,dict(read_cells=[1,4],write_cells=[4])) == [1]
    assert dependency_conflicts(a,dict(read_cells=[4],write_cells=[1])) == [1]
    assert dependency_conflicts(a,dict(read_cells=[0],write_cells=[0])) == [0]


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'local tools required')
class SceneDependencyOwnerIntegrationTests(unittest.TestCase):
    """Real chart producer/completion/parent, not a fake dependency certificate."""
    @classmethod
    def setUpClass(cls):
        from engine.tests import test_scene_source_owner as charts
        charts.SceneSourceIntegrationTests.setUpClass.__func__(cls)
        if isinstance(cls.result,source.family.PerformerFamilyRejection):
            errors=[row.get('error') for row in cls.result.evidence.get('candidate_alternatives',())]
            cls.temp.cleanup()
            raise AssertionError((cls.result.error,errors))

    @classmethod
    def tearDownClass(cls):
        from engine.tests import test_scene_source_owner as charts
        charts.SceneSourceIntegrationTests.tearDownClass.__func__(cls)

    def test_complete_owner_and_independent_parent_bind_actual_renderer_support(self):
        from crosscut.group_scene_dependency_support import derive_support
        from crosscut.render_plan_serialization import render_plan_from_json
        from crosscut.optional_native_worker import optional_native_scope
        from engine.tests import test_scene_source_owner as charts
        receipt=source.family.plain(self.result.performer_family_review)
        program=receipt['selected_scene_program']
        expected=derive_support(self.context.incumbent.render_plan,
            render_plan_from_json(program['numeric_plan']),retained_applied_look_edges=tuple(sorted(
                d.boundary_index for d in self.context.incumbent.boundary_look_match.decisions if d.applied)))
        self.assertEqual(program['dependency_support'],expected)
        self.assertEqual(receipt['refreshed_cells'],expected['descriptor_refresh_cells'])
        self.assertFalse(program['dependency_support']['multiple_repairs_selected_or_completed'])
        with optional_native_scope(monotonic()+90):
            proof=charts.scene_chart_selected_replay(self.context,self.result)
        self.assertTrue(proof['renderer_dependency_support_independently_rederived'])
        self.assertEqual(proof['selected_dependency_support_sha256'],expected['support_sha256'])
        self.assertFalse(proof['quality_approved'])

    def test_consistently_re_signed_support_omission_fails_before_native_assets(self):
        from crosscut.optional_native_worker import optional_native_scope
        receipt=deepcopy(source.family.plain(self.result.performer_family_review))
        program=receipt['selected_scene_program'];support=program['dependency_support']
        support['read_cells'].pop()
        support.pop('support_sha256');support['support_sha256']=source.family.digest(support)
        program.pop('program_sha256');program['program_sha256']=source.family.digest(program)
        altered=replace(self.result,performer_family_review=receipt)
        with optional_native_scope(monotonic()+90),patch.object(source,'installed_native_assets',
                side_effect=AssertionError('invalid support must reject before model asset/inference work')):
            with self.assertRaisesRegex(ValueError,'dependency support'):
                source.validate_parent(altered,self.context)

    def test_internal_support_cannot_erase_external_incumbent_active_look_reads(self):
        from crosscut.group_scene_dependency_support import derive_support
        from crosscut.render_plan_serialization import render_plan_from_json
        from crosscut.group_scene_selected_replay import validate_program
        from crosscut.optional_native_worker import optional_native_scope
        from engine.tests import test_scene_source_owner as charts
        receipt=deepcopy(source.family.plain(self.result.performer_family_review))
        program=receipt['selected_scene_program']
        self.assertTrue(program['dependency_support']['retained_applied_look_edges'])
        program['dependency_support']=derive_support(render_plan_from_json(program['original_numeric_plan']),
            render_plan_from_json(program['numeric_plan']),retained_applied_look_edges=())
        program.pop('program_sha256');program['program_sha256']=source.family.digest(program)
        # Internal consistency alone cannot know omitted old duties. The parent
        # must supply the actual external incumbent's applied-look inventory.
        validate_program(program,receipt['source_camera_refresh'])
        altered=replace(self.result,performer_family_review=receipt)
        from crosscut import group_scene_selected_replay as selected
        # A local pre-asset spy cannot cross a process boundary. The positive
        # test above uses the actual independent worker; this negative invokes
        # the same replay body directly to verify ordering with a real spy.
        with optional_native_scope(monotonic()+90),charts.scene_chart_sensors(), \
                patch.object(source,'installed_native_assets',
                    side_effect=AssertionError('external dependency mismatch must reject before model work')):
            with self.assertRaisesRegex(ValueError,'dependency support'):
                selected.replay_selected_scene.__wrapped__(self.context,altered)
