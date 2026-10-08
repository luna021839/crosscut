"""Normal controller lifecycle/clock wiring, not NN or media-quality QA."""
import ast
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from crosscut.first_result_controller import FirstResultController
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.group_request_work import GroupRequestWork, GroupWorkPolicy, install_group_request_work
from crosscut.render import RenderPlan, RenderSegment
from crosscut.request_work_lease import current_request_work_phase, RequestWorkExhausted
from crosscut.review_first_runtime import ReviewFirstRuntime, ReviewFirstDeadlineExceeded


def controller(root, clock=None):
    clock = [0.] if clock is None else clock
    runtime = ReviewFirstRuntime(enabled=True, _clock=lambda:clock[0])
    plan = RenderPlan((root/'not-real-input.mp4',), (RenderSegment(0,0.,1.,0.),),
        0, '16:9', root/'not-rendered.mp4')
    owner = FirstResultController(runtime, sources=plan.sources, source_durations=(2.,),
        master_start=0., duration=1., journal_path=root/'owner.json')
    owner.install_baseline(plan, plan=plan, authority='numeric-baseline-not-quality')
    return clock, owner, plan


def rows(session):
    return [json.loads(r) for r in session.lease.journal_path.read_text().splitlines()]


def test_normal_install_happens_once_before_optional_callback(tmp_path):
    _, owner, plan = controller(tmp_path)
    with ExitStack() as resources, patch('crosscut.first_result_controller.enforce_review_first_deadline', return_value=nullcontext()):
        work = install_group_request_work(controller=owner, resources=resources,
            journal_path=tmp_path/'work.jsonl')
        actual = []
        def operation():
            actual.append(current_request_work_phase())
            return plan
        assert owner.run_optional('normal_optional',operation) is plan
        assert actual == [work.lease.active]
        assert owner.request_work_lease is work.lease
        with pytest.raises(ValueError,match='once'):
            owner.reserve_request_work(total_limits={}, allocations={}, journal_path=tmp_path/'renewed.jsonl')
    assert work.closed and work.lease._stream.closed
    assert work.lease.request_terminal['outcome']=='returned'
    assert work.lease.request_terminal['quality_approved'] is False
    assert current_request_work_phase() is None
    assert work.lease.states['discovery']=='returned'


def test_one_phase_cutoff_does_not_spend_or_reset_future_render_budget(tmp_path):
    _, owner, plan = controller(tmp_path)
    with ExitStack() as resources, patch('crosscut.first_result_controller.enforce_review_first_deadline', return_value=nullcontext()):
        work = install_group_request_work(controller=owner,resources=resources,journal_path=tmp_path/'work.jsonl')
        port = work.lease.active
        port.charged['model_calls'] = port.allocation.resources['model_calls']
        def unadmitted():
            JointGroupWorkBudget(absolute_deadline=100.,clock=lambda:0.).spend('model_calls',model='numeric-admission-test')
            pytest.fail('unadmitted callback must not finish')
        assert owner.run_optional('no_capacity',unadmitted) is None
        assert owner.incumbent is plan
        assert work.lease.states['encode']=='not_requested'
        assert owner.attempts[-1]['observed_geometry_rejected'] is False
        encoded = work.run('encode',lambda:('same-original-lease',work.live_remaining_seconds))
        assert encoded[0]=='same-original-lease' and encoded[1]>0
        assert work.lease.states['discovery']=='budget_partial'
        assert work.lease.states['encode']=='returned'
    assert [r['payload']['status'] for r in rows(work) if r['kind']=='phase_terminal'][:2]==['budget_partial','returned']


@pytest.mark.parametrize('failure',[RuntimeError('outside-optional'),KeyboardInterrupt('cancel')])
def test_public_lifetime_cleans_context_and_records_primary_exception(tmp_path,failure):
    _,owner,_=controller(tmp_path)
    with pytest.raises(type(failure)):
        with ExitStack() as resources:
            work=install_group_request_work(controller=owner,resources=resources,journal_path=tmp_path/'work.jsonl')
            raise failure
    assert current_request_work_phase() is None
    assert work.lease.active is None and work.lease._stream.closed
    assert work.lease.states['discovery']=='failed'
    terminal=[r for r in rows(work) if r['kind']=='phase_terminal'][-1]
    assert terminal['payload']['error_type']==type(failure).__name__
    request_terminal=rows(work)[-1]
    assert request_terminal['kind']=='request_terminal'
    assert request_terminal['payload']['outcome']==('cancelled' if isinstance(failure,KeyboardInterrupt) else 'failed')


def test_encode_final_and_decoded_qa_use_distinct_once_only_phases(tmp_path):
    _,owner,plan=controller(tmp_path)
    with ExitStack() as resources, patch('crosscut.first_result_controller.enforce_review_first_deadline',return_value=nullcontext()):
        work=install_group_request_work(controller=owner,resources=resources,journal_path=tmp_path/'work.jsonl')
        phases=[]
        for name in ('final','encode','decoded_qa'):
            def operation(name=name):
                phases.append((name,current_request_work_phase().name))
                assert owner.incumbent is plan
                return plan
            assert work.run(name,operation) is plan
        assert phases==[(n,n) for n in ('final','encode','decoded_qa')]
        with pytest.raises(ValueError,match='already consumed'):
            work.run('encode',lambda:None)
    assert work.lease.states['selected_completion']=='not_requested'
    assert work.lease.to_dict()['all_model_call_sites_metered'] is False
    assert work.lease.to_dict()['runtime_calibrated'] is False
    assert work.lease.to_dict()['quality_approved'] is False


def test_normal_allocations_use_remaining_original_clock_not_a_new_600_seconds(tmp_path):
    clock,owner,_=controller(tmp_path)
    clock[0]=250.
    with ExitStack() as resources:
        work=install_group_request_work(controller=owner,resources=resources,journal_path=tmp_path/'work.jsonl')
        assert sum(a.wall_seconds for a in work.lease.allocations.values())==pytest.approx(315.)
        assert work.lease.deadline==600.
        assert work.lease.active.deadline==pytest.approx(287.8)
        assert work.lease.total_limits['native_completion_passes']==1
        encode=work.lease.allocations['encode'].resources
        assert encode['source_probes']>=len(owner.sources)
        assert encode['decoder_opens']>=len(owner.sources)
        assert encode['model_calls']==encode['model_loads']==0


def test_exclusive_work_history_cannot_be_replaced_by_a_new_normal_session(tmp_path):
    _,owner,_=controller(tmp_path)
    path=tmp_path/'work.jsonl'
    path.write_bytes(b'protected-old-history\n')
    with pytest.raises(FileExistsError):
        GroupRequestWork(owner,journal_path=path)
    assert path.read_bytes()==b'protected-old-history\n'
    assert owner.request_work_lease is None


def test_failed_owner_publication_before_context_registration_closes_raw_journal(tmp_path):
    _,owner,plan=controller(tmp_path)
    with patch.object(owner,'_persist',side_effect=OSError('owner publication failed')):
        with pytest.raises(OSError,match='publication failed'):
            GroupRequestWork(owner,journal_path=tmp_path/'work.jsonl')
    assert owner.incumbent is plan
    assert owner.request_work_lease.closed
    assert owner.request_work_lease._stream.closed
    assert current_request_work_phase() is None
    with pytest.raises(ValueError,match='closed'):
        with owner.request_work_lease.phase('discovery'):
            pytest.fail('failed admission must not launch work')


def test_initial_journal_failure_closes_the_handle_before_constructor_returns(tmp_path):
    _,owner,_=controller(tmp_path)
    captured=[]
    from crosscut import request_work_lease
    append=request_work_lease.RequestWorkLease._append
    def fail_header(lease,kind,*args,**kwargs):
        if kind=='lease':
            captured.append(lease)
            raise OSError('initial archive failed')
        return append(lease,kind,*args,**kwargs)
    with patch.object(request_work_lease.RequestWorkLease,'_append',fail_header):
        with pytest.raises(OSError,match='initial archive failed'):
            GroupRequestWork(owner,journal_path=tmp_path/'work.jsonl')
    assert captured[0].closed and captured[0]._stream.closed
    assert owner.request_work_lease is None


def test_between_phase_failure_is_persisted_after_original_deadline_not_success(tmp_path):
    clock,owner,_=controller(tmp_path)
    with pytest.raises(OSError,match='final envelope failed'):
        with ExitStack() as resources, patch('crosscut.first_result_controller.enforce_review_first_deadline',return_value=nullcontext()):
            work=install_group_request_work(controller=owner,resources=resources,journal_path=tmp_path/'work.jsonl')
            work.run('encode',lambda:None)
            clock[0]=601.
            raise OSError('final envelope failed')
    actual=rows(work)[-1]['payload']
    assert actual['outcome']=='failed'
    assert actual['last_entered_phase']=='encode'
    assert actual['phase_states']['encode']=='returned'
    assert actual['error_type']=='OSError'
    assert actual['original_deadline']==600.


def test_failed_phase_admission_records_request_not_phantom_execution(tmp_path):
    clock,owner,_=controller(tmp_path)
    with pytest.raises(ReviewFirstDeadlineExceeded):
        with ExitStack() as resources:
            work=install_group_request_work(controller=owner,resources=resources,journal_path=tmp_path/'work.jsonl')
            work.finish_optional()
            clock[0]=601.
            work.run('final',lambda:pytest.fail('expired request must not enter final work'))
    actual=rows(work)[-1]['payload']
    assert actual['last_requested_phase']=='final'
    assert actual['last_entered_phase']=='discovery'
    assert actual['phase_states']['final']=='not_requested'
    assert actual['error_type']=='ReviewFirstDeadlineExceeded'


def test_request_terminal_cannot_ack_unreaped_work_or_mint_a_new_phase(tmp_path):
    _,owner,_=controller(tmp_path)
    with ExitStack() as resources:
        work=install_group_request_work(controller=owner,resources=resources,journal_path=tmp_path/'work.jsonl')
        work.finish_optional()
        port=work.lease.ports['discovery']
        port.live_workers['numeric-unreaped']={}
        with pytest.raises(ValueError,match='reaped'):
            work.lease.finish_request(outcome='returned',last_entered_phase='discovery')
        assert not any(r['kind']=='request_terminal' for r in rows(work))
        port.live_workers.clear()
    with pytest.raises(ValueError,match='terminal'):
        with work.lease.phase('parent'):
            pytest.fail('terminal lease cannot be renewed')


@pytest.mark.parametrize('remaining',[True,0.,-1.,float('nan'),float('inf')])
def test_policy_has_no_unbounded_or_renewed_time_domain(remaining):
    with pytest.raises(ValueError):
        GroupWorkPolicy().allocations(remaining)


def test_normal_pipeline_installs_after_baseline_before_first_optional_and_uses_exact_plan():
    # Source wiring proof only, not a mocked full NN workflow or a media pass.
    source=Path(__file__).parents[1]/'crosscut/pipeline.py'
    tree=ast.parse(source.read_text())
    public=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='analyze')
    impl=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_analyze_impl')
    calls=sorted((n for n in ast.walk(impl) if isinstance(n,ast.Call)),key=lambda n:n.lineno)
    named=lambda n:getattr(n.func,'attr',getattr(n.func,'id',''))
    baseline=next(n for n in calls if named(n)=='install_baseline')
    install=next(n for n in calls if named(n)=='install_group_request_work')
    first_optional=next(n for n in calls if named(n)=='run_optional' and n.lineno>baseline.lineno)
    assert baseline.lineno<install.lineno<first_optional.lineno
    assert any(named(n)=='ExitStack' for n in ast.walk(public) if isinstance(n,ast.Call))
    assert any(named(n)=='_analyze_impl' for n in ast.walk(public) if isinstance(n,ast.Call))
    renders=[n for n in calls if named(n)=='render' and n.lineno>install.lineno]
    assert any(isinstance(n.args[0],ast.Name) and n.args[0].id=='plan' for n in renders)
    phase_names={n.args[0].value for n in calls if named(n)=='run' and n.args and isinstance(n.args[0],ast.Constant)
        and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='group_request_work'}
    assert {'selected_completion','final','encode','decoded_qa'}<=phase_names


def test_public_analyze_wrapper_really_closes_resources_on_impl_cancel(monkeypatch):
    from crosscut import pipeline
    closed=[]
    def cancel(request,reporter,**kwargs):
        kwargs['request_work_resources'].callback(lambda:closed.append('closed'))
        raise KeyboardInterrupt('cancel before any NN or media')
    monkeypatch.setattr(pipeline,'_analyze_impl',cancel)
    with pytest.raises(KeyboardInterrupt):
        pipeline.analyze(object(),object())
    assert closed==['closed']
