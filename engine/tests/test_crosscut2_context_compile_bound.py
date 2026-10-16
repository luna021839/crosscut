"""Original rank/order equivalence; these synthetic cues are NOT AV labels."""
import gc
import numpy as np
import pytest
from crosscut2.context_compile_bound import viable
from crosscut2.compiled_context_beam import _room
from crosscut2.compiled_context_port_dp import solve
from crosscut2.playback_workspace import PlaybackWorkspace
from test_crosscut2_compiled_context_port_dp import fixture,problem


@pytest.mark.parametrize('kernel',['numpy','numba'])
@pytest.mark.parametrize('same_episode',[False,True])
@pytest.mark.parametrize('positive_floor',[False,True])
def test_precompile_mask_is_exact_original_rank_oracle_including_free_slots_paid_reset_and_large_integer_backs(kernel,same_episode,positive_floor):
    if kernel=='numba':pytest.importorskip('numba')
    rng=np.random.default_rng(921);P,Q,C,L=5,3,6,8;origin=2**45;width=40
    shape=(Q,2,L)
    destination=tuple(np.empty(shape,d) for d in (float,np.int64,np.int64,np.int64,float,float,np.uint64))
    destination[0][:]=rng.normal(size=shape);destination[1][:]=rng.integers(0,3,size=shape)
    destination[2][:]=(origin*width*2*L)+rng.integers(-100,100,size=shape)
    destination[3][:]=rng.integers(0,6,size=shape);destination[4][:]=rng.choice([0.,.125],size=shape)
    destination[5][:]=0.;destination[6][:]=np.arange(1,1+np.prod(shape)).reshape(shape)
    destination[6][0,1,3]=0
    costs=rng.normal(size=(P,2,L));costs[0]=np.inf
    cuts=rng.integers(0,3,size=costs.shape);base=rng.normal(size=(P,Q,C))
    allowed=rng.random(base.shape)>.1;benefit=rng.choice([0.,.125,1.],size=(P,Q))
    # Equal cost/count candidates differ by integer backpointers beyond2**53.
    costs[1]=0.;base[1]=0.;cuts[1]=0
    destination[0][1]=0.;destination[1][1]=0;destination[4][1]=0.
    floors=np.abs(rng.normal(size=base.shape)) if positive_floor else None
    got=viable(base,allowed,costs,cuts,benefit,destination,same_episode=same_episode,
        origin=origin,width=width,workspace=PlaybackWorkspace(2**20),kernel=kernel,floors=floors)
    expected=np.zeros(allowed.shape,bool)
    for p,q,code in zip(*np.nonzero(allowed)):
        for bit in (0,1):
            for k in np.flatnonzero(np.isfinite(costs[p,bit])):
                floor=0. if floors is None else floors[p,q,code]
                back=(((origin*width+p)*2+bit)*L+int(k));raw=(costs[p,bit,k]+base[p,q,code])+floor
                for claim in (False,True) if bit==0 and benefit[p,q]>0 else (False,):
                    paid=benefit[p,q] if claim else 0.
                    outbit=(1 if bit or claim else 0) if same_episode else 0
                    group=tuple(v[q,outbit] for v in destination)
                    rank=(raw-paid,cuts[p,bit,k],back,int(code),-paid)
                    expected[p,q,code]|=_room(group,rank)>=0
                    if not expected[p,q,code]:
                        for extra in (np.nextafter(0.,1.),.1,1e5):
                            assert _room(group,(((costs[p,bit,k]+base[p,q,code])+(floor+extra))-paid,
                                cuts[p,bit,k],back,int(code),-paid))<0
    np.testing.assert_array_equal(got,expected);assert not got[0].any()


@pytest.mark.parametrize('kernel',['numpy','numba'])
@pytest.mark.parametrize('backend',['memory','spool'])
def test_actual_compiled_prebound_on_off_keeps_path_code_context_credit_and_total_bitwise(kernel,backend,tmp_path):
    context,c=fixture(np.array([0,18,36,54]),kernel=kernel)
    entry,exit,kw=problem(context)
    baseline=solve(context.nodes,entry,exit,compiler=c,precompile_rank_bound=False,**kw)
    c.arena.close();del c;gc.collect()
    if backend=='spool':
        from test_crosscut2_spooled_context import spooled
        c=spooled(context,tmp_path,kernel=kernel)
    else:
        from crosscut2.compiled_played_context import CompiledPlayback
        c=CompiledPlayback(context)
    enabled=solve(context.nodes,entry,exit,compiler=c,**kw)
    assert enabled[:2]==baseline[:2]
    for name in ('selected_arc_codes','opportunity_claims','selected_context_costs','camera_alternatives_scored'):
        assert enabled[2][name]==baseline[2][name]
    assert enabled[2]['nonnegative_precompile_rank_bound_enabled']
    assert enabled[2]['camera_recipes_skipped_before_compile_by_proved_rank']>0
    assert not baseline[2]['nonnegative_precompile_rank_bound_enabled']
    c.arena.close()


@pytest.mark.parametrize('replace_after_constructor',[False,True])
def test_unknown_temporal_rates_cannot_claim_nonnegative_precompile_bound(replace_after_constructor):
    from crosscut2.compiled_played_context import CompiledPlayback
    context,c=fixture(np.array([0,18,36]))
    original=context.model.rates
    context.model.rates=lambda *a,**kw:original(*a,**kw)
    if not replace_after_constructor:c.arena.close();c=CompiledPlayback(context)
    assert not c.proved_nonnegative_context
    entry,exit,kw=problem(context);out=solve(context.nodes,entry,exit,compiler=c,**kw)
    assert not out[2]['nonnegative_precompile_rank_bound_enabled']
    assert out[2]['camera_recipes_skipped_before_compile_by_proved_rank']==0
    c.arena.close()


def test_negative_cached_middle_rates_cannot_claim_the_zero_context_bound():
    context,c=fixture(np.array([0,18,180]));assert c.proved_nonnegative_context
    context.base_rates[0]=context.base_rates[0].copy();context.base_rates[0][0,0]=-1.
    assert not c.proved_nonnegative_context;c.arena.close()


def test_unreachable_actual_source_is_rank_rejected_before_any_raw_camera_compile(monkeypatch):
    from crosscut2.compiled_played_context import _Hold
    context,c=fixture(np.array([0,18,36]));entry,exit,kw=problem(context);old=_Hold.current
    def checked(hold,p,q,code,**opts):
        if hold.i==0 and hold.compiler.context.entries[0][p].source==1:
            raise AssertionError('unreachable recipe was compiled')
        return old(hold,p,q,code,**opts)
    monkeypatch.setattr(_Hold,'current',checked)
    out=solve(context.nodes,entry,exit,compiler=c,**kw)
    assert out[2]['camera_recipes_skipped_before_compile_by_proved_rank']>0
    c.arena.close()


@pytest.mark.parametrize('nonlinear',[False,True])
def test_proved_long_middle_can_reject_before_camera_compile_but_nonreference_and_direct_codes_keep_zero(nonlinear):
    # Packed destination handles below are finite rank fixtures only. They
    # are never installed in a beam or presented as owned physical roots.
    context,c=fixture(np.array([0,30,600,630]),nonlinear=nonlinear)
    a=c.arena;empty=a.empty();P=len(context.entries[0]);Q=len(context.exits[2]);L=8
    ids=np.zeros((P,2,L),np.uint64);costs=np.full(ids.shape,np.inf)
    chosen=next(p for p,v in enumerate(context.entries[0]) if v.source==1)
    ids[chosen,0,0]=empty.handle;costs[chosen,0,0]=0.
    with c.open_origin(0,ids,costs) as bank:
        base,allowed=context.arcs.camera_alternatives(0,2)
        with c.compile_hold(0,2,base,allowed) as hold:
            assert hold.prepare_long()
            destination=tuple(np.zeros((Q,2,L),d) for d in (float,np.int64,np.int64,np.int64,float,float,np.uint64))
            destination[6][:]=np.arange(1,1+Q*2*L).reshape(Q,2,L)
            for q in range(Q):destination[0][q]=base[chosen,q,0]+hold.long_middle[chosen,0]*.5
            zero=viable(base,allowed,costs,np.zeros(costs.shape,np.int64),np.zeros(base.shape[:2]),destination,
                same_episode=True,origin=0,width=max(P,Q),workspace=a.workspace,kernel='numpy')
            tighter=c.filter_current_domain(hold,base,allowed,costs,np.zeros(costs.shape,np.int64),np.zeros(base.shape[:2]),
                destination,same_episode=True,width=max(P,Q))
            assert not np.any(tighter&~zero)
            proved=allowed[chosen,:,:4]&hold.proved_long_domain[chosen]
            assert np.any(proved&zero[chosen,:,:4]&~tighter[chosen,:,:4])
            for q,code in zip(*np.nonzero(allowed[chosen])):
                q,code=int(q),int(code)
                with hold.current(chosen,q,code) as current:
                    score=float(c.score(bank,current,chosen,0,[0])[0])
                    floor=hold.long_middle[chosen,code] if code<4 and hold.proved_long_domain[chosen,q,code] else 0.
                    assert score>=floor
            assert np.array_equal(tighter[:,:,4:],zero[:,:,4:])
            if nonlinear:
                nonref=[p for p,port in enumerate(context.entries[0]) if not hold.reference_clock[port.source]]
                np.testing.assert_array_equal(tighter[nonref],zero[nonref])
    a.release(empty);a.close()
