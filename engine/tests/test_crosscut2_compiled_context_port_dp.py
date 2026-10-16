"""Finite actual-producer/ordered-model contracts, not real fits or films."""
import numpy as np
import pytest
from crosscut2.boundary_arc_costs import ArcCosts
from crosscut2.boundary_port_domain import make
from crosscut2.compiled_played_context import CompiledPlayback
from crosscut2.compiled_context_port_dp import solve as compiled_solve
from crosscut2.compiled_context_beam import CompiledBeam
from crosscut2.playback_suffix_arena import PlaybackSuffixArena
from crosscut2.context_port_dp import solve as reference_solve,ArcRef
from test_crosscut2_ordered_preference_consumer import ordered_fixture
from test_crosscut2_playback_suffix_arena import descriptor
from test_crosscut2_compiled_context_beam import fields


def fixture(nodes,*,kernel='numpy',nonlinear=False):
    if kernel=='numba':pytest.importorskip('numba')
    old=ordered_fixture(nodes,kernel=kernel);a=old.arcs
    entry,exit,dense=old.entries,old.exits,a.dense
    if nonlinear:
        # Real synthetic cache observations must describe the same nonlinear
        # source clock. The older nonlinear cost-only fixture deliberately
        # did NOT rebind those rasters and can legitimately have no programme.
        clock=old.cats[0].bank['programme_render_clock']
        offsets=np.linspace(.73,.11,len(nodes));offsets[1]=.61
        if len(nodes)>4:offsets[2]=.31;offsets[3]=.21
        clock['source_times']=np.asarray(nodes)/30.+offsets/30.;clock['count']+=2
        for name,stencil in (('native_indices',[-1,0]),('pose_native_indices',[-7,-1,0,6])):
            points=np.clip(np.asarray(nodes)[:,None]+stencil,0,int(nodes[-1])-1)
            seconds=np.interp(points.ravel(),nodes,clock['source_times']).reshape(points.shape)
            old.cats[0].bank[name]=(((seconds*1000.)/1000.)*30.+.5).astype(np.int64)
        paths=[[(nodes,curve[np.minimum(nodes,int(nodes[-1])-1)]) for curve in source] for source in dense]
        entry,exit,dense,_=make(nodes,old.cats,paths,fps=30.,coalesce_static_ports=True)
    arcs=ArcCosts(nodes,entry,exit,old.cats,dense,a.native,a.support,a.fragments,
        fps=a.fps,preserve_camera_alternatives=True,capture_playback_inputs=True,
        opportunity_episodes=np.zeros(len(nodes),int),roles=a.roles,point_geometry=a.point_geometry)
    context=old.model.prepare(nodes=nodes,entries=entry,exits=exit,catalogues=old.cats,arcs=arcs,
        native_ports=arcs.native,fps=30.,source_bindings=old.source_bindings,check=lambda:None,kernel=kernel)
    return context,CompiledPlayback(context)


def problem(context):
    nodes=context.nodes;entry=[[p.source for p in ports] for ports in context.entries]
    exit=[[p.source for p in ports] for ports in context.exits]
    def cuts(n):
        a=np.asarray(exit[n]);b=np.asarray(entry[n]);return np.zeros((len(a),len(b))),a[:,None]!=b[None,:]
    def holds(i,j):
        base,allowed=context.arcs.camera_alternatives(i,j);base=base.copy()
        base[:,:,4]+=np.finfo(float).eps
        return base,allowed
    kwargs=dict(start_cost=np.zeros(len(entry[0])),start_allowed=np.array(entry[0])==0,
        final_allowed=np.array(exit[-1])==1,hold_cost=holds,cut_cost=cuts,
        episodes=np.arange(len(nodes))//2,credit=lambda i,j:np.full(len(entry[i]),.025 if i else 0.),
        history_frames=120,label_limit=8)
    return entry,exit,kwargs


@pytest.mark.parametrize('kernel',['numpy','numba'])
@pytest.mark.parametrize('long',[False,True])
@pytest.mark.parametrize('nonlinear',[False,True])
def test_origin_major_owned_state_matches_legacy_path_codes_cost_paid_bits_and_independent_replay(kernel,long,nonlinear):
    nodes=np.array([0,30,180,210,240]) if long else np.array([0,18,36,54])
    context,compiler=fixture(nodes,kernel=kernel,nonlinear=nonlinear)
    entry,exit,kw=problem(context)
    old=reference_solve(nodes,entry,exit,context_cost=context.incremental,history_key=context.equivalent_key,
        context_lower_bound=context.lower_bound,long_context_batch=context.long_candidates,**kw)
    # The reference is finished; the new executor may use prepared source
    # arrays but may not consult tuple-history reconstruction/cost caches.
    def history_forbidden(*args,**kwargs):raise AssertionError('history reconstruction reached from opaque executor')
    context._encoded_history=history_forbidden;context._blocks=history_forbidden
    new=compiled_solve(nodes,entry,exit,compiler=compiler,**kw)
    assert new[0]==old[0] and new[1]==old[1]
    for key in ('selected_arc_codes','opportunity_claims','camera_alternatives_scored'):
        assert new[2][key]==old[2][key]
    np.testing.assert_array_equal(new[2]['selected_context_costs'],old[2]['selected_context_costs'])
    refs=[ArcRef(*run,code) for run,code in zip(new[0],new[2]['selected_arc_codes'])]
    parts,total=context.replay(refs)
    assert parts==pytest.approx(new[2]['selected_context_costs'],abs=2e-13)
    assert sum(parts)==pytest.approx(total,abs=2e-13)
    assert compiler.origin_decode_calls<=len(nodes)-1
    assert compiler.hold_compilations==new[2]['hold_calls']
    # Factored long alternatives still ALL enter relaxation. Their shared
    # head/tail/suffix pieces need not rebuild one entire recipe per p/q/code.
    full_recipes=compiler.recipe_compilations-sum(compiler.factored_piece_compilations.values())
    assert full_recipes+compiler.long_alternatives==new[2]['camera_alternatives_scored']
    if long:
        assert compiler.long_alternatives>0 and new[2]['factored_long_beam_kernel_calls']>0
        assert compiler.recipe_compilations<new[2]['camera_alternatives_scored']
    assert compiler.reference_frame_reuses>0
    if long and nonlinear:assert compiler.nonreference_recipes>0
    assert all(t['live']==0 for t in compiler.arena.receipt()['tables'].values())
    compiler.arena.close()


def test_after_recipe_compile_scoring_extension_and_decode_cannot_rebuild_raw_input(monkeypatch):
    nodes=np.array([0,18,36,54]);context,compiler=fixture(nodes,kernel='numba')
    a=compiler.arena;empty=a.empty();roots=np.zeros((len(context.entries[0]),2,8),np.uint64)
    costs=np.full(roots.shape,np.inf);roots[0,0,0]=empty.handle;costs[0,0,0]=0.
    with compiler.open_origin(0,roots,costs) as bank:
        base,allowed=context.arcs.camera_alternatives(0,1)
        p,q,code=map(int,np.argwhere(allowed&(np.arange(len(context.entries[0]))[:,None,None]==0))[0])
        with compiler.compile_hold(0,1,base,allowed) as hold:
            with hold.current(p,q,code) as current:
                def forbidden(*args,**kwargs):raise AssertionError('raw playback rebuilt after compile')
                monkeypatch.setattr(context,'_native',forbidden)
                monkeypatch.setattr(context,'_intent_projection',forbidden)
                monkeypatch.setattr(context.arcs,'sample_camera',forbidden)
                monkeypatch.setattr(context.model,'encode',forbidden)
                monkeypatch.setattr('crosscut2.compiled_played_context.pack',forbidden)
                extra=compiler.score(bank,current,p,0,[0]);assert np.isfinite(extra).all()
                root=a.append(empty,current.lease)
                with a.decode([root],end=18) as decoded:assert len(decoded.frames)==18
                a.release(root)
    a.release(empty);a.close()


def test_cancelled_search_releases_pending_future_roots_current_origin_and_current_leases():
    nodes=np.array([0,18,36,54]);context,compiler=fixture(nodes)
    entry,exit,kw=problem(context);calls=0
    def cancelled():
        nonlocal calls
        calls+=1
        if calls==5:raise RuntimeError('cancel finite fixture')
    with pytest.raises(RuntimeError,match='cancel finite'):
        compiled_solve(nodes,entry,exit,compiler=compiler,check=cancelled,**kw)
    assert compiler.hold_compilations>0 and compiler.active_origin is None
    assert all(t['live']==0 for t in compiler.arena.receipt()['tables'].values())
    compiler.arena.close()


def test_legacy_non_captured_context_cannot_silently_enter_new_executor():
    context=ordered_fixture(np.array([0,18,36]))
    with pytest.raises(ValueError,match='captured actual playback producer'):CompiledPlayback(context)


@pytest.mark.parametrize('kernel',['numpy','numba'])
def test_actual_general_current_compiles_but_cannot_demand_persistent_frames_before_beam_rank(kernel,monkeypatch):
    context,old=fixture(np.array([0,18,36]),kernel=kernel);old.arena.close()
    a=PlaybackSuffixArena(namespace=CompiledPlayback.namespace(context),embedding_width=2,history_frames=120,
        frame_capacity=96,cell_capacity=32,root_capacity=20,kernel=kernel)
    c=CompiledPlayback(context,arena=a);empty=a.empty();kept=[];beam=CompiledBeam(a);state=fields()
    for variant in range(8):
        with descriptor(a,0,12,lambda keys,v=variant:keys.__setitem__((slice(None),2),v+100)) as current:
            root=a.append(empty,current);kept.append(root)
        beam.cut(state,(float(variant+1),0,variant,0,0.,0.),root.handle)
    assert a.receipt()['tables']['frames']['live']==96
    P=len(context.entries[0]);ids=np.zeros((P,2,8),np.uint64);costs=np.full(ids.shape,np.inf)
    ids[0,0,0]=empty.handle;costs[0,0,0]=0.
    with c.open_origin(0,ids,costs):
        base,allowed=context.arcs.camera_alternatives(0,1)
        p,q,code=map(int,next(v for v in np.argwhere(allowed) if v[0]==0 and v[2]==4))
        with c.compile_hold(0,1,base,allowed) as hold:
            before=a.receipt()['tables']['frames']['requests'];snapshot=tuple(v.copy() for v in state)
            with hold.current(p,q,code) as current:
                assert current._lease is None and len(current.keys)==18
                def forbidden(*unused,**kwargs):raise AssertionError('rejected candidate persisted before rank')
                monkeypatch.setattr(a,'admit',forbidden)
                beam.offer_lazy(state,(100.,0,0,code,0.,.1),empty.handle,current)
                assert current._lease is None
            assert a.receipt()['tables']['frames']['requests']==before
            for actual,expected in zip(state,snapshot):np.testing.assert_array_equal(actual,expected)
    for handle in state[6]:a.release(type(empty)(a.owner,int(handle)))
    for root in kept:a.release(root)
    a.release(empty);a.close()


def test_compiled_actual_make_secondary_target_and_its_missing_face_gap_do_not_revert_to_native_focal():
    from test_crosscut2_playback_intent import secondary_target_fixture
    from test_crosscut2_ordered_preference_consumer import numeric_ordered_model
    scalar,_=secondary_target_fixture();old=scalar.arcs;nodes=scalar.nodes
    arcs=ArcCosts(nodes,scalar.entries,scalar.exits,scalar.cats,old.dense,old.native,old.support,old.fragments,
        fps=30.,preserve_camera_alternatives=True,capture_playback_inputs=True)
    context=numeric_ordered_model().prepare(nodes=nodes,entries=scalar.entries,exits=scalar.exits,catalogues=scalar.cats,
        arcs=arcs,native_ports=arcs.native,fps=30.,source_bindings=['a'*64,'b'*64],check=lambda:None)
    compiler=CompiledPlayback(context);a=compiler.arena;empty=a.empty()
    def origin(i,p,root):
        ids=np.zeros((len(context.entries[i]),2,8),np.uint64);costs=np.full(ids.shape,np.inf)
        ids[p,0,0]=root.handle;costs[p,0,0]=0.
        return compiler.open_origin(i,ids,costs)
    p=next(k for k,v in enumerate(context.entries[0]) if v.source==1)
    q=next(k for k,v in enumerate(context.exits[1]) if v.source==1)
    with origin(0,p,empty):
        base,allowed=arcs.camera_alternatives(0,1)
        with compiler.compile_hold(0,1,base,allowed) as hold:
            with hold.current(p,q,0) as current:past=a.append(empty,current.lease)
    p=next(k for k,v in enumerate(context.entries[1]) if v.source==0 and v.intent is not None
        and v.intent.pinned and v.observation['head_index']==1)
    q=next(k for k,v in enumerate(context.exits[2]) if v.source==0 and v.intent is None)
    with origin(1,p,past) as bank:
        base,allowed=arcs.camera_alternatives(1,2);assert allowed[p,q,4]
        with compiler.compile_hold(1,2,base,allowed) as hold:
            with hold.current(p,q,4) as current:
                seen=np.flatnonzero((current.frames==30)|(current.frames==31))
                assert (current.keys[seen,3:5]>0).any(axis=1).all()
                evidence=context._intent_projection(current.ref,current.frames[seen],current.native[seen],current.camera[seen],ordered=True)
                expected=context.model.encode(evidence)
                np.testing.assert_array_equal(current.encoded[seen],expected)
                original=context.model.encode(context.fields[0].project_ordered(
                    current.frames[seen],current.native[seen],current.camera[seen]))
                assert not np.array_equal(current.encoded[seen],original)
                assert np.isfinite(compiler.score(bank,current,p,0,[0])).all()
    a.release(past);a.release(empty);a.close()
