"""Actual make -> exact replay consumer, not a synthetic aesthetic verdict."""
from dataclasses import replace
import numpy as np
import pytest
from crosscut2.boundary_port_domain import make
from crosscut2.boundary_arc_costs import ArcCosts
from crosscut2.context_port_dp import ArcRef
from crosscut2.display_source_catalogue import SourceCatalogue
from crosscut2.event_holds import fragment_costs
from crosscut2.played_context import PlayedContext
from crosscut2.played_event_support import PlayedEventSupport
from crosscut2.sequence_context import PRIMITIVES
from test_crosscut2_sequence_preference_consumer import prepared, numeric_model


def secondary_target_fixture():
    nodes=np.array([0,30,60]);cats=prepared(nodes)
    for index,cat in enumerate(cats):
        cat.bank['heads'][...]=0.;cat.bank['vectors'][...]=0.;cat.bank['zoom_cap']=2.
        if index==0:
            cat.bank['heads'][:,:,0]=[.3,.375,.18,.99]
            cat.bank['heads'][:,:,1]=[.7,.375,.10,.99]
            cat.bank['vectors'][:,:,0,0]=1.;cat.bank['vectors'][:,:,1,1]=1.
        else:
            cat.bank['heads'][:,:,0]=[.7,.375,.20,.99]
            cat.bank['vectors'][:,:,0,1]=1.
        cats[index]=SourceCatalogue(nodes,cat.bank,None,[],cat.style,[],fps=30.)
    paths=[[(nodes,np.tile([1.,.5,.5],(len(nodes),1))) for _ in range(4)] for _ in cats]
    entries,exits,dense,domain=make(nodes,cats,paths,fps=30.,coalesce_static_ports=True)
    events=[np.zeros(0,int)]*2
    native=[dict(frames=np.zeros(0,int),heads=np.zeros((0,2,8,4)),native_event_keys=[]) for _ in cats]
    support=PlayedEventSupport(events,final=60,fps=30.)
    arcs=ArcCosts(nodes,entries,exits,cats,dense,native,support,
        fragment_costs(nodes,events,fps=30.,support_seconds=.4),fps=30.,preserve_camera_alternatives=True)
    scorer=PlayedContext(numeric_model(),nodes=nodes,entries=entries,exits=exits,catalogues=cats,
        arcs=arcs,native_ports=native,fps=30.,source_bindings=['a'*64,'b'*64],check=lambda:None)
    return scorer,domain


def test_actual_make_secondary_target_is_not_replaced_by_original_focal_in_context():
    scorer,domain=secondary_target_fixture()
    assert domain['retained_pairs_using_non_native_focal']==2
    for side,ports,ordinal in ((1,scorer.entries[1],30),(0,scorer.exits[1],29)):
        matches=[(k,p) for k,p in enumerate(ports) if p.source==0 and p.observation['target_pinned_before_camera']
            and p.observation['head_index']==1]
        assert len(matches)==1
        k,port=matches[0]
        assert port.intent.pinned and port.observation['target_is_displayed_focal']
        p=k if side else next(k for k,p in enumerate(scorer.entries[0]) if p.source==0)
        q=next(k for k,p in enumerate(scorer.exits[2]) if p.source==0) if side else k
        ref=ArcRef(1 if side else 0,2 if side else 1,p,q,4)
        _,allowed=scorer.arcs.camera_alternatives(ref.first,ref.final)
        assert allowed[p,q,4]
        played=scorer._part(ref,ordinal,ordinal+1)
        evidence=scorer._project_intents(ref,np.array([ordinal]),np.array([ordinal]),port.strip[:1])
        assert np.array_equal(played.values,evidence.values) and np.array_equal(played.observed,evidence.observed)
        def value(name):
            i=PRIMITIVES.index(name);assert evidence.observed[0,i];return evidence.values[0,i]
        assert value('feature_center_x')==pytest.approx(.7,abs=1e-7)
        assert value('target_visible_fraction')==pytest.approx(1.)
        assert value('known_head_crop_fraction')==0.
        assert not evidence.conflicted.any()


def test_intent_gaps_are_not_filled_with_competing_original_actor():
    scorer,_=secondary_target_fixture()
    p=next(k for k,p in enumerate(scorer.entries[1]) if p.source==0 and p.intent is not None and p.intent.pinned)
    q=next(k for k,p in enumerate(scorer.exits[2]) if p.source==0)
    ref=ArcRef(1,2,p,q,4);frames=np.array([31])
    camera=scorer.entries[1][p].strip[:1]
    evidence=scorer._project_intents(ref,frames,frames,camera)
    assert evidence.values[0,PRIMITIVES.index('role_feature')]==1.
    for name in ('feature_center_x','target_visible_fraction','known_head_crop_fraction'):
        assert not evidence.observed[0,PRIMITIVES.index(name)]


def test_intent_binding_is_immutable_and_camera_independent_and_native_cuts_bound_it():
    from crosscut2.playback_intent import before_camera
    cats=prepared(np.array([0,30,60]));cat=cats[0]
    cat.bank['native_cut_frames']=np.array([30])
    a=before_camera(cat,0,1,0,0);b=before_camera(cat,0,1,1,0)
    assert a.final_native<=30 and b.first_native>=30
    bound,key=a.bind('a'*64)
    assert bound.scene.binding=='a'*64 and not bound.scene.heads.flags.writeable
    assert key==a.bind('a'*64)[1] and key!=a.bind('b'*64)[1]
    assert replace(a,final_native=a.final_native+1).bind('a'*64)[1]!=key
    with pytest.raises(ValueError):bound.scene.heads[0,0,0]=.1


def test_actual_dense_source_binding_and_local_native_cut_survive_real_port_production():
    from crosscut2.playback_intent import before_camera
    from test_crosscut2_dense_subject_catalogue import prepared as dense_prepared
    a,_,_=dense_prepared(0,cut=(100,));b,_,_=dense_prepared(1,cut=(100,));cats=[a,b]
    for side in (0,1):
        intent=before_camera(a,0,1,side,0)
        assert intent.scene.binding=='0'
        assert intent.bind('0')[0].scene.binding=='0'
        with pytest.raises(ValueError,match='physical source binding changed'):intent.bind('different-source')
        if side==0:assert intent.final_native<=100
        else:assert intent.first_native>=100
    paths=[[(a.nodes,np.tile([1.,.5,.5],(len(a.nodes),1))) for _ in range(4)] for _ in cats]
    entry,exit,_,_=make(a.nodes,cats,paths,fps=30.,coalesce_static_ports=True)
    assert all(p.intent.scene.binding==str(p.source) for row in (*entry,*exit) for p in row if p.intent is not None)
    assert all(p.intent is None for row in (*entry,*exit) for p in row if not p.observation['target_pinned_before_camera'])


def test_original_input_equality_after_hash_collision_is_bitwise_not_signed_zero_or_dtype_alias():
    from crosscut2.playback_intent import before_camera,exact_array_input
    a=np.array([0.,1.]);b=a.copy();b[0]=-0.
    assert np.array_equal(a,b) and not exact_array_input(a,b)
    assert not exact_array_input(a,a.reshape(1,2))
    assert not exact_array_input(a,a.view(np.uint64))
    intent=before_camera(prepared(np.array([0,30,60]))[0],0,1,1,0)
    plus=intent.scene.heads.copy();minus=plus.copy();plus[0,0,0]=0.;minus[0,0,0]=-0.
    x=replace(intent,scene=replace(intent.scene,heads=plus));y=replace(intent,scene=replace(intent.scene,heads=minus))
    assert not x.same_input(y)
