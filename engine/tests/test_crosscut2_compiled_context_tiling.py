"""Lossless ragged frame tiling, not source/actor thinning or AV training."""
from dataclasses import replace
import gc
import numpy as np
import pytest
from crosscut2.ordered_context import project
from crosscut2.scene_targets import FORMATION
from crosscut2.compiled_played_context import CompiledPlayback
from test_crosscut2_ordered_context import observed_scene
from test_crosscut2_ordered_preference_consumer import numeric_ordered_model
from test_crosscut2_compiled_context_port_dp import fixture


@pytest.mark.parametrize('chunk',[1,3,7,17])
def test_pointwise_encoder_is_bitwise_invariant_with_seventeen_untruncated_people(chunk):
    scene=observed_scene(37)
    names=('heads','head_observed','body','body_evidence','body_ambiguity',
        'roi_readability','roi_measured','original_heads','original_observed')
    values={n:np.repeat(getattr(scene,n),17,axis=1) for n in names}
    values['heads'][:,:,0]+=np.linspace(-.18,.18,17)[None]
    values['original_heads'][:,:,0]=values['heads'][:,:,0]
    values['body'][:,:,:,0]+=np.linspace(-.18,.18,17)[None,:,None]
    values['body'][:,:,4,0]+=.03*np.sin(np.arange(37)[:,None]/5.+np.arange(17)[None])
    evidence=project(replace(scene,role=FORMATION,**values),np.tile([1.2,.5,.5],(37,1)))
    model=numeric_ordered_model();whole=model.encode(evidence)
    parts=[model.encode(evidence.take(np.arange(a,min(37,a+chunk)))) for a in range(0,37,chunk)]
    np.testing.assert_array_equal(np.concatenate(parts),whole)
    assert np.diff(evidence.person_splits).tolist()==[17]*37


def test_actual_projection_is_split_under_pressure_without_dropping_any_frame_or_changing_bits():
    context,c=fixture(np.array([0,18,36,54]));a=c.arena;empty=a.empty()
    P=len(context.entries[0]);ids=np.zeros((P,2,8),np.uint64);costs=np.full(ids.shape,np.inf)
    ids[0,0,0]=empty.handle;costs[0,0,0]=0.
    with c.open_origin(0,ids,costs):
        base,allowed=context.arcs.camera_alternatives(0,1)
        p,q,code=map(int,next(row for row in np.argwhere(allowed) if row[0]==0 and row[2]==4))
        with c.compile_hold(0,1,base,allowed) as hold:
            with hold.current(p,q,code) as current:
                expected=current.encoded.copy();used=np.arange(len(current.frames))
                single=max(c.projection_requirement(current.ref,current.native[k:k+1]) for k in used)
                room=single+64*2**10
                pressure=a.workspace.claim(a.workspace.limit_bytes-a.workspace.current_bytes-room,'finite-tiling-pressure-fixture')
                before=c.encoded_frames
                try:c.encode_missing(current,used)
                finally:a.workspace.release(pressure)
                np.testing.assert_array_equal(current.encoded,expected)
                assert c.encoded_frames-before==len(used) and c.projection_tile_splits>0
    a.release(empty);a.close()


def test_unknown_temporal_encoder_cannot_silently_claim_frame_tiling_is_safe():
    context,c=fixture(np.array([0,18,36]))
    context.model.pointwise_frame_encoder=False
    with pytest.raises(ValueError,match='pointwise encoder contract'):CompiledPlayback(context)
    c.arena.close()


def test_reference_only_long_hold_does_not_reserve_nonexistent_whole_length_direct_paths():
    context,c=fixture(np.array([0,30,180,210,240]));w=c.arena.workspace
    context.arcs.opportunity_episodes=np.arange(5)*10
    initial=w.current_bytes
    with c.reserve_hold(0,1) as short:short_bytes=w.current_bytes-initial
    assert short_bytes==c.producer_requirement(0,1)
    del short;gc.collect()
    assert w.current_bytes==initial
    with c.reserve_hold(0,4) as long:long_bytes=w.current_bytes-initial
    assert long_bytes==c.producer_requirement(0,4)
    assert long_bytes<short_bytes
    del long;gc.collect();c.arena.close()


def test_all_pair_producer_preflight_is_the_runtime_bound_without_producing_camera_inputs(monkeypatch):
    context,c=fixture(np.array([0,14,37,158,240]));w=c.arena.workspace
    initial=w.current_bytes;calls=context.arcs.calls
    def forbidden(*unused):raise AssertionError('preflight must not produce camera inputs')
    monkeypatch.setattr(context.arcs,'camera_alternatives',forbidden)
    maximum=0
    for i in range(len(context.nodes)-1):
        for j in range(i+1,len(context.nodes)):
            need=c.producer_requirement(i,j);maximum=max(maximum,need)
            with c.reserve_hold(i,j) as guard:assert w.current_bytes-initial==need
            del guard;gc.collect();assert w.current_bytes==initial
    assert maximum>0 and context.arcs.calls==calls
    c.arena.close()
