from dataclasses import replace
import numpy as np
from crosscut2.event_states import build
from crosscut2.physical_display_observations import bind, duplicate_wrists, focal_metadata
from test_crosscut2_event_holds import inputs


def make():
    nodes,banks,poses,styles=inputs()
    states,rows,_=build(nodes,banks,poses,styles,fps=30.,source_events=[[],[]],body_states=None)
    return nodes,banks,poses,styles,states,rows


def test_physical_group_content_pose_are_actor_alias_invariant():
    nodes,banks,_,styles,states,rows=make()
    ss,rr,report=bind(nodes,states,rows,banks,styles)
    renamed=tuple(replace(s,actor=None if s.actor is None else s.actor+1000) for s in states)
    other_s,other_r,_=bind(nodes,renamed,rows,banks,styles)
    assert len(ss)==len(other_s)==8 and report['actor_aliases_removed']>0
    for a,b in zip(rr,other_r):
        for k in ('heads','body','availability','event_loss_rate','primitive_evidence_kind'):
            np.testing.assert_array_equal(a[k],b[k])
        assert a['physical_group_focal_bound'] and not a['actor_alias_changes_cost']


def test_fourth_displayed_face_never_borrows_top_three_pose():
    nodes,banks,_,styles,states,rows=make()
    for b in banks:b['heads'][:,:,3]=[.5,.3,.4,.99]
    _,physical,_=bind(nodes,states,rows,banks,styles)
    for row in physical:
        assert (row['indices']==3).all()
        assert not row['body'].any() and not row['primitive_evidence_kind'].any()
        assert row['availability'].min()>0


def test_no_face_not_lower_body_and_ambiguous_two_heads_is_not_certified():
    heads=np.zeros((2,2,8,4));camera=np.tile([1.,.5,.5],(2,2,1))
    heads[1,:,0]=[.4,.3,.2,.99];heads[1,:,1]=[.6,.3,.2,.99]
    _,indices,ambiguity,scene=focal_metadata(heads,camera)
    assert (indices[0]==-1).all() and (scene[0]==0).all()
    assert (ambiguity[1]>.99).all() and (scene[1]==3).all()
    assert not (scene==4).any()


def test_duplicate_background_wrist_keeps_ownership_ambiguity():
    body=np.zeros((1,4,3,8,3));body[:,:,0,4]=[.5,.5,.9];body[:,:,1,4]=[.5,.5,.9]
    ambiguous=duplicate_wrists(body)
    assert ambiguous[:,:,0,4].all() and ambiguous[:,:,1,4].all()
    assert not ambiguous[:,:,2].any()
