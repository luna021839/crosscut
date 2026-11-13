import json
import pytest
from crosscut.group_review_realized_path import RealizedReviewVariant,select_realized_review_path
from crosscut.group_review_exact_path import (precompute_review_edge_graph,
    review_two_sided_reachability,select_exact_realized_review_path,nonblocking_beam_diagnostic)


def variant(a,b,name,utility=0.,changed=0):
    return RealizedReviewVariant(a,b,json.dumps(dict(camera_identity=name,sampler_identity='sampler',
        actor_path_identity='actor',source_identity='source',quality_approved=False)),utility,changed)


def edge(a,b,allowed=True,cuts=0):
    return dict(left_identity=a.identity,right_identity=b.identity,boundary_tick=b.first,
        cost=0.,allowed=allowed,visible_boundary_event_count=cuts)


def test_exact_path_preserves_low_prefix_bridge_lost_by_beam16():
    starts=[variant(0,10,str(i)) for i in range(17)]
    bridge=variant(0,10,'BRIDGE',-1.,10)
    keep=variant(10,20,'KEEP');gain=variant(10,20,'GAIN',10.,10)
    layers=[[*starts,bridge],[keep,gain]]
    def at(a,b):return edge(a,b,b is keep or a is bridge)
    graph=precompute_review_edge_graph(layers,at)
    exact=select_exact_realized_review_path(layers,graph,output_frame_count=20)
    beam=select_realized_review_path(layers,at,output_frame_count=20,maximum_beam=16)
    assert exact['utility']==9. and exact['changed_frames']==20
    assert beam['utility']==0. and beam['changed_frames']==0
    assert not exact['across_suffix_beam_pruning']


def test_third_candidate_with_entry_and_exit_survives_static_top2_failure():
    a=variant(0,10,'fixed');end=variant(20,30,'exit')
    best=variant(10,20,'high-utility',9.,10);second=variant(10,20,'high-scale',8.,10)
    third=variant(10,20,'two-sided',7.,10)
    layers=[[a],[best,second,third],[end]]
    def at(x,y):return edge(x,y,x is not best and y is not second)
    graph=precompute_review_edge_graph(layers,at)
    result=select_exact_realized_review_path(layers,graph,output_frame_count=30)
    assert result['variants'][1] is third
    reach=review_two_sided_reachability(layers,graph,maximum_changed_frames=10)
    donor={v['identity']:v for v in reach['variants']}
    assert donor[third.identity]['reachable_with_change_budget']
    assert donor[best.identity]['backward_minimum_changed'] is None
    assert donor[second.identity]['forward_minimum_changed'] is None


def test_adjacent_donors_that_only_work_together_are_not_separately_dropped():
    first=variant(0,10,'entry');last=variant(30,40,'exit')
    k1=variant(10,20,'keep1');k2=variant(20,30,'keep2')
    a=variant(10,20,'donor1',2.,10);b=variant(20,30,'donor2',3.,10)
    layers=[[first],[k1,a],[k2,b],[last]]
    graph=precompute_review_edge_graph(layers,lambda x,y:edge(x,y,
        not (x is a and y is k2 or x is k1 and y is b)))
    result=select_exact_realized_review_path(layers,graph,output_frame_count=40)
    assert result['variants'][1:3]==(a,b) and result['utility']==5.


def test_exact_selector_carries_change_and_actual_cut_budgets():
    a=variant(0,10,'keep');b=variant(0,10,'donor',10.,10);c=variant(10,20,'fixed')
    layers=[[a,b],[c]]
    graph=precompute_review_edge_graph(layers,lambda x,y:edge(x,y,cuts=2 if x is b else 0))
    chosen=select_exact_realized_review_path(layers,graph,output_frame_count=20,maximum_visible_cuts=1)
    assert chosen['variants'][0] is a
    chosen=select_exact_realized_review_path(layers,graph,output_frame_count=20,maximum_changed_frames=5)
    assert chosen['variants'][0] is a


def test_graph_budget_is_checked_before_scoring_and_truncation_is_not_infeasible():
    a,b=variant(0,10,'a'),variant(0,10,'b');c=variant(10,20,'c')
    calls=[]
    graph=precompute_review_edge_graph([[a,b],[c]],lambda x,y:calls.append((x,y)),maximum_edges=1)
    assert graph['state']=='TRUNCATED' and not calls
    full=precompute_review_edge_graph([[a,b],[c]],edge)
    result=select_exact_realized_review_path([[a,b],[c]],full,output_frame_count=20,maximum_states=1)
    assert result['state']=='TRUNCATED' and not result['no_feasible_path_claimed']


def test_actual_neighbor_receipt_cannot_be_substituted():
    a=variant(0,10,'a');b=variant(10,20,'b')
    with pytest.raises(ValueError,match='neighbor identity'):
        precompute_review_edge_graph([[a],[b]],lambda x,y:dict(edge(x,y),left_identity=y.identity))


def test_unconstrained_beam_does_not_report_false_search_loss_or_run():
    exact=dict(state='COMPLETE',utility=0.,changed_frames=0);calls=[]
    diagnostic=nonblocking_beam_diagnostic(exact,lambda:calls.append(1),constraints_equivalent=False)
    assert diagnostic['state']=='NOT_COMPARABLE_CONSTRAINTS' and not calls
    assert diagnostic['beam_lost_utility'] is None and diagnostic['exact_result_retained']


def test_beam_exception_never_discards_complete_exact_bridge():
    starts=[variant(0,10,str(i),i) for i in range(17)]
    final=variant(10,20,'only-bridge',100.)
    layers=[starts,[final]]
    def at(a,b):return edge(a,b,a is starts[0])
    graph=precompute_review_edge_graph(layers,at)
    exact=select_exact_realized_review_path(layers,graph,output_frame_count=20)
    diagnostic=nonblocking_beam_diagnostic(exact,lambda:select_realized_review_path(layers,at,
        output_frame_count=20,maximum_beam=16),constraints_equivalent=True)
    assert exact['utility']==100. and diagnostic['state']=='DIAGNOSTIC_FAILED'
    assert diagnostic['exact_result_retained']
