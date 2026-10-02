"""Pro F07: a reference vector must have existed as one measured comparison."""
import pytest

from crosscut.group_scene_observations import coherent_baseline, preserves_known


def edge(left,occupancy,attention,**kw):
    return dict(left_state=left,right_state='right',available=True,
        metrics=dict(occupancy=occupancy,attention=attention,**kw))


def test_reference_never_combines_incompatible_actor_minima():
    a,b=edge('a',.1,.8),edge('b',.7,.1)
    result=coherent_baseline([a,b])
    assert result['metrics']==b['metrics']
    assert result['metrics']!={'occupancy':.1,'attention':.1}
    assert result['coherent_reference_state_keys']==['b','right']
    assert result['observed_alternatives']==[a,b]
    assert preserves_known(result,edge('candidate',.6,.05))
    assert not preserves_known(result,edge('candidate',.8,.0))


def test_complete_observed_reference_precedes_cheaper_incomplete_one():
    partial=edge('partial',0.,0.);partial['available']=False
    full=edge('full',.3,.4)
    assert coherent_baseline([partial,full])['metrics']==full['metrics']


def test_ambiguous_focal_is_conditional_not_an_unconditional_actor_fact():
    row=edge('a',.1,.2,head_predicted=.05,focal_motion=.01)
    result=coherent_baseline([row],focal_assignment_unique=False)
    assert result['metrics']=={'occupancy':.1,'attention':.2}
    assert result['observed_alternatives'][0]['metrics']['head_predicted']==.05
    assert result['conditional_reference_is_not_actor_certainty']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.,True,'0'])
def test_nonfinite_or_non_numeric_metric_is_not_a_cheap_reference(value):
    bad=edge('bad',value,0.)
    result=coherent_baseline([bad])
    assert result['metrics']=={} and not result['available']
    assert not preserves_known({'metrics':{}},bad)


def test_missing_metrics_are_not_imputed_as_perfect_observations():
    assert coherent_baseline([])['coherent_reference_state_keys'] is None
    assert not coherent_baseline([{'metrics':{'occupancy':None},'available':True}])['available']
    assert not preserves_known({'metrics':{'occupancy':.2}},edge('c',None,.1))


def test_reference_tie_is_deterministic_and_keeps_exact_vector():
    a,b=edge('a',.2,.6),edge('b',.6,.2)
    assert coherent_baseline([b,a])['metrics']==a['metrics']
