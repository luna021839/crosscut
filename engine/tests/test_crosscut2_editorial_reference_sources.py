"""Editorial references guide intent, without excluding usable camera feeds."""
import pytest

from crosscut2.event_holds import select
from crosscut2.played_pose_port_programme import PlayedPosePortPrior
from test_crosscut2_preference_integration import producer


@pytest.mark.parametrize('bad',[(),(0,0),(-1,),(True,),('0',),(0.0,)])
def test_references_require_distinct_physical_indices(bad):
    with pytest.raises(ValueError,match='editorial reference'):
        PlayedPosePortPrior(editorial_opportunities=True,editorial_reference_sources=bad)


def test_explicit_references_cannot_silently_change_the_non_editorial_path():
    assert PlayedPosePortPrior().editorial_reference_sources is None
    with pytest.raises(ValueError,match='editorial reference'):
        PlayedPosePortPrior(editorial_reference_sources=(0,))


@pytest.mark.parametrize('cohort,references,expected',[
    (None,None,None),(None,(0,),(0,)),((1,),None,(1,)),((1,),(0,),(0,)),
])
def test_actual_producer_separates_reference_votes_and_render_admission(monkeypatch,cohort,references,expected):
    from crosscut2 import featured_member_state,programme_framing,played_pose_port_programme
    args,kw,native,_,_,_=producer(opposed=False,one_arm=True)
    for bank in args[1]:
        bank['programme_render_clock']=dict(fps=30.,count=int(args[0][-1]),source_times=args[0]/30.)
    calls=[]
    original_vote=featured_member_state.consensus
    original_frame=programme_framing.evidence
    original_domain=played_pose_port_programme.make

    def vote(*a,**k):
        calls.append(('vote',k['source_cohort']))
        return original_vote(*a,**k)

    def framing(*a,**k):
        calls.append(('framing',k['source_cohort']))
        return original_frame(*a,**k)

    def domain(*a,**k):
        result=original_domain(*a,**k)
        calls.append(('render_domain',k['source_cohort']))
        sources={p.source for row in (*result[0],*result[1]) for p in row}
        assert sources==({0,1} if cohort is None else set(cohort))
        return result

    monkeypatch.setattr(featured_member_state,'consensus',vote)
    monkeypatch.setattr(programme_framing,'evidence',framing)
    monkeypatch.setattr(played_pose_port_programme,'make',domain)
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,
        source_cohort=cohort,editorial_reference_sources=references)
    actual=select(*args,**kw,native_ports=native,music=[0,1,1,0],connection_prior=prior)
    receipt=actual[4]
    assert calls==[('vote',expected),('framing',expected),('render_domain',cohort)]
    expected_indices=[0,1] if expected is None else list(expected)
    assert receipt['editorial_reference_source_indices']==expected_indices
    assert receipt['featured_member_state']['voting_source_indices']==expected_indices
    assert receipt['original_programme_framing']['original_source_indices']==expected_indices
    assert prior.artifact['explicit_editorial_reference_sources']==references
    assert prior.artifact['editorial_reference_sources_do_not_exclude_render_sources']


def test_out_of_domain_references_fail_before_any_source_work():
    args,kw,native,_,_,_=producer(opposed=False)
    with pytest.raises(ValueError,match='editorial reference sources escape'):
        select(*args,**kw,native_ports=native,music=[0,1,1,0],
            connection_prior=PlayedPosePortPrior(editorial_opportunities=True,editorial_reference_sources=(2,)))
