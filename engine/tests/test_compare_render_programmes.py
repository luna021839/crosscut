"""Numeric programme comparisons must not confuse source index with identity."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from compare_render_programmes import verify_common_sources


def record(paths,hashes):return dict(render_plan=dict(sources=paths),source_sha256=hashes)


def test_added_or_reordered_sources_keep_physical_identity():
    old=record(['/one','/two'],['a'*64,'b'*64])
    new=record(['/three','/two','/one'],['c'*64,'b'*64,'a'*64])
    ids=verify_common_sources(old,new)
    assert len(ids)==3 and ids['/one']!=ids['/two']
    assert ids==verify_common_sources(new,old)


def test_changed_common_source_is_not_reported_as_an_editorial_improvement():
    with pytest.raises(ValueError,match='changed content'):
        verify_common_sources(record(['/one'],['a'*64]),record(['/one'],['b'*64]))


@pytest.mark.parametrize('paths,hashes',[(['/one','/one'],['a'*64,'a'*64]),(['/one'],[])])
def test_ambiguous_source_domain_rejected(paths,hashes):
    with pytest.raises(ValueError,match='unique original sources'):
        verify_common_sources(record(paths,hashes),record(['/two'],['b'*64]))
