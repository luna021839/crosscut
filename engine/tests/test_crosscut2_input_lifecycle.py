"""Integrity/cancellation lifecycle; not input attestation or video quality."""
import os
from types import SimpleNamespace
import numpy as np
import pytest
from crosscut2.input_lifecycle import FrozenInputCheck,identity
from crosscut2.playback_workspace import PlaybackWorkspace


def fixture(tmp_path,*,tick=lambda:None):
    paths=[tmp_path/f'source{i}.bin' for i in range(8)]
    for i,path in enumerate(paths):path.write_bytes(bytes([i])*16)
    check=FrozenInputCheck(sources=paths,identities=[identity(p) for p in paths],
        device_root=tmp_path,device=tmp_path.stat().st_dev,tick=tick)
    a=np.arange(12);a.setflags(write=False)
    return check,paths,((a,),)


def test_normal_source_reading_calls_still_verify_every_original(tmp_path):
    c,paths,groups=fixture(tmp_path);before=c.source_checks
    for _ in range(100):c()
    assert c.source_checks-before==800 and c.calls==100
    paths[3].write_bytes(b'changed')
    with pytest.raises(OSError,match='original source changed'):c()
    assert c.failed


def test_10000_numeric_checks_keep_device_and_cancel_live_without_80000_video_stats(tmp_path):
    ticks=[];c,paths,groups=fixture(tmp_path,tick=lambda:ticks.append(1));before=c.source_checks
    with c.frozen_numeric_phase(groups):
        entered=c.source_checks;devices=c.device_checks
        for _ in range(10000):c()
        assert c.source_checks==entered and c.device_checks-devices==10000
        assert len(ticks)==c.device_checks
    assert c.source_checks-before==16 and not c.numeric_active and not c.failed
    c();assert c.source_checks-before==24


def test_changed_source_during_numeric_phase_cannot_publish_after_exit(tmp_path):
    c,paths,groups=fixture(tmp_path);published=[]
    with pytest.raises(OSError,match='original source changed'):
        with c.frozen_numeric_phase(groups):
            paths[2].write_bytes(b'different');c()
        published.append('MUST NOT REACH publication')
    assert not published and c.failed and not c.numeric_active
    with pytest.raises(RuntimeError,match='already failed'):c()


def test_explicit_origin_checkpoint_detects_change_inside_phase(tmp_path):
    c,paths,groups=fixture(tmp_path)
    with pytest.raises(OSError,match='original source changed'):
        with c.frozen_numeric_phase(groups):
            paths[0].write_bytes(b'different');c.source_checkpoint()
    assert c.failed and not c.numeric_active


def test_device_change_fails_on_next_inner_check_not_only_at_phase_end(tmp_path,monkeypatch):
    c,paths,groups=fixture(tmp_path);stat=os.stat
    with pytest.raises(OSError,match='device changed'):
        with c.frozen_numeric_phase(groups):
            def changed(path,*a,**kw):
                if os.fspath(path)==os.fspath(tmp_path):return SimpleNamespace(st_dev=c.device+1)
                return stat(path,*a,**kw)
            monkeypatch.setattr(os,'stat',changed);c()
    assert c.failed and not c.numeric_active


def test_deadline_cancellation_and_original_error_survive_cleanup(tmp_path):
    cancelled=False
    def tick():
        if cancelled:raise TimeoutError('finite cancellation')
    c,paths,groups=fixture(tmp_path,tick=tick)
    with pytest.raises(TimeoutError,match='finite cancellation'):
        with c.frozen_numeric_phase(groups):cancelled=True;c()
    assert c.failed and not c.numeric_active


def test_writable_or_fake_buffers_cannot_enter_numeric_mode(tmp_path):
    c,paths,groups=fixture(tmp_path)
    for bad in ((),((np.zeros(1),),),((None,),),[groups[0]]):
        with pytest.raises(ValueError,match='immutable prepared'):
            with c.frozen_numeric_phase(bad):raise AssertionError('must not enter')
    assert not c.numeric_active and not c.failed


def test_nested_numeric_phase_and_return_to_strict_reading_keep_all_boundaries(tmp_path):
    c,paths,groups=fixture(tmp_path);before=c.source_checks
    with c.frozen_numeric_phase(groups):
        c()
        with c.frozen_numeric_phase(groups):c()
        assert c.numeric_active;c()
    assert c.source_checks-before==32 and not c.numeric_active
    c();assert c.source_checks-before==40


def test_workspace_groups_are_the_real_frozen_owners_not_copied_array_storage():
    w=PlaybackWorkspace(2**20);a=np.arange(8);w.pin_frozen_inputs(256,[a],owner='numeric-fixture')
    groups=w.frozen_input_groups();assert groups[0][0] is a and not a.flags.writeable


def test_actual_compiled_path_keeps_codes_costs_and_credit_with_source_checkpoints(tmp_path):
    from crosscut2.compiled_context_port_dp import solve
    from crosscut2.compiled_played_context import CompiledPlayback
    from test_crosscut2_compiled_context_port_dp import fixture as actual,problem
    context,c=actual(np.array([0,18,36,54]));entry,exit,kw=problem(context)
    before=solve(context.nodes,entry,exit,compiler=c,**kw);c.arena.close()
    guard,paths,groups=fixture(tmp_path);previous=context.arcs.check
    def checked():previous();guard()
    checked.input_lifecycle=guard;context.arcs.check=checked;c=CompiledPlayback(context)
    start=guard.source_checks
    with guard.frozen_numeric_phase(c.arena.workspace.frozen_input_groups()):
        after=solve(context.nodes,entry,exit,compiler=c,**kw)
    assert after[:2]==before[:2]
    for key in ('selected_arc_codes','opportunity_claims','selected_context_costs'):
        assert after[2][key]==before[2][key]
    assert guard.source_checks-start>=24 and not guard.failed and not guard.numeric_active
    assert all(v['live']==0 for v in c.arena.receipt()['tables'].values());c.arena.close()


def test_changed_source_at_actual_next_origin_aborts_and_releases_all_future_roots(tmp_path):
    from crosscut2.compiled_context_port_dp import solve
    from test_crosscut2_compiled_context_port_dp import fixture as actual,problem
    context,c=actual(np.array([0,18,36]));entry,exit,kw=problem(context)
    guard,paths,groups=fixture(tmp_path);previous=context.arcs.check
    def checked():previous();guard()
    checked.input_lifecycle=guard;context.arcs.check=checked;opened=c.open_origin
    def change(i,roots,costs):
        if i:paths[0].write_bytes(b'changed while numeric inputs remain immutable')
        return opened(i,roots,costs)
    c.open_origin=change
    with pytest.raises(OSError,match='original source changed'):
        with guard.frozen_numeric_phase(c.arena.workspace.frozen_input_groups()):
            solve(context.nodes,entry,exit,compiler=c,**kw)
    assert guard.failed and not guard.numeric_active and c.active_origin is None
    assert all(v['live']==0 for v in c.arena.receipt()['tables'].values());c.arena.close()
