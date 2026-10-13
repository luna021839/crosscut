from types import SimpleNamespace
from crosscut2.native_timing import flash_fragments


def schedule(start,size):
    return (start,SimpleNamespace(),SimpleNamespace(source_frame_indices=list(range(size))))


def test_one_frame_native_fragment_at_exit_is_not_hidden_by_authored_count():
    rows=flash_fragments([schedule(0,100),schedule(100,100)],[99])
    assert rows==[dict(segment=0,first_frame=99,final_frame=100,frames=1)]


def test_native_cut_exactly_on_authored_boundary_is_one_event_not_a_fragment():
    assert flash_fragments([schedule(0,100),schedule(100,100)],[100])==[]


def test_short_interior_native_burst_is_still_reported():
    assert flash_fragments([schedule(0,100)],[40,43])==[
        dict(segment=0,first_frame=40,final_frame=43,frames=3)]


def test_exact_schedule_repair_removes_exit_flash_without_new_observations(tmp_path,monkeypatch):
    from crosscut.render import AspectRatio, RenderPlan, RenderSegment
    from crosscut import render_frame_schedule
    from crosscut2.native_timing import repair_native_timing
    sources=(tmp_path/'one.mp4',tmp_path/'two.mp4')
    def build(bounds):
        return RenderPlan(sources,tuple(RenderSegment(i,float(bounds[i]/10),
            float((bounds[i+1]-bounds[i])/10),float(bounds[i]/10)) for i in range(2)),
            0,AspectRatio.LANDSCAPE,tmp_path/'final.mp4',fps=10.)
    monkeypatch.setattr(render_frame_schedule,'probe_render_frame_clock',
        lambda path:SimpleNamespace(frame_rate=10.,frame_count=300))
    bounds,receipt=repair_native_timing(build([0,100,200]),build,
        [dict(frames=[99]),dict(frames=[])],check=lambda:None)
    assert bounds.tolist()==[0,99,200]
    assert receipt['fragments_before'] and not receipt['fragments_after']
    assert receipt['new_source_inferences']==0
    assert receipt['changes'][0]['new_pose_port_measured'] is False
