"""App-final adapter for sequential native-clock programme rendering.

Only supported hard-cut/face-dissolve films enter this backend. It does not
choose a source, change a camera, retry a codec, or grant quality approval.
"""
import math
from crosscut.render import segment_frame_counts
from . import bounded_plan_render


def render(plan,*,ffmpeg,video_codec,maximum_runtime_seconds,progress_callback):
    counts=segment_frame_counts(plan);completed=0;last_pid=None;last_frame=-1
    def progress(event):
        nonlocal completed,last_pid,last_frame
        if event.get('pid') is not None:last_pid=event['pid']
        if event.get('phase')!='bounded_segment_complete':return
        completed=sum(counts[:event['completed']])
        # Final audio mux/whole AV decode still remain. Never publish100%
        # from a temporary pure block, or reset progress for every new PID.
        actual=min(plan.output_frame_count-1,completed)
        if progress_callback is not None and last_pid is not None and actual>last_frame:
            progress_callback(dict(phase='continue',pid=last_pid,frame=actual,
                out_time_us=round(actual/plan.fps*1e6),decoder_concurrency=1))
            last_frame=actual
    seconds=(max(660.,3.*plan.render_duration+120.) if maximum_runtime_seconds is None
        else float(maximum_runtime_seconds))
    if not math.isfinite(seconds) or seconds<=0:raise ValueError('finite final render deadline required')
    result=bounded_plan_render.render(plan,ffmpeg=ffmpeg,video_codec=video_codec,
        maximum_seconds=min(1800.,seconds),progress_callback=progress)
    if progress_callback is not None and last_pid is not None:
        progress_callback(dict(phase='end',pid=last_pid,frame=result['output_frames'],
            out_time_us=round(result['output_frames']/plan.fps*1e6),decoder_concurrency=1,
            encoder_peak_RSS_bytes=result.get('encoder_peak_RSS_bytes')))
    return video_codec
