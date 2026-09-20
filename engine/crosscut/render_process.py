"""Observe actual FFmpeg output progress without buffering stderr in memory."""
from pathlib import Path
import subprocess
import threading


def run_observed_render(
    command,
    *,
    stderr_path: Path,
    progress_callback,
    timeout_seconds: float | None = None,
):
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise ValueError("render timeout_seconds must be positive")
    command=list(command)
    command[1:1]=['-progress','pipe:1','-nostats']
    with stderr_path.open('w+b') as errors:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=errors,text=True)
        timed_out = threading.Event()
        timer = None
        if timeout_seconds is not None:
            def stop_timed_out_render() -> None:
                timed_out.set()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

            timer = threading.Timer(
                float(timeout_seconds),
                stop_timed_out_render,
            )
            timer.daemon = True
            timer.start()
        try:
            record={}
            assert process.stdout is not None
            for line in process.stdout:
                key,separator,value=line.strip().partition('=')
                if not separator:continue
                record[key]=value
                if key=='progress':
                    # These are encoder observations, not QA or publication.
                    payload={'phase':value,'pid':process.pid}
                    for field in ('frame','out_time_us','total_size'):
                        try:payload[field]=max(0,int(record[field]))
                        except (KeyError,ValueError):pass
                    if 'speed' in record:payload['speed']=record['speed']
                    progress_callback(payload)
                    record={}
            code=process.wait()
            if timed_out.is_set():
                raise subprocess.TimeoutExpired(command, timeout_seconds)
        except BaseException:
            # A cancelled observation must not leave its renderer orphaned.
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
        finally:
            if timer is not None:
                timer.cancel()
            if process.stdout is not None:process.stdout.close()
        errors.flush();errors.seek(0,2);size=errors.tell()
        errors.seek(max(0,size-65536))
        detail=errors.read().decode('utf-8',errors='replace')
    return subprocess.CompletedProcess(command,code,stdout='',stderr=detail)


def render_log_callback(reporter, *, total_frames):
    """Log actual progress at 5-percent milestones; never claim QA completion."""
    previous={}
    def report(event):
        frame=event.get('frame',0)
        percent=min(100,int(100*frame/max(1,total_frames)))
        bucket=percent//5
        pid=event['pid']
        if bucket>previous.get(pid,-1) or event['phase']=='end':
            previous[pid]=bucket
            reporter.log(f"인코딩 실측 · {frame}/{total_frames} 프레임 · {percent}% · {event.get('speed','속도 측정 중')} · 품질 QA 별도")
    return report
