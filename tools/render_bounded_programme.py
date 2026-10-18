"""Render a saved programme sequentially without selecting or observing again."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine'))
from crosscut.render_plan_serialization import render_plan_from_json
from crosscut2 import bounded_plan_render
from crosscut2.grid_perception import file_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','authorized-root'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--expected-plan-sha256',required=True)
    parser.add_argument('--maximum-seconds',type=float,default=660.)
    args=parser.parse_args();root=args.authorized_root.resolve(strict=True)
    raw=args.plan.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=args.expected_plan_sha256:raise ValueError('saved programme digest changed')
    record=json.loads(raw);plan=render_plan_from_json(record['render_plan'])
    for path in (args.plan,plan.output,*plan.sources):
        if not path.resolve().is_relative_to(root):raise ValueError('explicit authorized media root only')
    identities=record['source_file_identities'];receipt=plan.output.with_suffix('.render.json')
    claim=plan.output.with_suffix('.render.claim.json')
    if any(p.exists() for p in (plan.output,receipt,claim)):
        raise FileExistsError('recover the dispatched bounded render')
    def check():
        if root.stat().st_dev!=args.device:raise OSError('authorized media volume changed')
        if [list(file_identity(p)) for p in plan.sources]!=identities:raise OSError('saved programme sources changed')
        if args.plan.read_bytes()!=raw:raise OSError('saved programme changed during rendering')
    check();last=-10.;started=monotonic()
    def progress(event):
        nonlocal last
        elapsed=monotonic()-started
        if event.get('phase')=='bounded_segment_complete' and (elapsed-last>=10. or event.get('completed')==event.get('total')):
            print(json.dumps(event),flush=True);last=elapsed
    with claim.open('x') as f:json.dump(dict(plan_sha256=digest,maximum_seconds=args.maximum_seconds,
        automatic_retry=False,decoder_concurrency=1,source_NN=0),f,indent=2)
    try:
        result=bounded_plan_render.render(plan,ffmpeg='/opt/homebrew/bin/ffmpeg',
            maximum_seconds=args.maximum_seconds,check=check,progress_callback=progress)
        result.update(plan_sha256=digest,failure=None)
    except Exception as error:
        result=dict(plan_sha256=digest,failure=dict(type=type(error).__name__,message=str(error)),
            elapsed_seconds=monotonic()-started,movie=str(plan.output),automatic_retry=False)
    with receipt.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='blocks'}),flush=True)
    if result['failure'] is not None:raise SystemExit(1)


if __name__=='__main__':main()
