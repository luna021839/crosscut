"""Fit the experimental visible-shot duration model from explicit TRAIN audits.

Existing audit cut detections are weak reference evidence, not new manual cut
labels. This command reads no video, source observations, development or held-
out references and never activates a model in the app.
"""
import argparse
import hashlib
import json
from pathlib import Path

from crosscut2.visible_shot_duration import fit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-audit',action='append',required=True,metavar='REFERENCE_ID=PATH')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rows=[]
    for supplied in args.train_audit:
        identity,separator,name=supplied.partition('=')
        if not separator or not identity or not name:parser.error('each TRAIN audit needs REFERENCE_ID=PATH')
        raw=Path(name).read_bytes();record=json.loads(raw)
        rows.append(dict(reference_id=identity,split='train',input_sha256=hashlib.sha256(raw).hexdigest(),
            cut_seconds=[row['seconds'] for row in record['cuts']]))
    result=fit(rows)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),reference_ids=[r['reference_id'] for r in rows],
        geometric_mean_shot_seconds=__import__('math').exp(result['log_seconds_mean']),
        log_seconds_scale=result['log_seconds_scale'],aesthetic_weight_fitted=False)))


if __name__=='__main__':main()
