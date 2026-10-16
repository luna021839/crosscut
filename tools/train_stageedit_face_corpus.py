"""One actual production-trainer run with explicitly prepared face inputs.

Preserve the existing full-capacity model, temporal mode, losses, schedule,
supervision files and held-out partition. Change only prepared target-free
features and their schema width. No automatic qualification or activation.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from crosscut.stage_edit_transformer import StageEditTokenSchema, StageEditTransformerConfig


def training_arguments(original, prepared, schema, output):
    args = list(original)
    snapshots = prepared['feature_snapshot_by_reference']
    seen = set()
    config = None
    for i, option in enumerate(args):
        if option == '--feature-snapshot':
            reference = args[i+1].split('=',1)[0]
            if reference in seen or reference not in snapshots:
                raise ValueError('prepared input changed the exact reference set')
            seen.add(reference)
            args[i+1] = reference+'='+snapshots[reference]
        elif option == '--checkpoint-dir':
            args[i+1] = str(output/'checkpoint')
        elif option == '--model-config-json':
            old = StageEditTransformerConfig.from_dict(json.loads(args[i+1]))
            config = replace(old, token_schema_fingerprint=schema.fingerprint,
                source_input_dimension=schema.encoded_source_dimension,
                context_input_dimension=schema.encoded_context_dimension)
            args[i+1] = json.dumps(config.to_dict(), separators=(',',':'))
    if seen != set(snapshots) or config is None or '--authorize-production-activation' in args:
        raise ValueError('one complete explicit nonactivating production launch is required')
    if '--token-schema-json' in args:
        raise ValueError('source launch already has a custom schema; do not overwrite it implicitly')
    args.extend(('--token-schema-json', json.dumps(schema.to_dict(), separators=(',',':'))))
    return args, config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-launch', required=True, type=Path)
    p.add_argument('--prepared-inputs', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--ssd-device', required=True, type=int)
    p.add_argument('--train', action='store_true', help='Explicitly start the real model fit; no hidden training.')
    a = p.parse_args()
    volume = Path('/Volumes/CrossCutMedia')
    if (not volume.is_mount() or volume.stat().st_dev != a.ssd_device
            or not a.output.is_relative_to(volume) or a.output.exists()):
        raise OSError('one unused output on the authorized mounted SSD is required')
    original_bytes = a.original_launch.read_bytes()
    prepared_bytes = a.prepared_inputs.read_bytes()
    prepared = json.loads(prepared_bytes)
    if prepared['original_launch_file_sha256'] != hashlib.sha256(original_bytes).hexdigest():
        raise ValueError('prepared features came from another frozen launch')
    schema = StageEditTokenSchema.from_dict(json.loads(Path(prepared['token_schema_path']).read_bytes()))
    if schema.fingerprint != prepared['token_schema_fingerprint']:
        raise ValueError('prepared token schema changed')
    args, config = training_arguments(json.loads(original_bytes)['argv'], prepared, schema, a.output)
    a.output.mkdir()
    receipt = dict(actual_production_training=a.train, production_activation_authorized=False,
        source_commit=subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip(),
        original_launch_sha256=hashlib.sha256(original_bytes).hexdigest(),
        prepared_inputs_sha256=hashlib.sha256(prepared_bytes).hexdigest(),
        prepared_inputs=str(a.prepared_inputs), argv=args, model_config=config.to_dict(), pid=os.getpid(),
        limits=dict(seconds=1200, rss_bytes=7*2**30, mlx_bytes=6*2**30, minimum_free_bytes=1536*2**20))
    with (a.output/'launch.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    if not a.train:
        print(json.dumps(dict(prepared_only=True, launch=str(a.output/'launch.json'))))
        return
    if __import__('shutil').disk_usage(volume).free < receipt['limits']['minimum_free_bytes']:
        raise OSError('not enough bounded checkpoint/intermediate storage for this run')
    import mlx.core as mx
    import psutil
    from crosscut.stage_edit_production_train_cli import main as train
    process = psutil.Process()
    started = time.monotonic()
    done = threading.Event()
    mx.set_cache_limit(128*2**20)
    mx.set_memory_limit(receipt['limits']['mlx_bytes'])
    def watch():
        while not done.wait(.5):
            elapsed = time.monotonic()-started
            rss, peak = process.memory_info().rss, int(mx.get_peak_memory())
            if (not volume.is_mount() or volume.stat().st_dev != a.ssd_device
                    or elapsed > 1200 or rss > 7*2**30 or peak > 6*2**30):
                with (a.output/'resource-stop.json').open('w') as stream:
                    json.dump(dict(elapsed_seconds=elapsed, rss_bytes=rss, mlx_peak_bytes=peak,
                        actual_production_training=True, approved=False), stream)
                os._exit(75)
    threading.Thread(target=watch, daemon=True).start()
    result = dict(actual_production_training=True, professional_quality_approved=False)
    try:
        result['exit_code'] = train(args)
    except Exception as exc:
        result.update(exception=type(exc).__name__, error=str(exc))
        raise
    finally:
        done.set()
        result.update(elapsed_seconds=time.monotonic()-started, mlx_peak_bytes=int(mx.get_peak_memory()),
            rss_bytes=process.memory_info().rss)
        with (a.output/'runner-result.json').open('x') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
