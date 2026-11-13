"""One actual production-trainer run with explicitly prepared face inputs.

Preserve the existing full-capacity model, temporal mode, losses, schedule,
supervision files and held-out partition. Change only prepared target-free
features and their schema width. No automatic qualification or activation.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

from crosscut.stage_edit_transformer import StageEditTokenSchema, StageEditTransformerConfig


def checkpoint_layout(config,version=3):
    if type(version) is not int:raise ValueError('an explicit Dynamic Crop checkpoint version is required')
    if version==3:
        from crosscut.stage_edit_dynamic_crop_head_v3 import stage_edit_dynamic_crop_v3_parameter_count, DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
        return stage_edit_dynamic_crop_v3_parameter_count(config),DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
    if version==2:
        from crosscut.stage_edit_dynamic_crop_head_v2 import stage_edit_dynamic_crop_v2_parameter_count, DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME
        return stage_edit_dynamic_crop_v2_parameter_count(config),DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME
    raise ValueError('an explicit Dynamic Crop checkpoint version is required')


def checkpoint_working_space(config,version=3,*,reserve_bytes=2**30,written_weight_bytes=0,followup_payload_bytes=0):
    """Full FP32 weights,64MiB headers/reports, and an explicit free reserve."""
    if (type(reserve_bytes) is not int or reserve_bytes<256*2**20
            or type(written_weight_bytes) is not int or written_weight_bytes<0
            or type(followup_payload_bytes) is not int or followup_payload_bytes<0):
        raise ValueError('at least256MiB reserve and actual nonnegative weight progress required')
    count,_=checkpoint_layout(config,version)
    return reserve_bytes+64*2**20+followup_payload_bytes+max(0,count*4-written_weight_bytes)


def training_arguments(original, prepared, schema, output, *, dynamic_loss_override=None):
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
    if dynamic_loss_override is not None:
        from crosscut.stage_edit_dynamic_crop_head_v3 import DynamicCropV3LossContract
        from crosscut.stage_edit_transformer import StageEditLossContract
        if '--dynamic-crop-version' not in args or args[args.index('--dynamic-crop-version')+1] != '3':
            raise ValueError('an explicit v3 loss override requires the unchanged v3 model path')
        values = dict(dynamic_loss_override)
        values['base'] = StageEditLossContract(**values['base'])
        selected = DynamicCropV3LossContract(**values)
        if ('--base-loss-contract-json' not in args or selected.base != StageEditLossContract(
                **json.loads(args[args.index('--base-loss-contract-json')+1]))):
            raise ValueError('a trajectory-loss override cannot change the frozen base objective')
        flag = '--dynamic-crop-loss-contract-json'
        encoded = json.dumps(asdict(selected), separators=(',', ':'))
        if args.count(flag) > 1:
            raise ValueError('ambiguous existing trajectory loss arguments')
        if flag in args:
            args[args.index(flag)+1] = encoded
        else:
            args.extend((flag, encoded))
    args.extend(('--token-schema-json', json.dumps(schema.to_dict(), separators=(',',':'))))
    return args, config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-launch', required=True, type=Path)
    p.add_argument('--prepared-inputs', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--ssd-device', required=True, type=int)
    p.add_argument('--train', action='store_true', help='Explicitly start the real model fit; no hidden training.')
    p.add_argument('--workspace-reserve-mib',type=int,default=1024,
        help='Explicit free-space reserve; full FP32 weights and64MiB metadata are budgeted separately. Minimum256.')
    p.add_argument('--followup-payload-bytes',type=int,default=0,
        help='Keep this explicitly calculated later render payload free throughout model fitting.')
    p.add_argument('--dynamic-crop-loss-contract-json',
        help='Explicitly change the v3 trajectory objective only; preserve the frozen base loss, inputs and capacity.')
    a = p.parse_args()
    volume = Path('/Volumes/CrossCutMedia')
    if (not volume.is_mount() or volume.stat().st_dev != a.ssd_device
            or not a.output.resolve().is_relative_to(volume.resolve()) or a.output.exists()):
        raise OSError('one unused output on the authorized mounted SSD is required')
    original_bytes = a.original_launch.read_bytes()
    prepared_bytes = a.prepared_inputs.read_bytes()
    prepared = json.loads(prepared_bytes)
    if prepared['original_launch_file_sha256'] != hashlib.sha256(original_bytes).hexdigest():
        raise ValueError('prepared features came from another frozen launch')
    schema = StageEditTokenSchema.from_dict(json.loads(Path(prepared['token_schema_path']).read_bytes()))
    if schema.fingerprint != prepared['token_schema_fingerprint']:
        raise ValueError('prepared token schema changed')
    loss_override = None if a.dynamic_crop_loss_contract_json is None else json.loads(a.dynamic_crop_loss_contract_json)
    args, config = training_arguments(json.loads(original_bytes)['argv'], prepared, schema, a.output,
        dynamic_loss_override=loss_override)
    version=int(args[args.index('--dynamic-crop-version')+1]) if '--dynamic-crop-version' in args else 2
    reserve=a.workspace_reserve_mib*2**20
    needed=checkpoint_working_space(config,version,reserve_bytes=reserve,followup_payload_bytes=a.followup_payload_bytes)
    parameter_count,weight_name=checkpoint_layout(config,version)
    if a.train and shutil.disk_usage(volume).free<needed:
        raise OSError('not enough actual FP32 checkpoint, metadata and declared workspace reserve; no output created')
    a.output.mkdir()
    receipt = dict(actual_production_training=a.train, production_activation_authorized=False,
        source_commit=subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip(),
        original_launch_sha256=hashlib.sha256(original_bytes).hexdigest(),
        prepared_inputs_sha256=hashlib.sha256(prepared_bytes).hexdigest(),
        prepared_inputs=str(a.prepared_inputs), argv=args, model_config=config.to_dict(), pid=os.getpid(),
        explicit_dynamic_crop_loss_override=loss_override,
        storage_budget=dict(parameter_count=parameter_count,float32_weight_bytes=parameter_count*4,
            metadata_header_allowance_bytes=64*2**20,reserve_bytes=reserve,
            followup_payload_bytes=a.followup_payload_bytes,
            full_capacity_and_precision_unchanged=True),
        limits=dict(seconds=1200, rss_bytes=7*2**30, mlx_bytes=6*2**30, minimum_free_bytes=needed))
    with (a.output/'launch.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    if not a.train:
        print(json.dumps(dict(prepared_only=True, launch=str(a.output/'launch.json'))))
        return
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
            remaining=needed;source_live=False;free=0
            try:
                source_live=volume.is_mount() and volume.stat().st_dev==a.ssd_device
                weight=a.output/'checkpoint'/weight_name
                try:written=weight.stat().st_size
                except FileNotFoundError:written=0
                remaining=checkpoint_working_space(config,version,reserve_bytes=reserve,written_weight_bytes=written,
                    followup_payload_bytes=a.followup_payload_bytes)
                free=shutil.disk_usage(volume).free
            except OSError:source_live=False
            if (not source_live
                    or elapsed > 1200 or rss > 7*2**30 or peak > 6*2**30
                    or free<remaining):
                try:
                    with (a.output/'resource-stop.json').open('w') as stream:
                        json.dump(dict(elapsed_seconds=elapsed, rss_bytes=rss, mlx_peak_bytes=peak,
                            remaining_checkpoint_space_bytes=remaining,actual_production_training=True, approved=False), stream)
                finally:os._exit(75)
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
