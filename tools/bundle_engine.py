#!/usr/bin/env python3
"""Verify a staged app engine before replacing any existing built resources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

REQUIRED = ('crosscut_cli.py','crosscut/__init__.py','crosscut2/__init__.py',
    'crosscut/models/face_detection_yunet_2023mar.onnx','crosscut/models/LICENSE.yunet',
    'crosscut/models/face_recognition_sface_2021dec.onnx','crosscut/models/LICENSE.opencv-zoo-sface',
    'crosscut/models/person_detection_mediapipe_2023mar.onnx',
    'crosscut/models/pose_estimation_mediapipe_2023mar.onnx',
    'crosscut/models/human_segmentation_pphumanseg_2023mar.onnx',
    'crosscut/models/LICENSE.opencv-zoo-mediapipe-pose','crosscut/models/LICENSE.opencv-zoo-pphumanseg',
    'crosscut/models/PROVENANCE.pose.md')


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            chunk=stream.read(1024*1024)
            if not chunk:break
            value.update(chunk)
    return value.hexdigest()


def bundle(source,destination,build_root):
    source=Path(source).resolve(strict=True);destination=Path(destination);build_root=Path(build_root).resolve(strict=True)
    if (destination.is_symlink() or not destination.resolve().is_relative_to(build_root)
            or destination.name!='engine' or destination.parent.name!='Resources'
            or destination.parent.parent.name!='Contents' or destination.parent.parent.parent.suffix!='.app'):
        raise ValueError('refusing an engine destination outside this built app')
    for name in REQUIRED:
        if not (source/name).is_file() or (source/name).is_symlink():raise ValueError('required engine asset missing: '+name)
    paths=[source/'crosscut_cli.py']
    for package in ('crosscut','crosscut2'):paths.extend(sorted((source/package).glob('*.py')))
    for folder in ('crosscut/models','crosscut/licenses'):
        paths.extend(sorted(p for p in (source/folder).rglob('*') if p.is_file() and not any(v.startswith('.') for v in p.relative_to(source).parts)))
    if any(p.is_symlink() for p in paths):raise ValueError('engine source assets cannot be symlinks')
    expected={str(p.relative_to(source)):digest(p) for p in paths}
    for path in paths:
        if path.suffix=='.py' and b'\0' in path.read_bytes():raise ValueError('null bytes in engine source: '+str(path.relative_to(source)))
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.engine-staging-',dir=destination.parent))
    backup=destination.with_name('.engine-backup-'+uuid.uuid4().hex)
    published=False
    reused_bytes=0
    try:
        for path in paths:
            relative=path.relative_to(source);target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True)
            previous=destination/relative
            linked=False
            if (previous.is_file() and not previous.is_symlink()
                    and previous.resolve().is_relative_to(destination.resolve())
                    and digest(previous)==expected[str(relative)]):
                # Staging and the previous built tree share this filesystem.
                # Reuse verified immutable model/source bytes, not a second
                # 80 MiB copy for a few changed Python files on a nearly-full
                # external disk. Hash verification still precedes publication.
                try:os.link(previous,target,follow_symlinks=False);linked=True
                except OSError:pass
            if linked:reused_bytes+=path.stat().st_size
            else:shutil.copyfile(str(path),str(target))
            with target.open('rb') as stream:os.fsync(stream.fileno())
            if digest(target)!=expected[str(relative)]:raise ValueError('bundled engine copy differs: '+str(relative))
        if any(digest(source/name)!=sha for name,sha in expected.items()):raise ValueError('engine source changed during packaging')
        if destination.exists():destination.rename(backup)
        try:stage.rename(destination)
        except BaseException:
            if backup.exists():backup.rename(destination)
            raise
        published=True
        # The old built tree is the only deletion target, never sources or a
        # workspace/cache root. Its replacement has already been verified.
        if backup.exists():shutil.rmtree(backup)
        receipt=dict(contract='verified_app_engine_bundle_v1',files=expected,
            bytes=sum(p.stat().st_size for p in paths),verified_existing_bytes_reused=reused_bytes)
        (destination.parent/'python-engine-bundle.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
        return receipt
    finally:
        if not published and stage.exists():shutil.rmtree(stage)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--destination',required=True,type=Path);parser.add_argument('--build-root',required=True,type=Path)
    args=parser.parse_args();result=bundle(args.source,args.destination,args.build_root)
    print('Verified %s engine files (%s bytes) before app publication.'%(len(result['files']),result['bytes']))
