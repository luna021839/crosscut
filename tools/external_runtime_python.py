"""Launch the selected Python from a small, externally stored APFS image.

The local JSON configuration is not shipped with the repository. No install,
download, source deletion, model load, or internal-disk fallback occurs here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import uuid


def load_configuration(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=8192:
        raise ValueError('a bounded regular local runtime configuration is required')
    value=json.loads(path.read_bytes())
    if (not isinstance(value,dict) or set(value)!={'schema','image_path','mount_path','python_relative','volume_uuid'}
            or type(value['schema']) is not int or value['schema']!=1
            or any(not isinstance(value[k],str) for k in ('image_path','mount_path','python_relative','volume_uuid'))):
        raise ValueError('external runtime configuration schema differs')
    image=Path(value['image_path']);mount=Path(value['mount_path']);relative=Path(value['python_relative'])
    if (not image.is_absolute() or image.is_symlink() or not image.is_file() or image.suffix!='.sparseimage'
            or mount.parent!=Path('/Volumes') or mount.name in ('','.', '..')
            or relative.is_absolute() or not relative.parts or '..' in relative.parts):
        raise ValueError('explicit external image, volume and relative Python are required')
    return image,mount,relative,str(uuid.UUID(value['volume_uuid'])).lower()


def mounted_uuid(mount):
    if not mount.is_mount():return None
    result=subprocess.run(['/usr/sbin/diskutil','info','-plist',str(mount)],
        capture_output=True,timeout=2.)
    if result.returncode:return None
    try:return str(plistlib.loads(result.stdout)['VolumeUUID']).lower()
    except (KeyError,ValueError,TypeError):return None


def resolve_python(configuration):
    image,mount,relative,expected=load_configuration(configuration)
    actual=mounted_uuid(mount);attached=False
    if actual is None:
        if mount.is_mount():raise ValueError('configured mount point belongs to an unidentified volume')
        if mount.exists() and any(mount.iterdir()):
            raise ValueError('configured mount point is occupied; nothing was replaced')
        result=subprocess.run(['/usr/bin/hdiutil','attach','-nobrowse','-owners','on',
            '-mountpoint',str(mount),str(image)],capture_output=True,timeout=8.)
        actual=mounted_uuid(mount);attached=True
        # Another simultaneous locator may have attached the same image.
        # Recover that verified mount, never issue a second attach/retry.
        if result.returncode and actual!=expected:
            raise OSError('external Python image could not be mounted')
    if actual!=expected:raise ValueError('external runtime volume UUID differs; nothing was executed')
    python=mount/relative
    if not python.is_file() or not os.access(python,os.X_OK):
        raise OSError('selected external Python is unavailable')
    return python,attached


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    args,command=parser.parse_known_args()
    if command[:1]==['--']:command=command[1:]
    try:
        python,attached=resolve_python(args.config)
        if args.check:
            print(json.dumps(dict(external_python=str(python),mounted_during_launch=attached,
                volume_uuid_verified=True)));return
        os.execv(str(python),[str(python),*command])
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        print('CrossCut external Python: '+str(error),file=sys.stderr);raise SystemExit(1)


if __name__=='__main__':main()
