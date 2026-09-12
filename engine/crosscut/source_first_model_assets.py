"""Acquire the two digest-pinned source-first models in the selected cache.

No home-directory or built-in model-cache fallback is used. Existing files
with an unexpected digest are preserved and reported, never overwritten.
Only the exact ONNX member is read from the official ZIP; no archive paths
are extracted. Official MIT / Apache-2.0 licenses accompany published assets.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import tempfile
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

from .body_appearance import OSNET_AIN_MSMT17_SHA256
from .body_pose_2d import RTMPOSE_BODY7_MODEL_SHA256

BODY_URL = 'https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'
POSE_URL = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip'
POSE_ARCHIVE_SHA256 = 'f7fbb6c5c11a1bb70f3d445e4ddec5d144ea89ad5649c081ef976f9b24a0b741'
POSE_MEMBER = '20230831/rtmpose_onnx/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504/end2end.onnx'


@dataclass(frozen=True, slots=True)
class SourceFirstModelAssets:
    body_model_path: Path
    pose_model_path: Path


def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _download(url, maximum_bytes):
    started=time.monotonic(); blocks=[]; size=0
    with urlopen(Request(url,headers={'User-Agent':'CrossCut-model-assets/1'}),timeout=20.) as response:
        while True:
            block=response.read(1024*1024)
            if not block:break
            size+=len(block)
            if size>maximum_bytes or time.monotonic()-started>120.:
                raise OSError('source-first model download exceeded its bounded budget')
            blocks.append(block)
    return b''.join(blocks)


def _publish_bytes(path, data, expected_sha256):
    if hashlib.sha256(data).hexdigest()!=expected_sha256:
        raise ValueError('source-first model download digest mismatch')
    if path.exists():
        if _sha256(path)!=expected_sha256:
            raise ValueError('existing source-first model digest mismatch; preserved')
        return
    descriptor, temporary=tempfile.mkstemp(prefix='.source-first-download-',dir=path.parent)
    temporary=Path(temporary)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        # Only our private, digest-verified temporary file is renamed.
        if path.exists():
            if _sha256(path)!=expected_sha256:
                raise ValueError('source-first model destination changed; preserved')
        else:
            os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def ensure_source_first_model_assets(cache_directory: Path) -> SourceFirstModelAssets:
    root=Path(cache_directory)
    if not root.is_dir():
        raise OSError('selected source-first cache is unavailable; no internal fallback')
    directory=root/'models'/'source-first'
    directory.mkdir(parents=True,exist_ok=True)
    body=directory/'osnet_ain_msmt17.pth'
    pose=directory/'rtmpose_body7.onnx'
    for path,digest in ((body,OSNET_AIN_MSMT17_SHA256),(pose,RTMPOSE_BODY7_MODEL_SHA256)):
        if path.exists() and _sha256(path)!=digest:
            raise ValueError('existing source-first model digest mismatch; preserved')
    if not body.exists():
        _publish_bytes(body,_download(BODY_URL,25*1024*1024),OSNET_AIN_MSMT17_SHA256)
    if not pose.exists():
        archive=_download(POSE_URL,80*1024*1024)
        if hashlib.sha256(archive).hexdigest()!=POSE_ARCHIVE_SHA256:
            raise ValueError('source-first pose archive digest mismatch')
        with ZipFile(io.BytesIO(archive)) as bundle:
            matches=[info for info in bundle.infolist() if info.filename==POSE_MEMBER]
            if len(matches)!=1 or matches[0].file_size!=54330655:
                raise ValueError('source-first pose archive member differs')
            data=bundle.read(matches[0])
        _publish_bytes(pose,data,RTMPOSE_BODY7_MODEL_SHA256)
    for name in ('OSNet-MIT.txt','RTMPose-Apache-2.0.txt'):
        data=(Path(__file__).parent/'licenses'/name).read_bytes()
        _publish_bytes(directory/name,data,hashlib.sha256(data).hexdigest())
    return SourceFirstModelAssets(body,pose)
