"""Bounded, single-decode final-frame PTS and sampled pixels.

This is an observation primitive, not a quality gate, identity/pose model or
production approval. showinfo runs BEFORE selection, so the complete decoded
frame count/PTS are checked while only bounded analysis rasters leave FFmpeg.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
import os
from pathlib import Path
import re
import subprocess
import numpy as np


@dataclass(frozen=True)
class RenderedFrameObservation:
    pts_ticks: np.ndarray
    time_base: Fraction
    selected_indexes: np.ndarray
    pixels: np.ndarray
    pixel_format: str = 'gray'

    @property
    def grayscale(self):
        if self.pixel_format!='gray':raise ValueError('this observation contains BGR color, not a grayscale plane')
        return self.pixels

    @property
    def selected_seconds(self):
        return self.pts_ticks[self.selected_indexes].astype(np.float64)*float(self.time_base)


def _parsed_pts(stderr: str, expected_frames: int):
    bases=re.findall(r'config in time_base:\s*(\d+)/(\d+)',stderr)
    if not bases or any(b!=bases[0] for b in bases):
        raise ValueError('one explicit decoded-frame time base required')
    numerator,denominator=map(int,bases[0])
    if numerator<=0 or denominator<=0:
        raise ValueError('decoded-frame time base must be positive')
    records=re.findall(r'\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:',stderr)
    if len(records)!=expected_frames or [int(n) for n,p in records]!=list(range(expected_frames)):
        raise ValueError('complete ordered decoded-frame census differs from plan')
    ticks=np.asarray([int(p) for n,p in records],dtype=np.int64)
    if len(ticks)>1 and np.any(ticks[1:]<=ticks[:-1]):
        raise ValueError('decoded-frame PTS must increase strictly')
    ticks.setflags(write=False)
    return ticks,Fraction(numerator,denominator)


def observe_rendered_frames(path, *, expected_frames, stride=8, width=320,height=180,
                            max_rasters=2048,timeout_seconds=120.,ffmpeg='ffmpeg',
                            selected_indexes=None,pixel_format='gray'):
    """Read a local completed film once, without changing it or saving rasters.

    Container PTS (including a nonzero origin) are retained with -copyts.
    Integer ticks and the filter's rational time base are authoritative, not
    rounded pts_time strings or an assumed constant frame rate. The extra
    sample limit bounds a wrongly declared/long input before capture grows.
    """
    for name,value in (('expected_frames',expected_frames),('stride',stride),
                       ('width',width),('height',height),('max_rasters',max_rasters)):
        if type(value) is not int or value<1:
            raise ValueError(f'{name} must be a positive integer')
    if expected_frames>20000:
        raise ValueError('bounded observation supports at most 20000 decoded frames')
    if pixel_format not in ('gray','bgr24'):
        raise ValueError('bounded gray or bgr24 pixel format required')
    if selected_indexes is None:
        indexes=np.arange(0,expected_frames,stride,dtype=np.int64)
        selection=f'not(mod(n\\,{stride}))'
    else:
        if not isinstance(selected_indexes,(tuple,list,np.ndarray)):
            raise ValueError('bounded explicit output-frame sample sequence required')
        if isinstance(selected_indexes,np.ndarray) and (selected_indexes.ndim!=1 or selected_indexes.dtype.kind not in 'iu'):
            raise ValueError('strictly ordered actual output-frame sample indexes required')
        if not len(selected_indexes) or len(selected_indexes)>max_rasters:
            raise ValueError('analysis raster/byte allowance exceeded before decode')
        if any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or not 0<=n<expected_frames
                for n in selected_indexes):
            raise ValueError('strictly ordered actual output-frame sample indexes required')
        indexes=np.asarray(selected_indexes,dtype=np.int64).copy()
        if np.any(np.diff(indexes)<=0):
            raise ValueError('strictly ordered actual output-frame sample indexes required')
        # The next-frame sentinel also bounds a wrongly declared long input:
        # selection cannot silently discard millions of extra decoded frames.
        selection='+'.join(f'eq(n\\,{int(n)})' for n in np.r_[indexes,expected_frames])
    channels=1 if pixel_format=='gray' else 3
    if len(indexes)>max_rasters or (len(indexes)+1)*width*height*channels>128*2**20:
        raise ValueError('analysis raster/byte allowance exceeded before decode')
    if isinstance(timeout_seconds,bool) or not isinstance(timeout_seconds,(int,float)) or not math.isfinite(timeout_seconds) or not 0<timeout_seconds<=120:
        raise ValueError('explicit observation deadline in (0,120] required')
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('observation requires an ordinary local completed movie')
    before=os.stat(path)
    args=[os.fspath(ffmpeg),'-hide_banner','-nostdin','-v','info','-xerror','-copyts','-i',os.fspath(path),
          '-map','0:v:0','-an','-sn','-dn','-vf',
          f'showinfo,select={selection},scale={width}:{height},format={pixel_format}',
          '-fps_mode','passthrough','-frames:v',str(len(indexes)+1),'-f','rawvideo','pipe:1']
    result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout_seconds)
    after=os.stat(path)
    fields=('st_dev','st_ino','st_size','st_mtime_ns')
    if any(getattr(before,k)!=getattr(after,k) for k in fields):
        raise OSError('movie changed during frame observation')
    if result.returncode:
        raise RuntimeError(f'frame observation decoder failed (exit {result.returncode})')
    ticks,time_base=_parsed_pts(result.stderr.decode('utf-8',errors='replace'),expected_frames)
    if len(result.stdout)!=len(indexes)*width*height*channels:
        raise ValueError('sampled decoded pixel count differs from plan')
    shape=(len(indexes),height,width) if channels==1 else (len(indexes),height,width,channels)
    pixels=np.frombuffer(result.stdout,dtype=np.uint8).reshape(shape)
    indexes.setflags(write=False)
    return RenderedFrameObservation(ticks,time_base,indexes,pixels,pixel_format)
