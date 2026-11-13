"""Source integrity at explicit frozen-numeric phase boundaries.

This does NOT attest content hashes, bless a cache or weaken source-reading
checks. The caller must already bind hash-verified inputs. Ordinary calls
verify every original file. ONLY an explicit phase over immutable prepared
arrays uses cheap cancel/deadline/device checks. Source checkpoints at phase
entry/exit and origins prevent a changed input from publishing a result.
No acquisition, decoding, rendering or output publication belongs in a phase.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import numpy as np


def identity(path):
    stat=os.stat(path)
    return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns


class FrozenInputCheck:
    def __init__(self,*,sources,identities,device_root,device,tick=lambda:None):
        self.sources=tuple(os.fspath(p) for p in sources)
        self.identities=tuple(tuple(v) for v in identities)
        self.device_root=os.fspath(device_root);self.device=device;self.tick=tick
        if (not self.sources or len(self.sources)!=len(self.identities)
                or any(not Path(p).is_absolute() for p in (*self.sources,self.device_root))
                or any(len(v)!=4 or any(type(x) is not int or x<0 for x in v) for v in self.identities)
                or type(device) is not int or device<0 or not callable(tick)):
            raise ValueError('ordered original file identities, root device and live cancellation callback required')
        self.depth=0;self.failed=False;self.calls=self.source_checks=self.device_checks=self.numeric_phases=0
        self.source_checkpoint()

    @property
    def numeric_active(self):return self.depth>0

    def _device(self):
        if self.failed:raise RuntimeError('input lifecycle already failed; no output may be published')
        self.tick();self.device_checks+=1
        if os.stat(self.device_root).st_dev!=self.device:
            self.failed=True;raise OSError('frozen input device changed')

    def source_checkpoint(self):
        try:
            self._device()
            for path,expected in zip(self.sources,self.identities,strict=True):
                self.source_checks+=1
                if identity(path)!=expected:raise OSError('frozen original source changed')
        except BaseException:self.failed=True;raise

    def __call__(self):
        self.calls+=1
        if self.numeric_active:
            try:self._device()
            except BaseException:self.failed=True;raise
        else:self.source_checkpoint()

    @contextmanager
    def frozen_numeric_phase(self,groups):
        if (not isinstance(groups,tuple) or not groups or any(not isinstance(g,tuple) or not g for g in groups)
                or any(not isinstance(a,np.ndarray) or a.flags.writeable for g in groups for a in g)):
            raise ValueError('actual immutable prepared input groups required; not source-reading mode')
        self.source_checkpoint();prior=self.depth;self.depth=prior+1;self.numeric_phases+=1
        try:yield self
        except BaseException:self.depth=prior;self.failed=True;raise
        else:
            self.depth=prior
            self.source_checkpoint()

    def receipt(self):
        return dict(contract='immutable_numeric_phase_source_boundary_verification_v1',
            source_count=len(self.sources),calls=self.calls,source_identity_checks=self.source_checks,
            device_checks=self.device_checks,numeric_phases=self.numeric_phases,
            numeric_phase_active=self.numeric_active,failed=self.failed,
            content_hash_verification_replaced=False,source_reading_or_rendering_in_numeric_phase_allowed=False,
            normal_app_or_media_quality_approved=False)
