"""Bounded private native facts, reusable without repeating an NN pass.

No pixels, training labels or quality approval. SFace vectors stay in the
user's source cache; callers must never stage/upload this runtime file.
Archive reuse preserves actual raw parents and adverse decode/conflict facts.
"""
import json
import os
from pathlib import Path
import tempfile

from .group_performer_family_owner import digest
from .group_review_recorded_index import RecordedGroupReviewIndex

MAX_BYTES=32*1024**2
MAX_ROWS=16384


class WholeNativeFactCache:
    def __init__(self,*,root,sensors):
        self.sensors=sensors
        session=sensors.session
        self.header=dict(contract='private_whole_native_fact_cache_v1',
            sources={str(i):dict(path=str(session.domain.request_sources[i]),
                file_identity=list(session._assert_source_identity(i)),
                native_clock=session.domain.clocks_by_request[i].to_dict()) for i in session.domain.eligible_indices},
            models={name:dict(sha256=session.models[name][2],adapter=session.model_adapters[name])
                for name in sorted(session.models)},quality_approved=False)
        root=Path(root).resolve(strict=True)
        if not root.is_dir():raise ValueError('native fact cache root must already exist')
        self.storage_identity=(root.stat().st_dev,root.stat().st_ino)
        self.root=root
        # The existing visual-feature `derived` directory may deliberately be
        # linked to a mounted APFS cache image. Do not reject or alter that
        # independent owner. These bounded private facts stay in this source
        # cache root, without following that link or materializing pixels.
        folder=root/'whole-native-facts-v1'
        # Never follow an unverified link, or recreate an absent SSD root.
        for part in (folder,):
            if part.is_symlink():raise ValueError('native fact cache cannot follow a directory link')
            part.mkdir(exist_ok=True)
        self.path=folder/(digest(self.header)+'.json')
        self.records=[];self.failures=[]

    def _check(self):
        session=self.sensors.session
        session.budget.check()
        if session.closed or (self.root.stat().st_dev,self.root.stat().st_ino)!=self.storage_identity:
            raise ValueError('native fact cache lost its live storage/session binding')
        for source,row in self.header['sources'].items():
            if list(session._assert_source_identity(int(source)))!=row['file_identity']:
                raise ValueError('native fact cache source changed')

    def bind_existing(self):
        self._check()
        if not self.path.exists():return False
        if self.path.is_symlink() or not self.path.is_file() or self.path.stat().st_size>MAX_BYTES:
            raise ValueError('private native fact cache is not a bounded regular file')
        raw=self.path.read_bytes()
        if len(raw)>MAX_BYTES:raise ValueError('private native fact cache byte bound changed')
        port=self.sensors.session.budget.request_work_phase
        if port is not None:port.charge('ipc_bytes',len(raw))
        payload=json.loads(raw)
        if payload.get('header')!=self.header or set(payload)!={'header','records','decode_failures'}:
            raise ValueError('private native fact cache changed its source/model header')
        records,failures=payload['records'],payload['decode_failures']
        if (not isinstance(records,list) or not isinstance(failures,list)
                or not 1<=len(records)+len(failures)<=MAX_ROWS):
            raise ValueError('private native fact cache row bound changed')
        index=RecordedGroupReviewIndex(maximum_rows=MAX_ROWS,maximum_bytes=MAX_BYTES,work_port=port)
        index.add_native_completion_records(records,decode_failures=failures,provenance='private-whole-native-cache')
        self.sensors.bind_recorded_motion_index(index,
            source_identities={int(i):tuple(row['file_identity']) for i,row in self.header['sources'].items()},
            source_content_sha256={int(i):None for i in self.header['sources']})
        self.records=records;self.failures=failures
        self._check()
        return True

    def preserve_completed(self,inventory):
        """Persist facts BEFORE graph/metadata/encode can fail or be cancelled.

        Existing adverse alternatives are unioned, never replaced by the
        cheapest observation. Too large an archive is explicitly not written.
        The next request must still consume one completion allowance.
        """
        self._check()
        records={digest(r):r for r in self.records}
        failures={digest(r):r for r in self.failures}
        for supply in inventory.supplies:
            if supply.observation is None:continue
            raw=supply.observation.to_dict()
            if raw['native_pixel_sha256'] is None:failures[digest(raw)]=raw
            else:
                row=dict(native_observation=raw,appearance=supply.appearance)
                records[digest(row)]=row
        # An empty pass is still consumed, but contains no reusable fact. Do
        # not publish an archive that the strict nonempty importer rejects.
        if not 1<=len(records)+len(failures)<=MAX_ROWS:return False
        payload=json.dumps(dict(header=self.header,records=list(records.values()),decode_failures=list(failures.values())),
            sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if len(payload)>MAX_BYTES:return False
        port=self.sensors.session.budget.request_work_phase
        if port is not None:port.charge('ipc_bytes',len(payload))
        handle,name=tempfile.mkstemp(prefix='.whole-native-',suffix='.tmp',dir=self.path.parent)
        try:
            with os.fdopen(handle,'wb') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
            self._check()
            os.replace(name,self.path)
        finally:
            # Only our uncommitted generated atomic-write temporary, not
            # user files, source videos or an old fact archive.
            if os.path.exists(name):os.unlink(name)
        return True
