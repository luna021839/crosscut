"""Bounded source-native review reuse, never live observation authority.

An output tick belongs to an edit, not to a camera-independent detector row.
The index retains every distinct recorded result at a physical address. It
cannot select the most favorable capture, borrow another clock/file, or turn
a missing appearance vector into a missing visible person.
"""
import json
import math
from hashlib import file_digest

from .group_performer_family_owner import digest
from .group_joint_observations import _file_identity
from .group_scene_static_query import reproject_recorded_static_query


def native_analysis_variant(raw):
    """Declared detector/model/adapter alternative, not the favorable output."""
    models=raw.get('model_content_sha256',())
    adapters=raw.get('model_adapter_fingerprints',())
    if not models or not adapters:return None
    return digest(dict(models=sorted(map(tuple,models)),adapters=sorted(map(tuple,adapters)),
        ownership_policy=raw.get('person_ownership_policy_sha256')))


def validate_recorded_appearance(raw,appearance,*,expected_space=None):
    """Keep exact raw parent, all head IDs and pinned descriptor space."""
    from .native_face_descriptor import NativeFaceDescriptorSpace,SFACE_MODEL_NAME
    if appearance is None:return None
    for field in ('request_source_index','native_frame_index','source_clock_identity',
            'source_file_identity','native_pixel_sha256'):
        if appearance.get(field)!=raw.get(field):
            raise ValueError('recorded appearance borrowed a different native parent')
    if (appearance.get('recorded_native_observation_sha256')!=digest(raw)
            or appearance.get('quality_approved') is not False):
        raise ValueError('recorded appearance borrowed a different native parent or approval')
    space=NativeFaceDescriptorSpace(**appearance.get('descriptor_space',{}))
    if space.model_name!=SFACE_MODEL_NAME or expected_space is not None and space!=expected_space:
        raise ValueError('recorded appearance changed its declared SFace model/adapter space')
    heads=appearance.get('heads',())
    expected={h['observation_id'] for h in raw['heads']}
    if (not isinstance(heads,(list,tuple)) or len(heads)!=len(expected)
            or {h.get('observation_id') for h in heads}!=expected):
        raise ValueError('recorded appearance omitted or repeated native head occurrences')
    for head in heads:
        v=head.get('embedding')
        if v is not None and (not isinstance(v,(list,tuple)) or len(v)!=space.dimension
                or any(type(x) not in (int,float) or not math.isfinite(x) for x in v)
                or sum(x*x for x in v)<=1e-12):
            raise ValueError('recorded appearance needs a finite nonzero vector in its exact space')
    return space


def verify_recorded_source_file(path, recorded_identity, *, content_sha256=None, work_port=None):
    """Retain archived addresses across a verified device-number-only remount.

    A matching filename, volume label or stat tail alone is not enough. On a
    remount the complete file must match an independently retained content
    digest. The archived detector/appearance/body parents are never rewritten.
    This verifies reusable source bytes, not actor identity or media quality.
    """
    recorded = tuple(recorded_identity)
    if len(recorded) != 5 or any(type(v) is not int or v < 0 for v in recorded):
        raise ValueError('recorded source needs its complete file identity')
    if work_port is not None:
        work_port.charge('identity_reads')
    current = _file_identity(path)
    receipt = dict(recorded_file_identity=list(recorded), current_file_identity=list(current),
        archived_observations_rewritten=False, quality_approved=False)
    if current == recorded:
        return dict(receipt, state='EXACT_FILE_IDENTITY', content_hash_bytes=0)
    if current[1:] != recorded[1:]:
        raise ValueError('recorded source changed beyond a device-number remount')
    if (not isinstance(content_sha256, str) or len(content_sha256) != 64
            or any(c not in '0123456789abcdef' for c in content_sha256)):
        raise ValueError('remounted source needs its retained complete content digest')
    if work_port is not None:
        # The request's existing hash-I/O budget also bounds source hashing.
        work_port.charge('model_hash_bytes', current[2])
        work_port.charge('identity_reads')
    with path.open('rb') as stream:
        actual = file_digest(stream, 'sha256').hexdigest()
    if _file_identity(path) != current:
        raise ValueError('source changed during remount content verification')
    if actual != content_sha256:
        raise ValueError('remounted source content differs from the recorded source')
    return dict(receipt, state='CONTENT_VERIFIED_DEVICE_REMOUNT', content_sha256=actual,
        content_hash_bytes=current[2])


def _freeze(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _key(source, native, clock, identity):
    if (type(source) is not int or source < 0 or type(native) is not int or native < 0
            or not isinstance(clock, str) or len(clock) != 64
            or any(c not in '0123456789abcdef' for c in clock)
            or not isinstance(identity, (list, tuple)) or len(identity) != 5
            or any(type(v) is not int or v < 0 for v in identity)):
        raise ValueError('recorded index needs an explicit physical source/native/clock/file address')
    return source, native, clock, tuple(identity)


def _validate_exact_decode_failure(raw):
    """A measured failed exact read, never an unobserved missing input.

    This validates persisted recorded evidence, not a fresh live decode. Keep
    its full source/file/clock/model/adapter/reason preimage; no neighboring
    frame, head, feature or successful-raster identity may be attached to it.
    """
    from .group_scene_focal_pose_motion import restore_native_row
    restored=restore_native_row(raw)
    if (restored.native_pixel_sha256 is not None or restored.frame_size_wh!=(0,0)
            or restored.people or restored.heads or restored.reason!='exact_native_decode_unavailable'
            or restored.raw_person_detector_json is not None or native_analysis_variant(raw) is None
            or raw.get('quality_approved') is not False
            or _freeze(restored.to_dict())!=_freeze(raw)):
        raise ValueError('recorded decode failure requires its exact unavailable native attempt preimage')


class RecordedGroupReviewIndex:
    def __init__(self, *, maximum_rows=16384, maximum_bytes=128*1024*1024, work_port=None):
        if (type(maximum_rows) is not int or not 1 <= maximum_rows <= 65536
                or type(maximum_bytes) is not int or not 1 <= maximum_bytes <= 256*1024*1024):
            raise ValueError('recorded index needs finite declared row/byte bounds')
        self.maximum_rows = maximum_rows
        self.maximum_bytes = maximum_bytes
        self.work_port = work_port
        self.rows = {}
        self.imported_rows = 0
        self.stored_bytes = 0

    def add_capture(self, capture, *, provenance):
        if (not isinstance(provenance, str) or not provenance or len(provenance) > 1024
                or capture.get('quality_approved') is not False
                or not isinstance(capture.get('rows'), dict)):
            raise ValueError('recorded index needs a bounded unapproved capture and explicit provenance')
        if self.imported_rows + len(capture['rows']) > self.maximum_rows:
            raise ValueError('recorded index imported-row budget exceeded before reuse')
        rows=[]
        for tick,row in capture['rows'].items():
            if str(row['output_tick']) != tick:
                raise ValueError('recorded index capture changed its output occurrence address')
            rows.append((row['native_observation'],capture.get('patches',{}).get(tick),
                capture.get('appearance',{}).get(tick),row['output_tick']))
        self._import_records(rows,provenance=provenance)

    def add_native_record(self,raw,appearance=None,*,provenance):
        """Import a real native payload without inventing an output occurrence."""
        from .group_scene_focal_pose_motion import restore_native_row
        restore_native_row(raw)
        if native_analysis_variant(raw) is None:
            raise ValueError('native record needs its exact model and adapter variant')
        validate_recorded_appearance(raw,appearance)
        self._import_records(((raw,None,appearance,None),),provenance=provenance)

    def add_native_decode_failure(self,raw,*,provenance):
        self.add_native_completion_records((),decode_failures=(raw,),provenance=provenance)

    def add_native_completion_records(self,records,*,decode_failures=(),provenance):
        """Atomically reload explicit success/failure cache entries, without NN.

        Persisted failure origin survives as an occurrence with no invented
        output tick. Success and failure at the same declared physical variant
        stay conflicting; a subsequent request cannot silently retry until one
        wins. This importer neither reads directories nor rewrites raw parents.
        """
        from .group_scene_focal_pose_motion import restore_native_row
        if (not isinstance(records,(tuple,list)) or not isinstance(decode_failures,(tuple,list))
                or not 1<=len(records)+len(decode_failures)<=self.maximum_rows-self.imported_rows):
            raise ValueError('native completion reload requires bounded explicit records')
        rows=[]
        for record in records:
            if not isinstance(record,dict) or set(record)!={'native_observation','appearance'}:
                raise ValueError('native completion success record changed its explicit schema')
            raw,appearance=record['native_observation'],record['appearance']
            restore_native_row(raw)
            if raw.get('native_pixel_sha256') is None or native_analysis_variant(raw) is None:
                raise ValueError('native completion success needs its exact pixel and analysis parent')
            validate_recorded_appearance(raw,appearance)
            rows.append((raw,None,appearance,None))
        for failure in decode_failures:
            _validate_exact_decode_failure(failure)
            rows.append((failure,None,None,None))
        self._import_records(tuple(rows),provenance=provenance,allow_decode_failures=True)

    def _import_records(self,rows,*,provenance,allow_decode_failures=False):
        if not isinstance(provenance,str) or not provenance or len(provenance)>1024:
            raise ValueError('recorded index needs explicit bounded provenance')
        if self.imported_rows+len(rows)>self.maximum_rows:
            raise ValueError('recorded index imported-row budget exceeded before reuse')
        pending = []
        byte_count = 0
        for raw,patches,appearance,tick in rows:
            if raw.get('quality_approved') is not False:
                raise ValueError('recorded index cannot grant or import quality approval')
            key = _key(raw['request_source_index'], raw['native_frame_index'],
                raw['source_clock_identity'], raw['source_file_identity'])
            pixel = raw.get('native_pixel_sha256')
            if pixel is None and allow_decode_failures:
                _validate_exact_decode_failure(raw)
                if patches is not None or appearance is not None:
                    raise ValueError('recorded failed decode cannot have raster-dependent features')
            elif (not isinstance(pixel, str) or len(pixel) != 64
                    or any(c not in '0123456789abcdef' for c in pixel)):
                raise ValueError('recorded index raw observation has no exact pixel identity')
            if appearance is not None:
                if (_key(appearance['request_source_index'], appearance['native_frame_index'],
                        appearance['source_clock_identity'], appearance['source_file_identity']) != key
                        or appearance['native_pixel_sha256'] != pixel
                        or appearance['recorded_native_observation_sha256'] != digest(raw)
                        or appearance.get('quality_approved') is not False):
                    raise ValueError('recorded appearance borrowed a different native parent')
                if 'descriptor_space' in appearance:
                    validate_recorded_appearance(raw,appearance)
            # Visibility/patch data survives missing appearance and tracking.
            value = dict(raw=raw, patches=patches, appearance=appearance)
            frozen = {name: None if payload is None else _freeze(payload) for name, payload in value.items()}
            byte_count += sum(len(payload.encode()) for payload in frozen.values() if payload is not None)
            pending.append((key, frozen, provenance, tick,native_analysis_variant(raw)))
        if self.stored_bytes + byte_count > self.maximum_bytes:
            raise ValueError('recorded index byte budget exceeded before reuse')
        if self.work_port is not None:
            self.work_port.charge('pose_evidence_bytes', byte_count)
        for key, frozen, origin, tick,variant in pending:
            record = self.rows.setdefault(key, dict(raw={}, patches={}, appearance={}, occurrences=[],
                analysis_variants={},features_by_raw={}))
            raw_sha=digest(frozen['raw'])
            record['analysis_variants'][raw_sha]=variant
            features=record['features_by_raw'].setdefault(raw_sha,dict(patches={},appearance={}))
            for name, payload in frozen.items():
                if payload is not None:
                    record[name].setdefault(digest(payload), payload)
                    if name!='raw':features[name].setdefault(digest(payload),payload)
            record['occurrences'].append(dict(provenance=origin, output_tick=tick))
        self.imported_rows += len(pending)
        # Count import bytes, including aliases. De-duplication is not free work.
        self.stored_bytes += byte_count

    def lookup(self, *, source, native, clock, identity,analysis_variant_sha256=None):
        key = _key(source, native, clock, identity)
        record = self.rows.get(key)
        result = dict(state='MISSING', raw=None, patches=None, appearance=None,
            recorded_observations_only=True, live_native_authority=False,
            fresh_inference_performed=False, quality_approved=False)
        if record is None:
            return result
        result['occurrences'] = tuple(dict(r) for r in record['occurrences'])
        result['raw_variant_sha256'] = tuple(sorted(record['raw']))
        result['available_analysis_variants']=tuple(sorted({v for v in record['analysis_variants'].values() if v is not None}))
        raws=record['raw']
        if analysis_variant_sha256 is not None:
            if (not isinstance(analysis_variant_sha256,str) or len(analysis_variant_sha256)!=64
                    or any(c not in '0123456789abcdef' for c in analysis_variant_sha256)):
                raise ValueError('recorded lookup needs its prior exact analysis variant')
            raws={k:v for k,v in raws.items() if record['analysis_variants'][k]==analysis_variant_sha256}
            if not raws:return dict(result,state='MISSING_DECLARED_ANALYSIS_VARIANT')
        if len(raws) != 1:
            return dict(result, state='CONFLICTING_NATIVE_RECORDS')
        raw_sha=next(iter(raws))
        result['raw'] = json.loads(raws[raw_sha])
        features=record['features_by_raw'][raw_sha]
        for name in ('patches', 'appearance'):
            variants = features[name]
            result[name + '_variant_sha256'] = tuple(sorted(variants))
            if len(variants) == 1:
                result[name] = json.loads(next(iter(variants.values())))
        result['state'] = ('CONFLICTING_FEATURE_RECORDS'
            if len(features['patches']) > 1 or len(features['appearance']) > 1
            else 'RECORDED_EXACT_DECODE_UNAVAILABLE' if result['raw']['native_pixel_sha256'] is None
            else 'RECORDED_NATIVE_REVIEW_INPUT')
        return result

    def reproject(self, query, tick, *, domain, source_file_identity,compact=False,analysis_variant_sha256=None):
        _, _, native = query.sample_at(tick)
        source = domain.request_index(query.eligible_source_index)
        found = self.lookup(source=source, native=native,
            clock=query.source_clock_identity, identity=source_file_identity,
            analysis_variant_sha256=analysis_variant_sha256)
        if found['raw'] is not None and found['raw']['native_pixel_sha256'] is not None:
            found['row'] = reproject_recorded_static_query(query, tick, found['raw'], domain=domain,compact=compact)
        else:
            found['row'] = None
        return found

    def diagnostics(self):
        return dict(imported_rows=self.imported_rows, physical_addresses=len(self.rows),
            import_evidence_bytes=self.stored_bytes,
            conflicting_native_addresses=sum(len(v['raw']) != 1 for v in self.rows.values()),
            conflicting_feature_addresses=sum(len(v['patches']) > 1 or len(v['appearance']) > 1
                for v in self.rows.values()), approval_granted=False)
