"""Bounded source-native review reuse, never live observation authority.

An output tick belongs to an edit, not to a camera-independent detector row.
The index retains every distinct recorded result at a physical address. It
cannot select the most favorable capture, borrow another clock/file, or turn
a missing appearance vector into a missing visible person.
"""
import json

from .group_performer_family_owner import digest
from .group_scene_static_query import reproject_recorded_static_query


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
        pending = []
        byte_count = 0
        for tick, row in capture['rows'].items():
            if str(row['output_tick']) != tick:
                raise ValueError('recorded index capture changed its output occurrence address')
            raw = row['native_observation']
            if raw.get('quality_approved') is not False:
                raise ValueError('recorded index cannot grant or import quality approval')
            key = _key(raw['request_source_index'], raw['native_frame_index'],
                raw['source_clock_identity'], raw['source_file_identity'])
            pixel = raw.get('native_pixel_sha256')
            if (not isinstance(pixel, str) or len(pixel) != 64
                    or any(c not in '0123456789abcdef' for c in pixel)):
                raise ValueError('recorded index raw observation has no exact pixel identity')
            appearance = capture.get('appearance', {}).get(tick)
            if appearance is not None:
                if (_key(appearance['request_source_index'], appearance['native_frame_index'],
                        appearance['source_clock_identity'], appearance['source_file_identity']) != key
                        or appearance['native_pixel_sha256'] != pixel
                        or appearance['recorded_native_observation_sha256'] != digest(raw)
                        or appearance.get('quality_approved') is not False):
                    raise ValueError('recorded appearance borrowed a different native parent')
            # Visibility/patch data survives missing appearance and tracking.
            value = dict(raw=raw, patches=capture.get('patches', {}).get(tick), appearance=appearance)
            frozen = {name: None if payload is None else _freeze(payload) for name, payload in value.items()}
            byte_count += sum(len(payload.encode()) for payload in frozen.values() if payload is not None)
            pending.append((key, frozen, provenance, row['output_tick']))
        if self.stored_bytes + byte_count > self.maximum_bytes:
            raise ValueError('recorded index byte budget exceeded before reuse')
        if self.work_port is not None:
            self.work_port.charge('pose_evidence_bytes', byte_count)
        for key, frozen, origin, tick in pending:
            record = self.rows.setdefault(key, dict(raw={}, patches={}, appearance={}, occurrences=[]))
            for name, payload in frozen.items():
                if payload is not None:
                    record[name].setdefault(digest(payload), payload)
            record['occurrences'].append(dict(provenance=origin, output_tick=tick))
        self.imported_rows += len(pending)
        # Count import bytes, including aliases. De-duplication is not free work.
        self.stored_bytes += byte_count

    def lookup(self, *, source, native, clock, identity):
        key = _key(source, native, clock, identity)
        record = self.rows.get(key)
        result = dict(state='MISSING', raw=None, patches=None, appearance=None,
            recorded_observations_only=True, live_native_authority=False,
            fresh_inference_performed=False, quality_approved=False)
        if record is None:
            return result
        result['occurrences'] = tuple(dict(r) for r in record['occurrences'])
        result['raw_variant_sha256'] = tuple(sorted(record['raw']))
        if len(record['raw']) != 1:
            return dict(result, state='CONFLICTING_NATIVE_RECORDS')
        result['raw'] = json.loads(next(iter(record['raw'].values())))
        for name in ('patches', 'appearance'):
            variants = record[name]
            result[name + '_variant_sha256'] = tuple(sorted(variants))
            if len(variants) == 1:
                result[name] = json.loads(next(iter(variants.values())))
        result['state'] = ('CONFLICTING_FEATURE_RECORDS'
            if len(record['patches']) > 1 or len(record['appearance']) > 1
            else 'RECORDED_NATIVE_REVIEW_INPUT')
        return result

    def reproject(self, query, tick, *, domain, source_file_identity,compact=False):
        _, _, native = query.sample_at(tick)
        source = domain.request_index(query.eligible_source_index)
        found = self.lookup(source=source, native=native,
            clock=query.source_clock_identity, identity=source_file_identity)
        if found['raw'] is not None:
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
