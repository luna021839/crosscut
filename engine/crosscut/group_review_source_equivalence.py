"""Verified file continuity across archived and live native observations.

A mount's device number is not a new camera cut. Archived raw/feature/body
parents remain unchanged. Only a live sensor's explicitly verified source
manifest can bind their physical continuity to the current file identity.
This is file equivalence, never actor, motion or perceptual-quality authority.
"""
from dataclasses import dataclass,field
import json

from .group_performer_family_owner import digest

_OWNER=object()


@dataclass(frozen=True,slots=True)
class VerifiedNativeFileBinding:
    request_source_index: int
    source_clock_identity: str
    raw_parent_identity: str
    recorded_file_identity: tuple[int,...]
    physical_file_identity: tuple[int,...]
    _owner: object=field(repr=False,compare=False)

    def __post_init__(self):
        if self._owner is not _OWNER:
            raise ValueError('physical file continuity requires the live verified sensor binding')

    def identity_for(self,raw):
        if (raw['request_source_index']!=self.request_source_index
                or raw['source_clock_identity']!=self.source_clock_identity
                or tuple(raw['source_file_identity'])!=self.recorded_file_identity
                or digest(raw)!=self.raw_parent_identity):
            raise ValueError('physical file continuity borrowed another native raw parent')
        return self.physical_file_identity


def freeze_verified_native_file_binding(sensors,raw,source_equivalence_json):
    session=sensors.session;session.budget.check()
    if session.closed:raise ValueError('physical file binding needs its live native session')
    source=raw['request_source_index']
    current=tuple(session._assert_source_identity(source))
    recorded=tuple(raw['source_file_identity'])
    if raw['source_clock_identity']!=digest(session.domain.clocks_by_request[source].to_dict()):
        raise ValueError('physical file equivalence changed its source clock')
    if recorded!=current:
        owned=(getattr(sensors,'recorded_motion_bindings',{}).get(source),
            getattr(sensors,'recorded_motion_additional_bindings',{}).get(source))
        if source_equivalence_json is None or source_equivalence_json not in owned:
            raise ValueError('archived parent lacks its live sensor-verified source equivalence')
        receipt=json.loads(source_equivalence_json)
        sha=receipt.get('content_sha256')
        if (receipt.get('state')!='CONTENT_VERIFIED_DEVICE_REMOUNT'
                or tuple(receipt.get('recorded_file_identity',()))!=recorded
                or tuple(receipt.get('current_file_identity',()))!=current
                or recorded[1:]!=current[1:]
                or not isinstance(sha,str) or len(sha)!=64
                or any(c not in '0123456789abcdef' for c in sha)
                or receipt.get('archived_observations_rewritten') is not False):
            raise ValueError('archived parent changed beyond verified device-only remount')
    return VerifiedNativeFileBinding(source,raw['source_clock_identity'],digest(raw),recorded,current,_OWNER)


def physical_review_file_identity(view):
    raw=view['row']['native_observation'];binding=view.get('verified_native_file_binding')
    if binding is None:
        identity=raw.get('source_file_identity')
        return None if identity is None else tuple(identity)
    if not isinstance(binding,VerifiedNativeFileBinding):
        raise ValueError('untyped source alias cannot grant actor-path file continuity')
    return binding.identity_for(raw)
