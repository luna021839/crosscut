"""Received scene-archive reference closure, NOT pre-receipt history authority.

An internal digest alone accepts a consistently re-signed omission. Bind the
selected program to the archive and check every retained query, hypothesis,
raw anchor, dense row and link reference. Independent selected/baseline pixel
replay remains mandatory. This is not a parent-owned append-only sensor journal
and cannot certify observations that were removed before receipt creation.
"""
from . import group_performer_family_owner as family
from .group_joint_source_camera import GroupOutputRole

CONTRACT = 'received_scene_discovery_reference_binding_v1'
SECTIONS = ('discoveries', 'hypotheses', 'native_rows',
    'parallel_articulation_rows', 'endpoint_pixel_links')


def archive_binding(archive):
    """A frozen inventory root; structural checks are separate and mandatory."""
    if not isinstance(archive, dict):
        raise ValueError('scene discovery archive must be an explicit mapping')
    return dict(contract=CONTRACT, archive_sha256=family.digest(archive),
        sections={name:dict(count=len(archive.get(name, ())),
            sha256=family.digest(archive.get(name))) for name in SECTIONS},
        pre_receipt_history_completeness_certified=False, quality_approved=False)


def validate_archive(program, dense):
    from .group_scene_observations import (SCENE_CONTRACT, DISCOVERY_QUERY_CONTRACT,
        _discovery_rolls, scene_anchor_ticks)
    from .group_scene_intent import validate_intent
    from .group_scene_composition import LINK_CONTRACT, validate_support_link
    from .group_native_detector_result import validate_serialized_detector_result
    archive = dense.get('scene_discovery')
    if (not isinstance(archive, dict) or archive.get('contract') != SCENE_CONTRACT
            or archive.get('quality_approved') is not False
            or archive.get('association_hypotheses_are_not_named_actors') is not True
            or any(not isinstance(archive.get(name), list) for name in SECTIONS if name != 'hypotheses')
            or not isinstance(archive.get('hypotheses'), dict)
            or not archive['discoveries'] or not archive['native_rows'] or not archive['hypotheses']):
        raise ValueError('scene discovery archive inventory or uncertainty authority missing')
    if program.get('discovery_archive_binding') != archive_binding(archive):
        raise ValueError('selected program does not bind the received scene discovery archive')
    domain = dense.get('source_domain') or {}
    eligible = domain.get('eligible_request_indices', [])
    clocks = domain.get('render_frame_clocks', [])
    if (not isinstance(eligible, list) or not eligible or len(set(eligible)) != len(eligible)
            or any(type(i) is not int or i < 0 for i in eligible)
            or not isinstance(clocks, list) or len(clocks) != len(eligible)):
        raise ValueError('scene archive needs the bound eligible/native clock domain')
    clock_hashes = dict(zip(eligible, map(family.digest, clocks), strict=True))
    raw = {}
    def ids(row, name):
        values = row.get(name)
        if (not isinstance(values, (list, tuple)) or any(not isinstance(v, dict)
                or type(v.get('observation_id')) is not int or v['observation_id'] < 0 for v in values)
                or len({v['observation_id'] for v in values}) != len(values)):
            raise ValueError('scene archive raw cohort lost frame-local observation IDs')
        return [v['observation_id'] for v in values]
    for row in (*archive['native_rows'], *archive['parallel_articulation_rows']):
        if row is None:  # An explicitly unavailable parallel modality is not zero people.
            continue
        if (not isinstance(row, dict) or row.get('request_source_index') not in clock_hashes
                or row.get('source_clock_identity') != clock_hashes[row['request_source_index']]
                or type(row.get('native_frame_index')) is not int or row['native_frame_index'] < 0
                or any(row.get(key) is not False for key in ('persistent_subject_correspondence_verified',
                    'named_member_identity_certified', 'motion_certified', 'visibility_certified', 'quality_approved'))):
            raise ValueError('scene archive raw source/clock/uncertainty changed')
        ids(row, 'heads'); ids(row, 'people')
        validate_serialized_detector_result(row)
        raw[family.digest(row)] = row
    def references(refs):
        if not isinstance(refs, list) or any(ref is not None and ref not in raw for ref in refs):
            raise ValueError('scene archive is missing a referenced original raw observation')
        return [None if ref is None else raw[ref] for ref in refs]
    def walk(value):
        if isinstance(value, dict):
            for name in ('raw_pair', 'raw_observation_sha256'):
                if name in value:
                    references(value[name])
            if 'native_observation' in value and family.digest(value['native_observation']) not in raw:
                raise ValueError('scene archive omitted a retained dense native observation')
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
    hypotheses = archive['hypotheses']
    for key, h in hypotheses.items():
        if (not isinstance(h, dict) or family.digest(h) != key or h.get('contract') != SCENE_CONTRACT
                or any(h.get(name) is not False for name in ('temporal_identity_certified',
                    'cross_camera_identity_certified', 'whole_head_extent_certified',
                    'body_phase_certified', 'quality_approved'))):
            raise ValueError('scene archive hypothesis hash or uncertainty changed')
        span = h.get('span_key')
        if (not isinstance(span, list) or len(span) != 6 or any(type(v) is not int for v in span[:5])
                or not 0 <= span[0] < len(eligible) or min(span[1:5]) < 0
                or span[2] <= span[1] or span[4] < span[3] or span[-1] != h.get('role')):
            raise ValueError('scene archive hypothesis source/native span missing')
        scope = h.get('endpoint_scope', 'whole')
        ticks = list(scene_anchor_ticks(span[1], span[2], scope))
        anchors = references(h.get('raw_observation_sha256'))
        head_ids = h.get('anchor_head_ids')
        if (h.get('anchor_output_ticks') != ticks or len(anchors) != len(ticks)
                or not isinstance(head_ids, list) or len(head_ids) != len(ticks)
                or any(row is None or row['request_source_index'] != eligible[span[0]]
                    or not span[3] <= row['native_frame_index'] <= span[4] for row in anchors)):
            raise ValueError('scene archive hypothesis raw anchors do not own its source span')
        validate_intent(h.get('shot_intent'), role=h['role'], first=span[1], end=span[2], endpoint_scope=scope)
        for tick, row, head in zip(ticks, anchors, head_ids, strict=True):
            if ((tick == span[1] and row['native_frame_index'] != span[3])
                    or (tick == span[2]-1 and row['native_frame_index'] != span[4])
                    or h['role'] not in (GroupOutputRole.FORMATION.value, GroupOutputRole.COMPOSITION.value)
                        and head not in ids(row, 'heads')):
                raise ValueError('scene archive hypothesis selected an absent native head/endpoint')
        if h['role'] == GroupOutputRole.COMPOSITION.value and h.get('body_anchor_ids') != [ids(row, 'people') for row in anchors]:
            raise ValueError('scene archive composition hypothesis filtered its raw cohort')
    referenced_hypotheses = set()
    query_keys = set()
    for query in archive['discoveries']:
        if not isinstance(query, dict) or query.get('query_contract') != DISCOVERY_QUERY_CONTRACT:
            raise ValueError('scene archive discovery lacks explicit ordered query ownership')
        rolls = list(_discovery_rolls(tuple(query.get('analysis_roll_query', ()))))
        span, scope = query.get('span_key'), query.get('endpoint_scope')
        if not isinstance(span, list) or len(span) != 6:
            raise ValueError('scene archive discovery span missing')
        query_key = family.digest([span, scope, rolls])
        if query_key in query_keys:
            raise ValueError('scene archive duplicated one owned discovery query')
        query_keys.add(query_key)
        variants = query.get('variants'); keys = query.get('hypothesis_ids')
        if (not isinstance(variants, list) or not isinstance(keys, list) or len(set(keys)) != len(keys)
                or any(key not in hypotheses for key in keys)):
            raise ValueError('scene archive discovery omitted its referenced hypotheses')
        if query.get('reason') == 'scene_span_needs_endpoint_window':
            try:
                scene_anchor_ticks(span[1], span[2], scope)
            except ValueError:
                if variants or keys:
                    raise ValueError('unavailable scene query invented hypotheses')
                continue
            raise ValueError('scene query falsely claims a missing endpoint window')
        ticks = list(scene_anchor_ticks(span[1], span[2], scope))
        if (query.get('anchor_output_ticks') != ticks
                or [v.get('analysis_roll') for v in variants] != rolls):
            raise ValueError('scene archive discovery changed its requested roll/anchor inventory')
        for variant in variants:
            anchors = references(variant.get('raw_observation_sha256'))
            if (len(anchors) != len(ticks) or any(row is None for row in anchors)
                    or variant.get('heads_by_anchor') != [ids(row, 'heads') for row in anchors]
                    or variant.get('people_by_anchor') != [ids(row, 'people') for row in anchors]):
                raise ValueError('scene archive discovery filtered raw or unmatched cohort members')
        for key in keys:
            h = hypotheses[key]
            if h['span_key'] != span or h.get('endpoint_scope', 'whole') != scope or h['analysis_roll'] not in rolls:
                raise ValueError('scene archive query borrowed a hypothesis from another scope')
        referenced_hypotheses.update(keys)
    if referenced_hypotheses != set(hypotheses):
        raise ValueError('scene archive hypothesis inventory lost its discovery owner')
    required = set(dense.get('hypothesis_keys', ())) | {
        key for group in program.get('baseline_hypothesis_keys', ()) for key in group}
    if not required <= set(hypotheses) or any(hypotheses.get(key) != value
            for key, value in dense.get('scene_hypotheses', {}).items()):
        raise ValueError('scene archive omitted selected or original baseline hypotheses')
    walk(archive); walk(dense.get('mutable_shots')); walk(dense.get('incident_projected_observations'))
    for link in archive['endpoint_pixel_links']:
        if link.get('contract') == LINK_CONTRACT and link.get('observed_support_paths') is True:
            a, b = references(link['raw_observation_sha256'])
            validate_support_link(link, a, b)
    return archive_binding(archive)
