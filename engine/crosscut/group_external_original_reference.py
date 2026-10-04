"""Explicit external GROUP original, frozen before optional authoring.

This is a numeric preservation policy, not human quality approval. Its native
clocks are caller-supplied; live source/clock validation still belongs to the
native owner. A fresh automatic request is not implicitly an absence-preserving
external-reference transaction merely because its baseline has no look fit.
"""
from dataclasses import dataclass, field, fields, replace
import json
from typing import Any

from .group_source_domain import GroupSourceDomain
from .group_reference_interval import compare_group_reference_interval
from .render import RenderPlan


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _header(plan):
    # Future executable header fields must not silently escape preservation.
    return tuple((f.name,getattr(plan,f.name)) for f in fields(RenderPlan)
        if f.name not in ('segments','transitions','output')) + (
            ('output_frame_count',plan.output_frame_count),)


def _validate_protected_spans(spans, output_frame_count):
    previous, total = -1, 0
    for span in spans:
        if (not isinstance(span, tuple) or len(span) != 2
                or any(type(t) is not int for t in span)
                or not 0 <= span[0] < span[1] <= output_frame_count
                or span[0] < previous):
            raise ValueError('external protected ranges must be ordered disjoint exact output ticks')
        previous = span[1]
        total += span[1]-span[0]
    if total > 16384:
        raise ValueError('external protected range comparison exceeds its declared frame cap')


@dataclass(frozen=True, slots=True)
class GroupExternalOriginalReference:
    original_bundle: Any
    source_domain: GroupSourceDomain
    protected_output_intervals: tuple[tuple[int, int], ...]
    _frozen_preimage: str = field(init=False, repr=False)

    def __post_init__(self):
        from .pipeline import _GroupRenderCandidateBundle
        if (not isinstance(self.original_bundle, _GroupRenderCandidateBundle)
                or not isinstance(self.source_domain, GroupSourceDomain)
                or self.original_bundle.render_plan.sources != self.source_domain.request_sources
                or self.original_bundle.render_plan.aspect_ratio.value != '16:9'
                or not isinstance(self.protected_output_intervals, tuple)):
            raise ValueError('external original requires a complete typed GROUP bundle/domain/policy')
        plan = self.original_bundle.render_plan
        if (plan.authorized_shot_video_clocks
                or any(t.kind.uses_overlap for t in plan.transitions)
                or not 1 <= plan.output_frame_count <= 36000):
            raise ValueError('external original needs dedicated authority for overlap/local clocks')
        _validate_protected_spans(self.protected_output_intervals, plan.output_frame_count)
        object.__setattr__(self, '_frozen_preimage', _canonical(self._preimage()))

    def _preimage(self):
        from . import group_performer_family_owner as family
        bundle = self.original_bundle
        return dict(original_plan_sha256=family.digest(family.plan_payload(bundle.render_plan)),
            original_metadata_sha256=family.digest(family.metadata_payload(bundle)),
            source_domain_sha256=family.digest(self.source_domain.to_dict()),
            source_identities=[list(family._file_identity(path)) for path in self.source_domain.eligible_sources],
            original_boundary_look=family.boundary_look_inventory(bundle.boundary_look_match, bundle.render_plan),
            protected_output_intervals=[list(span) for span in self.protected_output_intervals])

    def validate_original(self):
        if _canonical(self._preimage()) != self._frozen_preimage:
            raise ValueError('external original reference changed after its policy was frozen')

    def to_dict(self):
        from .group_performer_family_owner import digest, metadata_payload
        self.validate_original()
        value = json.loads(self._frozen_preimage)
        # The native archive freezes tuples as JSON arrays. Return that same
        # canonical scalar representation BEFORE parent/child transport.
        return json.loads(_canonical(dict(value, contract='external_original_group_look_and_protected_range_policy_v1',
            policy_sha256=digest(value),
            mode='ABSENT_PRESERVED' if value['original_boundary_look']['mode']=='ORIGINAL_ABSENT'
                else 'ACTIVE_EXISTING',
            implicit_boundary_look_creation_allowed=False,
            original_numeric_plan=self.original_bundle.render_plan.to_json(),
            original_metadata=metadata_payload(self.original_bundle),
            original_source_domain=self.source_domain.to_dict(),
            source_clock_authority='caller_supplied_requires_live_native_owner_validation',
            encoded_pixel_identity_verified=False, quality_approved=False)))

    def baseline_for_output(self, output):
        self.validate_original()
        return replace(self.original_bundle, render_plan=replace(self.original_bundle.render_plan, output=output))

    def validate_baseline(self, bundle, plan):
        from . import group_performer_family_owner as family
        self.validate_original()
        original = json.loads(self._frozen_preimage)
        if (not isinstance(plan, RenderPlan) or bundle.render_plan != plan
                or family.digest(family.plan_payload(plan)) != original['original_plan_sha256']
                or family.digest(family.metadata_payload(bundle)) != original['original_metadata_sha256']):
            raise ValueError('external original was replaced before optional work began')
        self.validate_candidate(bundle, plan)

    def validate_source_domain(self, domain):
        from .group_performer_family_owner import digest
        self.validate_original()
        if (not isinstance(domain, GroupSourceDomain)
                or domain.request_sources != self.source_domain.request_sources
                or digest(domain.to_dict()) != json.loads(self._frozen_preimage)['source_domain_sha256']):
            raise ValueError('external original lost its live request source/clock domain')

    def validate_candidate(self, bundle, plan):
        from .group_performer_family_owner import boundary_look_inventory
        self.validate_original()
        if not isinstance(plan, RenderPlan) or getattr(bundle, 'render_plan', None) != plan:
            raise ValueError('external-reference candidate lost its complete owned GROUP plan')
        if (_header(plan) != _header(self.original_bundle.render_plan)
                or plan.authorized_shot_video_clocks
                or any(t.kind.uses_overlap for t in plan.transitions)):
            raise ValueError('external reference common output/source clock authority changed')
        old = json.loads(self._frozen_preimage)['original_boundary_look']
        current = boundary_look_inventory(bundle.boundary_look_match, plan)
        if current['mode'] != old['mode']:
            raise ValueError('external original boundary-look policy forbids implicit creation/removal')
        if self.protected_output_intervals:
            comparison = compare_group_reference_interval(reference=self.original_bundle.render_plan,
                candidate=plan, reference_source_clocks=self.source_domain.clocks_by_request,
                candidate_source_clocks=self.source_domain.clocks_by_request,
                output_intervals=self.protected_output_intervals)
            if not comparison.planned_parameters_preserved:
                raise ValueError('candidate changed protected external source/clock/camera/look parameters')


def validate_external_original_reference_payload(policy, *, plan, look, source_domain_payload,
        source_identities, live_clocks=None):
    """Re-derive the serialized root, actual tracks and protected native grid.

    The fresh parent ALSO compares this entire record to its actual external
    reference; a self-consistent child hash cannot establish that authority.
    Final replay uses the existing content-bound receipt and fresh native
    clock/source checks, never promotes this numeric result to visual approval.
    """
    if policy is None:
        return
    from . import group_performer_family_owner as family
    from .render_plan_serialization import render_plan_from_json
    from .group_scene_edit_operations import numeric_replay_domain
    if not isinstance(policy, dict):
        raise ValueError('external original policy lost its serialized complete root')
    try:
        original = render_plan_from_json(policy['original_numeric_plan'])
        metadata = policy['original_metadata']
        domain_payload = policy['original_source_domain']
        inventory = family.boundary_look_inventory(metadata['boundary_look_match'], original)
        preimage = dict(original_plan_sha256=family.digest(family.plan_payload(original)),
            original_metadata_sha256=family.digest(metadata),
            source_domain_sha256=family.digest(domain_payload),
            source_identities=policy['source_identities'], original_boundary_look=inventory,
            protected_output_intervals=policy['protected_output_intervals'])
        expected_mode = 'ABSENT_PRESERVED' if inventory['mode']=='ORIGINAL_ABSENT' else 'ACTIVE_EXISTING'
        if (any(policy[key] != value for key, value in preimage.items())
                or policy['policy_sha256'] != family.digest(preimage)
                or policy['contract'] != 'external_original_group_look_and_protected_range_policy_v1'
                or policy['mode'] != expected_mode
                or policy['implicit_boundary_look_creation_allowed'] is not False
                or policy['source_clock_authority'] != 'caller_supplied_requires_live_native_owner_validation'
                or policy['encoded_pixel_identity_verified'] is not False
                or policy['quality_approved'] is not False
                or domain_payload != source_domain_payload
                or policy['source_identities'] != source_identities
                or family.boundary_look_inventory(look, plan)['mode'] != inventory['mode']
                or _header(plan) != _header(original)
                or original.authorized_shot_video_clocks or plan.authorized_shot_video_clocks
                or any(t.kind.uses_overlap for t in (*original.transitions,*plan.transitions))
                or not 1 <= original.output_frame_count <= 36000):
            raise ValueError('external original boundary-look/root/source authority changed')
        domain = numeric_replay_domain(original, domain_payload)
        if live_clocks is not None and live_clocks != domain.clocks_by_request:
            raise ValueError('external original lost its live native frame clocks')
        spans = tuple(tuple(span) for span in policy['protected_output_intervals'])
        _validate_protected_spans(spans, original.output_frame_count)
        if spans:
            comparison = compare_group_reference_interval(reference=original, candidate=plan,
                reference_source_clocks=domain.clocks_by_request,
                candidate_source_clocks=domain.clocks_by_request, output_intervals=spans)
            if not comparison.planned_parameters_preserved:
                raise ValueError('serialized candidate changed protected external reference parameters')
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('external original policy lost its explicit root preimage') from exc
