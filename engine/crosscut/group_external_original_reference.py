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


def _global_header(plan):
    # Local clocks are compared separately against a frozen, root-bound
    # phase-zero parent. Every other current/future executable header remains.
    return tuple(row for row in _header(plan) if row[0]!='authorized_shot_video_clocks')


def _compare_spans(original,candidate,domain,spans,*,clock_context=None):
    # Keep the existing 16,384-tick comparison cap. A complete exterior may
    # require several bounded chunks; no tiny protected sample stands in for it.
    for first,end in spans:
        for start in range(first,end,16384):
            result=compare_group_reference_interval(reference=original,candidate=candidate,
                reference_source_clocks=domain.clocks_by_request,
                candidate_source_clocks=domain.clocks_by_request,
                output_intervals=((start,min(end,start+16384)),),
                candidate_shot_clock_context=clock_context)
            if not result.planned_parameters_preserved:
                raise ValueError('candidate changed protected external source/clock/camera/look parameters')


def _validate_root_bound_phase(*,original,plan,domain,protected_intervals,context):
    from . import group_performer_family_owner as family
    from .shot_video_clock import FrozenShotVideoClockContext,validate_authorized_shot_video_clocks
    if (not isinstance(context,FrozenShotVideoClockContext)
            or context.original_reference_plan is None or context.source_domain!=domain
            or family.plan_payload(context.original_reference_plan)!=family.plan_payload(original)
            or context.protected_output_intervals!=protected_intervals
            or _global_header(context.incumbent_plan)!=_global_header(original)):
        raise ValueError('external local clock lacks its matching original root and phase-zero context')
    authorities=validate_authorized_shot_video_clocks(plan=plan,context=context)
    for clock in plan.authorized_shot_video_clocks:
        if (clock.first_output_frame,clock.end_output_frame) not in context.mutable_output_intervals:
            raise ValueError('external local clock escaped its frozen whole-shot mutable scope')
    if not authorities:
        from .group_review_interval_compile import verify_realized_sampler
        # A zero-phase programme can carry its root context without a local
        # clock. It must then use EXACT phase-zero source times/native IDs.
        # Deleting a nonzero clock is not permission to keep its retimed rows.
        verify_realized_sampler(context.incumbent_plan,plan,domain,
            tuple(t for first,end in context.mutable_output_intervals for t in range(first,end)),
            require_source_request_time=True)
    outside=[];cursor=0
    for first,end in context.mutable_output_intervals:
        if cursor<first:outside.append((cursor,first))
        cursor=end
    if cursor<original.output_frame_count:outside.append((cursor,original.output_frame_count))
    # The new unretimed topology AND the actual retimed/camera programme must
    # retain the root's entire exterior, including crop/roll/look, not only its
    # source ordinals. Never strip clocks for these exact sampler comparisons.
    _compare_spans(original,context.incumbent_plan,domain,tuple(outside))
    _compare_spans(original,plan,domain,tuple(outside),clock_context=context)
    _compare_spans(original,plan,domain,protected_intervals,clock_context=context)


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
        context=None;production=getattr(bundle,'native_motion_production',None)
        if production is not None or plan.authorized_shot_video_clocks:
            from .group_joint_bundle_metadata import validate_native_motion_production
            validate_native_motion_production(production,attached_bundle=bundle)
            context=production.shot_clock_context
        from .pipeline import _validate_review_first_group_bundle
        _validate_review_first_group_bundle(bundle,eligible_indices=self.source_domain.eligible_indices,
            offsets=self.source_domain.audio_offsets,timewarps=self.source_domain.source_timewarps,
            common_start=self.original_bundle.render_plan.master_audio_start,
            phrase_times=bundle.phrase_times,shot_clock_context=context)
        self.validate_numeric_plan(plan,shot_clock_context=context)
        old = json.loads(self._frozen_preimage)['original_boundary_look']
        current = boundary_look_inventory(bundle.boundary_look_match, plan)
        if current['mode'] != old['mode']:
            raise ValueError('external original boundary-look policy forbids implicit creation/removal')

    def validate_numeric_plan(self,plan,*,shot_clock_context=None):
        """Exclude protected writes before selected feature/completion work.

        This does not admit look metadata, native duties, source freshness or
        completed learned authority. Those remain mandatory later consumers.
        """
        self.validate_original()
        if not isinstance(plan,RenderPlan):
            raise TypeError('external numeric preservation needs a complete RenderPlan')
        if (_global_header(plan) != _global_header(self.original_bundle.render_plan)
                or any(t.kind.uses_overlap for t in plan.transitions)):
            raise ValueError('external reference common output/source clock authority changed')
        if plan.authorized_shot_video_clocks or shot_clock_context is not None:
            _validate_root_bound_phase(original=self.original_bundle.render_plan,plan=plan,
                domain=self.source_domain,protected_intervals=self.protected_output_intervals,
                context=shot_clock_context)
        else:
            _compare_spans(self.original_bundle.render_plan,plan,self.source_domain,self.protected_output_intervals)


def validate_external_original_reference_payload(policy, *, plan, look, source_domain_payload,
        source_identities, live_clocks=None,shot_clock_context_payload=None):
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
                or _global_header(plan) != _global_header(original)
                or original.authorized_shot_video_clocks
                or any(t.kind.uses_overlap for t in (*original.transitions,*plan.transitions))
                or not 1 <= original.output_frame_count <= 36000):
            raise ValueError('external original boundary-look/root/source authority changed')
        domain = numeric_replay_domain(original, domain_payload)
        if live_clocks is not None and live_clocks != domain.clocks_by_request:
            raise ValueError('external original lost its live native frame clocks')
        spans = tuple(tuple(span) for span in policy['protected_output_intervals'])
        _validate_protected_spans(spans, original.output_frame_count)
        if plan.authorized_shot_video_clocks or shot_clock_context_payload is not None:
            from .shot_video_clock import frozen_shot_video_clock_context_from_dict
            context=frozen_shot_video_clock_context_from_dict(shot_clock_context_payload,source_domain=domain)
            _validate_root_bound_phase(original=original,plan=plan,domain=domain,
                protected_intervals=spans,context=context)
        else:
            _compare_spans(original,plan,domain,spans)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('external original policy lost its explicit root preimage') from exc
