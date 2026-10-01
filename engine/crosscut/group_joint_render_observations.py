"""Fresh projected visual observations for an atomic two-shot source edit.

The fixed rendered path is materialized directly, not fed through the legacy
solo-source selector. This is one part of the complete owned bundle producer:
it grants no dense role, correspondence, motion or production admission.
"""
from dataclasses import dataclass, replace

from .group_joint_native_preparation import _plan_identity
from .group_joint_source_camera import GroupSourceCameraState
from .group_joint_state_factory import assemble_joint_group_states
from .group_source_domain import GroupSourceDomain
from .group_source_span_proposal import refresh_fixed_group_render_observations
from .optional_native_worker import isolated_optional
from .render import RenderPlan
from .visual_score import VisualTimelinePlan


@dataclass(frozen=True, slots=True)
class GroupJointRenderedObservations:
    incumbent_plan_identity: str
    plan: RenderPlan
    timeline: VisualTimelinePlan
    path: tuple[GroupSourceCameraState, ...]
    changed_segment_indices: tuple[int, ...]
    refreshed_segment_indices: tuple[int, ...]

    def to_dict(self):
        return dict(contract='atomic_joint_group_fixed_rendered_observations_v1',
            incumbent_plan_identity=self.incumbent_plan_identity,
            plan_identity=_plan_identity(self.plan),
            changed_segment_indices=list(self.changed_segment_indices),
            refreshed_segment_indices=list(self.refreshed_segment_indices),
            full_path=[state.to_dict() for state in self.path],
            visual_timeline=self.timeline.to_dict(),
            observations_basis='fixed_source_native_clock_and_actual_complete_render_camera',
            solo_replacement_required=False, dense_role_evidence_verified=False,
            production_selected=False, quality_approved=False)


@isolated_optional()
def refresh_joint_group_render_observations(*, plan, timeline, domain, path,
        sync_scores, feature_cache_directory=None, feature_derived_cache_directory=None,
        require_persistent_feature_cache=False):
    """Assemble both new sources/cameras before sensing dependent descriptors.

    Replacing B with C's source does not have to pass a solo constructor while
    C still holds that source. Whole-state assembly first protects actual cut
    cadence and every native sampling tick. Only then are changed cells and
    their adjacent boundary descriptors sensed under the new render cameras.
    The full ordinary bundle owner must still refresh review, transitions,
    motion/face receipts and exact music authority before final admission.
    """
    if (not isinstance(plan,RenderPlan) or not isinstance(timeline,VisualTimelinePlan)
            or not isinstance(domain,GroupSourceDomain) or len(timeline.choices)!=len(plan.segments)):
        raise ValueError('joint rendered observations need their complete incumbent timeline and source domain')
    # Validate the external incumbent, not a source-index-mutated stale row.
    for index,(choice,segment) in enumerate(zip(timeline.choices,plan.segments,strict=True)):
        if (choice.segment_index!=index or domain.request_index(choice.source_index)!=segment.source_index
                or any(abs(a-b)>1e-6 for a,b in (
                    (choice.canonical_master_start,plan.master_audio_start+segment.master_start),
                    (choice.canonical_master_end,plan.master_audio_start+segment.master_start+segment.duration),
                    (choice.source_local_start,segment.source_start),(choice.source_local_end,segment.source_end)))):
            raise ValueError('joint rendered observations have a stale incumbent source/clock timeline')
    proposed,changed=assemble_joint_group_states(plan=plan,domain=domain,path=path)
    if not changed:
        return GroupJointRenderedObservations(_plan_identity(plan),plan,timeline,path,(),())
    first=max(0,min(changed)-1);last=min(len(plan.segments)-1,max(changed)+1)
    choices=list(timeline.choices)
    for index in changed:
        segment=proposed.segments[index]
        # These rows are only a numeric clock scaffold; the shared materializer
        # replaces every source-dependent descriptor before returning them.
        choices[index]=replace(choices[index],source_index=domain.eligible_index(segment.source_index),
            source_local_start=segment.source_start,source_local_end=segment.source_end)
    scaffold=replace(timeline,choices=tuple(choices),candidate_review_summaries=(),
        analysis_basis='UNREFRESHED_joint_source_camera_scaffold_not_a_selected_timeline')
    fresh=refresh_fixed_group_render_observations(plan=proposed,clock_timeline=scaffold,
        first_segment_index=first,last_segment_index=last,source_domain=domain,
        sync_scores=sync_scores,feature_cache_directory=feature_cache_directory,
        feature_derived_cache_directory=feature_derived_cache_directory,
        require_persistent_feature_cache=require_persistent_feature_cache)
    return GroupJointRenderedObservations(_plan_identity(plan),proposed,fresh,path,
        changed,tuple(range(first,last+1)))
