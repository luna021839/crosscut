"""Fresh projected visual observations for an atomic two-shot source edit.

The fixed rendered path is materialized directly, not fed through the legacy
solo-source selector. This is one part of the complete owned bundle producer:
it grants no dense role, correspondence, motion or production admission.
"""
from dataclasses import dataclass, replace

from .group_joint_native_preparation import _plan_identity
from .group_joint_source_camera import GroupSourceCameraState
from .group_joint_state_factory import (assemble_joint_group_states,
    assemble_single_group_source_camera_state)
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
    dependency_support: dict | None = None

    def to_dict(self):
        return dict(contract='atomic_joint_group_fixed_rendered_observations_v1',
            incumbent_plan_identity=self.incumbent_plan_identity,
            plan_identity=_plan_identity(self.plan),
            changed_segment_indices=list(self.changed_segment_indices),
            refreshed_segment_indices=list(self.refreshed_segment_indices),
            dependency_support=self.dependency_support,
            full_path=[state.to_dict() for state in self.path],
            visual_timeline=self.timeline.to_dict(),
            observations_basis='fixed_source_native_clock_and_actual_complete_render_camera',
            solo_replacement_required=False, dense_role_evidence_verified=False,
            production_selected=False, quality_approved=False)


@isolated_optional()
def refresh_joint_group_render_observations(*, plan, timeline, domain, path,
        sync_scores, feature_cache_directory=None, feature_derived_cache_directory=None,
        require_persistent_feature_cache=False, shot_clock_context=None, shot_video_clocks=(),
        maximum_refreshed_cells=None, retained_applied_look_edges=()):
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
    from .group_scene_execution_scope import GlobalSceneExecutionPath
    if isinstance(path,GlobalSceneExecutionPath):
        # Explicit global ownership and atomic union, not len(path) relaxation.
        # The ordinary complete owner externally rederives original intent
        # before this numeric/descriptor materializer receives the program.
        assembler=path.assemble
    else:
        if not isinstance(path, tuple) or len(path) not in (3, 4):
            raise ValueError('joint rendered observations require three or four actual whole-shot states')
        assembler = (assemble_single_group_source_camera_state if len(path) == 3
            else assemble_joint_group_states)
    proposed,changed=assembler(plan=plan,domain=domain,path=path)
    if shot_video_clocks:
        from .shot_video_clock import FrozenShotVideoClockContext, attach_authorized_shot_video_clocks
        if (not isinstance(shot_clock_context, FrozenShotVideoClockContext)
                or shot_clock_context.incumbent_plan != plan or shot_clock_context.source_domain != domain):
            raise ValueError('joint clock context differs from the external incumbent/source domain')
        proposed = attach_authorized_shot_video_clocks(plan=proposed,
            context=shot_clock_context, clocks=shot_video_clocks)
    if not changed:
        return GroupJointRenderedObservations(_plan_identity(plan),plan,timeline,path,(),())
    from .group_scene_dependency_support import derive_support,cell_intervals
    support = (None if plan.authorized_shot_video_clocks or proposed.authorized_shot_video_clocks
        else derive_support(plan,proposed,retained_applied_look_edges=retained_applied_look_edges))
    refreshed = (tuple(range(max(0,min(changed)-1),min(len(plan.segments),max(changed)+2)))
        if support is None else tuple(support['descriptor_refresh_cells']))
    if maximum_refreshed_cells is not None and (type(maximum_refreshed_cells) is not int
            or maximum_refreshed_cells < 1 or len(refreshed) > maximum_refreshed_cells):
        raise ValueError('source-family refresh cell budget exceeded before descriptor work')
    choices=list(timeline.choices)
    for index in changed:
        segment=proposed.segments[index]
        # These rows are only a numeric clock scaffold; the shared materializer
        # replaces every source-dependent descriptor before returning them.
        choices[index]=replace(choices[index],source_index=domain.eligible_index(segment.source_index),
            source_local_start=segment.source_start,source_local_end=segment.source_end)
    scaffold=replace(timeline,choices=tuple(choices),candidate_review_summaries=(),
        analysis_basis='UNREFRESHED_joint_source_camera_scaffold_not_a_selected_timeline')
    fresh = scaffold
    for first,end in cell_intervals(refreshed):
        fresh=refresh_fixed_group_render_observations(plan=proposed,clock_timeline=fresh,
            first_segment_index=first,last_segment_index=end-1,source_domain=domain,
            sync_scores=sync_scores,feature_cache_directory=feature_cache_directory,
            feature_derived_cache_directory=feature_derived_cache_directory,
            require_persistent_feature_cache=require_persistent_feature_cache, shot_clock_context=shot_clock_context)
    return GroupJointRenderedObservations(_plan_identity(plan),proposed,fresh,path,
        changed,refreshed,support)
