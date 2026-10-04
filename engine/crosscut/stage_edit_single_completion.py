"""Select a complete learned beam path BEFORE its one expensive completion.

This is numerical retention ownership, not native scene/motion admission.
Original-duty sensing, mixed-role lifecycle, independent native parent/final
and real decoded quality remain separate obligations. No beam fragments are
spliced and no old path score is reused for a changed programme.
"""
from dataclasses import dataclass,replace
from hashlib import sha256
import json
import math

from .render import aligned_render_segment
from .stage_edit_pipeline_bridge import (StageEditPipelineBridgeResult,
    RetainedStageEditCandidate,STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT)
from .stage_edit_semimarkov_beam import StageEditBeamDecision
from .stage_edit_dynamic_crop_semimarkov_adapter import DynamicCropTrackDecodeV2
from .stage_edit_dynamic_crop_semimarkov_adapter_v3 import DynamicCropTrackDecodeV3

CONTRACT='complete_learned_beam_numeric_selection_before_single_completion_v1'


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True,slots=True)
class SelectedCompleteLearnedPath:
    candidate: RetainedStageEditCandidate
    generation_sha256: str
    compared_path_records: tuple

    def to_dict(self):
        return dict(contract=CONTRACT,generation_sha256=self.generation_sha256,
            selected_rank=self.candidate.rank,selected_score=self.candidate.score,
            selected_stage_edit_sha256=_digest(self.candidate.edit.to_dict()),
            compared_complete_paths=[dict(row) for row in self.compared_path_records],
            selection_basis='unchanged_complete_semimarkov_path_score',
            beam_fragments_spliced=False,completion_before_selection=False,
            maximum_selected_completions=1,native_scene_selection_performed=False,
            original_native_duties_frozen=False,quality_approved=False)


def select_complete_learned_path(generation,*,sources,source_offsets,timewarps,
        source_durations,maximum_paths=8,numeric_admission=None):
    """Compare the finite FULL family once; never complete ranks until one works.

    A failed numerical admission is recorded without native/feature work. A
    later materialization/completion failure is terminal for this family; it
    is not permission to choose another rank with a renewed budget.
    """
    if not isinstance(generation,StageEditPipelineBridgeResult):
        raise TypeError('single completion needs the actual typed learned generation')
    generation.__post_init__()
    candidates=generation.retained_candidates
    if (type(maximum_paths) is not int or not 1<=maximum_paths<=8
            or not 1<=len(candidates)<=maximum_paths
            or generation.semimarkov.audit.request_fulfilled is not True
            or generation.semimarkov.audit.returned_timeline_count!=len(generation.semimarkov.timelines)
            or generation.preflight.edit_mode!='group'
            or generation.learned_section_director is None):
        raise ValueError('single completion lost its finite complete GROUP beam family')
    sources=tuple(sources);source_offsets=tuple(source_offsets);timewarps=tuple(timewarps)
    source_durations=tuple(source_durations)
    if (len(sources)!=len(generation.preflight.source_ids)
            or any(len(v)!=len(sources) for v in (source_offsets,timewarps,source_durations))):
        raise ValueError('single completion changed the complete eligible source clock domain')
    timelines={t.rank:t for t in generation.semimarkov.timelines}
    if (len(timelines)!=len(generation.semimarkov.timelines)
            or len({c.rank for c in candidates})!=len(candidates)):
        raise ValueError('single completion cannot alias decoded path ranks')
    from .request_work_lease import current_request_work_phase
    port=current_request_work_phase()
    records=[];admitted=[]
    for candidate in candidates:
        if port is not None:port.charge('states')
        if (type(candidate.rank) is not int or candidate.rank<1
                or type(candidate.score) not in (int,float) or not math.isfinite(candidate.score)
                or candidate.candidate_origin!='stage_edit_semimarkov'
                or candidate.exact_clock_reprojected is not True
                or candidate.rank not in timelines):
            raise ValueError('single completion requires an unchanged decoded complete path')
        timeline=timelines[candidate.rank]
        if (type(timeline.score) not in (int,float) or not math.isfinite(timeline.score)
                or abs(candidate.score-timeline.score)>1e-9
                or not timeline.score_components
                or any(type(v) not in (int,float) or not math.isfinite(v)
                    for v in timeline.score_components.values())
                or abs(math.fsum(timeline.score_components.values())-timeline.score)>1e-6
                or len(timeline.decisions)!=len(timeline.edit.shots)
                or any(not isinstance(d,StageEditBeamDecision) for d in timeline.decisions)):
            raise ValueError('retained score/history is not its complete decoded path score')
        projected=[];warped=0
        for decision,shot in zip(timeline.decisions,timeline.edit.shots,strict=True):
            index=decision.source_index
            if type(index) is not int or not 0<=index<len(sources):
                raise ValueError('decoded complete path changed source indices')
            decode=decision.dynamic_crop_decode if generation.preflight.dynamic_crop_version==2 else decision.dynamic_crop_decode_v3
            expected_type=DynamicCropTrackDecodeV2 if generation.preflight.dynamic_crop_version==2 else DynamicCropTrackDecodeV3
            if (not isinstance(decode,expected_type) or not decode.eligible
                    or decode.track!=decision.editorial_transform
                    or shot.segment.editorial_transform!=decision.editorial_transform):
                raise ValueError('decoded complete path lacks its actual typed applied camera track')
            warp=timewarps[index]
            if warp is not None and abs(warp.global_offset_seconds-source_offsets[index])>1e-6:
                raise ValueError('decoded complete path lost its external alignment clock')
            segment=aligned_render_segment(source_index=index,
                reference_start=decision.master_start_seconds,
                common_master_start=timeline.edit.section_plan.start_seconds,
                source_offset=source_offsets[index],duration=decision.duration_seconds,
                editorial_transform=decision.editorial_transform,timewarp=warp)
            if (segment.source_start<0 or segment.source_end>source_durations[index]+1e-6
                    or shot.shot_index!=decision.shot_index
                    or shot.dynamic_crop_decode_v3!=decision.dynamic_crop_decode_v3):
                raise ValueError('decoded complete path lost source bounds or typed camera decode')
            projected.append(replace(shot,segment=segment))
            warped+=int(warp is not None and not warp.scalar_fallback)
        event=dict(bridge=STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT,visual_baseline_replaced=False,
            selected_for_render=False,exact_source_clock_reprojection=True,
            timewarped_segment_count=warped,dynamic_crop_version=generation.preflight.dynamic_crop_version,
            continuity_authority=generation.preflight.continuity_authority)
        expected=replace(timeline.edit,sources=sources,shots=tuple(projected),
            candidate_history=tuple(timeline.edit.candidate_history)+(event,))
        if (candidate.timewarped_segment_count!=warped
                or _digest(expected.to_dict())!=_digest(candidate.edit.to_dict())
                or expected.section_plan!=generation.learned_section_director.section_plan):
            raise ValueError('retained full edit differs from its own beam and exact clock reprojection')
        record=dict(rank=candidate.rank,score=candidate.score,
            stage_edit_sha256=_digest(candidate.edit.to_dict()),numerically_admitted=True,
            native_feature_work_performed=False)
        if numeric_admission is not None:
            try:numeric_admission(candidate.edit.to_render_plan())
            except (TypeError,ValueError) as exc:
                record.update(numerically_admitted=False,rejection_type=type(exc).__name__,
                    rejection=str(exc)[:1000])
        records.append(record)
        if record['numerically_admitted']:admitted.append(candidate)
    if not admitted:
        raise ValueError('no complete learned path preserves the required external numeric domain')
    selected=min(admitted,key=lambda c:(-c.score,c.rank))
    return SelectedCompleteLearnedPath(selected,_digest(generation.to_dict()),
        tuple(tuple(row.items()) for row in records))
