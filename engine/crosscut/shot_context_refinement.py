"""Bounded GROUP framing refinement, independent of timing/identity quality.

One complete shot is composed against both unchanged neighboring pictures.
Cached timeline geometry only shortlists work. Admission uses exact native
endpoint observations projected through both the old and new camera plans.
The final encoded-output QA remains required; this is not its replacement.
"""
from dataclasses import dataclass, replace
import math
from time import monotonic

from .editorial_transform_events import rendered_transform_event_records
from .optional_native_worker import isolated_optional
from .render import segment_frame_counts
from .shot_context_camera import propose_shot_context_camera
from .shot_context_binding import build_shot_context_binding, bound_shot_context_cameras
from .timeline import ShotType


def _framing_rows_improve(proposal,rows,basis,*,position_tolerance=1e-6):
    if len(rows)!=2 or [r.get('boundary_index') for r in rows]!=[proposal.segment_index-1,proposal.segment_index]:
        return False
    gains=[]
    for row in rows:
        if row.get('basis')!=basis:
            return False
        before,after=row.get('before'),row.get('after')
        if (not isinstance(before,(tuple,list)) or not isinstance(after,(tuple,list))
                or len(before)!=2 or len(after)!=2
                or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1
                    for v in (*before,*after))):
            return False
        # Neither cut nor scale/position can buy a regression in the other.
        if after[0]>before[0]+1e-6 or after[1]>before[1]+position_tolerance:
            return False
        gains.append(sum(before)-sum(after))
    return sum(gains)>.01


def both_boundary_framings_improve(proposal):
    return (_framing_rows_improve(proposal,proposal.boundary_framing,
        'same_native_face_projected_through_both_camera_plans_not_encoded_qa')
        and _framing_rows_improve(proposal,proposal.pixel_boundary_framing,
        'redetected_full_output_coordinate_pixels_not_encoded_qa',
        # The pixel path may have a different peak-zoom intermediate raster
        # even when this endpoint is exactly neutral. Do not turn subpixel
        # detector jitter into a regression. Analytic geometry must still be
        # nonworse, scale stays strict, and total improvement must be > .01.
        # _box_discontinuity normalizes center distance by half a frame.
        # Use the longer edge so this never admits more than one output pixel
        # on either axis, including landscape and portrait outputs.
        position_tolerance=2./max(proposal.plan.width,proposal.plan.height)))


def shot_context_shortlist(plan,timeline):
    """Cheap ordering only; no cached face is accepted as exact cut evidence."""
    if len(timeline.choices)!=len(plan.segments):
        raise ValueError('shot-context timeline and render plan cardinality differ')
    counts=segment_frame_counts(plan)
    focal={ShotType.CLOSE_UP,ShotType.MEDIUM}
    rows=[]
    for index in range(1,len(plan.segments)-1):
        left,middle,right=plan.segments[index-1:index+2]
        choices=timeline.choices[index-1:index+2]
        if (not 2<=counts[index]<=300 or middle.source_index in (left.source_index,right.source_index)
                or any(s.crop is not None for s in (left,middle,right))
                or any(t.kind.value!='hard_cut' for t in plan.transitions[index-1:index+1])
                or choices[1].internal_scene_cut_master_seconds):
            continue
        endpoints=((choices[0],'exit'),(choices[1],'entry'),(choices[1],'exit'),(choices[2],'entry'))
        if any((getattr(choice,side+'_shot_type') or choice.shot_type) not in focal for choice,side in endpoints):
            continue
        scales=[value if (value:=getattr(choice,side+'_shot_scale')) is not None else choice.shot_scale
            for choice,side in endpoints]
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in scales):
            continue
        # Favor face-dominant cuts and the existing pan/short-settle conflict.
        # Never hard-code a song, broadcaster, member, second or segment ID.
        priority=min(scales)+(.2 if middle.pan_crop is not None else 0.)
        rows.append((-priority,index))
    return tuple(index for _,index in sorted(rows))


@dataclass(frozen=True)
class ShotContextRefinement:
    plan: object
    motion: object
    changed_segments: tuple
    attempts: tuple
    elapsed_seconds: float

    def to_dict(self):
        return dict(contract='bounded_two_sided_whole_shot_framing_v1',
            changed_segments=list(self.changed_segments),attempts=list(self.attempts),
            elapsed_seconds=self.elapsed_seconds,audio_and_source_phase_unchanged=True,
            identity_or_gesture_certified=False,quality_approved=False,
            learned_transformer_inference=False)


def attach_shot_context_proposal(plan,motion,proposal):
    """Commit camera, receipt and event inventory together or not at all."""
    import json
    from .post_render_qa import _bound_local_alignment_tracks
    index=proposal.segment_index
    if not both_boundary_framings_improve(proposal):
        raise ValueError('shot-context candidate does not improve both boundary framings')
    if (replace(proposal.plan,segments=plan.segments)!=plan
            or proposal.plan.sources!=plan.sources or proposal.plan.transitions!=plan.transitions
            or proposal.plan.master_audio_start!=plan.master_audio_start
            or proposal.plan.output_frame_count!=plan.output_frame_count
            or len(proposal.plan.segments)!=len(plan.segments)
            or any(a!=b for i,(a,b) in enumerate(zip(plan.segments,proposal.plan.segments)) if i!=index)
            or replace(proposal.plan.segments[index],pan_crop=plan.segments[index].pan_crop,
                editorial_transform=plan.segments[index].editorial_transform)!=plan.segments[index]):
        raise ValueError('shot-context proposal changed source/audio clock or a fixed neighbor')
    receipt=build_shot_context_binding(proposal)
    old=_bound_local_alignment_tracks(motion.to_dict(),json.loads(json.dumps(plan.to_json())))
    events=rendered_transform_event_records(proposal.plan.segments,proposal.plan.fps)
    changed=replace(motion,segments=tuple(replace(item,
        track=proposal.plan.segments[i].editorial_transform,
        reasons=(*item.reasons,'two_sided_whole_shot_face_framing') if i==index else item.reasons)
        for i,item in enumerate(motion.segments)),
        shot_context_cameras=(*motion.shot_context_cameras,receipt),
        zoom_event_count=len(events['zoom']),pan_event_count=len(events['pan']),roll_event_count=len(events['roll']))
    new=_bound_local_alignment_tracks(changed.to_dict(),json.loads(json.dumps(proposal.plan.to_json())))
    if not old[0]<=new[0] or not old[1]<=new[1] or index not in new[0]:
        raise ValueError('shot-context proposal lost a complete dense camera binding')
    if len(bound_shot_context_cameras(changed.to_dict(),proposal.plan))!=len(changed.shot_context_cameras):
        raise ValueError('shot-context proposal invalidated a previous fixed-neighbor binding')
    added=len(new[1]-old[1])
    authored=dict(motion.authored_event_counts) or dict(zoom=motion.zoom_event_count,
        pan=motion.pan_event_count,roll=motion.roll_event_count)
    authored['zoom']+=added
    return replace(changed,boundary_match_count=motion.boundary_match_count+added,
        authored_event_counts=tuple((key,authored[key]) for key in ('zoom','pan','roll')))


def attach_coupled_shot_proposal(plan,motion,proposal):
    """Replace one context receipt with an atomic pair, retaining all others."""
    import json
    from .post_render_qa import _bound_local_alignment_tracks,_bound_context_cameras
    from .coupled_shot_binding import build_coupled_shot_binding
    index=proposal.segment_index;candidate=proposal.plan;changed_indices={index,index+1}
    singles=bound_shot_context_cameras(motion.to_dict(),plan)
    if index not in singles or not both_boundary_framings_improve(proposal.outgoing):
        raise ValueError('coupled camera needs an existing bound shot and improved framing')
    if (replace(candidate,segments=plan.segments)!=plan
            or len(candidate.segments)!=len(plan.segments)
            or any(a!=b for i,(a,b) in enumerate(zip(plan.segments,candidate.segments)) if i not in changed_indices)
            or any(replace(candidate.segments[i],editorial_transform=plan.segments[i].editorial_transform)!=plan.segments[i]
                for i in changed_indices)
            or plan.segments[index+1].editorial_transform is not None
            or candidate.segments[index].editorial_transform.evaluate(0.)!=plan.segments[index].editorial_transform.evaluate(0.)):
        raise ValueError('coupled camera changed source/audio timing or an unrelated camera')
    receipt=build_coupled_shot_binding(proposal)
    events=rendered_transform_event_records(candidate.segments,candidate.fps)
    changed=replace(motion,segments=tuple(replace(item,track=candidate.segments[i].editorial_transform,
        reasons=(*item.reasons,'coupled_whole_shot_and_tracked_entry_framing') if i in changed_indices else item.reasons)
        for i,item in enumerate(motion.segments)),
        shot_context_cameras=tuple(row for row in motion.shot_context_cameras if row['segment_index']!=index),
        coupled_shot_cameras=(*motion.coupled_shot_cameras,receipt),
        zoom_event_count=len(events['zoom']),pan_event_count=len(events['pan']),roll_event_count=len(events['roll']))
    before=_bound_local_alignment_tracks(motion.to_dict(),json.loads(json.dumps(plan.to_json())))
    after=_bound_local_alignment_tracks(changed.to_dict(),json.loads(json.dumps(candidate.to_json())))
    contexts=_bound_context_cameras(changed.to_dict(),json.loads(json.dumps(candidate.to_json())))
    if (not before[0]<=after[0] or not before[1]<=after[1] or not changed_indices<=after[0]
            or len(contexts)!=len(changed.shot_context_cameras)+2*len(changed.coupled_shot_cameras)):
        raise ValueError('coupled camera invalidated a complete camera binding')
    # The same two cut events are refined, not counted again as new matches.
    return changed


@isolated_optional()
def refine_shot_context_cameras(*,plan,timeline,motion,scene_cut_times,
        maximum_attempts=24,maximum_runtime_seconds=120.):
    if (type(maximum_attempts) is not int or not 1<=maximum_attempts<=64
            or type(maximum_runtime_seconds) not in (float,int)
            or not math.isfinite(maximum_runtime_seconds) or not 0<maximum_runtime_seconds<=180):
        raise ValueError('invalid bounded shot-context refinement budget')
    if len(scene_cut_times)!=len(plan.sources):
        raise ValueError('shot-context cuts must cover the full render source domain')
    started=monotonic();attempts=[];changed=[];coupled_attempts=0
    existing={row['segment_index'] for row in motion.to_dict().get('shot_context_cameras',())}
    existing.update(i for row in motion.to_dict().get('coupled_shot_cameras',())
        for i in (row['segment_index'],row['segment_index']+1))
    for index in shot_context_shortlist(plan,timeline):
        if len(attempts)>=maximum_attempts or monotonic()-started>=maximum_runtime_seconds:
            break
        if any(abs(index-other)<=1 for other in (*existing,*changed)):
            continue
        row=dict(segment_index=index)
        try:
            proposal=propose_shot_context_camera(plan=plan,segment_index=index,
                scene_cut_times=scene_cut_times,experimental=True,
                maximum_runtime_seconds=min(30.,maximum_runtime_seconds-(monotonic()-started)))
            row['boundary_framing']=list(proposal.boundary_framing)
            row['pixel_boundary_framing']=list(proposal.pixel_boundary_framing)
            next_motion=attach_shot_context_proposal(plan,motion,proposal)
            plan,motion=proposal.plan,next_motion
            changed.append(index);row['status']='geometry_admitted_pending_encoded_qa'
            # Refine a remaining visible position jump only after the whole
            # shot succeeds. Keep already matched/liked contexts unchanged.
            # This shares the same stage deadline; there is no retry loop.
            residual=proposal.pixel_boundary_framing[-1]['after'][1]
            from .boundary_subject_alignment import BoundarySubjectAlignmentConfig
            unmatched_position=2.*BoundarySubjectAlignmentConfig().already_matched_center_distance
            if (residual>unmatched_position and coupled_attempts<6 and index+2<len(plan.segments)
                    and not any(abs(index+1-other)<=1 for other in (*existing,*changed) if other!=index)
                    and maximum_runtime_seconds-(monotonic()-started)>.1):
                from .coupled_shot_camera import propose_coupled_shot_camera
                coupled_attempts+=1
                try:
                    coupled=propose_coupled_shot_camera(previous=proposal,scene_cut_times=scene_cut_times,
                        maximum_runtime_seconds=min(20.,maximum_runtime_seconds-(monotonic()-started)))
                    paired_motion=attach_coupled_shot_proposal(plan,motion,coupled)
                    plan,motion=coupled.plan,paired_motion;changed.append(index+1)
                    row['coupled_entry']=coupled.to_dict()
                except (ValueError,OSError) as exc:
                    row['coupled_entry']=dict(status='unavailable',reason=str(exc))
        except (ValueError,OSError) as exc:
            row.update(status='unavailable',reason=str(exc))
        attempts.append(row)
    return ShotContextRefinement(plan,motion,tuple(changed),tuple(attempts),monotonic()-started)
