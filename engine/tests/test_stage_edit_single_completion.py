"""Actual beam/clock selection with explicitly synthetic head/readiness inputs.

No trained human model, native performer quality or professional approval is
claimed. The public decoder, typed bridge, exact clocks and selector are real.
"""
from dataclasses import replace
from pathlib import Path
import math
from unittest.mock import patch

import pytest

from crosscut.stage_edit_single_completion import select_complete_learned_path
from crosscut.stage_edit_pipeline_bridge import (StageEditPipelineBridgePreflight,
    StageEditPipelineBridgeResult,_exact_clock_candidate)
from crosscut.stage_edit_semimarkov_beam import (StageEditSemiMarkovConfig,
    decode_stage_edit_semimarkov_beam)
from crosscut.stage_edit_transformer import StageEditSequence,DEFAULT_STAGE_EDIT_SCHEMA
from crosscut.section_director import SectionPlan
from engine.tests.test_stage_edit_pipeline_bridge import (_snapshot_sequence,_candidate,
    _snapshot_identity,_ready,_typed_head_result,_typed_section_director)
from engine.tests.stage_edit_causal_sensing_test_support import synthetic_production_snapshot
from engine.tests.test_stage_edit_semimarkov_beam import _frames,_metadata
from engine.tests import test_stage_edit_semimarkov_beam as beam_fixture


def synthetic_complete_generation(root,*,sources=None,section_plan=None,duration=6,
        source_offsets=None,width=1920,height=1080,fps=30.,prefer_long_spans=False):
    sources=tuple(sources or (root/f'NONHUMAN-UNOPENED-{i}.mp4' for i in range(4)))
    count=len(sources);ids=tuple(f'source-{i}' for i in range(count))
    director=_typed_section_director()
    section_plan=section_plan or SectionPlan((replace(director.section_plan.sections[0],
        end_seconds=float(duration)),))
    chorus=tuple(s.section_index for s in section_plan.sections if s.role.value=='chorus_like')
    allocated=tuple(s.section_index for s in section_plan.sections
        if s.role.value=='chorus_like' and s.establish_with_wide_shot)
    assert len(allocated)==math.ceil(director.chorus_wide_anchor_requested_fraction*len(chorus))
    director=replace(director,section_plan=section_plan,
        evidence=tuple(replace(director.evidence[0],section_index=s.section_index,
            chorus_wide_anchor_selected=s.section_index in allocated) for s in section_plan.sections),
        chorus_section_count=len(chorus),chorus_wide_anchor_allocated_count=len(allocated),
        chorus_wide_anchor_allocated_section_indices=allocated)
    prediction=beam_fixture.StageEditSemiMarkovBeamTests._dynamic_prediction()
    frames=[]
    for frame in _frames(duration,count):
        directive=section_plan.directive_at(frame.master_time_seconds)
        frames.append(replace(frame,dynamic_crop_v2=prediction,section_role=directive.role.value,
            section_progress=(frame.master_time_seconds-directive.start_seconds)/directive.duration_seconds))
    if prefer_long_spans:
        # Explicit synthetic head inputs favour the ORIGINAL 15/min BUILD
        # schedule. Do not change production clocks, gates or section clamps.
        bucket=DEFAULT_STAGE_EDIT_SCHEMA.duration_bucket(3.)
        frames=[replace(frame,cut_logit=8. if index%3==0 else -8.,
            duration_logits=tuple(8. if b==bucket else -8.
                for b in range(DEFAULT_STAGE_EDIT_SCHEMA.duration_bucket_count)))
            for index,frame in enumerate(frames)]
    frames=tuple(frames)
    first=_snapshot_sequence().tokens[0]
    sequence=StageEditSequence('EXPLICIT-SYNTHETIC-COMPLETE-BEAM',tuple(replace(first,
        master_time_seconds=frame.master_time_seconds,source_candidates=tuple(_candidate(s,.3) for s in ids))
        for frame in frames))
    snapshot=synthetic_production_snapshot(sequence)
    checkpoint=root/'EXPLICIT-SYNTHETIC-READINESS-NOT-TRAINED'
    preflight=StageEditPipelineBridgePreflight(snapshot_path=root/'NONHUMAN-snapshot.json',
        checkpoint_directory=checkpoint,source_ids=ids,snapshot=snapshot,readiness=_ready(checkpoint),
        snapshot_identity=_snapshot_identity(snapshot,root/'NONHUMAN-snapshot.json'),
        edit_mode='group',target_member=None)
    config=StageEditSemiMarkovConfig(beam_width=16,timeline_count=8,minimum_unique_sources=3,
        enforce_montage_grammar=False)
    with patch('crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report',return_value=_ready(checkpoint)):
        decoded=decode_stage_edit_semimarkov_beam(checkpoint_directory=checkpoint,sources=sources,
            source_metadata=_metadata(count,float(duration)),head_frames=frames,section_plan=section_plan,
            output=root/'NONHUMAN-NOT-RENDERED.mp4',config=config,width=width,height=height,fps=fps)
    offsets=tuple(source_offsets or (0.,)*count);durations=(float(duration)+2.,)*count
    retained=tuple(_exact_clock_candidate(timeline=timeline,sources=sources,source_offsets=offsets,
        source_durations=durations,internal_cuts=((),)*count,timewarps=(None,)*count,
        section_plan=section_plan,config=config,dynamic_crop_version=2,
        continuity_authority=preflight.continuity_authority) for timeline in decoded.timelines)
    head=replace(_typed_head_result(checkpoint),frames=frames,sequence_id=sequence.sequence_id,
        canonical_source_ids=ids)
    generation=StageEditPipelineBridgeResult(preflight=preflight,head_materialization=head,
        semimarkov=decoded,retained_candidates=retained,learned_section_director=director,
        selection_policy=dict(visual_baseline_replaced=False,selected_for_render=False,
            dynamic_crop_version=2,continuity_authority=preflight.continuity_authority,
            continuity_scorer_contract=preflight.continuity_scorer_contract,
            continuity_scorer_fingerprint=preflight.continuity_scorer_fingerprint,
            contextual_source_pair_identity_v2=None,temporal_pose_sidecar=snapshot.temporal_pose_sidecar_audit()))
    args=dict(sources=sources,source_offsets=offsets,timewarps=(None,)*count,source_durations=durations)
    return generation,args


def test_actual_complete_beam_is_compared_before_selection_not_tuple_order(tmp_path):
    generation,args=synthetic_complete_generation(tmp_path)
    generation=replace(generation,retained_candidates=tuple(reversed(generation.retained_candidates)))
    calls=[]
    selection=select_complete_learned_path(generation,**args,numeric_admission=lambda p:calls.append(p))
    assert len(calls)==8
    assert selection.candidate==max(generation.retained_candidates,key=lambda c:c.score)
    receipt=selection.to_dict()
    assert len(receipt['compared_complete_paths'])==8
    assert receipt['maximum_selected_completions']==1
    assert not receipt['native_scene_selection_performed']
    assert not receipt['quality_approved']
    assert all(row['native_feature_work_performed'] is False for row in receipt['compared_complete_paths'])


def test_numeric_preservation_rejects_best_before_any_selected_materialization(tmp_path):
    generation,args=synthetic_complete_generation(tmp_path,source_offsets=(.5,)*4)
    denied=generation.retained_candidates[0].edit.to_render_plan()
    def protect(plan):
        if plan==denied:raise ValueError('explicit protected original sampler mismatch')
    selection=select_complete_learned_path(generation,**args,numeric_admission=protect)
    assert selection.candidate.rank!=generation.retained_candidates[0].rank
    failed=[r for r in selection.to_dict()['compared_complete_paths'] if not r['numerically_admitted']]
    assert len(failed)==1
    assert 'protected original sampler' in failed[0]['rejection']


@pytest.mark.parametrize('mutation',['score','clock','rank_alias','truncated','spliced_history'])
def test_old_scores_or_spliced_unredecoded_programmes_never_reach_completion(tmp_path,mutation):
    generation,args=synthetic_complete_generation(tmp_path)
    candidates=list(generation.retained_candidates);first=candidates[0]
    if mutation=='score':candidates[0]=replace(first,score=first.score+1.)
    if mutation=='clock':
        shots=list(first.edit.shots);shots[0]=replace(shots[0],segment=replace(shots[0].segment,source_start=.25))
        candidates[0]=replace(first,edit=replace(first.edit,shots=tuple(shots)))
    if mutation=='rank_alias':candidates[1]=replace(candidates[1],rank=first.rank)
    if mutation=='truncated':
        candidates[0]=replace(first,edit=replace(first.edit,shots=first.edit.shots[:-1],boundaries=first.edit.boundaries[:-1]))
    if mutation=='spliced_history':
        history=list(first.edit.candidate_history);history.insert(0,dict(unredecoded_fragment_splice=True))
        candidates[0]=replace(first,edit=replace(first.edit,candidate_history=tuple(history)))
    generation=replace(generation,retained_candidates=tuple(candidates))
    with pytest.raises(ValueError):select_complete_learned_path(generation,**args)


def test_numeric_no_go_is_terminal_not_a_complete_rank_retry(tmp_path):
    generation,args=synthetic_complete_generation(tmp_path)
    calls=[]
    def deny(plan):
        calls.append(plan);raise ValueError('explicit original KEEP constraint')
    with pytest.raises(ValueError,match='no complete learned path'):
        select_complete_learned_path(generation,**args,numeric_admission=deny)
    assert len(calls)==8
