"""Label-only admission and sampler measure; never opens heldout media."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np


def hierarchical_prior_class_balance(tracks: Sequence[Any]) -> dict[str, Any]:
    references=defaultdict(set)
    sizes=Counter((track.song_id,track.reference_id) for track in tracks)
    for song,reference in sizes:references[song].add(reference)
    count_histogram=np.zeros(8,np.int64);motion_histogram=np.zeros(2,np.int64)
    count_mass=np.zeros(8);motion_mass=np.zeros(2)
    for track in tracks:
        count=len(track.values)
        if not 1<=count<=8 or track.motion_label not in {'static','dynamic'}:
            raise ValueError('invalid structured prior training label')
        mass=1./(len(references)*len(references[track.song_id])*sizes[track.song_id,track.reference_id])
        count_histogram[count-1]+=1;count_mass[count-1]+=mass
        if not track.motion_label_loss_masked:
            motion=int(track.motion_label=='dynamic')
            motion_histogram[motion]+=1;motion_mass[motion]+=mass
    def weights(mass):
        present=mass>0
        if not present.any():raise ValueError('Dynamic Crop v3 prior class labels are empty')
        result=np.zeros_like(mass)
        result[present]=float(mass.sum())/(int(present.sum())*mass[present])
        return result.tolist()
    return dict(contract='training_only_hierarchical_sampler_conditional_expected_weight_one_v2',
        count_histogram_1_to_8=count_histogram.tolist(),motion_histogram_static_dynamic=motion_histogram.tolist(),
        count_sampler_mass_1_to_8=count_mass.tolist(),motion_sampler_mass_static_dynamic=motion_mass.tolist(),
        count_class_weights_1_to_8=weights(count_mass),motion_class_weights_static_dynamic=weights(motion_mass),
        validation_labels_consumed=False,heldout_labels_consumed=False)


def require_trajectory_selection_coverage(tracks: Sequence[Any], *, partition: str) -> dict[str, Any]:
    """Necessary conditions of the existing gate, not a weakened replacement."""
    if partition not in {'train','validation'}:
        raise ValueError('trajectory admission must not inspect heldout labels')
    counts=sorted({len(track.values) for track in tracks})
    motion=sorted({track.motion_label for track in tracks if not track.motion_label_loss_masked})
    has_range=False
    for track in tracks:
        values=np.asarray(track.values,float)
        if values.ndim!=2 or values.shape[1]!=4 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f'data_contract_unsatisfiable: partition={partition}, invalid trajectory values')
        has_range|=bool(np.any(np.ptp(values,axis=0)>1e-8))
    reasons=[]
    if len(counts)<2:reasons.append('insufficient_target_count_classes')
    if motion!=['dynamic','static']:reasons.append('insufficient_unmasked_motion_classes')
    if not has_range:reasons.append('no_target_trajectory_range')
    if reasons:
        raise ValueError(f'data_contract_unsatisfiable: partition={partition}, reasons={reasons}')
    return dict(partition=partition,track_count=len(tracks),target_count_classes=counts,
        unmasked_motion_classes=motion,nonzero_target_range=has_range,
        label_only=True,model_forward_performed=False,heldout_labels_consumed=False)


def require_releasable_training_plan(*, epochs: int, batches_per_epoch: int,
        minimum_finetune_steps: int, checkpoint_burn_in_steps: int,
        prior_may_be_retained: bool, minimum_joint_steps_after_release: int) -> dict[str, int]:
    for name,value in (('epochs',epochs),('batches_per_epoch',batches_per_epoch),
            ('minimum_finetune_steps',minimum_finetune_steps),('checkpoint_burn_in_steps',checkpoint_burn_in_steps),
            ('minimum_joint_steps_after_release',minimum_joint_steps_after_release)):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f'{name} must be a positive integer')
    if not isinstance(prior_may_be_retained,bool):raise ValueError('explicit prior retention possibility required')
    if checkpoint_burn_in_steps>minimum_finetune_steps:
        raise ValueError('checkpoint selection burn-in cannot exceed the finetune step floor')
    retention=((checkpoint_burn_in_steps+batches_per_epoch-1)//batches_per_epoch*batches_per_epoch
        if prior_may_be_retained else 0)
    total=epochs*batches_per_epoch
    floor=max(minimum_finetune_steps,retention+minimum_joint_steps_after_release if retention else 0)
    if total<floor:
        raise ValueError(f'insufficient_release_budget: planned={total}, retention={retention}, required={floor}')
    return dict(planned_steps=total,maximum_retention_steps=retention,
        minimum_stop_step_when_retained=floor,minimum_joint_steps_after_release=minimum_joint_steps_after_release)
