"""Lossless ROI-track wire storage and raw-return/view reference closure.

The compressed tracks are the bytes retained in the link cache/archive, not
an estimate of RSS. Expanding a view never executes a model or tracking call.
Received closure is not controller-owned pre-receipt history authentication.
"""
from hashlib import sha256
import json

import numpy as np

from .group_joint_primitive_journal import encoded,packed,unpacked,primitive_inputs,restore_numeric_return

CONTRACT='lossless_native_roi_track_wire_v1'
MAX_TRACK_DECODE_BYTES=6*1024*1024


def pack_region_tracks(value):
    if 'region_track_storage' in value:
        expand_region_tracks(value)
        return value
    tracks=value.get('region_tracks')
    if not isinstance(tracks,list):raise ValueError('ROI tracks require their actual bounded inventory')
    payload=encoded(tracks)
    if len(payload)>MAX_TRACK_DECODE_BYTES:raise ValueError('ROI track view exceeds its wire contract')
    result=dict(value);result.pop('region_tracks')
    result['region_track_storage']=dict(contract=CONTRACT,sha256=sha256(payload).hexdigest(),
        lossless_json=packed(payload),pre_receipt_history_certified=False,quality_approved=False)
    return result


def expand_region_tracks(value):
    """Decode a bounded semantic projection; keep raw float bits in the journal."""
    storage=value.get('region_track_storage')
    if storage is None:
        if not isinstance(value.get('region_tracks'),list):raise ValueError('ROI track inventory is absent')
        return value
    if ('region_tracks' in value or not isinstance(storage,dict)
            or set(storage)!={'contract','sha256','lossless_json','pre_receipt_history_certified','quality_approved'}
            or storage['contract']!=CONTRACT or storage['pre_receipt_history_certified'] is not False
            or storage['quality_approved'] is not False):
        raise ValueError('ROI wire storage duplicated tracks or promoted authority')
    payload=unpacked(storage['lossless_json'],limit=MAX_TRACK_DECODE_BYTES)
    if sha256(payload).hexdigest()!=storage['sha256']:
        raise ValueError('ROI track storage lost its exact canonical bytes')
    tracks=json.loads(payload)
    if not isinstance(tracks,list) or encoded(tracks)!=payload:
        raise ValueError('ROI stored tracks are not a lossless canonical inventory')
    result=dict(value);result.pop('region_track_storage');result['region_tracks']=tracks
    return result


class RegionPrimitiveInventory:
    """Replay received arrays once per blob, after validate_journal()."""
    def __init__(self,journal):
        self.journal=journal;self.results={};self.inputs={};self.references={}
        for row in journal['attempts']:
            if row['status']=='returned':
                self.references.setdefault(row['measurement_ref_sha256'],[]).append(row['sequence'])

    def returned(self,reference,label,*,after_sequence=-1):
        sequences=self.references.get(reference,()) if isinstance(reference,str) else ()
        sequence=next((s for s in sequences if s>after_sequence),None)
        if sequence is None:
            raise ValueError('ROI view has an absent primitive attempt')
        row=self.journal['attempts'][sequence]
        if row['sequence']!=sequence or row['label']!=label or row['status']!='returned':
            raise ValueError('ROI view borrowed a failed/different primitive')
        ref=row['raw_result_sha256']
        if ref not in self.results:self.results[ref]=restore_numeric_return(self.journal['blobs'][ref])
        if sequence not in self.inputs:self.inputs[sequence]=primitive_inputs(row)
        return self.results[ref],self.inputs[sequence],sequence


def validate_region_primitive_views(value,inventory):
    """Join each seed/forward/backward projection to exact retained raw returns."""
    from .group_scene_region_flow import _step_summary,POLICY
    value=expand_region_tracks(value)
    size=np.asarray(value['frame_size_wh'],float)
    raw_pair=value['raw_observation_sha256'];hashes=value['native_pixel_sha256']
    first=value['native_frame_indices'][0]
    for track in value['region_tracks']:
        seeds,binding,last_sequence=inventory.returned(track.get('seed_primitive_ref'),'opencv_region_gftt')
        expected=dict(raw_pair=raw_pair,native=first,native_pixel_sha256=hashes[0],
            roi_observation_id=track['previous_body_observation_id'],
            roi_xywh=track['original_box_xywh'],policy_sha256=value['policy_sha256'])
        if binding!=expected:raise ValueError('ROI seed view changed its actual primitive input')
        if seeds is None:points=[]
        elif (isinstance(seeds,np.ndarray) and seeds.ndim==3 and seeds.shape[1:]==(1,2)
                and len(seeds)<=POLICY['maximum_features_per_region'] and np.isfinite(seeds).all()):
            points=seeds[:,0].tolist()
        else:raise ValueError('invalid raw seed return cannot support a finite ROI projection')
        if track['seed_points_xy']!=points:raise ValueError('ROI seeds differ from committed raw points')
        previous={i:np.asarray(p) for i,p in enumerate(points)}
        for offset,step in enumerate(track['steps']):
            ids=list(previous);source_points=np.asarray(list(previous.values()),np.float32).reshape(-1,1,2)
            forward,binding,sequence=inventory.returned(step.get('forward_primitive_ref'),
                'opencv_region_lk_forward',after_sequence=last_sequence)
            expected=dict(raw_pair=raw_pair,native_step_offset=offset,native=first+offset+1,
                roi_observation_id=track['previous_body_observation_id'],feature_ids=ids,
                source_points_sha256=sha256(memoryview(source_points)).hexdigest(),
                source_pixel_sha256=hashes[offset],target_pixel_sha256=hashes[offset+1],
                policy_sha256=value['policy_sha256'])
            if binding!=expected or type(forward) is not tuple or len(forward)!=3:
                raise ValueError('ROI forward view differs from its actual primitive input/return')
            target,fs,_=forward
            backward_ref=step.get('backward_primitive_ref')
            if target is not None and np.isfinite(target).all():
                backward,binding,backward_sequence=inventory.returned(backward_ref,
                    'opencv_region_lk_backward',after_sequence=sequence)
                if (binding!=dict(expected,source_points_sha256=sha256(memoryview(target)).hexdigest())
                        or type(backward) is not tuple or len(backward)!=3):
                    raise ValueError('ROI backward view changed its actual primitive input/return')
                back,bs,_=backward;last_sequence=backward_sequence
            else:
                if backward_ref is not None:raise ValueError('invalid forward output invented backward work')
                back=bs=None;last_sequence=sequence
            def finite(data,index):
                return (isinstance(data,np.ndarray) and data.shape==(len(ids),1,2)
                    and np.isfinite(data[index,0]).all())
            def succeeded(status,index):
                return bool(isinstance(status,np.ndarray) and status.shape==(len(ids),1) and status[index,0])
            attempts=[dict(feature_id=fid,source_feature_xy=previous[fid].tolist(),
                tracked_target_xy=target[i,0].tolist() if finite(target,i) else None,
                backward_feature_xy=back[i,0].tolist() if finite(back,i) else None,
                forward_lk_succeeded=bool(succeeded(fs,i) and finite(target,i)),
                backward_lk_succeeded=bool(succeeded(bs,i) and finite(back,i))) for i,fid in enumerate(ids)]
            if step.get('attempts')!=attempts:
                raise ValueError('ROI feature view differs from committed coordinate/status arrays')
            previous=_step_summary(step,previous_points=previous,size=size)
