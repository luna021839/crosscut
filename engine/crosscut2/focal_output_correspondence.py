"""Match a rendered detection to the planned physical focal, or say UNKNOWN.

Geometry correspondence is not member identity. A background-only detection
must not turn a missed foreground face into a fictitious scale jump.
"""
import math
import numpy as np


def match(faces,expected):
    receipt=dict(policy='planned_physical_focal_geometry_correspondence_v1',
        expected_focal=expected,candidates=list(faces),selected_output_face_index=None,
        matched=False,ownership_certified=False,named_identity_certified=False,
        maximum_center_error_face_heights=.75,maximum_height_ratio=1.8,
        correspondence_gate_is_not_editorial_quality_threshold=True)
    if expected.get('height',0.)<=.019 or expected.get('confidence',0.)<.75:
        return None,dict(receipt,reason='planned_focal_unmeasured_or_conflicting')
    if not faces:return None,dict(receipt,reason='no_output_face_detection')
    center=np.asarray([f['center'] for f in faces]);height=np.asarray([f['height'] for f in faces])
    wanted=np.asarray(expected['center']);h=float(expected['height'])
    distance=np.linalg.norm((center-wanted)*[16/9,1.],axis=-1)/np.maximum(.005,(height+h)/2.)
    size=np.abs(np.log(np.maximum(.001,height)/h))
    allowed=(distance<=.75)&(size<=math.log(1.8))
    score=distance+size;order=np.argsort(np.where(allowed,score,np.inf),kind='stable')
    selected=int(order[0]);valid=np.flatnonzero(allowed)
    ambiguity=len(valid)>1 and score[order[1]]-score[selected]<=.1
    receipt.update(candidate_center_error_face_heights=distance.tolist(),candidate_scale_log_error=size.tolist(),
        correspondence_ambiguous=bool(ambiguity))
    if not len(valid) or ambiguity:
        return None,dict(receipt,reason='ambiguous_correspondence' if ambiguity else 'detected_faces_do_not_match_planned_focal')
    return selected,dict(receipt,matched=True,selected_output_face_index=selected,reason='geometric_correspondence_only')


def predictions_by_tick(seams,frame_count):
    candidates={}
    for seam in seams:
        predicted=seam.get('expected_output_focal')
        if predicted is None:continue
        if len(predicted)!=4:raise ValueError('four played focal predictions required')
        for offset,value in zip((-7,-1,0,6),predicted):
            tick=max(0,min(frame_count-1,seam['output_frame']+offset))
            candidates.setdefault(tick,[]).append(value)
    result={}
    for tick,values in candidates.items():
        exact=[v for v in values if v.get('face_native_sample_is_actual_endpoint',False)]
        chosen=(exact or values)[0]
        conflict=False
        for other in (exact or values)[1:]:
            scale=max(.005,(chosen['height']+other['height'])/2.)
            distance=np.linalg.norm((np.asarray(chosen['center'])-other['center'])*[16/9,1.])/scale
            conflict|=distance>.5 or abs(math.log(max(.001,chosen['height'])/max(.001,other['height'])))>math.log(1.5)
        result[tick]=dict(chosen,prediction_conflict=bool(conflict),contributing_prediction_count=len(values))
        if conflict:result[tick].update(height=0.,confidence=0.)
    return result
