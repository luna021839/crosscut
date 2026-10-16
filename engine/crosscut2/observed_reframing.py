"""Bounded observed IMAGE reframing, not a camera-cause or actor certificate.

Distributed forward/backward-consistent image features support an affine
summary. Moving LED content or a large foreground person can also dominate it.
Only bounded 2D correspondence coordinates and affine scalars persist, never
pixels, feature descriptors, neural embeddings or people labels. Coordinates
let the consumer refuse a crop with no actually visible correspondence support.
UNKNOWN is None, not a stationary camera or an unusable source.

API: OpenCV calcOpticalFlowPyrLK and estimateAffinePartial2D; documented in
https://docs.opencv.org/4.10.0/dc/d6b/group__video__track.html and
https://docs.opencv.org/5.0/javadoc/org/opencv/geometry/Geometry.html .
"""
import hashlib
import json
from pathlib import Path
from time import monotonic
import cv2
import numpy as np


def estimate(first,last,*,native_cut_crossed=False):
    a,b=np.asarray(first),np.asarray(last)
    if a.dtype!=np.uint8 or b.dtype!=np.uint8 or a.shape!=b.shape or a.ndim!=3 or a.shape[2]!=3:
        raise ValueError('two same-size uint8 BGR rasters required')
    result=dict(state='UNOBSERVED',affine=None,inlier_samples=None,feature_count=0,correspondence_loss=None,
        consistent_tracks=0,inliers=0,occupied_cells=0,camera_cause_certified=False,
        actor_motion_or_identity_certified=False,neural_calls=0)
    if native_cut_crossed:return dict(result,state='NATIVE_DISCONTINUITY')
    a=cv2.resize(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),(320,180),interpolation=cv2.INTER_AREA)
    b=cv2.resize(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),(320,180),interpolation=cv2.INTER_AREA)
    points=cv2.goodFeaturesToTrack(a,maxCorners=600,qualityLevel=.01,minDistance=6,blockSize=7)
    if points is None:return result
    result['feature_count']=len(points)
    options=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01))
    following,valid,_=cv2.calcOpticalFlowPyrLK(a,b,points,None,**options)
    if following is None:return result
    returned,back,_=cv2.calcOpticalFlowPyrLK(b,a,following,None,**options)
    if returned is None:return result
    keep=valid.ravel().astype(bool)&back.ravel().astype(bool)
    keep&=(np.linalg.norm(returned-points,axis=-1).ravel()<=.75)
    keep&=np.isfinite(following).all((1,2))
    keep&=((following[:,0]>=0)&(following[:,0]<[320,180])).all(-1)
    p,q=points[keep,0],following[keep,0];result['consistent_tracks']=len(p)
    if result['feature_count']>=12:
        result['correspondence_loss']=1.-len(p)/result['feature_count']
    if len(p)<12:return result
    affine,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,
        ransacReprojThreshold=1.5,maxIters=500,confidence=.99,refineIters=10)
    if affine is None or inliers is None or not np.isfinite(affine).all():return result
    mask=inliers.ravel().astype(bool);p,q=p[mask],q[mask];result['inliers']=len(p)
    cells=np.minimum(2,np.maximum(0,(p/[320,180]*3).astype(int)))
    result['occupied_cells']=len(set(map(tuple,cells)))
    # Observation support, NOT an editorial pass gate/source ban. Logos or
    # one tiny patch cannot stand in for distributed image-space motion.
    if len(p)<12 or len(p)<.5*result['consistent_tracks'] or result['occupied_cells']<4:return result
    source=np.diag([320.,180.,1.]);matrix=np.vstack((affine,[0.,0.,1.]))
    normal=np.linalg.solve(source,matrix@source)
    residual=np.linalg.norm(q-(p@affine[:,:2].T+affine[:,2]),axis=-1)
    samples=np.column_stack((p/[320,180],q/[320,180])).astype(np.float32)
    return dict(result,state='OBSERVED_IMAGE_REFRAMING',affine=normal.tolist(),inlier_samples=samples.tolist(),
        median_fit_error_pixels=float(np.median(residual)),fit_raster=[320,180],
        point_descriptors_or_identity_persisted=False,uncalibrated_image_support_not_perceptual_quality=True)


def displayed_energy(observation,first_camera,last_camera,*,seconds):
    """Squared observed output-image displacement / actual interval seconds.

    Compose native image motion and proposed digital view transforms. Do not
    double-count a compensating digital pan as extra apparent image motion.
    This is measured only at the two rasters; no interior trajectory claim.
    """
    cameras=np.asarray([first_camera,last_camera],float)
    if cameras.shape!=(2,3) or not np.isfinite(cameras).all() or np.any(cameras[:,0]<1.-1e-9) or not np.isfinite(seconds) or seconds<=0:
        raise ValueError('two finite actual cameras and positive played duration required')
    if observation['state']!='OBSERVED_IMAGE_REFRAMING':return None
    affine=np.asarray(observation['affine'],float);samples=np.asarray(observation['inlier_samples'],float)
    if (affine.shape!=(3,3) or not np.isfinite(affine).all() or not np.allclose(affine[2],[0,0,1])
            or samples.ndim!=2 or samples.shape[1]!=4 or len(samples)>600
            or not np.isfinite(samples).all() or np.any((samples<0)|(samples>1))):
        raise ValueError('finite source-bound affine and normalized correspondence samples required')
    before=.5+cameras[0,0]*(samples[:,:2]-cameras[0,1:])
    after=.5+cameras[1,0]*(samples[:,2:]-cameras[1,1:])
    keep=((before>=0)&(before<=1)&(after>=0)&(after<=1)).all(-1)
    before,after=before[keep],after[keep]
    if len(before)<12:return None
    for points in (before,after):
        cells=np.minimum(2,(points*3).astype(int))
        if len(set(map(tuple,cells)))<4:return None
    # Only the same physical correspondences visible in BOTH actual views.
    # Full-frame motion moments cannot certify an arbitrary new crop.
    delta=(after-before)*[16/9,1.]
    return float(np.mean(np.sum(delta**2,axis=-1)))/seconds


def displayed_energy_many(observation,first_cameras,last_cameras,*,seconds):
    """Same correspondence/crop support for a batch of actual view pairs.

    Returns energy AND a support mask. Zero in an unsupported slot is only
    storage, never evidence of a stationary/safe source. No actor/flow label.
    """
    first,last=np.asarray(first_cameras,float),np.asarray(last_cameras,float)
    if (first.ndim!=2 or first.shape[1]!=3 or last.shape!=first.shape
            or not np.isfinite(first).all() or not np.isfinite(last).all()
            or np.any(first[:,0]<1.-1e-9) or np.any(last[:,0]<1.-1e-9)
            or not np.isfinite(seconds) or seconds<=0):
        raise ValueError('finite batch actual camera pairs and duration required')
    energy=np.zeros(len(first));supported=np.zeros(len(first),bool)
    if observation['state']!='OBSERVED_IMAGE_REFRAMING':return energy,supported
    # Validate the bound observation with the existing scalar consumer first.
    displayed_energy(observation,[1.,.5,.5],[1.,.5,.5],seconds=seconds)
    points=np.asarray(observation['inlier_samples'],float)
    before=.5+first[:,None,:1]*(points[None,:,:2]-first[:,None,1:])
    after=.5+last[:,None,:1]*(points[None,:,2:]-last[:,None,1:])
    keep=((before>=0)&(before<=1)&(after>=0)&(after<=1)).all(-1)
    count=keep.sum(-1);supported=count>=12
    for values in (before,after):
        cells=np.minimum(2,np.maximum(0,(values*3).astype(int)))
        codes=cells[...,0]*3+cells[...,1]
        occupied=((codes[...,None]==np.arange(9))&keep[...,None]).any(1).sum(-1)
        supported&=occupied>=4
    square=np.sum(((after-before)*[16/9,1.])**2,axis=-1)
    energy=np.where(supported,(square*keep).sum(-1)/np.maximum(1,count)/seconds,0.)
    return energy,supported


def measure(path,pairs,*,source_sha256,source_identity,native_cuts,cache,budget,check=lambda:None):
    """Explicit source-native pairs, one SHARED caller-owned raster/deadline cap.

    A complete warm bank returns before packet probing or raster decoding.
    Caller has already attested source_sha256 in its acquisition/analysis pass.
    A missing pair never authorizes whole-source sequential decoding.
    """
    from .grid_perception import file_identity
    from . import sparse_native_frames as sparse
    path=Path(path);cache=Path(cache);domain=np.asarray(pairs)
    if domain.ndim!=2 or domain.shape[1]!=2 or domain.dtype.kind not in 'iu':
        raise ValueError('exact integer source-native pairs required')
    pairs=[tuple(map(int,p)) for p in domain]
    if len(pairs)>24 or len(set(pairs))!=len(pairs) or any(p[0]<0 or p[1]<=p[0] for p in pairs):
        raise ValueError('at most24 distinct increasing exact native pairs required')
    if len(source_sha256)!=64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise ValueError('attested source SHA256 required')
    if file_identity(path)!=tuple(source_identity):raise OSError('source changed before reframing observation')
    producer=hashlib.sha256(Path(__file__).read_bytes()+cv2.__version__.encode()+
        Path(sparse.__file__).read_bytes()).hexdigest()
    boundaries=np.asarray(native_cuts,int);records=[];decodes=[];clock=None;started=monotonic()
    cache.mkdir(parents=True,exist_ok=True)
    for first,last in pairs:
        check();budget.live()
        if file_identity(path)!=tuple(source_identity):raise OSError('source changed during reframing cache recovery')
        binding=dict(source_sha256=source_sha256,source_identity=list(source_identity),
            native=[first,last],native_cut_crossed=bool(np.any((boundaries>first)&(boundaries<=last))),
            producer=producer,raster=[960,540],opencv=cv2.__version__)
        target=cache/(hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()+'.json')
        if target.is_file():
            record=json.loads(target.read_text())
            if record['binding']!=binding:raise ValueError('cached reframing binding changed')
            records.append(dict(record,reused=True));continue
        if binding['native_cut_crossed']:
            observation=dict(state='NATIVE_DISCONTINUITY',affine=None,inlier_samples=None,
                camera_cause_certified=False,neural_calls=0);raster_hashes=[]
        else:
            try:
                budget.reserve('raster',2);budget.arrays(2*960*540*3)
                if clock is None:clock=sparse.probe(path,check=budget.live)
                receipt={};pixels=dict(sparse.frames(path,[first,last],clock=clock,check=budget.live,
                    receipt=receipt,maximum_seconds=min(20.,max(1.,budget.seconds-(monotonic()-budget.started)))))
                if any(pixels[t].shape!=(540,960,3) or pixels[t].dtype!=np.uint8 for t in (first,last)):
                    raise ValueError('actual reframing rasters differ from the bound canonical dimensions')
                observation=estimate(pixels[first],pixels[last]);budget.live()
                raster_hashes=[hashlib.sha256(pixels[t].tobytes()).hexdigest() for t in (first,last)]
                decodes.append(receipt)
            except (TimeoutError,sparse.UnsupportedSeek):
                records.append(dict(binding=binding,observation=dict(state='DEFERRED',affine=None,inlier_samples=None,
                    camera_cause_certified=False,neural_calls=0),reused=False));continue
        check()
        if file_identity(path)!=tuple(source_identity):raise OSError('source changed during reframing observation')
        record=dict(binding=binding,observation=observation,raster_sha256=raster_hashes)
        with target.open('x') as output:output.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
        records.append(dict(record,reused=False))
    if file_identity(path)!=tuple(source_identity):raise OSError('source changed before reframing result')
    return records,dict(new_neural_calls=0,actual_native_pairs=len(pairs),reused_pairs=sum(r['reused'] for r in records),
        returned_source_rasters=2*len(decodes),decode_receipts=decodes,elapsed_seconds=monotonic()-started,
        camera_motion_or_normal_speed_quality_certified=False)
