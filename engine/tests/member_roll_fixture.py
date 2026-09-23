"""Synthetic observed screen joints, not a real member or model measurement."""
from dataclasses import dataclass, replace
import math
import numpy as np

from crosscut.direct_member_editorial_motion import DirectMemberFaceSample
from crosscut.direct_member_target_face import FOCAL_FACE_BASIS
from crosscut.focal_selector_geometry import FocalSelectorGeometry
from crosscut.joint_dissolve_pose import OverlapPoseObservation
from crosscut.joint_dissolve_registration import OverlapFaceObservation
from crosscut.pose import FormationPersonDescriptor

SIZE = (180, 320)
FACE = (.43, .19, .14, .12)
BODY = (.25, .15, .50, .70)


def rotate(points, degrees, size=SIZE):
    """Independent pixel-coordinate construction for the synthetic input."""
    radians = math.radians(degrees)
    matrix = np.array(((math.cos(radians), -math.sin(radians)),
                       (math.sin(radians), math.cos(radians))))
    return ((np.asarray(points) - .5) * size) @ matrix.T / size + .5


def points(source, seconds, degrees=2.4, size=SIZE):
    value = np.array(((.39,.38),(.61,.38),(.34,.48),(.66,.48),
                      (.30,.56),(.70,.56),(.43,.70),(.57,.70)))
    value[:, 0] += .002 * math.sin(seconds)
    return rotate(value, degrees if source % 2 else 0., size)


def box(source, value, degrees=2.4):
    x,y,w,h = value
    corners = rotate(((x,y),(x+w,y),(x+w,y+h),(x,y+h)), degrees if source % 2 else 0.)
    low, high = corners.min(axis=0), corners.max(axis=0)
    return tuple(float(x) for x in (*low, *(high-low)))


def sample(source, _seconds):
    return DirectMemberFaceSample(box(source,FACE),1,'yunet',box(source,BODY),.95,
        'source_local_focal_body_track',FOCAL_FACE_BASIS)


def focal(source, seconds):
    face, body = box(source,FACE), box(source,BODY)
    screen = tuple((float(x),float(y),.95) for x,y in points(source, seconds))
    person = FormationPersonDescriptor(body,(body[0]+body[2]/2,body[1]+body[3]/2),.95,screen)
    return FocalSelectorGeometry(source,seconds,face,person=person,
        pose_backend='rtmpose_body7',pose_screen_xyc=screen)


def windows(n=7, degrees=2.4, size=SIZE):
    faces, poses = [], []
    for source in (0,1):
        fs, ps = [], []
        for i in range(n):
            master, local = 2+i/30, 5+3*source+i/30
            fs.append(OverlapFaceObservation(source,master,local,box(source,FACE,degrees)))
            ps.append(OverlapPoseObservation(source,master,local,
                tuple((float(x),float(y),.95) for x,y in points(source,master,degrees,size))))
        faces.append(tuple(fs)); poses.append(tuple(ps))
    return tuple(faces), tuple(poses)


def direct_inputs():
    from types import SimpleNamespace
    from crosscut.direct_member_transform import direct_member_source_frame_geometry
    from crosscut.direct_member_visual import DirectMemberVisualConfig
    from crosscut.render import RenderSegment, RenderTransition
    from tests.test_direct_member_editorial_motion import DirectMemberEditorialMotionTests
    selection=DirectMemberEditorialMotionTests._selection((0,1),(box(0,FACE),box(1,FACE)))
    selection.source_frame_geometries=(direct_member_source_frame_geometry((360,640),output_size=SIZE),)*2
    selection.span_validity=tuple(SimpleNamespace(sources=tuple(SimpleNamespace(eligible_source_index=s,
        native_scene_cut_source_seconds=()) for s in range(2))) for _ in range(2))
    selection.config=DirectMemberVisualConfig()
    selection.selected_boundary_master_seconds=(2.,)
    for boundary in selection.boundaries:
        boundary.mismatch.cinematic_bridge=None
    return selection,(RenderSegment(2,3.,2.,0.),RenderSegment(0,6.,2.,2.)),(RenderTransition(),)


class SyntheticFocalProvider:
    sample = staticmethod(sample)
    selector_geometry = staticmethod(focal)

    def paired_samples(self,a,ta,b,tb):
        self.captures.append((a,b,None,None))
        return self.sample(a,ta),self.sample(b,tb)
    def begin_focal_boundary_reference_capture(self):self.captures=[]
    def focal_boundary_reference_capture(self):return tuple(self.captures)
    def recover_exact_focal_pair_from_registered_references(self,*args):return None
    def close(self):pass


@dataclass(frozen=True)
class SyntheticProviderSpec:
    """Only model/observation ownership is synthetic; IPC executes production code."""
    sources: tuple
    identities: tuple

    def open(self):
        self.validate_files()
        return SyntheticFocalProvider()

    def validate_files(self):
        from crosscut.member_camera_worker import _identity
        if tuple(_identity(p) for p in self.sources)!=self.identities:
            raise ValueError('synthetic source identity changed')


class IdentityLook:
    def apply(self,segments):
        from crosscut.render import SourceLookAdjustment
        return tuple(replace(s,look_adjustment=SourceLookAdjustment()) for s in segments)
    def to_dict(self):return {'fixture':'identity look; color solving is outside rotation test'}


def worker_proposal(sources,output,dissolve=False,segment_duration=1.):
    """Actual worker entry and pipeline camera construction in a separate PID."""
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch
    import os
    from crosscut.direct_member_transform import direct_member_source_frame_geometry
    from crosscut.direct_member_visual import DirectMemberVisualConfig
    from crosscut.ive import AspectRatio
    from crosscut.member_camera_worker import propose_member_camera_in_worker,_identity
    from crosscut.render import RenderSegment,RenderPlan,RenderTransition,TransitionKind,DIRECT_MEMBER_POSE_DISSOLVE_REASON
    from tests.test_direct_member_editorial_motion import DirectMemberEditorialMotionTests
    sources=tuple(Path(p) for p in sources)
    path=tuple(range(len(sources)))
    selection=DirectMemberEditorialMotionTests._selection(path,tuple(box(i,FACE) for i in path))
    selection.source_frame_geometries=tuple(direct_member_source_frame_geometry((360,640),output_size=SIZE) for _ in path)
    entries=tuple(SimpleNamespace(eligible_source_index=i,native_scene_cut_source_seconds=()) for i in path)
    selection.span_validity=tuple(SimpleNamespace(sources=entries) for _ in path)
    selection.config=DirectMemberVisualConfig()
    selection.selected_boundary_master_seconds=tuple(i*segment_duration for i in path[1:])
    for boundary in selection.boundaries:boundary.mismatch.cinematic_bridge=None
    segments=tuple(RenderSegment(i,i*segment_duration,segment_duration,i*segment_duration) for i in path)
    hard=tuple(RenderTransition() for _ in path[1:])
    effects=tuple(RenderTransition(TransitionKind.FACE_DISSOLVE,7,DIRECT_MEMBER_POSE_DISSOLVE_REASON)
                  if dissolve and i%2==0 else RenderTransition() for i in range(len(hard)))
    primitive=RenderPlan(sources,segments,0,AspectRatio.PORTRAIT,Path(output),width=SIZE[0],height=SIZE[1],fps=30.,
        transitions=hard,member_pose_dissolve_requested=dissolve,member_pose_dissolve_effective=False,member_pose_dissolve_count=0)
    spec=SyntheticProviderSpec(sources,tuple(_identity(p) for p in sources))
    arguments=dict(direct_visual_selection=selection,segments=segments,member_transitions=effects,
        member_common_start=0.,eligible_indices=path,source_timewarps=(None,)*len(path),offsets=(0.,)*len(path),
        direct_full_frame_ratios=(9/16,)*len(path),primitive_plan=primitive)
    # The real camera planner, registration, containment, attachment and parent
    # validation run; only unrelated color measurement uses a synthetic result.
    with patch('crosscut.member_look_preflight.build_member_look_proxy_proposal',return_value=IdentityLook()):
        proposal=propose_member_camera_in_worker(provider_spec=spec,arguments=arguments)
    return os.getpid(),primitive,proposal
