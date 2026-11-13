"""Counted native face appearance for review associations, not member identity.

Persisted detector boxes are explicitly persisted inputs, never new detection
authority. The actual source raster is decoded and matched to its recorded
pixel hash before the pinned local SFace model sees it. No downloads, geometry
substitutes, named member gallery or professional-approval threshold is used.
"""
import hashlib
from pathlib import Path

import cv2

from .face_identity import SFACE_SHA256, SFaceEmbedder, _validate_normalized_face
from .native_face_descriptor import SFACE_MODEL_NAME, native_descriptor_space

CONTRACT = 'recorded_head_actual_native_sface_review_appearance_v1'


def register_review_sface(session, model_path):
    path=Path(model_path)
    before=session._identity(path)
    port=getattr(session.budget,'request_work_phase',None)
    if port is not None:port.charge('model_hash_bytes',before[2])
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=SFACE_SHA256 or session._identity(path)!=before:
        raise ValueError('review face appearance needs the pinned unchanged local SFace asset')
    session.budget.check()
    if port is not None:port.charge('model_loads',1,model=SFACE_MODEL_NAME)
    embedder=SFaceEmbedder(path)
    if session._identity(path)!=before:
        raise ValueError('review SFace asset changed while loading')
    session.register_model(SFACE_MODEL_NAME,path,embedder.embedding,
        expected_asset_sha256=SFACE_SHA256,expected_file_identity=before,
        adapter_fingerprint=dict(contract=CONTRACT,opencv_version=cv2.__version__,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            embedding_implementation_sha256=hashlib.sha256(Path(__file__).with_name('face_identity.py').read_bytes()).hexdigest(),
            space='pinned_sface_128d_l2',camera_independent=True,
            boxes_clamped=False,appearance_is_semantic_identity=False))
    return native_descriptor_space(session,SFACE_MODEL_NAME)


def observe_recorded_head_appearances(session,raw):
    """Bounded actual native embeddings, keeping unobservable heads UNKNOWN."""
    if session.closed:raise ValueError('review appearance needs a live native owner')
    source,native=raw['request_source_index'],raw['native_frame_index']
    local=session.domain.eligible_index(source)
    clock=session.domain.render_frame_clocks[local]
    from .group_performer_family_owner import digest
    if raw['source_clock_identity']!=digest(clock.to_dict()):
        raise ValueError('recorded appearance uses another source clock')
    if list(session._assert_source_identity(source))!=list(raw['source_file_identity']):
        raise ValueError('recorded appearance source changed')
    pixels=session.native_pixels(source,native)
    if pixels is None or hashlib.sha256(memoryview(pixels)).hexdigest()!=raw['native_pixel_sha256']:
        raise ValueError('recorded appearance exact native raster changed or unavailable')
    if len(raw['heads'])>24:raise ValueError('review appearance head count is unbounded')
    space=native_descriptor_space(session,SFACE_MODEL_NAME)
    rows=[]
    for head in raw['heads']:
        box,points=head['box_xywh'],head['landmarks_xy']
        try:_validate_normalized_face(box,points)
        except ValueError:
            rows.append(dict(observation_id=head['observation_id'],embedding=None,
                reason='recorded_geometry_outside_native_descriptor_input'));continue
        vector=session.observe_native(source,native,model=SFACE_MODEL_NAME,
            arguments=dict(bbox_xywh=box,landmarks_xy=points))
        rows.append(dict(observation_id=head['observation_id'],embedding=vector,
            reason='actual_native_sface' if vector is not None else 'native_embedding_unavailable'))
    session._assert_source_identity(source);session._assert_model_assets(SFACE_MODEL_NAME)
    return dict(contract=CONTRACT,request_source_index=source,native_frame_index=native,
        source_clock_identity=raw['source_clock_identity'],source_file_identity=raw['source_file_identity'],
        native_pixel_sha256=raw['native_pixel_sha256'],recorded_native_observation_sha256=digest(raw),
        descriptor_space=space.to_dict(),heads=rows,actual_native_raster_redecoded=True,
        detector_input_is_recorded=True,head_detection_reexecuted=False,
        named_member_identity_certified=False,persistent_actor_identity_certified=False,
        motion_certified=False,candidate_admitted=False,quality_approved=False)
