"""Construct one member target authority for proposal and dense camera work."""
import time
from .source_first_focal_geometry import SourceFirstFocalGeometryProvider
from .source_first_model_assets import ensure_source_first_model_assets
from .direct_member_visual import _VideoDescriptorProvider


def open_source_first_member_session(sources, *, feature_cache_directory=None,
        feature_derived_cache_directory=None, require_persistent_feature_cache=False,
        progress_callback=None, body_execution_device='auto'):
    if feature_cache_directory is None:
        raise ValueError('source-first member editing requires an explicit source cache')
    assets=ensure_source_first_model_assets(feature_cache_directory)
    arguments=dict(feature_cache_directory=feature_cache_directory,
        feature_derived_cache_directory=feature_derived_cache_directory,
        require_persistent_feature_cache=require_persistent_feature_cache)
    focal=SourceFirstFocalGeometryProvider(sources,body_model_path=assets.body_model_path,
        pose_model_path=assets.pose_model_path,body_execution_device=body_execution_device,**arguments)
    resolver=focal.selector_geometry
    if progress_callback is not None:
        started=time.monotonic();last=started;queries=0
        def resolver(source,seconds,descriptor=None):
            nonlocal last,queries
            result=focal.selector_geometry(source,seconds,descriptor)
            queries+=1
            now=time.monotonic()
            if now-last>=30.:
                progress_callback(dict(elapsed_seconds=round(now-started,3),queries=queries,
                    body_frame_inference_count=focal.bodies.frame_inference_count,
                    native_face_inference_count=focal.native_faces.inference_frame_count,
                    persistent=focal._persistent.diagnostics() if focal._persistent else None))
                last=now
            return result
    try:
        visual=_VideoDescriptorProvider(sources,focal_geometry_provider=resolver,**arguments)
    except BaseException:
        focal.close()
        raise
    return focal,visual
