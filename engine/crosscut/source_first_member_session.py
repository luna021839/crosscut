"""Construct one member target authority for proposal and dense camera work."""
import time
from .source_first_focal_geometry import SourceFirstFocalGeometryProvider
from .source_first_model_assets import ensure_source_first_model_assets
from .source_first_unresolved_memo import install_source_first_unresolved_memo
from .source_first_observation_batch import SourceFirstObservationBatch
from .source_first_anchor_prefilter import install_source_first_anchor_prefilter
from .direct_member_visual import _VideoDescriptorProvider


def _decoder_locality_diagnostics(focal):
    """Report the actual reader state, not an inferred code-version promise."""
    def describe(index, reader, sampler=None):
        if reader is None:
            return dict(source_index=index, initialized=False)
        return dict(source_index=index, initialized=True, enabled=reader.enabled,
                    seek_count=reader.seek_count, read_count=reader.read_count,
                    pixel_cache_hit_count=reader.pixel_cache_hit_count,
                    fallback_count=reader.fallback_count,
                    legacy_direct_read_count=getattr(sampler, 'dense_sequential_frame_decode_count', 0))
    return dict(
        working=[describe(i, getattr(sampler, '_seek_locality', None), sampler)
                 for i, sampler in enumerate(focal.faces._samplers)],
        native=[describe(i, reader) for i, reader in sorted(focal.native_faces._readers.items())],
        shared=[dict(source_index=i, **owner.diagnostics())
                for i, owner in enumerate(getattr(focal, 'shared_source_frames', None) or ())],
    )


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
    try:
        install_source_first_unresolved_memo(focal)
        install_source_first_anchor_prefilter(focal)
    except BaseException:
        focal.close()
        raise
    resolver=focal.selector_geometry
    if progress_callback is not None:
        started=time.monotonic();last=started;queries=0
        def resolver(source,seconds,descriptor=None):
            nonlocal last,queries
            result=focal.selector_geometry(source,seconds,descriptor)
            queries+=1
            now=time.monotonic()
            if now-last>=30.:
                progress_callback(dict(phase='focal_geometry',elapsed_seconds=round(now-started,3),queries=queries,
                    body_frame_inference_count=focal.bodies.frame_inference_count,
                    native_face_inference_count=focal.native_faces.inference_frame_count,
                    unresolved_query_memo=focal._source_first_unresolved_memo.diagnostics(),
                    anchor_prefilter=focal._source_first_anchor_prefilter.diagnostics(),
                    decoder_locality=_decoder_locality_diagnostics(focal),
                    persistent=focal._persistent.diagnostics() if focal._persistent else None))
                last=now
            return result
    try:
        visual=_VideoDescriptorProvider(sources,focal_geometry_provider=resolver,
            progress_callback=progress_callback,
            shared_source_frames=getattr(focal, 'shared_source_frames', None),**arguments)
        visual.prepare_focal_observations = SourceFirstObservationBatch(focal,
            progress_callback=progress_callback)
        visual.member_observation_progress = progress_callback
        from .member_cached_observations import attach_cached_member_observations
        attach_cached_member_observations(focal, visual)
    except BaseException:
        focal.close()
        raise
    return focal,visual
