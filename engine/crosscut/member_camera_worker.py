"""Private immutable inputs for optional member-camera native work.

Only source-bound frozen authority crosses the anonymous pipe. No live decoder,
model, callable or raw descriptor cache is serialized or written to disk.
"""
from dataclasses import dataclass
from pathlib import Path
from .optional_native_worker import isolated_optional


def _identity(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def validate_member_camera_clock(proposal, baseline):
    """Optional camera work may transform pixels, not silently reselect a path."""
    from .render import RenderPlan
    if not isinstance(proposal, (tuple, list)) or len(proposal) != 5:
        raise ValueError('member camera proposal must contain the complete bundle')
    plan = proposal[0]
    if (not isinstance(plan, RenderPlan)
            or any(getattr(plan, key) != getattr(baseline, key) for key in (
                'sources', 'reference_source', 'master_audio_start', 'fps',
                'aspect_ratio', 'width', 'height', 'output', 'member_pose_dissolve_requested'))
            or len(plan.segments) != len(baseline.segments)):
        raise ValueError('member camera changed the selected render authority')
    for before, after in zip(baseline.segments, plan.segments):
        if any(getattr(before, key) != getattr(after, key) for key in (
                'source_index', 'source_start', 'master_start', 'duration', 'effective_source_duration')):
            raise ValueError('member camera changed the selected source/time path')


@dataclass(frozen=True)
class MemberCameraProviderSpec:
    sources: tuple
    source_identities: tuple
    body_model: Path
    pose_model: Path
    model_identities: tuple
    body_execution_device: str
    cache_arguments: tuple
    face_banks: tuple
    body_references: tuple
    observation_namespace: str

    @classmethod
    def from_provider(cls, provider, cache_arguments):
        from .source_first_focal_geometry import SourceFirstFocalGeometryProvider
        if not isinstance(provider, SourceFirstFocalGeometryProvider):
            raise TypeError('member camera requires source-first authority')
        return cls(provider.sources, provider._identities,
            Path(provider.encoder.model_path), Path(provider.pose.model_path),
            (_identity(provider.encoder.model_path), _identity(provider.pose.model_path)),
            provider.body_execution_device, tuple(sorted(cache_arguments.items())),
            tuple(sorted(provider.faces._frozen_focal_banks.items())),
            tuple((index, tuple(values)) for index, values in sorted(provider.bodies._references.items())),
            provider.observation_namespace)

    def validate_files(self):
        if (tuple(_identity(path) for path in self.sources) != self.source_identities
                or tuple(_identity(path) for path in (self.body_model, self.pose_model)) != self.model_identities):
            raise ValueError('member camera source/model identity changed')

    def open(self):
        from .source_first_focal_geometry import SourceFirstFocalGeometryProvider
        from .source_first_unresolved_memo import install_source_first_unresolved_memo
        self.validate_files()
        provider = SourceFirstFocalGeometryProvider(self.sources, body_model_path=self.body_model,
            pose_model_path=self.pose_model, body_execution_device=self.body_execution_device,
            **dict(self.cache_arguments))
        try:
            if provider.observation_namespace != self.observation_namespace:
                raise ValueError('member camera observation authority changed')
            provider.faces._frozen_focal_banks = dict(self.face_banks)
            for index, bank in self.face_banks:
                provider.faces._focal_references[index] = [entry.candidate for entry in bank.references] if bank else []
            provider.bodies._references = dict(self.body_references)
            install_source_first_unresolved_memo(provider)
            return provider
        except BaseException:
            provider.close()
            raise


@isolated_optional()
def propose_member_camera_in_worker(*, provider_spec, arguments):
    from .pipeline import _build_member_camera_with_provider
    provider = provider_spec.open()
    try:
        result = _build_member_camera_with_provider(dense_face_provider=provider, **arguments)
        provider_spec.validate_files()
        return result
    finally:
        provider.close()
