"""Physical-camera projection and original-role queries on the FULL renderer.

A role epoch is not a new RenderSegment, source seek or camera time origin.
These observation records may contain one output tick. They grant neither
mutable edit ownership nor an endpoint-motion/actor/quality certificate.
"""
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from bisect import bisect_right

from .group_actual_shots import visible_group_shots
from .group_joint_native_preparation import _plan_identity
from .group_joint_source_camera import GroupOutputRole
from .group_source_domain import GroupSourceDomain
from .render import RenderPlan, RenderSegment, segment_frame_counts
from .render_boundary_clock import native_frame_clocks_for_output


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return sha256(_json(value).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class GroupWholeCameraObservation:
    """One actual camera unit; canonical segment phase is never re-based."""
    first_output_frame: int
    end_output_frame: int
    eligible_source_index: int
    segments: tuple[RenderSegment, ...]
    canonical_segment_first_output_frames: tuple[int, ...]
    canonical_segment_output_frame_counts: tuple[int, ...]
    native_frame_indices: tuple[int, ...]
    source_clock_identity: str
    source_domain_json: str
    full_renderer_identity: str
    output_fps: float
    output_size: tuple[int, int]

    @classmethod
    def from_plan(cls, *, plan, domain, physical_shot):
        return PreparedGroupCameraQueries(plan,domain).camera_for(physical_shot)

    def __post_init__(self):
        if (any(type(t) is not int for t in (self.first_output_frame,self.end_output_frame,
                    self.eligible_source_index))
                or self.first_output_frame < 0 or self.end_output_frame <= self.first_output_frame
                or self.eligible_source_index < 0
                or not isinstance(self.segments,tuple) or not self.segments
                or any(not isinstance(s,RenderSegment) for s in self.segments)
                or not isinstance(self.canonical_segment_first_output_frames,tuple)
                or not isinstance(self.canonical_segment_output_frame_counts,tuple)
                or len(self.segments) != len(self.canonical_segment_first_output_frames)
                or len(self.segments) != len(self.canonical_segment_output_frame_counts)
                or any(type(t) is not int or t < 0 for t in self.canonical_segment_first_output_frames)
                or any(type(t) is not int or t < 1 for t in self.canonical_segment_output_frame_counts)
                or any(a+n != b for a,n,b in zip(self.canonical_segment_first_output_frames,
                    self.canonical_segment_output_frame_counts,self.canonical_segment_first_output_frames[1:]))
                or not self.canonical_segment_first_output_frames[0] <= self.first_output_frame
                    < self.canonical_segment_first_output_frames[0]+self.canonical_segment_output_frame_counts[0]
                or not self.canonical_segment_first_output_frames[-1] < self.end_output_frame
                    <= self.canonical_segment_first_output_frames[-1]+self.canonical_segment_output_frame_counts[-1]
                or not isinstance(self.native_frame_indices,tuple)
                or len(self.native_frame_indices) != self.end_output_frame-self.first_output_frame
                or any(type(t) is not int or t < 0 for t in self.native_frame_indices)
                or any(a > b for a,b in zip(self.native_frame_indices,self.native_frame_indices[1:]))):
            raise ValueError('whole camera query lost original physical span/canonical phase/native samples')
        if (type(self.output_fps) not in (float,int) or not 0 < self.output_fps < float('inf')
                or not isinstance(self.output_size,tuple) or len(self.output_size) != 2
                or any(type(t) is not int or t < 1 for t in self.output_size)
                or any(not isinstance(s,str) or len(s) != 64 or any(c not in '0123456789abcdef' for c in s)
                    for s in (self.source_clock_identity,self.full_renderer_identity))):
            raise ValueError('whole camera query lost complete clock/canvas identity')

    @property
    def key(self):
        return _digest(self.to_dict())

    def to_dict(self):
        return dict(contract='physical_camera_full_renderer_observation_v1',
            first_output_frame=self.first_output_frame,end_output_frame=self.end_output_frame,
            eligible_source_index=self.eligible_source_index,segments=[asdict(s) for s in self.segments],
            canonical_segment_first_output_frames=list(self.canonical_segment_first_output_frames),
            canonical_segment_output_frame_counts=list(self.canonical_segment_output_frame_counts),
            native_frame_indices=list(self.native_frame_indices),source_clock_identity=self.source_clock_identity,
            source_domain=json.loads(self.source_domain_json),full_renderer_identity=self.full_renderer_identity,
            output_fps=self.output_fps,output_size=list(self.output_size),
            mutable_edit_authority=False,quality_approved=False)

    def bind_source(self, domain):
        if not isinstance(domain,GroupSourceDomain) or _json(domain.to_dict()) != self.source_domain_json:
            raise ValueError('whole camera query changed its complete external source/clock domain')
        source = domain.request_index(self.eligible_source_index)
        clock = domain.render_frame_clocks[self.eligible_source_index]
        if (any(s.source_index != source for s in self.segments)
                or self.source_clock_identity != _digest(clock.to_dict())
                or self.native_frame_indices[-1] >= clock.frame_count):
            raise ValueError('whole camera query changed its actual source/native clock')
        return self

    def sample_at(self, tick):
        if type(tick) is not int or not self.first_output_frame <= tick < self.end_output_frame:
            raise ValueError('camera projection borrowed a tick outside its actual physical span')
        index = bisect_right(self.canonical_segment_first_output_frames,tick)-1
        return (self.segments[index],(tick-self.canonical_segment_first_output_frames[index])/self.output_fps,
            self.native_frame_indices[tick-self.first_output_frame])


@dataclass(frozen=True, slots=True)
class PreparedGroupCameraQueries:
    """One numerical inventory/schedule per plan, not per role/camera query."""
    plan: RenderPlan
    domain: GroupSourceDomain
    shots: tuple = field(init=False)
    starts: tuple = field(init=False)
    counts: tuple = field(init=False)
    samples: tuple = field(init=False)
    plan_identity: str = field(init=False)
    domain_json: str = field(init=False)

    def __post_init__(self):
        if (not isinstance(self.plan,RenderPlan) or not isinstance(self.domain,GroupSourceDomain)
                or self.plan.sources!=self.domain.request_sources
                or not 1<=self.plan.output_frame_count<=36000
                or any(t.kind.uses_overlap for t in self.plan.transitions)):
            raise ValueError('camera query requires one bounded complete hard-cut renderer/source domain')
        counts=segment_frame_counts(self.plan);starts=[0]
        for count in counts:starts.append(starts[-1]+count)
        samples=native_frame_clocks_for_output(self.plan,self.domain.clocks_by_request,
            range(self.plan.output_frame_count))
        object.__setattr__(self,'shots',visible_group_shots(self.plan,self.domain))
        object.__setattr__(self,'starts',tuple(starts))
        object.__setattr__(self,'counts',tuple(counts))
        object.__setattr__(self,'samples',tuple(samples.values()))
        object.__setattr__(self,'plan_identity',_plan_identity(self.plan))
        object.__setattr__(self,'domain_json',_json(self.domain.to_dict()))

    def camera_for(self,shot):
        if (_plan_identity(self.plan)!=self.plan_identity or _json(self.domain.to_dict())!=self.domain_json):
            raise ValueError('prepared camera queries changed their complete renderer/domain preimage')
        if shot not in self.shots:
            raise ValueError('camera query span is not one actual selected physical camera')
        first,end=shot.first_output_frame,shot.end_output_frame
        indices=tuple(i for i in range(len(self.counts)) if self.starts[i]<end and self.starts[i+1]>first)
        samples=self.samples[first:end];source=samples[0].source_index
        if any(s.source_index!=source or s.terminal_clone for s in samples):
            raise ValueError('camera query cannot borrow a different source or terminal clone')
        local=self.domain.eligible_index(source)
        return GroupWholeCameraObservation(first,end,local,tuple(self.plan.segments[i] for i in indices),
            tuple(self.starts[i] for i in indices),tuple(self.counts[i] for i in indices),
            tuple(s.native_frame_index for s in samples),_digest(self.domain.render_frame_clocks[local].to_dict()),
            self.domain_json,self.plan_identity,self.plan.fps,(self.plan.width,self.plan.height)).bind_source(self.domain)


@dataclass(frozen=True, slots=True)
class GroupSceneRoleQuery:
    """A readonly original-role query, including a one-frame static query."""
    camera: GroupWholeCameraObservation
    original_role_program_json: str
    first_output_frame: int
    end_output_frame: int
    epoch_index: int
    role: GroupOutputRole | None
    editorial_contract_sha256: str | None = None
    raw_shot_family: str | None = None
    slot: int = 1

    def __post_init__(self):
        if not isinstance(self.camera,GroupWholeCameraObservation):
            raise ValueError('role query requires a complete whole-camera projection owner')
        from .group_scene_epoch_program import validate_original_role_epochs
        program=self.original_role_program
        validate_original_role_epochs(program)
        epochs = program['epochs']
        if (type(self.epoch_index) is not int or not 0 <= self.epoch_index < len(epochs)
                or type(self.first_output_frame) is not int or type(self.end_output_frame) is not int
                or self.role is not None and not isinstance(self.role,GroupOutputRole)
                or self.slot != 1 or type(self.slot) is not int):
            raise ValueError('role query lost its original epoch address')
        epoch = epochs[self.epoch_index]
        if (epoch['role'] != (None if self.role is None else self.role.value)
                or not epoch['first_output_frame'] <= self.first_output_frame < self.end_output_frame <= epoch['end_output_frame']
                or not self.camera.first_output_frame <= self.first_output_frame < self.end_output_frame <= self.camera.end_output_frame):
            raise ValueError('role query cannot weaken or borrow original duties across epochs')
        if self.editorial_contract_sha256 is not None and (not isinstance(self.editorial_contract_sha256,str)
                or len(self.editorial_contract_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.editorial_contract_sha256)):
            raise ValueError('role query subject hypothesis hash is invalid')

    @property
    def original_role_program(self): return json.loads(self.original_role_program_json)

    @property
    def original_unit_id(self):
        program=self.original_role_program
        return _digest(dict(original_plan_identity=program['original_plan_identity'],original_role_program=program))

    def require_resolved_role(self):
        if self.role is None:
            raise ValueError('unresolved original role cannot become a donor-selected context duty')
        return self.role

    @property
    def eligible_source_index(self): return self.camera.eligible_source_index
    @property
    def source_clock_identity(self): return self.camera.source_clock_identity
    @property
    def output_fps(self): return self.camera.output_fps
    @property
    def output_size(self): return self.camera.output_size
    @property
    def segments(self): return self.camera.segments
    @property
    def native_frame_indices(self):
        return self.camera.native_frame_indices[self.first_output_frame-self.camera.first_output_frame:
            self.end_output_frame-self.camera.first_output_frame]

    def sample_at(self,tick):
        if type(tick) is not int or not self.first_output_frame <= tick < self.end_output_frame:
            raise ValueError('role query borrowed a non-epoch output tick')
        return self.camera.sample_at(tick)

    def bind_source(self,domain):
        self.camera.bind_source(domain)
        return self

    def key_preimage(self):
        return dict(slot=self.slot,first=self.first_output_frame,end=self.end_output_frame,
            source=self.eligible_source_index,native=list(self.native_frame_indices),
            role=None if self.role is None else self.role.value,
            clock=self.source_clock_identity,fps=self.output_fps,output_size=self.output_size,
            whole_camera_sha256=self.camera.key,original_unit_id=self.original_unit_id,
            original_epoch_index=self.epoch_index,editorial_contract_sha256=self.editorial_contract_sha256)

    @property
    def key(self): return _digest(self.key_preimage())

    def to_dict(self):
        return dict(contract='original_role_readonly_camera_query_v1',state_key=self.key,
            key_preimage=self.key_preimage(),raw_shot_family=self.raw_shot_family,
            camera=self.camera.to_dict(),original_role_program=self.original_role_program,
            render_topology_materialized=False,
            motion_observed=False,candidate_admitted=False,quality_approved=False)
