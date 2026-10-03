"""Bounded local GROUP graph over complete source/clock/camera/role states.

This kernel has no raw-family gate and no prerequisite single-shot success.
Its owning media worker must construct projected observations and complete
dense verification. A numeric path alone does not select a production bundle,
approve source identity/phase or certify a professional edit.
"""
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict,dataclass
from enum import Enum
import hashlib
from itertools import product
import json
import math
from time import monotonic

from .group_source_domain import GroupSourceDomain
from .render import RenderPlan,RenderSegment,segment_frame_counts,_can_share_source_clock
from .render_boundary_clock import native_frame_clocks_for_output


class GroupOutputRole(str,Enum):
    FOCAL_HEAD='focal_head'
    FOCAL_HEAD_ARMS='focal_head_and_observed_arms'
    FORMATION='required_formation'
    COMPOSITION='observed_body_composition'


@dataclass(frozen=True,slots=True)
class GroupSourceCameraState:
    slot: int
    first_output_frame: int
    end_output_frame: int
    eligible_source_index: int
    segments: tuple[RenderSegment,...]
    native_frame_indices: tuple[int,...]
    role: GroupOutputRole
    source_clock_identity: str
    output_fps: float = 30.
    output_size: tuple[int,int] = (1920,1080)
    segment_output_frame_counts: tuple[int,...] | None = None
    raw_shot_family: str | None = None
    editorial_contract_sha256: str | None = None

    def __post_init__(self):
        if (type(self.slot) is not int or not 0<=self.slot<=3
                or type(self.first_output_frame) is not int or self.first_output_frame<0
                or type(self.end_output_frame) is not int or self.end_output_frame-self.first_output_frame<2
                or type(self.eligible_source_index) is not int or self.eligible_source_index<0
                or not isinstance(self.segments,tuple) or not self.segments
                or any(not isinstance(s,RenderSegment) for s in self.segments)
                or not isinstance(self.native_frame_indices,tuple)
                or len(self.native_frame_indices)!=self.end_output_frame-self.first_output_frame
                or any(type(n) is not int or n<0 for n in self.native_frame_indices)
                or any(a>b for a,b in zip(self.native_frame_indices,self.native_frame_indices[1:]))
                or not isinstance(self.role,GroupOutputRole)
                or not isinstance(self.source_clock_identity,str) or not self.source_clock_identity
                or type(self.output_fps) not in (int,float) or not math.isfinite(self.output_fps) or self.output_fps<=0
                or not isinstance(self.output_size,tuple) or len(self.output_size)!=2
                or any(type(n) is not int or n<1 for n in self.output_size)
                or abs(sum(s.duration for s in self.segments)-
                    (self.end_output_frame-self.first_output_frame)/self.output_fps)>1/self.output_fps
                or self.editorial_contract_sha256 is not None and (not isinstance(self.editorial_contract_sha256,str)
                    or len(self.editorial_contract_sha256)!=64
                    or any(c not in '0123456789abcdef' for c in self.editorial_contract_sha256))):
            raise ValueError('joint GROUP state needs one complete shot, role and exact native/output clock')
        counts=self.segment_output_frame_counts
        if counts is None and len(self.segments)==1:
            counts=(self.end_output_frame-self.first_output_frame,)
            object.__setattr__(self,'segment_output_frame_counts',counts)
        if (not isinstance(counts,tuple) or len(counts)!=len(self.segments)
                or any(type(n) is not int or n<2 for n in counts)
                or sum(counts)!=self.end_output_frame-self.first_output_frame):
            raise ValueError('joint whole camera needs exact per-cell output frame counts')

    def key_preimage(self):
        # Whole trajectories and native clocks participate, not merely source
        # index/endpoints. Raw camera labels are diagnostics, NOT authority.
        payload=dict(slot=self.slot,first=self.first_output_frame,end=self.end_output_frame,
            source=self.eligible_source_index,segments=[asdict(s) for s in self.segments],
            native=list(self.native_frame_indices),role=self.role.value,clock=self.source_clock_identity,
            fps=self.output_fps,output_size=self.output_size,cell_frame_counts=self.segment_output_frame_counts)
        if self.editorial_contract_sha256 is not None:
            payload['editorial_contract_sha256']=self.editorial_contract_sha256
        return payload

    @property
    def key(self):
        return hashlib.sha256(json.dumps(self.key_preimage(),sort_keys=True,separators=(',',':'),
            allow_nan=False).encode()).hexdigest()

    def to_dict(self):
        return dict(contract='complete_joint_source_camera_state_hash_preimage_v1',
            key_preimage=self.key_preimage(),state_key=self.key,raw_shot_family=self.raw_shot_family,
            named_identity_or_motion_certified=False,quality_approved=False)

    def bind_source(self,domain):
        if not isinstance(domain,GroupSourceDomain):raise TypeError('joint state requires a GROUP source domain')
        source=domain.request_index(self.eligible_source_index)
        clock=domain.render_frame_clocks[self.eligible_source_index]
        identity=hashlib.sha256(json.dumps(clock.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (any(s.source_index!=source for s in self.segments) or identity!=self.source_clock_identity
                or self.native_frame_indices[-1]>=clock.frame_count):
            raise ValueError('joint state source file/index/native clock domain differs')
        if any(self.native_frame_indices[0]<math.ceil(t*float(clock.frame_rate)-1e-9)<=self.native_frame_indices[-1]
                for t in domain.scene_cut_times[self.eligible_source_index]):
            raise ValueError('joint state crosses an observed actual native camera cut')
        return self


def joint_state_from_render_plan(*,plan,domain,slot,first_output_frame,end_output_frame,
        role,raw_shot_family=None,editorial_contract_sha256=None):
    """Bind a complete cell-aligned actual shot using the FULL renderer clock.

    An arbitrary shorter subshot inside a cell cannot expand to the whole cell
    and silently erase its broadcaster cut. Such windows need explicit owned
    cell splitting first. Same-source coalescing is sampled on the renderer's
    global unit clock, not a newly rounded per-state source seek.
    """
    if (not isinstance(plan,RenderPlan) or not isinstance(domain,GroupSourceDomain)
            or plan.sources!=domain.request_sources
            or type(first_output_frame) is not int or type(end_output_frame) is not int
            or not 0<=first_output_frame<end_output_frame<=plan.output_frame_count
            or any(t.kind.uses_overlap for t in plan.transitions)):
        raise ValueError('joint state requires complete hard-cut renderer/source-clock authority')
    counts=segment_frame_counts(plan);ticks=[0]
    for count in counts:ticks.append(ticks[-1]+count)
    if first_output_frame not in ticks or end_output_frame not in ticks:
        raise ValueError('actual shot ends inside a canonical cell; owned splitting required')
    first,last=ticks.index(first_output_frame),ticks.index(end_output_frame)-1
    if not _can_share_source_clock(plan,first_index=first,candidate_last_index=last):
        raise ValueError('joint state is not one continuous actual source clock')
    segments=plan.segments[first:last+1]
    source=segments[0].source_index;local=domain.eligible_index(source)
    clock=domain.render_frame_clocks[local]
    samples=native_frame_clocks_for_output(plan,domain.clocks_by_request,range(first_output_frame,end_output_frame))
    if any(s.source_index!=source or s.terminal_clone for s in samples.values()):
        raise ValueError('joint state cannot borrow other-source or terminal-cloned frames')
    identity=hashlib.sha256(json.dumps(clock.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return GroupSourceCameraState(slot,first_output_frame,end_output_frame,local,segments,
        tuple(s.native_frame_index for s in samples.values()),role,identity,plan.fps,
        (plan.width,plan.height),counts[first:last+1],raw_shot_family,
        editorial_contract_sha256).bind_source(domain)


@dataclass(frozen=True,slots=True)
class JointGroupEdgeObservation:
    left_state_key: str
    right_state_key: str
    first_incoming_output_frame: int
    framing_cost: float
    required_visibility_observed: bool
    motion_preservation_observed: bool
    correspondence_ambiguous: bool = False
    analysis_basis: str = 'projected_complete_state_pixels_not_raw_family'

    def __post_init__(self):
        if (type(self.first_incoming_output_frame) is not int or self.first_incoming_output_frame<0
                or type(self.framing_cost) not in (int,float) or not math.isfinite(self.framing_cost)
                or not 0<=self.framing_cost<=1
                or any(type(v) is not bool for v in (self.required_visibility_observed,
                    self.motion_preservation_observed,self.correspondence_ambiguous))
                or self.analysis_basis!='projected_complete_state_pixels_not_raw_family'):
            raise ValueError('joint edge requires measured projected geometry and explicit unknownness')

    @property
    def feasible(self):
        return self.required_visibility_observed and self.motion_preservation_observed and not self.correspondence_ambiguous


@dataclass(frozen=True,slots=True)
class JointGroupDenseVerification:
    state_keys: tuple[str,str,str,str]
    verified_output_frames: tuple[tuple[int,...],tuple[int,...]]
    complete: bool
    reason: str

    def validates(self,path):
        return (self.complete is True and self.state_keys==tuple(s.key for s in path)
            and self.verified_output_frames==tuple(tuple(range(s.first_output_frame,s.end_output_frame))
                for s in path[1:3]) and isinstance(self.reason,str) and bool(self.reason))


@dataclass(frozen=True,slots=True)
class SingleGroupDenseVerification:
    """All ticks of ONE whole mutable shot, not two endpoint/cell checks."""
    state_keys: tuple[str,str,str]
    verified_output_frames: tuple[int,...]
    complete: bool
    reason: str

    def validates(self,path):
        return (len(path)==3 and self.complete is True
            and self.state_keys==tuple(s.key for s in path)
            and self.verified_output_frames==tuple(range(path[1].first_output_frame,path[1].end_output_frame))
            and isinstance(self.reason,str) and bool(self.reason))


class JointGroupBudgetExhausted(ValueError):
    pass


class JointGroupWorkBudget:
    """Shared native/time caps; explicitly bounded scene state/edge frontiers.

    Default callers still own at most one 24-state graph. A normal scene
    request can explicitly own at most three such graphs, without multiplying
    the decoder/model/dense/evidence limits or refreshing its absolute clock.
    Enlarged state caps require a declared active frontier, not a silent reset.
    """
    def __init__(self,*,absolute_deadline,maximum_states=24,maximum_edges=512,
            maximum_dense_paths=4,maximum_new_decodes=1200,maximum_model_calls=2400,
            maximum_pose_comparisons=4096,
            maximum_region_flow_feature_updates=65536,
            maximum_pose_evidence_bytes=8*1024*1024,
            maximum_association_nodes=1536, maximum_association_edges=18000,
            maximum_projected_cache_bytes=64*1024*1024,maximum_frontiers=1,clock=monotonic):
        values=dict(states=maximum_states,edges=maximum_edges,dense_paths=maximum_dense_paths,
            new_decodes=maximum_new_decodes,model_calls=maximum_model_calls,
            pose_comparisons=maximum_pose_comparisons,pose_evidence_bytes=maximum_pose_evidence_bytes,
            region_flow_feature_updates=maximum_region_flow_feature_updates,
            association_nodes=maximum_association_nodes,association_edges=maximum_association_edges)
        if (type(absolute_deadline) not in (float,int) or not math.isfinite(absolute_deadline)
                or any(type(v) is not int or v<1 for v in (*values.values(),maximum_projected_cache_bytes))
                or type(maximum_frontiers) is not int or not 1<=maximum_frontiers<=3
                or maximum_states>24*maximum_frontiers or maximum_edges>512 or maximum_dense_paths>4
                or maximum_pose_comparisons>4096 or maximum_pose_evidence_bytes>8*1024*1024
                or maximum_region_flow_feature_updates>131072
                or maximum_association_nodes>1536 or maximum_association_edges>18000):
            raise ValueError('joint GROUP requires explicit bounded state/edge/dense/decode/model/time budgets')
        self.clock=clock;self.deadline=absolute_deadline;self.limits=values
        self.counts={k:0 for k in values};self.model_calls={}
        self.projected_cache_bytes=maximum_projected_cache_bytes
        self.maximum_frontiers=maximum_frontiers;self.frontiers=[];self.active_frontier=None
        self.reserved={k:0 for k in values};self._primitive_reservations={}
        from .group_joint_primitive_journal import LocalPrimitiveJournal
        self.primitive_journal=LocalPrimitiveJournal(self)

    def check(self):
        if self.clock()>=self.deadline:raise JointGroupBudgetExhausted('joint GROUP absolute work deadline reached')

    def spend(self,kind,count=1,*,model=None):
        self.check()
        if kind not in self.limits or type(count) is not int or count<0:
            raise ValueError('unknown or malformed joint GROUP work dimension')
        if kind=='model_calls' and (not isinstance(model,str) or not model):
            raise ValueError('model work must name its measured model')
        if kind in ('states','edges'):
            if self.maximum_frontiers>1 and self.active_frontier is None:
                raise ValueError('multi-frontier state/edge work requires an explicit active frontier')
            if self.active_frontier is not None and count>self.frontier_remaining(kind):
                raise JointGroupBudgetExhausted('joint GROUP frontier '+kind+' work budget reached')
        if self.counts[kind]+self.reserved[kind]+count>self.limits[kind]:
            raise JointGroupBudgetExhausted('joint GROUP '+kind+' work budget reached')
        self.counts[kind]+=count
        if kind=='model_calls':
            self.model_calls[model]=self.model_calls.get(model,0)+count

    def require_primitive_capacity(self,work,maximum_evidence_bytes):
        self.check()
        if (not isinstance(work,dict) or not work or 'pose_evidence_bytes' in work
                or type(maximum_evidence_bytes) is not int or maximum_evidence_bytes<1
                or any(k not in self.limits or k in ('states','edges','model_calls')
                    or type(n) is not int or n<0 for k,n in work.items())):
            raise ValueError('local numeric primitive needs explicit bounded work/evidence dimensions')
        amounts=dict(work,pose_evidence_bytes=maximum_evidence_bytes)
        for kind,count in amounts.items():
            if self.counts[kind]+self.reserved[kind]+count>self.limits[kind]:
                raise JointGroupBudgetExhausted('joint GROUP primitive '+kind+' reservation unavailable')
        return amounts

    def reserve_primitive(self,work,maximum_evidence_bytes):
        amounts=self.require_primitive_capacity(work,maximum_evidence_bytes)
        ticket=object()
        for kind,count in amounts.items():self.reserved[kind]+=count
        self._primitive_reservations[ticket]=dict(amounts=amounts,started=False)
        return ticket

    def _owned_primitive_ticket(self,ticket):
        try:owned=self._primitive_reservations.get(ticket)
        except TypeError:owned=None
        if owned is None:
            raise ValueError('primitive reservation is not owned by this live budget')
        return owned

    def start_primitive(self,ticket):
        owned=self._owned_primitive_ticket(ticket);self.check()
        if owned['started']:raise ValueError('primitive reservation was already started')
        owned['started']=True
        for kind,count in owned['amounts'].items():
            if kind=='pose_evidence_bytes':continue
            self.reserved[kind]-=count;self.counts[kind]+=count

    def settle_primitive(self,ticket,evidence_bytes):
        owned=self._owned_primitive_ticket(ticket)
        if type(evidence_bytes) is not int or not 0<=evidence_bytes<=owned['amounts']['pose_evidence_bytes']:
            raise ValueError('primitive returned more evidence than its owned reservation')
        # Completed raw/terminal storage is reserved already. It must not be
        # discarded because its computation used the last available tick.
        for kind,count in owned['amounts'].items():
            if kind=='pose_evidence_bytes' or not owned['started']:self.reserved[kind]-=count
        self.counts['pose_evidence_bytes']+=evidence_bytes
        del self._primitive_reservations[ticket]

    def measure_primitive(self,**kwargs):
        return self.primitive_journal.measure(**kwargs)

    def begin_frontier(self,*,label,maximum_states,maximum_edges):
        """Declare the next finite graph; keep ALL request work/time counters.

        Only state/edge accounting changes scope. Fixed/donor pixels, model
        observations, dense checks and proof work keep their shared counters.
        The previous frontier is closed with its actual spent amounts.
        """
        self.check()
        if (type(label) not in (int,str) or not str(label)
                or type(maximum_states) is not int or not 1<=maximum_states<=24
                or type(maximum_edges) is not int or not 1<=maximum_edges<=512):
            raise ValueError('joint GROUP frontier requires explicit bounded label/state/edge limits')
        if len(self.frontiers)>=self.maximum_frontiers:
            raise JointGroupBudgetExhausted('joint GROUP frontier count reached')
        if self.active_frontier is not None:
            row=self.frontiers[self.active_frontier]
            row['spent']={k:self.counts[k]-row['start'][k] for k in ('states','edges')}
        self.active_frontier=len(self.frontiers)
        self.frontiers.append(dict(label=label,limits=dict(states=maximum_states,edges=maximum_edges),
            start={k:self.counts[k] for k in ('states','edges')}))

    def frontier_remaining(self,kind):
        if kind not in ('states','edges'):
            raise ValueError('only state/edge work has frontier-local allocation')
        remaining=self.limits[kind]-self.counts[kind]
        if self.active_frontier is None:return remaining
        row=self.frontiers[self.active_frontier]
        return min(remaining,row['limits'][kind]-(self.counts[kind]-row['start'][kind]))

    def to_dict(self):
        frontiers=[dict(label=row['label'],limits=dict(row['limits']),start=dict(row['start']),
            spent=dict(row.get('spent',{k:self.counts[k]-row['start'][k] for k in ('states','edges')})))
            for row in self.frontiers]
        return dict(limits=dict(self.limits),counts=dict(self.counts),reserved=dict(self.reserved),model_calls=dict(self.model_calls),
            absolute_deadline=self.deadline,maximum_projected_cache_bytes=self.projected_cache_bytes,
            maximum_frontiers=self.maximum_frontiers,state_edge_frontiers=frontiers,
            native_and_time_caps_not_multiplied=True)


class JointProjectedPixelCache:
    """Byte-bounded transient projected pixels; never overwrites raw caches."""
    def __init__(self,budget):
        self.budget=budget;self.rows=OrderedDict();self.bytes=0

    def get_or_observe(self,*,state,output_tick,raw_identity,observer:Callable):
        self.budget.check()
        if type(output_tick) is not int or not state.first_output_frame<=output_tick<state.end_output_frame:
            raise ValueError('projected observation is outside its complete shot')
        key=(state.key,output_tick,raw_identity)
        if key in self.rows:
            value=self.rows.pop(key);self.rows[key]=value;return value
        value=observer()
        self.budget.check()
        size=0 if value is None else getattr(value,'nbytes',None)
        if type(size) is not int or size<0:raise ValueError('projected cache requires pixels with measured byte size')
        if value is not None:
            value=value.copy();value.setflags(write=False)
        if size<=self.budget.projected_cache_bytes:
            self.rows[key]=value;self.bytes+=size
            while self.bytes>self.budget.projected_cache_bytes or len(self.rows)>1200:
                _,old=self.rows.popitem(last=False);self.bytes-=0 if old is None else old.nbytes
        return value


@dataclass(frozen=True)
class JointGroupSelection:
    path: tuple | None
    edges: tuple
    attempts: tuple
    status: str
    budget: dict
    baseline_edge_costs: tuple = ()

    def to_dict(self):
        return dict(contract='bounded_two_actual_shot_source_clock_camera_role_graph_v1',status=self.status,
            selected_state_keys=[s.key for s in self.path] if self.path else None,
            attempts=list(self.attempts),budget=self.budget,raw_family_reachability_gate=False,
            prior_solo_success_required=False,complete_middle_camera_shared_by_both_edges=True,
            baseline_edge_costs=list(self.baseline_edge_costs),
            per_edge_nonregression_checked=bool(self.baseline_edge_costs),
            production_selected=False,quality_approved=False)


@dataclass(frozen=True)
class SingleGroupSelection(JointGroupSelection):
    def to_dict(self):
        result=super().to_dict()
        result['contract']='bounded_one_actual_shot_source_clock_camera_role_graph_v1'
        result['mutable_actual_shot_count']=1
        result['actual_boundary_count']=2
        return result


def select_joint_group_source_camera_path(*,domain,left,right,states_by_slot,declared_roles,
        observe_edge,verify_dense,budget,editorial_contract=None,baseline_path=None,
        baseline_unknown_policy="retain"):
    """Search A -> B -> C -> D without first solving B or C on its own.

    Every edge owns the SAME complete state and shared exact boundary tick.
    None is UNKNOWN, never zero loss. The role is fixed before observation;
    failed matches cannot be relabelled intentional. Callback media work must
    debit the SAME budget and own identity-bound native readers/models.
    """
    from .group_joint_editorial_contract import GroupJointEditorialContract
    return _select_group_source_camera_path(domain=domain,left=left,right=right,
        states_by_slot=states_by_slot,declared_roles=declared_roles,observe_edge=observe_edge,
        verify_dense=verify_dense,budget=budget,editorial_contract=editorial_contract,
        baseline_path=baseline_path,baseline_unknown_policy=baseline_unknown_policy,state_count=4,editorial_type=GroupJointEditorialContract,
        dense_type=JointGroupDenseVerification,selection_type=JointGroupSelection)


def select_single_group_source_camera_path(*,domain,left,right,states,declared_roles,
        observe_edge,verify_dense,budget,editorial_contract=None,baseline_path=None,
        baseline_unknown_policy="retain"):
    """Fixed L -> the SAME complete X -> fixed R, across any internal cells.

    The two ends cannot choose different cameras, subjects or source clocks.
    This numeric sibling is not the normal app owner or raw-evidence admission.
    """
    from .group_joint_editorial_contract import GroupSingleShotEditorialContract
    return _select_group_source_camera_path(domain=domain,left=left,right=right,
        states_by_slot=(states,),declared_roles=declared_roles,observe_edge=observe_edge,
        verify_dense=verify_dense,budget=budget,editorial_contract=editorial_contract,
        baseline_path=baseline_path,baseline_unknown_policy=baseline_unknown_policy,state_count=3,editorial_type=GroupSingleShotEditorialContract,
        dense_type=SingleGroupDenseVerification,selection_type=SingleGroupSelection)


def _select_group_source_camera_path(*,domain,left,right,states_by_slot,declared_roles,
        observe_edge,verify_dense,budget,editorial_contract,baseline_path,baseline_unknown_policy,state_count,
        editorial_type,dense_type,selection_type):
    if (not isinstance(budget,JointGroupWorkBudget) or not isinstance(states_by_slot,tuple)
            or len(states_by_slot)!=state_count-2 or not isinstance(declared_roles,tuple) or len(declared_roles)!=state_count
            or any(not isinstance(role,GroupOutputRole) for role in declared_roles)
            or not isinstance(left,GroupSourceCameraState) or left.slot!=0
            or not isinstance(right,GroupSourceCameraState) or right.slot!=state_count-1):
        raise ValueError('joint GROUP requires the declared actual-shot slots and fixed roles/neighbors')
    if baseline_unknown_policy not in ('retain', 'observe_candidate'):
        raise ValueError('unknown baseline observation policy')
    # The opt-in policy permits a fully observed replacement of UNKNOWN. It
    # does NOT turn missing measurements into zero, or waive ANY known cost.
    # Domain owners must additionally preserve each independently known motion
    # component: a scalar framing score cannot represent that obligation.
    attempts=[];edges={};seen=set();states=[];baseline_costs=();baseline_unknown=()
    if baseline_path is not None:
        if (not isinstance(baseline_path,tuple) or len(baseline_path)!=state_count
                or any(not isinstance(s,GroupSourceCameraState) or s.slot!=i
                    or s.role!=declared_roles[i] for i,s in enumerate(baseline_path))
                or baseline_path[0].key!=left.key or baseline_path[-1].key!=right.key
                or any(a.end_output_frame!=b.first_output_frame for a,b in zip(baseline_path,baseline_path[1:]))):
            raise ValueError('joint comparison requires the same actual-shot incumbent and fixed outer states')
        for state in baseline_path:state.bind_source(domain)
    if editorial_contract is not None:
        if not isinstance(editorial_contract,editorial_type):
            raise ValueError('joint search needs its pre-search typed editorial declaration')
        editorial_contract.validate_states((left,*(s for values in states_by_slot for s in values),right))
        if baseline_path is not None:editorial_contract.validate_states(baseline_path)
    try:
        for slot,values in enumerate(((left,),*states_by_slot,(right,))):
            if not isinstance(values,tuple):raise ValueError('joint states must be immutable bounded tuples')
            unique=[]
            for state in values:
                budget.check()
                if (not isinstance(state,GroupSourceCameraState) or state.slot!=slot
                        or state.role!=declared_roles[slot] or state.output_fps!=left.output_fps
                        or state.output_size!=left.output_size):
                    raise ValueError('joint source/camera state changed its predeclared role or slot')
                state.bind_source(domain)
                if state.key in seen:continue
                seen.add(state.key);budget.spend('states');unique.append(state)
            states.append(tuple(unique))
        def edge(a,b,*,raw=False):
            if a.end_output_frame!=b.first_output_frame:return None
            key=(a.key,b.key)
            if key not in edges:
                budget.spend('edges');value=observe_edge(a,b,budget)
                budget.check()
                if value is not None and (not isinstance(value,JointGroupEdgeObservation)
                        or (value.left_state_key,value.right_state_key,value.first_incoming_output_frame)!=
                        (a.key,b.key,b.first_output_frame)):
                    raise ValueError('joint edge observation belongs to another camera or native/output clock')
                edges[key]=value
            value=edges[key]
            return value if raw or value is not None and value.feasible else None
        if baseline_path is not None:
            baseline_edges=tuple(edge(a,b,raw=True) for a,b in zip(baseline_path,baseline_path[1:]))
            baseline_unknown=tuple(i for i,value in enumerate(baseline_edges)
                if value is None or not value.feasible)
            if baseline_unknown and baseline_unknown_policy == 'retain':
                return selection_type(None,(),({'complete':False,'reason':'baseline_edge_unknown'},),
                    'baseline_edge_unknown_retained',budget.to_dict())
            baseline_costs=tuple(None if value is None else value.framing_cost for value in baseline_edges)
            if baseline_unknown:
                attempts.append(dict(complete=False,reason='baseline_unknown_preserved_not_imputed',
                    unknown_edge_slots=list(baseline_unknown),baseline_edge_costs=list(baseline_costs)))
        paths=[]
        for mutable in product(*states[1:-1]):
            budget.check();path=(left,*mutable,right)
            observed=[]
            for a,b in zip(path,path[1:]):
                value=edge(a,b)
                if value is None:break
                observed.append(value)
            if len(observed)!=state_count-1:continue
            observed=tuple(observed)
            if baseline_costs:
                costs=tuple(e.framing_cost for e in observed)
                regressed=tuple(i for i,(old,new) in enumerate(zip(baseline_costs,costs)) if old is not None and new>old+1e-9)
                if regressed or not (baseline_unknown or any(
                        old is not None and new<old-1e-6 for old,new in zip(baseline_costs,costs))):
                    attempts.append(dict(state_keys=[s.key for s in path],complete=False,
                        reason='protected_edge_framing_regression' if regressed else 'no_observed_edge_gain',
                        regressed_edge_slots=list(regressed),baseline_edge_costs=list(baseline_costs),
                        proposed_edge_costs=list(costs)))
                    continue
            paths.append((sum(e.framing_cost for e in observed),tuple(s.key for s in path),path,observed))
        for _,_,path,observed in sorted(paths,key=lambda p:p[:2]):
            budget.spend('dense_paths');verification=verify_dense(path,observed,budget);budget.check()
            if not isinstance(verification,dense_type):
                raise ValueError('joint GROUP dense verification must return complete typed evidence')
            complete=verification.validates(path)
            attempts.append(dict(state_keys=[s.key for s in path],complete=complete,reason=verification.reason))
            if complete:return selection_type(path,observed,tuple(attempts),'complete_geometric_candidate',budget.to_dict(),baseline_costs)
    except JointGroupBudgetExhausted as exc:
        attempts.append(dict(complete=False,reason=str(exc)))
        return selection_type(None,(),tuple(attempts),'budget_retained_baseline',budget.to_dict(),baseline_costs)
    return selection_type(None,(),tuple(attempts),'no_complete_geometric_candidate',budget.to_dict(),baseline_costs)
