"""Bounded missing-input admission, not an editorial or identity selector.

Complete shared original context first, then coherent predeclared programmes.
Cache coverage may determine which programme can be measured within a lease;
pose, face similarity and camera scores may not determine this input schedule.
An unavailable observation remains unavailable and cannot invite a retry.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

COMPLETION_RESOURCES=('new_decodes','model_calls','pose_evidence_bytes')


@dataclass(frozen=True, slots=True)
class NativeMotionCompletionCost:
    """Conservative work bounds, not a reservation or a successful observation."""
    new_decodes: int
    model_calls: int
    pose_evidence_bytes: int

    def __post_init__(self):
        if any(type(getattr(self,k)) is not int or not 0<=getattr(self,k)<=2**40
                for k in COMPLETION_RESOURCES):
            raise ValueError('native completion cost needs finite nonnegative resource bounds')

    def resources(self):return {k:getattr(self,k) for k in COMPLETION_RESOURCES}


def _physical_key(key):
    return (isinstance(key, tuple) and len(key) == 3
        and all(type(n) is int and n >= 0 for n in key[:2])
        and (key[2] is None or type(key[2]) in (int, float) and isfinite(key[2])))


@dataclass(frozen=True, slots=True)
class NativeMotionCompletionPackage:
    identity: str
    native_keys: tuple
    # The ORIGINAL dependency union, frozen before any input cap, is never
    # intersected down to the inputs that happened to enter the inventory.
    # Omitted addresses remain explicit unrequested obligations, not success.
    unrequested_native_keys: tuple = ()

    def __post_init__(self):
        if (not isinstance(self.identity, str) or not 1 <= len(self.identity) <= 128
                or not isinstance(self.native_keys, tuple) or not 1 <= len(self.native_keys) <= 1024
                or any(not _physical_key(k) for k in self.native_keys)
                or len(set(self.native_keys)) != len(self.native_keys)
                or not isinstance(self.unrequested_native_keys,tuple)
                or any(not _physical_key(k) for k in self.unrequested_native_keys)
                or len(set(self.unrequested_native_keys))!=len(self.unrequested_native_keys)
                or not set(self.unrequested_native_keys)<=set(self.native_keys)):
            raise ValueError('native completion package needs a finite unique physical inventory')


def validate_completion_packages(inventory, required_native_keys, packages,critical_required_native_keys=()):
    keys = {s.demand.native_key for s in inventory.supplies}
    if (not isinstance(required_native_keys, tuple) or len(required_native_keys) > 16384
            or any(not _physical_key(k) for k in required_native_keys)
            or len(set(required_native_keys)) != len(required_native_keys)
            or not set(required_native_keys) <= keys
            or not isinstance(packages, tuple) or len(packages) > 64
            or any(not isinstance(p, NativeMotionCompletionPackage)
                or set(p.native_keys)-keys!=set(p.unrequested_native_keys)
                for p in packages)
            or len({p.identity for p in packages}) != len(packages)):
        raise ValueError('native completion obligations differ from the frozen request inventory')
    if (not isinstance(critical_required_native_keys,tuple)
            or len(set(critical_required_native_keys))!=len(critical_required_native_keys)
            or not set(critical_required_native_keys)<=set(required_native_keys)):
        raise ValueError('critical native dependencies differ from the frozen request inventory')


def coherent_completion_schedule(inventory, *, required_native_keys, packages,
        maximum_observations, maximum_raw_observations,resource_costs,remaining_resources,
        critical_required_native_keys=(),excluded_native_keys=(),active_package_identity=None):
    """Admit complete feasible packages, or ONE explicit partial package.

    A small remaining budget must not buy isolated frames from every donor and
    leave every programme unmeasurable. Required contexts retain caller order;
    package ordering uses missing inputs, conservative resource vectors and
    frozen source families, never actor similarity or editorial scores.
    Admission does NOT atomically reserve parent capacity. Every actual charge
    still needs its live parent ACK; execution receipts must report cutoffs.
    """
    validate_completion_packages(inventory, required_native_keys, packages,critical_required_native_keys)
    if (active_package_identity is not None and (not isinstance(active_package_identity,str)
            or active_package_identity not in {p.identity for p in packages})):
        raise ValueError('active completion package differs from the frozen request inventory')
    keys={s.demand.native_key for s in inventory.supplies}
    if (not isinstance(excluded_native_keys,tuple)
            or any(not _physical_key(k) for k in excluded_native_keys)
            or len(set(excluded_native_keys))!=len(excluded_native_keys)
            or not set(excluded_native_keys)<=keys):
        raise ValueError('previous chunk inputs differ from the frozen request inventory')
    excluded=set(excluded_native_keys)
    aliases = {}; physical = {}
    for supply in inventory.supplies:
        physical[supply.demand.native_key] = supply
        if supply.missing_modalities and supply.demand.native_key not in excluded:
            aliases.setdefault(supply.demand.native_key, []).append(supply.demand)
    # Exclusion means already admitted/attempted or live-preflight skipped,
    # NEVER measured success. Failed partial observations cannot invite a retry
    # or make a package's actual completion receipt appear successful.
    unresolved_excluded=tuple(k for k in excluded_native_keys if physical[k].missing_modalities
        or physical[k].state!='NATIVE_ROWS_AVAILABLE')
    if (not isinstance(resource_costs,dict) or set(resource_costs)!=set(aliases)
            or any(not isinstance(c,NativeMotionCompletionCost) for c in resource_costs.values())
            or not isinstance(remaining_resources,dict) or set(remaining_resources)!=set(COMPLETION_RESOURCES)
            or any(type(n) is not int or not 0<=n<=2**40 for n in remaining_resources.values())
            or any(type(n) is not int or not 0<=n<=1200
                for n in (maximum_observations,maximum_raw_observations))):
        raise ValueError('native completion admission needs all physical resource costs and live capacity')
    spent={k:0 for k in COMPLETION_RESOURCES}
    def totals(keys):
        return {r:sum(getattr(resource_costs[k],r) for k in keys) for r in COMPLETION_RESOURCES}
    def fits_resources(cost):
        return all(cost[r]<=remaining_resources[r]-spent[r] for r in COMPLETION_RESOURCES)
    pending = dict(aliases); scheduled = []; raw_count = 0; family_counts = {}
    terminal=tuple(dict(native_key=k,state=physical[k].state) for k in critical_required_native_keys
        if physical[k].state in ('CONFLICTING_NATIVE_RECORDS','CONFLICTING_FEATURE_RECORDS','EXACT_DECODE_UNAVAILABLE'))
    if terminal:
        # Only explicitly declared fixed entry/return dependencies stop the
        # programme. An original interior UNKNOWN is not a universal veto.
        return (),dict(policy='shared_context_then_resource_feasible_declared_programmes_v2',
            basis='cache_modalities_resource_vectors_and_frozen_families_not_pose_face_or_camera_cost',
            required_native_keys=required_native_keys,unadmitted_required_native_keys=tuple(
                k for k in required_native_keys if k in pending),terminal_required_dependencies=terminal,
            scheduled_unique_facts=0,scheduled_new_raw_facts=0,selected_packages=[],unretryable_packages=[],
            initial_live_resource_capacity=remaining_resources,admitted_resource_upper_bounds=spent,
            excluded_native_keys=excluded_native_keys,unresolved_excluded_native_keys=unresolved_excluded,
            resource_admission_is_not_atomic_parent_reservation=True,
            successful_observations_or_connected_actors_certified=False,quality_approved=False)

    def admit(key):
        nonlocal raw_count
        if key not in pending:
            return True
        is_raw = physical[key].observation is None
        cost=resource_costs[key].resources()
        if (len(scheduled) >= maximum_observations or is_raw and raw_count >= maximum_raw_observations
                or not fits_resources(cost)):
            return False
        rows = pending.pop(key)
        demand = min(rows, key=lambda d: (d.output_tick, d.source_family, d.opportunity))
        scheduled.append(demand); raw_count += is_raw
        for r in COMPLETION_RESOURCES:spent[r]+=cost[r]
        for lane in {d.source_family for d in rows}:
            family_counts[lane] = family_counts.get(lane, 0) + 1
        return True

    for key in required_native_keys:
        admit(key)
    required_left = tuple(k for k in required_native_keys if k in pending)
    remaining = list(packages); selected = []; blocked = []
    while remaining and not required_left and len(scheduled) < maximum_observations:
        available = []
        for package in remaining:
            failed = tuple(k for k in package.native_keys if k in physical and physical[k].state in
                ('CONFLICTING_NATIVE_RECORDS', 'CONFLICTING_FEATURE_RECORDS', 'EXACT_DECODE_UNAVAILABLE'))
            if failed:
                blocked.append(dict(identity=package.identity,unretryable_native_keys=failed))
                continue
            missing = tuple(k for k in package.native_keys if k in pending)
            raw = sum(physical[k].observation is None for k in missing)
            families = {physical[k].demand.source_family for k in package.native_keys if k in physical}
            available.append((package, missing, raw, families,totals(missing)))
        remaining = [row[0] for row in available]
        if not available:
            break
        fits = [row for row in available if not row[0].unrequested_native_keys
            and len(row[1]) <= maximum_observations-len(scheduled)
            and row[2] <= maximum_raw_observations-raw_count and fits_resources(row[4])]
        # A memory-bounded chunk is not a renewed programme choice. Once an
        # immutable dependency union is active, finish its requestable inputs
        # before a newly cheaper/feasible donor can steal the next chunk.
        active=next((row for row in available if row[0].identity==active_package_identity
            and row[1]),None)
        pool = (active,) if active is not None else fits or available
        package, missing, raw, families,cost = min(pool, key=lambda row: (
            bool(row[0].unrequested_native_keys),len(row[1]),
            sum(family_counts.get(f, 0) for f in row[3]), row[0].identity))
        remaining.remove(package)
        for key in missing:
            admit(key)
        requestable_left=tuple(k for k in package.native_keys if k in pending)
        left=tuple(k for k in package.native_keys if k in pending or k in package.unrequested_native_keys)
        selected.append(dict(identity=package.identity,admitted_missing_native_keys=tuple(
            k for k in missing if k not in pending),unadmitted_missing_native_keys=left,
            original_required_native_keys=package.native_keys,
            original_required_native_input_count=len(package.native_keys),
            unrequested_native_keys=package.unrequested_native_keys,
            unadmitted_requestable_native_keys=requestable_left,
            entire_missing_package_resource_upper_bounds=cost,
            admitted_missing_input_resource_upper_bounds=totals(tuple(k for k in missing if k not in pending)),
            entire_missing_input_package_admitted=not left))
        if requestable_left:
            break  # No scattered partial work on a second programme.
    receipt = dict(policy='shared_context_then_resource_feasible_declared_programmes_v2',
        basis='cache_modalities_resource_vectors_and_frozen_families_not_pose_face_or_camera_cost',
        required_native_keys=required_native_keys,unadmitted_required_native_keys=required_left,
        terminal_required_dependencies=terminal,
        scheduled_unique_facts=len(scheduled),scheduled_new_raw_facts=raw_count,
        initial_live_resource_capacity=remaining_resources,admitted_resource_upper_bounds=spent,
        excluded_native_keys=excluded_native_keys,unresolved_excluded_native_keys=unresolved_excluded,
        resource_admission_is_not_atomic_parent_reservation=True,
        selected_packages=selected,unretryable_packages=blocked,
        active_package_identity=next((r['identity'] for r in selected
            if r['unadmitted_requestable_native_keys']),None),
        continuing_package_is_not_reselected_after_chunk_settlement=True,
        successful_observations_or_connected_actors_certified=False,quality_approved=False)
    return tuple(scheduled), receipt


def balanced_completion_schedule(inventory, *, maximum_observations, maximum_raw_observations,
        resource_costs, remaining_resources, excluded_native_keys=(), lane_counts=None):
    """One finite time/source-balanced chunk, not a renewed work allocation."""
    # Reuse validation and the physical resource contract without using package
    # ordering. The caller retains lane counts across chunks in this pass.
    _,receipt=coherent_completion_schedule(inventory,required_native_keys=(),packages=(),
        maximum_observations=maximum_observations,maximum_raw_observations=maximum_raw_observations,
        resource_costs=resource_costs,remaining_resources=remaining_resources,
        excluded_native_keys=excluded_native_keys)
    counts={} if lane_counts is None else lane_counts
    if (not isinstance(counts,dict) or any(not isinstance(k,tuple) or len(k)!=2
            or any(not isinstance(v,str) or not v for v in k)
            or type(n) is not int or n<0 for k,n in counts.items())):
        raise ValueError('completion time bands need retained source/opportunity counts')
    excluded=set(excluded_native_keys);pending={};physical={}
    for supply in inventory.supplies:
        key=supply.demand.native_key;physical[key]=supply
        if supply.missing_modalities and key not in excluded:
            pending.setdefault(key,[]).append(supply.demand)
    limit=min(maximum_observations,len(pending));scheduled=[];raw_count=0
    capacity=dict(remaining_resources)
    first=min(s.demand.output_tick for s in inventory.supplies)
    end=max(s.demand.output_tick for s in inventory.supplies)+1
    bits=max(1,(limit-1).bit_length())
    bands=sorted(range(limit),key=lambda n:int(format(n,'0'+str(bits)+'b')[::-1],2))
    for band in bands:
        eligible={key:rows for key,rows in pending.items()
            if (physical[key].observation is not None or raw_count<maximum_raw_observations)
            and all(getattr(resource_costs[key],r)<=capacity[r] for r in COMPLETION_RESOURCES)}
        if not eligible:break
        target=first+(band+.5)*(end-first)/max(1,limit)
        def priority(d):
            return (counts.get((d.source_family,d.opportunity),0),abs(d.output_tick-target),
                d.source_family,d.opportunity,d.source_index,d.native_frame_index,
                -1000. if d.analysis_roll is None else d.analysis_roll)
        key,demand=min(((key,min(rows,key=priority)) for key,rows in eligible.items()),
            key=lambda row:priority(row[1]))
        aliases=pending.pop(key);scheduled.append(demand)
        raw_count+=physical[key].observation is None
        for resource in COMPLETION_RESOURCES:capacity[resource]-=getattr(resource_costs[key],resource)
        for lane in {(d.source_family,d.opportunity) for d in aliases}:
            counts[lane]=counts.get(lane,0)+1
    receipt.update(policy='source_opportunity_resource_feasible_time_band_balanced_v3',
        scheduled_unique_facts=len(scheduled),scheduled_new_raw_facts=raw_count,
        admitted_resource_upper_bounds={r:remaining_resources[r]-capacity[r] for r in COMPLETION_RESOURCES})
    return tuple(scheduled),receipt
