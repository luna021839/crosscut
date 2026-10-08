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

    def __post_init__(self):
        if (not isinstance(self.identity, str) or not 1 <= len(self.identity) <= 128
                or not isinstance(self.native_keys, tuple) or not 1 <= len(self.native_keys) <= 1024
                or any(not _physical_key(k) for k in self.native_keys)
                or len(set(self.native_keys)) != len(self.native_keys)):
            raise ValueError('native completion package needs a finite unique physical inventory')


def validate_completion_packages(inventory, required_native_keys, packages,critical_required_native_keys=()):
    keys = {s.demand.native_key for s in inventory.supplies}
    if (not isinstance(required_native_keys, tuple) or len(required_native_keys) > 16384
            or any(not _physical_key(k) for k in required_native_keys)
            or len(set(required_native_keys)) != len(required_native_keys)
            or not set(required_native_keys) <= keys
            or not isinstance(packages, tuple) or len(packages) > 64
            or any(not isinstance(p, NativeMotionCompletionPackage) or not set(p.native_keys) <= keys
                for p in packages)
            or len({p.identity for p in packages}) != len(packages)):
        raise ValueError('native completion obligations differ from the frozen request inventory')
    if (not isinstance(critical_required_native_keys,tuple)
            or len(set(critical_required_native_keys))!=len(critical_required_native_keys)
            or not set(critical_required_native_keys)<=set(required_native_keys)):
        raise ValueError('critical native dependencies differ from the frozen request inventory')


def coherent_completion_schedule(inventory, *, required_native_keys, packages,
        maximum_observations, maximum_raw_observations,resource_costs,remaining_resources,
        critical_required_native_keys=()):
    """Admit complete feasible packages, or ONE explicit partial package.

    A small remaining budget must not buy isolated frames from every donor and
    leave every programme unmeasurable. Required contexts retain caller order;
    package ordering uses missing inputs, conservative resource vectors and
    frozen source families, never actor similarity or editorial scores.
    Admission does NOT atomically reserve parent capacity. Every actual charge
    still needs its live parent ACK; execution receipts must report cutoffs.
    """
    validate_completion_packages(inventory, required_native_keys, packages,critical_required_native_keys)
    aliases = {}; physical = {}
    for supply in inventory.supplies:
        physical[supply.demand.native_key] = supply
        if supply.missing_modalities:
            aliases.setdefault(supply.demand.native_key, []).append(supply.demand)
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
            failed = tuple(k for k in package.native_keys if physical[k].state in
                ('CONFLICTING_NATIVE_RECORDS', 'CONFLICTING_FEATURE_RECORDS', 'EXACT_DECODE_UNAVAILABLE'))
            if failed:
                blocked.append(dict(identity=package.identity,unretryable_native_keys=failed))
                continue
            missing = tuple(k for k in package.native_keys if k in pending)
            raw = sum(physical[k].observation is None for k in missing)
            families = {physical[k].demand.source_family for k in package.native_keys}
            available.append((package, missing, raw, families,totals(missing)))
        remaining = [row[0] for row in available]
        if not available:
            break
        fits = [row for row in available if len(row[1]) <= maximum_observations-len(scheduled)
            and row[2] <= maximum_raw_observations-raw_count and fits_resources(row[4])]
        pool = fits or available
        package, missing, raw, families,cost = min(pool, key=lambda row: (
            len(row[1]), sum(family_counts.get(f, 0) for f in row[3]), row[0].identity))
        remaining.remove(package)
        for key in missing:
            admit(key)
        left = tuple(k for k in package.native_keys if k in pending)
        selected.append(dict(identity=package.identity,admitted_missing_native_keys=tuple(
            k for k in missing if k not in pending),unadmitted_missing_native_keys=left,
            entire_missing_package_resource_upper_bounds=cost,
            admitted_missing_input_resource_upper_bounds=totals(tuple(k for k in missing if k not in pending)),
            entire_missing_input_package_admitted=not left))
        if left:
            break  # No scattered partial work on a second programme.
    receipt = dict(policy='shared_context_then_resource_feasible_declared_programmes_v2',
        basis='cache_modalities_resource_vectors_and_frozen_families_not_pose_face_or_camera_cost',
        required_native_keys=required_native_keys,unadmitted_required_native_keys=required_left,
        terminal_required_dependencies=terminal,
        scheduled_unique_facts=len(scheduled),scheduled_new_raw_facts=raw_count,
        initial_live_resource_capacity=remaining_resources,admitted_resource_upper_bounds=spent,
        resource_admission_is_not_atomic_parent_reservation=True,
        selected_packages=selected,unretryable_packages=blocked,
        successful_observations_or_connected_actors_certified=False,quality_approved=False)
    return tuple(scheduled), receipt
