"""Bounded missing-input admission, not an editorial or identity selector.

Complete shared original context first, then coherent predeclared programmes.
Cache coverage may determine which programme can be measured within a lease;
pose, face similarity and camera scores may not determine this input schedule.
An unavailable observation remains unavailable and cannot invite a retry.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


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


def validate_completion_packages(inventory, required_native_keys, packages):
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


def coherent_completion_schedule(inventory, *, required_native_keys, packages,
        maximum_observations, maximum_raw_observations):
    """Admit complete feasible packages, or ONE explicit partial package.

    A small remaining budget must not buy isolated frames from every donor and
    leave every programme unmeasurable. Required contexts retain caller order;
    package ordering uses only missing-input count and frozen source families.
    This reserves inputs, not NN success, actor connectivity or quality.
    """
    validate_completion_packages(inventory, required_native_keys, packages)
    aliases = {}; physical = {}
    for supply in inventory.supplies:
        physical[supply.demand.native_key] = supply
        if supply.missing_modalities:
            aliases.setdefault(supply.demand.native_key, []).append(supply.demand)
    pending = dict(aliases); scheduled = []; raw_count = 0; family_counts = {}

    def admit(key):
        nonlocal raw_count
        if key not in pending:
            return True
        is_raw = physical[key].observation is None
        if len(scheduled) >= maximum_observations or is_raw and raw_count >= maximum_raw_observations:
            return False
        rows = pending.pop(key)
        demand = min(rows, key=lambda d: (d.output_tick, d.source_family, d.opportunity))
        scheduled.append(demand); raw_count += is_raw
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
            available.append((package, missing, raw, families))
        remaining = [row[0] for row in available]
        if not available:
            break
        fits = [row for row in available if len(row[1]) <= maximum_observations-len(scheduled)
            and row[2] <= maximum_raw_observations-raw_count]
        pool = fits or available
        package, missing, raw, families = min(pool, key=lambda row: (
            len(row[1]), sum(family_counts.get(f, 0) for f in row[3]), row[0].identity))
        remaining.remove(package)
        for key in missing:
            admit(key)
        left = tuple(k for k in package.native_keys if k in pending)
        selected.append(dict(identity=package.identity,admitted_missing_native_keys=tuple(
            k for k in missing if k not in pending),unadmitted_missing_native_keys=left,
            entire_missing_input_package_admitted=not left))
        if left:
            break  # No scattered partial work on a second programme.
    receipt = dict(policy='shared_context_then_coherent_declared_programmes_v1',
        basis='cache_modalities_and_frozen_families_not_pose_face_or_camera_cost',
        required_native_keys=required_native_keys,unadmitted_required_native_keys=required_left,
        scheduled_unique_facts=len(scheduled),scheduled_new_raw_facts=raw_count,
        selected_packages=selected,unretryable_packages=blocked,
        successful_observations_or_connected_actors_certified=False,quality_approved=False)
    return tuple(scheduled), receipt
