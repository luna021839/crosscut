"""Final pixel/clock replay for selected native metadata, not actor approval.

The native metadata producer never claimed a pose-model receipt. Dispatching
its receipt to the older model-based camera owner invented that dependency.
This consumer checks its actual selected sampler and projected pixel parents;
it does not infer identity, body phase or professional quality from hashes.
"""
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Mapping

from .group_joint_observations import GroupJointObservationSession, _file_identity
from .group_joint_source_camera import JointGroupWorkBudget
from .group_scene_camera_queries import PreparedGroupCameraQueries
from .group_source_domain import GroupSourceDomain
from .optional_native_worker import isolated_optional
from .render_frame_schedule import VerifiedRenderFrameClock, probe_render_frame_clock

CONTRACT = "selected_native_motion_complete_metadata_v1"
MAX_FINAL_SAMPLES = 1536


def validate_serialized_native_metadata(complete):
    from .group_performer_family_owner import digest, plain
    if not isinstance(complete, Mapping):
        raise ValueError("selected native complete metadata missing")
    metadata = dict(complete)
    receipt = plain(metadata.pop("performer_family_review", None))
    # The production backlink has its own complete-selection seal. It was not
    # present when the family payload was hashed and cannot be folded into it.
    metadata.pop("native_motion_production", None)
    dense = plain((metadata.get("editorial_motion") or {}).get("source_camera_refresh"))
    if (not isinstance(receipt, dict) or receipt.get("contract") != CONTRACT
            or metadata.get("candidate_origin") != CONTRACT
            or not isinstance(dense, dict) or dense.get("contract") != CONTRACT
            or receipt.get("source_camera_refresh") != dense
            or receipt.get("metadata_sha256") != digest(metadata)
            or any(receipt.get(k) is not False for k in ("quality_approved", "raw_parent_admitted",
                "cross_camera_identity_certified", "body_phase_certified", "decoded_output_quality_certified"))):
        raise ValueError("selected native serialized family or uncertainty binding changed")
    if metadata.get("boundary_look_match") is not None or any(
            row.get("boundary_look_track") is not None for row in metadata.get("final_render_segments", ())):
        raise ValueError("selected native metadata has an unsupported live look owner")
    return receipt


def _final_inputs(plan_json, receipt):
    from .group_performer_family_owner import digest, plan_payload, plain
    from .group_joint_native_preparation import _plan_identity
    from .render_plan_serialization import render_plan_from_json
    receipt = plain(receipt)
    plan = render_plan_from_json(plan_json)
    dense = receipt.get("source_camera_refresh") or {}
    evidence = dense.get("timeline_evidence") or {}
    source = receipt.get("source_domain") or {}
    binding = evidence.get("live_source_binding")
    if (receipt.get("contract") != CONTRACT or dense.get("contract") != CONTRACT
            or receipt.get("plan_sha256") != digest(plan_payload(plan))
            or evidence.get("selected_plan_identity") != _plan_identity(plan)
            or not isinstance(binding, dict) or binding.get("contract") != "native_metadata_live_sources_v1"
            or binding.get("source_domain") != source):
        raise ValueError("selected native final replay lacks its exact live source/plan preimage")
    indices = source.get("eligible_request_indices")
    clocks = source.get("render_frame_clocks")
    identities = binding.get("source_identities")
    if (not isinstance(indices, list) or not indices or len(indices) > 64 or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(plan.sources) for i in indices)
            or source.get("eligible_sources") != [str(plan.sources[i]) for i in indices]
            or not isinstance(clocks, list) or len(clocks) != len(indices)
            or not isinstance(identities, list) or len(identities) != len(indices)):
        raise ValueError("selected native final source mapping changed")
    samples = evidence.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_FINAL_SAMPLES:
        raise ValueError("selected native final pixel footprint is absent or unbounded")
    seen = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("selected native final sample schema changed")
        tick, source_index, native = (sample.get(k) for k in
            ("output_tick", "request_source_index", "native_frame_index"))
        if (type(tick) is not int or not 0 <= tick < plan.output_frame_count or tick in seen
                or source_index not in indices or type(source_index) is not int
                or type(native) is not int or native < 0
                or any(not isinstance(sample.get(k), str) or len(sample[k]) != 64
                    or any(c not in "0123456789abcdef" for c in sample[k])
                    for k in ("native_pixel_sha256", "projected_pixel_sha256"))):
            raise ValueError("selected native final sample address or pixel parent changed")
        seen.add(tick)
    return plan, source, identities, samples


@isolated_optional()
def replay_selected_native_metadata(plan_json, receipt, deadline):
    """Independent exact raster replay under the final request-work phase."""
    from .group_performer_family_owner import digest
    plan, source, identities, samples = _final_inputs(plan_json, receipt)
    indices = tuple(source["eligible_request_indices"])
    before = [list(_file_identity(plan.sources[i])) for i in indices]
    if before != identities:
        raise ValueError("selected native final source file changed")
    clocks = []
    for i, row in zip(indices, source["render_frame_clocks"], strict=True):
        expected = VerifiedRenderFrameClock(Fraction(row["frame_rate"]), row["frame_count"],
            row["packet_grid_sha256"], row["grid_basis"])
        remaining = deadline-monotonic()
        if remaining <= 0:
            raise TimeoutError("selected native final clock replay deadline exceeded")
        if probe_render_frame_clock(plan.sources[i], timeout_seconds=min(10., remaining)) != expected:
            raise ValueError("selected native final source presentation clock changed")
        clocks.append(expected)
    # The complete plan owns fixed source times. Dummy alignment values here
    # are not reused for editorial selection or claimed as sync validation.
    domain = GroupSourceDomain(tuple(plan.sources), indices, (0.,)*len(indices), (None,)*len(indices),
        tuple(tuple(c) for c in source["scene_cut_times"]), tuple(clocks))
    prepared = PreparedGroupCameraQueries(plan, domain)
    session = GroupJointObservationSession(domain=domain, budget=JointGroupWorkBudget(
        absolute_deadline=deadline, maximum_new_decodes=len(samples)))
    try:
        for row in sorted(samples, key=lambda r: (r["request_source_index"], r["native_frame_index"], r["output_tick"])):
            tick = row["output_tick"]
            actual = prepared.samples[tick]
            if (actual.source_index != row["request_source_index"]
                    or actual.native_frame_index != row["native_frame_index"] or actual.terminal_clone):
                raise ValueError("selected native final actual sampler differs from observed parent")
            shot = next(s for s in prepared.shots if s.first_output_frame <= tick < s.end_output_frame)
            camera = prepared.camera_for(shot)
            native = session.native_pixels(actual.source_index, actual.native_frame_index)
            projected = session.projected_pixels(camera, tick)
            if (native is None or projected is None
                    or sha256(memoryview(native)).hexdigest() != row["native_pixel_sha256"]
                    or sha256(memoryview(projected)).hexdigest() != row["projected_pixel_sha256"]):
                raise ValueError("selected native final native/projected pixel parent changed")
    finally:
        session.close()
    if [list(_file_identity(plan.sources[i])) for i in indices] != before:
        raise ValueError("selected native source changed during final pixel replay")
    result = dict(contract=CONTRACT, binding_valid=True, source_live_replay=True,
        sampled_native_pixel_count=len(samples), quality_approved=False,
        raw_parent_admitted=False, body_phase_certified=False, cross_camera_identity_certified=False,
        decoded_output_quality_certified=False, final_quality_status="UNAPPROVED_NATIVE_REVIEW_CANDIDATE",
        plan_sha256=receipt["plan_sha256"], metadata_sha256=receipt["metadata_sha256"])
    result["evidence_sha256"] = digest(result)
    return result
