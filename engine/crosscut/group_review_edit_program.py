"""Review-only edited topology over an immutable original duty programme.

R (original roles, clocks and protected pixels) is not E (the authored edit).
An original physical cut is a source fact, not a mandate to show that cut in
every alternative montage. This owner allows partial-cell and mixed-role edits
without turning numeric closure into normal GROUP admission or quality proof.
"""
from dataclasses import dataclass

from .group_actual_shots import visible_group_cut_inventory
from .group_joint_native_preparation import _plan_identity
from .group_performer_family_owner import digest
from .group_reference_interval import compare_group_reference_interval
from .group_scene_epoch_program import compile_common_clock_camera_queries
from .group_scene_owned_program import _header, validate_owned_scene_program
from .render import RenderPlan


CONTRACT = 'review_edit_topology_over_immutable_original_program_v1'


def _intervals(values, count, *, allow_empty=False):
    if not isinstance(values, tuple) or (not values and not allow_empty) or len(values) > 256:
        raise ValueError('edit scope needs explicit bounded intervals')
    previous = -1
    for span in values:
        if (not isinstance(span, tuple) or len(span) != 2 or any(type(t) is not int for t in span)
                or not 0 <= span[0] < span[1] <= count or span[0] < previous):
            raise ValueError('edit scope needs ordered disjoint output intervals')
        previous = span[1]
    return values


def _runs(ticks):
    result = []
    for tick in ticks:
        if result and result[-1][1] == tick:
            result[-1] = (result[-1][0], tick+1)
        else:
            result.append((tick, tick+1))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class GroupReviewEditProgram:
    plan: RenderPlan
    original_program_sha256: str
    original_plan_identity: str
    changed_output_intervals: tuple[tuple[int, int], ...]
    mutable_output_intervals: tuple[tuple[int, int], ...]
    protected_output_intervals: tuple[tuple[int, int], ...]
    original_cuts_json: str
    candidate_cuts_json: str
    query_receipt_json: str
    maximum_changed_frames: int
    maximum_visible_cuts: int

    def to_dict(self):
        import json
        before, after = json.loads(self.original_cuts_json), json.loads(self.candidate_cuts_json)
        original_keys = {(r['first_incoming_output_frame'], r['kind']) for r in before}
        candidate_keys = {(r['first_incoming_output_frame'], r['kind']) for r in after}
        return dict(contract=CONTRACT, original_plan_identity=self.original_plan_identity,
            original_program_sha256=self.original_program_sha256,
            edited_plan_identity=_plan_identity(self.plan),
            changed_output_intervals=[list(s) for s in self.changed_output_intervals],
            mutable_output_intervals=[list(s) for s in self.mutable_output_intervals],
            protected_output_intervals=[list(s) for s in self.protected_output_intervals],
            changed_output_frame_count=sum(b-a for a, b in self.changed_output_intervals),
            maximum_changed_frames=self.maximum_changed_frames,
            maximum_visible_cuts=self.maximum_visible_cuts,
            original_visible_cuts=before, candidate_visible_cuts=after,
            original_visible_events_removed=[r for r in before
                if (r['first_incoming_output_frame'], r['kind']) not in candidate_keys],
            candidate_visible_events_added=[r for r in after
                if (r['first_incoming_output_frame'], r['kind']) not in original_keys],
            original_source_native_inventory_changed=False,
            original_roles_reinterpreted=False,
            original_role_queries=json.loads(self.query_receipt_json),
            original_native_duties_still_require_measured_producer=True,
            normal_complete_bundle_created=False, candidate_admitted=False,
            exact_decoded_pixel_identity_verified=False, quality_approved=False)


def own_review_edit_program(*, context, domain, original_program, candidate_plan,
        mutable_output_intervals, protected_output_intervals,
        maximum_changed_frames, maximum_visible_cuts, maximum_segments=256):
    """Validate one whole edited plan, with the actual new neighbors.

    Candidate cuts may differ. Per-tick original unit/epoch references cannot.
    This does not freeze required cast from a role label, synthesize sensor
    evidence, or copy the original bundle's completion authority to the edit.
    """
    import json
    original = context.incumbent.render_plan
    validate_owned_scene_program(original_program.to_dict(), context, domain)
    if (not isinstance(candidate_plan, RenderPlan) or _header(original) != _header(candidate_plan)
            or original.output_frame_count != candidate_plan.output_frame_count
            or original.authorized_shot_video_clocks or candidate_plan.authorized_shot_video_clocks
            or any(t.kind.uses_overlap for p in (original, candidate_plan) for t in p.transitions)):
        raise ValueError('review edit changed immutable audio/grid/source/canvas headers or clock authority')
    if (type(maximum_segments) is not int or not 1 <= maximum_segments <= 256
            or len(candidate_plan.segments) > maximum_segments
            or type(maximum_changed_frames) is not int or not 1 <= maximum_changed_frames <= 16384
            or type(maximum_visible_cuts) is not int or not 0 <= maximum_visible_cuts <= 255
            or not 1 <= original.output_frame_count <= 16384):
        raise ValueError('review edit exceeds its explicit whole-program bounds')
    count = original.output_frame_count
    mutable = _intervals(mutable_output_intervals, count)
    protected = _intervals(protected_output_intervals, count, allow_empty=True)
    if any(a < d and c < b for a, b in mutable for c, d in protected):
        raise ValueError('review mutable scope overlaps protected output')
    comparison = compare_group_reference_interval(reference=original, candidate=candidate_plan,
        reference_source_clocks=domain.clocks_by_request, candidate_source_clocks=domain.clocks_by_request,
        output_intervals=((0, count),), maximum_output_frames=count)
    changes = tuple(r.output_frame_index for r in comparison.changes)
    if not comparison.audio_reference_parameters_equal:
        raise ValueError('review edit changed original audio')
    if len(changes) > maximum_changed_frames:
        raise ValueError('review edit changed-frame budget exceeded')
    if any(not any(a <= t < b for a, b in mutable) for t in changes):
        raise ValueError('review edit changed pixels/clocks outside its explicit mutable scope')
    if set(changes).intersection(comparison.terminal_clone_output_frames):
        raise ValueError('review edit introduced an unmeasured terminal-clone occurrence')
    before = visible_group_cut_inventory(original, domain)
    after = visible_group_cut_inventory(candidate_plan, domain)
    if len(after) > maximum_visible_cuts:
        raise ValueError('review edit visible-cut budget exceeded')
    # Recompile against the EXTERNAL original, not candidate labels. This
    # explicitly handles an edited cut inside an original cell or mixed role.
    queries = compile_common_clock_camera_queries(context=context, domain=domain,
        original_program=original_program, candidate_plan=candidate_plan,
        output_intervals=((0, count),))
    receipts = tuple(dict(first=q.first_output_frame, end=q.end_output_frame,
        original_role_program_sha256=digest(q.original_role_program),
        original_epoch_index=q.epoch_index,
        role=None if q.role is None else q.role.value,
        selected_source_index=domain.request_index(q.camera.eligible_source_index))
        for q in queries.queries)
    dump = lambda v: json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return GroupReviewEditProgram(candidate_plan, original_program.program_sha256,
        _plan_identity(original), _runs(changes), mutable, protected,
        dump([r.to_dict() for r in before]), dump([r.to_dict() for r in after]), dump(receipts),
        maximum_changed_frames, maximum_visible_cuts)
