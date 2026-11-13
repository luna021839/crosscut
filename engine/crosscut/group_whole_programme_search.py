"""Variable-duration GROUP search over actual source/actor/camera recipes.

This is the numeric search seam, not another acquisition/render pipeline or a
learned quality oracle. Callers supply source-native runs and frozen measured
recipes. Every opportunity is optional: a candidate can hold across it. A
beam is explicitly approximate; no source-only dominance silently erases
actor, phase, camera, duration or earlier edit history.
"""
from collections import defaultdict
from dataclasses import dataclass
import math
from time import monotonic
from typing import Callable


@dataclass(frozen=True, slots=True)
class ProgrammeOpportunity:
    output_tick: int
    reasons: tuple[str, ...]

    def __post_init__(self):
        if (type(self.output_tick) is not int or not 0 <= self.output_tick <= 36000
                or not isinstance(self.reasons, tuple) or not 1 <= len(self.reasons) <= 16
                or any(not isinstance(reason, str) or not reason for reason in self.reasons)
                or len(set(self.reasons)) != len(self.reasons)):
            raise ValueError('programme opportunity needs an exact tick and distinct event reasons')


def programme_opportunities(*, output_frame_count, fps, beat_seconds=(), phrase_seconds=(),
        section_seconds=(), native_cut_ticks=(), action_ticks=(), handoff_ticks=(), protected_spans=(),
        protected_recipe_boundaries=()):
    """Union exact native/action events with music nodes; do not snap them away."""
    if (type(output_frame_count) is not int or not 2<=output_frame_count<=36000
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0):
        raise ValueError('whole programme requires its explicit output clock')
    reasons=defaultdict(set)
    reasons[0].add('output_start');reasons[output_frame_count].add('output_end')
    for name,seconds in (('beat',beat_seconds),('phrase',phrase_seconds),('section',section_seconds)):
        if not isinstance(seconds, (tuple, list)) or len(seconds) > 32768:
            raise ValueError('musical opportunity inputs must be finite sequences')
        for value in seconds:
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError('musical opportunities need finite output seconds')
            tick=round(value*fps)
            if 0<=tick<=output_frame_count:reasons[tick].add(name)
    for name,ticks in (('native_camera_cut',native_cut_ticks),('observed_action',action_ticks),
            ('part_handoff',handoff_ticks),('protected_recipe',protected_recipe_boundaries)):
        if not isinstance(ticks, (tuple, list)) or len(ticks) > 32768:
            raise ValueError('native/action opportunity inputs must be finite sequences')
        for tick in ticks:
            if type(tick) is not int or not 0<=tick<=output_frame_count:
                raise ValueError('native/action opportunities need exact output ticks')
            reasons[tick].add(name)
    previous=0
    if not isinstance(protected_spans, (tuple, list)) or len(protected_spans) > 1024:
        raise ValueError('whole programme protection requires a finite sequence')
    for span in protected_spans:
        if (not isinstance(span,tuple) or len(span)!=2 or any(type(t) is not int for t in span)
                or not previous<=span[0]<span[1]<=output_frame_count):
            raise ValueError('whole programme protection requires ordered exact spans')
        for tick in span:reasons[tick].add('protected_boundary')
        previous=span[1]
    if len(reasons)>4096:
        raise ValueError('whole programme opportunity union exceeds its declared bound')
    return tuple(ProgrammeOpportunity(t,tuple(sorted(r))) for t,r in sorted(reasons.items()))


@dataclass(frozen=True, slots=True)
class ProgrammeRun:
    identity: str
    first_output_frame: int
    end_output_frame: int
    request_source_index: int
    native_camera_run_index: int
    actor_path_identity: str | None
    camera_recipe_identity: str
    phase_native_frames: int
    unary_cost: float
    protected_reference: bool = False

    def __post_init__(self):
        if (not isinstance(self.identity,str) or not self.identity
                or not isinstance(self.camera_recipe_identity,str) or not self.camera_recipe_identity
                or self.actor_path_identity is not None and (
                    not isinstance(self.actor_path_identity,str) or not self.actor_path_identity)
                or any(type(n) is not int for n in (self.first_output_frame,self.end_output_frame,
                    self.request_source_index,self.native_camera_run_index,self.phase_native_frames))
                or not 0<=self.first_output_frame<self.end_output_frame<=36000
                or self.end_output_frame-self.first_output_frame<2
                or not 0<=self.request_source_index<64 or self.native_camera_run_index<0
                or abs(self.phase_native_frames)>2
                or type(self.unary_cost) not in (int,float) or not math.isfinite(self.unary_cost)
                or type(self.protected_reference) is not bool):
            raise ValueError('programme run needs a finite frozen source/actor/phase/camera recipe')


@dataclass(frozen=True, slots=True)
class ProgrammeSearchBudget:
    beam_width: int | None = 48
    maximum_runs: int = 32768
    maximum_expansions: int = 2000000
    maximum_seconds: float = 45.

    def __post_init__(self):
        if (self.beam_width is not None and (type(self.beam_width) is not int or not 1<=self.beam_width<=128)
                or type(self.maximum_runs) is not int or not 1<=self.maximum_runs<=32768
                or type(self.maximum_expansions) is not int or not 1<=self.maximum_expansions<=2000000
                or type(self.maximum_seconds) not in (int,float)
                or not math.isfinite(self.maximum_seconds) or not 0<self.maximum_seconds<=60):
            raise ValueError('programme search needs explicit finite work bounds')


@dataclass(frozen=True, slots=True)
class WholeProgrammeSelection:
    runs: tuple[ProgrammeRun, ...]
    cost: float
    expansions: int
    beam_truncated: bool
    elapsed_seconds: float
    physically_incompatible_paths: int = 0

    def to_dict(self):
        from dataclasses import asdict
        return dict(contract='group_whole_programme_variable_duration_search_v1',
            runs=[asdict(r) for r in self.runs],cost=self.cost,expansions=self.expansions,
            approximate=self.beam_truncated,elapsed_seconds=self.elapsed_seconds,
            physically_incompatible_paths=self.physically_incompatible_paths,
            optional_nodes_are_not_forced_cuts=True,full_edit_history_preserved=True,
            source_only_state_merging=False,camera_refit_after_selection=False,
            named_identity_certified=False,professional_quality_approved=False)


def select_whole_programme(*, opportunities, runs, transition_cost: Callable,
        protected_spans=(), budget=ProgrammeSearchBudget(), cancelled=None,
        physically_compatible=None, work_port=None):
    """Select variable-length runs, retaining all history consumed by scoring.

    ``transition_cost(history, incoming)`` sees the exact selected whole runs,
    including their actors/cameras/phases and durations. It may score a part
    handoff or uncertainty; missing anatomy/identity is not an infeasibility
    gate in this solver. Hard physical eligibility belongs to run production.
    With ``beam_width=None`` no histories are pruned, under the same declared
    work bound. Exhaustion is terminal for this call, not a retry invitation.
    ``physically_compatible`` may reject a renderer clock/coverage conflict,
    not missing identity, anatomy or another uncertain artistic measurement.
    """
    if (not isinstance(opportunities,tuple) or not 2<=len(opportunities)<=4096
            or any(not isinstance(o,ProgrammeOpportunity) for o in opportunities)
            or opportunities[0].output_tick!=0
            or tuple(o.output_tick for o in opportunities)!=tuple(sorted({o.output_tick for o in opportunities}))
            or not isinstance(runs,tuple) or not runs or not isinstance(budget,ProgrammeSearchBudget)
            or len(runs)>budget.maximum_runs or not callable(transition_cost)
            or cancelled is not None and not callable(cancelled)
            or physically_compatible is not None and not callable(physically_compatible)
            or not isinstance(protected_spans, tuple) or len(protected_spans) > 1024):
        raise ValueError('whole programme needs a frozen finite opportunity/run inventory')
    ticks={o.output_tick for o in opportunities};end=opportunities[-1].output_tick
    if not 2 <= end <= 36000:
        raise ValueError('whole programme output coverage exceeds its bound')
    started=monotonic();deadline=started+budget.maximum_seconds

    def check_budget():
        if work_port is not None:
            work_port._check()
        if cancelled is not None and cancelled():
            raise InterruptedError('whole programme cancelled')
        if monotonic() > deadline:
            raise TimeoutError('whole programme original search budget exhausted; no retry')

    check_budget()
    previous=0
    for span in protected_spans:
        if (not isinstance(span, tuple) or len(span) != 2
                or any(type(t) is not int for t in span)
                or not previous <= span[0] < span[1] <= end):
            raise ValueError('whole programme protected scope changed')
        previous=span[1]
    by_end=defaultdict(list);identities=set()
    for run in runs:
        check_budget()
        if (not isinstance(run,ProgrammeRun) or run.identity in identities
                or run.first_output_frame not in ticks or run.end_output_frame not in ticks
                or run.end_output_frame>end):
            raise ValueError('programme run escaped its frozen unique opportunity inventory')
        identities.add(run.identity)
        if any(run.first_output_frame<b and run.end_output_frame>a for a,b in protected_spans):
            if not run.protected_reference or not any(
                    a <= run.first_output_frame < run.end_output_frame <= b
                    for a, b in protected_spans):
                continue
        elif run.protected_reference:
            raise ValueError('protected recipe flag escaped its declared protected interval')
        by_end[run.end_output_frame].append(run)
    # No source/run-only recombination: distinct complete histories can carry
    # different later A-B-A, actor, duration and camera costs.
    states={0:[(0.,(),())]};expansions=0;truncated=False;physical_rejections=0
    for opportunity in opportunities[1:]:
        check_budget()
        tick=opportunity.output_tick;current=[]
        incoming_runs=sorted(by_end[tick],key=lambda r:r.identity)
        # One durable admission for this finite numeric layer, not a fsync
        # per arithmetic edge. Successful searches charge exactly their
        # measured expansions. Interrupted layers retain the admitted upper
        # work bound; no NN/decode work is refunded or invented.
        planned_edges=sum(len(states.get(run.first_output_frame,())) for run in incoming_runs)
        if expansions+planned_edges>budget.maximum_expansions:
            raise TimeoutError('whole programme original search budget exhausted; no retry')
        if work_port is not None and planned_edges:
            work_port.charge('edges',planned_edges)
        for run in incoming_runs:
            for cost,path,tie in states.get(run.first_output_frame,()):
                expansions+=1
                if expansions>budget.maximum_expansions:
                    raise TimeoutError('whole programme original search budget exhausted; no retry')
                check_budget()
                if physically_compatible is not None:
                    eligible = physically_compatible(path, run)
                    check_budget()
                    if type(eligible) is not bool:
                        raise ValueError('physical path compatibility requires an explicit boolean')
                    if not eligible:
                        physical_rejections += 1
                        continue
                edge=transition_cost(path,run)
                check_budget()
                if type(edge) not in (int,float) or not math.isfinite(edge):
                    raise ValueError('programme edge needs an actual finite cost, not an unknown veto')
                total = cost+run.unary_cost+edge
                if not math.isfinite(total):
                    raise ValueError('programme accumulated cost is not finite')
                current.append((total,(*path,run),(*tie,run.identity)))
        current.sort(key=lambda state:(state[0],state[2]))
        check_budget()
        if budget.beam_width is not None and len(current)>budget.beam_width:
            current=current[:budget.beam_width];truncated=True
        if work_port is not None:
            # Count retained search states, and every attempted edge above.
            # Pruning cannot refund edge work or renew the parent phase.
            work_port.charge('states',len(current))
        states[tick]=current
    if not states[end]:
        raise ValueError('whole programme has no complete physically eligible source path')
    cost,path,_=states[end][0]
    return WholeProgrammeSelection(path,cost,expansions,truncated,monotonic()-started,physical_rejections)
