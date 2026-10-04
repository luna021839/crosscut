"""Bounded editorial beam for a close-up-priority GROUP review.

The inputs are measured/ranked hypotheses, not a professional-quality oracle.
KEEP participates at every cell; protected cells permit only KEEP. A source-ID
change alone earns no reward. Motivated near-shot matches and long static near
holds are explicit style terms, separate from comparative quality evidence.
"""
from dataclasses import asdict, dataclass
import math

from .group_performer_family_owner import digest


@dataclass(frozen=True, slots=True)
class CloseupReviewStyle:
    closeup_weight: float = 5.
    blur_weight: float = .6
    uncertainty_weight: float = .7
    seam_weight: float = .9
    matched_moving_cut_reward: float = 1.15
    near_hold_seconds: float = 2.4
    near_hold_weight: float = .35
    change_weight: float = .025
    maximum_beam: int = 48
    maximum_added_source_cuts: int = 48

    def __post_init__(self):
        values = asdict(self)
        if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError('closeup style needs explicit finite nonnegative values')
        if not 1 <= self.maximum_beam <= 96 or type(self.maximum_beam) is not int:
            raise ValueError('closeup beam must be explicitly bounded')
        if type(self.maximum_added_source_cuts) is not int or not 0 <= self.maximum_added_source_cuts <= 128:
            raise ValueError('closeup cut exploration must be explicitly bounded')

    def to_dict(self):
        return dict(contract='closeup_priority_review_style_v1', **asdict(self),
            source_id_change_is_quality_reward=False, calibrated_quality_score=False,
            fixed_cut_count_required=False, quality_approved=False)


def select_closeup_review_path(*, cells, edge_at, fps, style, maximum_changed_frames,
        work_port=None):
    """One bounded whole programme, not sequential local incumbent mutations.

    A cell has ordered [first,end), original_source and alternatives with
    identity/source/keep, near_fraction [0,1], blur_risk, uncertainty_risk.
    edge_at owns actual adjacent-hypothesis measurements. Its moving_match in
    [0,1] must describe evidence, not just different source names. Unknown
    evidence is an explicit exploration penalty, never comparative BETTER.
    """
    if (not cells or len(cells) > 256 or not isinstance(style, CloseupReviewStyle)
            or not math.isfinite(fps) or fps <= 0
            or type(maximum_changed_frames) is not int or not 0 <= maximum_changed_frames <= 16384):
        raise ValueError('closeup graph needs a bounded exact output domain')
    edge_cache, seen = {}, set()
    for i, cell in enumerate(cells):
        if (any(type(cell[k]) is not int for k in ('first', 'end', 'original_source'))
                or not 0 <= cell['first'] < cell['end']
                or i and cells[i-1]['end'] != cell['first']
                or not 1 <= len(cell['alternatives']) <= 8
                or sum(bool(a['keep']) for a in cell['alternatives']) != 1):
            raise ValueError('every closeup cell needs exact contiguous scope and one real KEEP')
        if cell.get('protected') and (len(cell['alternatives']) != 1 or not cell['alternatives'][0]['keep']):
            raise ValueError('protected closeup cell cannot enter donor exploration')
        for alt in cell['alternatives']:
            if (alt['identity'] in seen or type(alt['source']) is not int
                    or any(type(alt[k]) not in (float, int) or not math.isfinite(alt[k]) or alt[k] < 0
                        for k in ('near_fraction', 'blur_risk', 'uncertainty_risk'))
                    or alt['near_fraction'] > 1):
                raise ValueError('closeup alternatives need distinct bound identities and finite observed risks')
            seen.add(alt['identity'])
    # State: actual complete path/cost/last camera run/change frames/added cuts.
    beam = [dict(cost=0., path=(), hold=0, changed=0, added=0)]
    visited_states = visited_edges = 0
    for i, cell in enumerate(cells):
        duration = (cell['end']-cell['first'])/fps
        next_beam = []
        for parent in beam:
            prev = None if not parent['path'] else parent['path'][-1]
            for alt in cell['alternatives']:
                changed = parent['changed']+(0 if alt['keep'] else cell['end']-cell['first'])
                if changed > maximum_changed_frames: continue
                switched = prev is not None and prev['source'] != alt['source']
                added = parent['added']+int(switched and cells[i-1]['original_source'] == cell['original_source'])
                if added > style.maximum_added_source_cuts: continue
                hold = cell['end']-cell['first']+(0 if switched else parent['hold'])
                node = duration*(style.closeup_weight*(1-alt['near_fraction'])
                    +style.blur_weight*alt['blur_risk']+style.uncertainty_weight*alt['uncertainty_risk']
                    +style.change_weight*(not alt['keep']))
                # Hold duration is actual output time, not numeric cell count.
                node += (style.near_hold_weight*alt['near_fraction']*duration
                    *max(0., hold/fps-style.near_hold_seconds))
                edge_cost = 0.
                if prev is not None:
                    key = (prev['identity'], alt['identity'])
                    if key not in edge_cache:
                        result = edge_at(i, prev, alt)
                        if (any(type(result[k]) not in (float, int) or not math.isfinite(result[k]) or result[k] < 0
                                for k in ('risk', 'moving_match')) or result['moving_match'] > 1):
                            raise ValueError('closeup edge must retain actual finite risks and match strength')
                        if not switched and result['moving_match']:
                            raise ValueError('unchanged camera cannot earn authored crosscut reward')
                        edge_cache[key] = result
                    result = edge_cache[key]
                    edge_cost = style.seam_weight*result['risk']-style.matched_moving_cut_reward*result['moving_match']
                    visited_edges += 1
                    if work_port is not None: work_port.charge('edges')
                next_beam.append(dict(cost=parent['cost']+node+edge_cost,
                    path=(*parent['path'], alt), hold=min(hold, round(8*fps)), changed=changed, added=added))
                visited_states += 1
                if work_port is not None: work_port.charge('states')
        if not next_beam: raise ValueError('closeup path has no feasible complete KEEP/donor branch')
        # Content-derived tie: source indices are not aesthetic preferences.
        next_beam.sort(key=lambda p: (p['cost'], digest([a['identity'] for a in p['path']])))
        representatives, keys = [], set()
        for item in next_beam:
            key = (item['path'][-1]['identity'], item['hold'], item['changed'], item['added'])
            if key in keys: continue
            keys.add(key); representatives.append(item)
            if len(representatives) == style.maximum_beam: break
        beam = representatives
    best = beam[0]
    return dict(contract='bounded_complete_closeup_review_path_v1',
        alternatives=best['path'], exploration_cost=best['cost'], changed_output_frames=best['changed'],
        added_source_switches=best['added'], visited_states=visited_states, visited_edges=visited_edges,
        measured_edges=len(edge_cache), style=style.to_dict(), whole_program_selected_once=True,
        quality_approved=False, normal_candidate_admitted=False)
