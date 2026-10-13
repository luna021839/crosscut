"""Finite editing preferences from existing sparse observations, not QA labels."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class HoldPolicy:
    focal_lower: float = .12
    focal_upper: float = .28
    formation_lower: float = .055
    formation_upper: float = .14
    owner_switch: float = .85
    salience_weight: float = 1.1


DEFAULT_POLICY = HoldPolicy()


def known_switch(a, b):
    """An observed local-label change is a soft event; equal -1 is not a match."""
    a, b = np.asarray(a), np.asarray(b)
    return (a >= 0) & (b >= 0) & (a != b)


def source_local_salience(bank, labels, nodes, source_events, *, fps, radius_seconds=1.2):
    """Native-shot-bounded persistence tempers one-frame largest-head priority.

    No nearest-position fallback, imputed identity, exported vectors or new NN.
    Duplicate physical inputs do not masquerade as independent persistence.
    """
    heads = bank['heads']
    ticks = np.clip(np.asarray(nodes)[:, None] + [-1, 0], 0, nodes[-1]-1)
    shot = np.searchsorted(np.asarray(source_events), ticks, side='right')
    result = np.zeros(heads.shape[:2] + (3,))
    for n in range(len(nodes)):
        for side in (0, 1):
            near = [(k, flank) for k in range(max(0,n-4),min(len(nodes),n+5)) for flank in (0,1)
                if abs(ticks[k,flank]-ticks[n,side])/fps <= radius_seconds
                and shot[k,flank]==shot[n,side]
                and bank['native_indices'][k,flank]!=bank['native_indices'][n,side]]
            distinct = {}
            for k, flank in near:
                distinct.setdefault(int(bank['native_indices'][k,flank]), (k,flank))
            maximum = max(.001,float(heads[n,side,:,2].max()))
            for head in range(3):
                if heads[n,side,head,2] < .019:
                    continue
                owner = labels[n,side,head]
                persistence = (.85 if owner < 0 or not distinct else
                    .25 + .75 * np.mean([np.any(labels[k,flank]==owner) for k,flank in distinct.values()]))
                result[n,side,head] = (float(heads[n,side,head,2])/maximum)**.75 * persistence
    return result


def soft_hold_utility(height, zoom=1., *, formation=False, salience=1., body_states=None, projected_center=None,
                      policy=DEFAULT_POLICY):
    """Continuous role/size preference plus finite observation uncertainty.

    State 0 is unattempted, 1 observed, 2 association failed, 3 other failure.
    None means unavailable. No state establishes real occlusion or visibility.
    Unknown inputs never veto a source or receive fabricated motion evidence.
    """
    if not math.isfinite(height*zoom) or height < 0 or zoom < 1:
        raise ValueError('finite no-padding candidate size required')
    lower, upper = ((policy.formation_lower,policy.formation_upper) if formation else
        (policy.focal_lower,policy.focal_upper))
    size = max(.005,height*zoom)
    scale = (-.05 if formation else -.55) + .85*max(0.,math.log(lower/size)) + .8*max(0.,math.log(size/upper))
    persistence = 0. if formation else -policy.salience_weight*math.log(max(.02,float(salience)))
    uncertainty = 0.
    if body_states is not None and not formation:
        states = np.asarray(body_states)
        if np.any((states<0)|(states>3)):
            raise ValueError('declared cached body observation states required')
        uncertainty = float(np.mean(np.asarray([.08,0.,.32,.24])[states.astype(int)]))
    position=0.
    if projected_center is not None and not formation:
        x,y=map(float,projected_center)
        position=2.*max(0.,.15-x,x-.85)+2.5*max(0.,.18-y,y-.55)
    return dict(scale=float(scale),salience=float(persistence),visibility_uncertainty=uncertainty,
        framing_position=float(position))
