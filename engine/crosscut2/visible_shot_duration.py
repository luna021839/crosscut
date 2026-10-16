"""Soft, reference-fitted pacing of visible shots, including native cuts.

This is a duration distribution, not a calibrated aesthetic critic. It neither
sets a cut quota nor forbids a short highlight. Only explicitly supplied TRAIN
references fit the two log-duration statistics; no same-song edit timestamps
are used at inference. Camera and action costs still select the actual film.
"""
import hashlib
import json
import math

import numpy as np


CONTRACT = 'train_reference_visible_shot_log_duration_v1'
TAIL_CONTRACT = 'train_reference_visible_shot_interquartile_tail_surprise_v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def fit(references):
    """Fit equally weighted reference distributions, ignoring censored ends.

    Each row supplies its physical reference id, TRAIN assignment, input hash,
    and strictly ordered actual visible-cut seconds. Duplicate reference ids
    are rejected rather than reweighted as independent editing examples.
    """
    values = []; identities = set(); bindings = set(); provenance = []
    for row in references:
        identity = row.get('reference_id'); binding = row.get('input_sha256', '')
        cuts = np.asarray(row.get('cut_seconds'), float)
        if (row.get('split') != 'train' or not isinstance(identity, str) or not identity
                or identity in identities or binding in bindings or len(binding) != 64
                or any(c not in '0123456789abcdef' for c in binding)
                or cuts.ndim != 1 or len(cuts) < 3 or not np.isfinite(cuts).all()
                or np.any(cuts < 0) or np.any(np.diff(cuts) <= 0)):
            raise ValueError('distinct explicit TRAIN references and actual chronological cuts required')
        identities.add(identity); bindings.add(binding)
        durations = np.diff(cuts)
        values.append(np.log(durations))
        provenance.append(dict(reference_id=identity, split='train', input_sha256=binding,
            uncensored_shots=len(durations)))
    if not values:
        raise ValueError('no TRAIN durations supplied')
    # Each editor/video has equal mass, not each cut or duplicated source.
    mean = float(np.mean([x.mean() for x in values]))
    variance = float(np.mean([np.mean((x-mean)**2) for x in values]))
    if variance <= 0 or not math.isfinite(variance):
        raise ValueError('reference duration variation is required')
    result = dict(contract=CONTRACT, log_seconds_mean=mean,
        log_seconds_scale=math.sqrt(variance), references=provenance,
        loss='seconds_times_one_minus_gaussian_log_duration_affinity',
        aesthetic_weight_fitted=False, cut_times_used_at_inference=False,
        professional_quality_approved=False)
    result['content_sha256'] = _digest(result)
    return result


class Model:
    def __init__(self, artifact):
        payload = dict(artifact); expected = payload.pop('content_sha256', None)
        if (expected != _digest(payload) or payload.get('contract') not in (CONTRACT,TAIL_CONTRACT)
                or not payload.get('references')
                or any(r.get('split') != 'train' for r in payload['references'])
                or len({r['reference_id'] for r in payload['references']}) != len(payload['references'])
                or len({r['input_sha256'] for r in payload['references']}) != len(payload['references'])):
            raise ValueError('explicit fitted TRAIN duration model required')
        self.mean = float(payload['log_seconds_mean']); self.scale = float(payload['log_seconds_scale'])
        if not math.isfinite(self.mean) or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('finite fitted log-duration distribution required')
        self.artifact = dict(artifact)
        self.tails=payload['contract']==TAIL_CONTRACT
        if self.tails:
            fps=payload.get('output_fps',0.)
            if (not math.isfinite(fps) or fps<=0 or payload.get('perceptual_unit_seconds')!=6./fps
                    or payload.get('tail_probability')!=.25 or not payload.get('fit_content_sha256')):
                raise ValueError('explicit fitted-duration tails and actual six-frame output unit required')

    def loss(self, seconds, *, censored=False):
        seconds = np.asarray(seconds, float)
        if not np.isfinite(seconds).all() or np.any(seconds < 0):
            raise ValueError('finite nonnegative visible shot durations required')
        delta = np.log(np.maximum(np.finfo(float).tiny, seconds))-self.mean
        if self.tails:
            z=delta/self.scale
            erfc=np.vectorize(math.erfc,otypes=[float])
            lower=np.maximum(np.finfo(float).tiny,.5*erfc(-z/math.sqrt(2.)))
            upper=np.maximum(np.finfo(float).tiny,.5*erfc(z/math.sqrt(2.)))
            short=np.where(censored,0.,np.maximum(0.,-np.log(lower)+math.log(.25)))
            long=np.maximum(0.,-np.log(upper)+math.log(.25))
            return np.where(seconds>0,self.artifact['perceptual_unit_seconds']*(short+long),0.)
        # At film boundaries the observed shot may be only its remaining
        # fragment. A short censored fragment is not an intentionally short
        # edit; a long one still contains that much observed screen time.
        delta = np.where(censored, np.maximum(0., delta), delta)/self.scale
        return seconds * -np.expm1(-.5*delta**2)

    def table(self, nodes, events, *, fps):
        """All hold/source costs; native fragments, not just authored holds.

        Uses O(S*N*N) finite numeric storage and event prefixes. It adds no
        state, camera family, NN observation, minimum shot, or cut count rule.
        """
        nodes = np.asarray(nodes)
        if self.tails and fps!=self.artifact['output_fps']:
            raise ValueError('duration-tail output clock changed')
        if (nodes.ndim != 1 or nodes.dtype.kind not in 'iu' or len(nodes) < 2
                or nodes[0] != 0 or np.any(np.diff(nodes) <= 0)
                or not math.isfinite(fps) or fps <= 0):
            raise ValueError('actual chronological output nodes and fps required')
        first = nodes[:, None]; final = nodes[None, :]; chronological = final > first
        count = len(nodes); result = np.zeros((count, count, len(events)))
        for source, raw in enumerate(events):
            xs = np.asarray(raw)
            if xs.size == 0:
                xs = np.zeros(0, np.int64)
            if (xs.ndim != 1 or xs.dtype.kind not in 'iu' or np.any(np.diff(xs) <= 0)
                    or np.any((xs < 0) | (xs > nodes[-1]))):
                raise ValueError('unique source-native output cut frames required')
            xs = xs[(xs > 0) & (xs < nodes[-1])]
            base = self.loss(np.maximum(0, final-first)/fps,
                censored=(first == 0) | (final == nodes[-1]))
            if len(xs):
                lo = np.searchsorted(xs, first, side='right')
                stop = np.searchsorted(xs, final, side='left'); hi = stop-1
                internal = stop > lo
                left = np.maximum(0, xs[np.minimum(lo, len(xs)-1)]-first)/fps
                right = np.maximum(0, final-xs[np.maximum(hi, 0)])/fps
                prefix = np.r_[0., np.cumsum(self.loss(np.diff(xs)/fps))]
                middle = prefix[np.maximum(hi, 0)]-prefix[np.minimum(lo, len(xs)-1)]
                pieces = (self.loss(left, censored=first == 0)
                    + self.loss(right, censored=final == nodes[-1])+middle)
                base = np.where(internal, pieces, base)
            result[:, :, source] = np.where(chronological, base, 0.)
        result.setflags(write=False)
        return result


def tail_artifact(artifact,*,fps):
    """Reuse TRAIN statistics with a broad flat middle, not a target length.

    Only lower/upper quartile surprises pay. Six output frames provide the
    existing boundary-effort unit; its aesthetic weight is NOT fitted.
    """
    original=Model(artifact)
    if original.tails or not math.isfinite(fps) or fps<=0:
        raise ValueError('original TRAIN fit and actual positive output fps required')
    value=dict(artifact);value.pop('content_sha256')
    value.update(contract=TAIL_CONTRACT,fit_content_sha256=artifact['content_sha256'],
        output_fps=float(fps),perceptual_unit_seconds=6./fps,tail_probability=.25,
        loss='six_frame_unit_times_log_tail_surprise_outside_reference_interquartile_range',
        preferred_single_duration=False,aesthetic_weight_fitted=False)
    value['content_sha256']=_digest(value)
    return value
