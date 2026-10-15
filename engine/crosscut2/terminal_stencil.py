"""Bounded ending-pixel observation, not a general occlusion/identity gate.

Only a large newly opaque near-black component inside one native shot counts.
Static letterboxes, ordinary dark footage and unobserved tails are not banned.
This is an explicit finishing heuristic, not a certified broadcaster classifier.
"""
import hashlib
from time import monotonic
import cv2
import numpy as np
from . import sparse_native_frames as sparse


def new_stencil(reference,ending):
    a,b=np.asarray(reference),np.asarray(ending)
    if a.shape!=(540,960,3) or b.shape!=a.shape or a.dtype!=np.uint8 or b.dtype!=np.uint8:
        raise ValueError('two actual960x540 uint8 native rasters required')
    old=cv2.dilate(cv2.inRange(a,(0,0,0),(3,3,3)),np.ones((5,5),np.uint8))
    black=cv2.inRange(b,(0,0,0),(3,3,3))
    new=cv2.bitwise_and(black,cv2.bitwise_not(old));new[:54]=0;new[-27:]=0;new[:,:48]=0;new[:,-48:]=0
    _,_,stats,_=cv2.connectedComponentsWithStats(new,8)
    area=0 if len(stats)<2 else int(stats[1:,cv2.CC_STAT_AREA].max())
    fraction=area/(960*540)
    return dict(largest_new_component_fraction=fraction,new_opaque_ending_stencil=bool(fraction>=.015),
        minimum_component_fraction=.015,near_black_channel_maximum=3,semantic_logo_identity_certified=False)


def observe(sources,banks,clocks,*,check=lambda:None,maximum_seconds=45.):
    """Two small native rasters/source, no NN and no whole-video recensus."""
    if not len(sources)==len(banks)==len(clocks):raise ValueError('one actual bank/clock per source required')
    if not np.isfinite(maximum_seconds) or not 1<=maximum_seconds<=90:raise ValueError('bounded optional ending allowance required')
    started=monotonic();receipts=[]
    def live():
        check()
        if monotonic()-started>=maximum_seconds:raise TimeoutError('optional ending observation allowance exhausted')
    for s,(path,bank,clock) in enumerate(zip(sources,banks,clocks)):
        check();last=int(bank['native_indices'][-1,0]);first=last-round(float(clock['fps']))
        row=dict(source=s,known=False,new_opaque_ending_stencil=False,native_ordinals=[first,last],
            source_NN_calls=0,whole_source_analysis_repeated=False)
        if monotonic()-started>=maximum_seconds:
            receipts.append(dict(row,reason='optional_ending_allowance_unmeasured'));continue
        cuts=np.asarray(bank.get('native_cut_frames',[]),int)
        if first<0 or last>=clock['frame_count'] or np.any((cuts>first)&(cuts<=last)):
            receipts.append(dict(row,reason='unobserved_or_different_native_shot_not_an_overlay_verdict'));continue
        decoded={};receipt={}
        try:
            native=sparse.probe(path,check=live)
            if native.count!=clock['frame_count'] or not np.isclose(float(native.rate),clock['fps'],rtol=1e-9):
                raise ValueError('ending observation source clock changed')
            for n,pixels in sparse.frames(path,[first,last],clock=native,check=live,receipt=receipt,
                    maximum_seconds=min(20.,max(1.,maximum_seconds-(monotonic()-started)))):decoded[n]=pixels
        except (sparse.UnsupportedSeek,TimeoutError) as exc:
            check();receipts.append(dict(row,reason=type(exc).__name__,observation_error=str(exc)));continue
        receipts.append(dict(row,known=True,**new_stencil(decoded[first],decoded[last]),
            actual_raster_sha256=[hashlib.sha256(decoded[n].tobytes()).hexdigest() for n in (first,last)],
            decoder_receipt=receipt))
    return receipts


def eligible(banks):
    rows=[bank.get('terminal_stencil_observation',{}) for bank in banks]
    clean=any(r.get('known',False) and not r['new_opaque_ending_stencil'] for r in rows)
    # No source-count requirement. Unknown remains eligible; if all observed
    # tails are questionable this heuristic does not destroy full coverage.
    return np.asarray([not(clean and r.get('known',False) and r['new_opaque_ending_stencil']) for r in rows],bool)
