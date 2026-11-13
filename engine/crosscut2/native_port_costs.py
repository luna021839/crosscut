"""Editing policy over cached native observations; no sensor invalidation."""
import numpy as np

from .hold_utility import DEFAULT_POLICY
from .native_ports import native_geometry_events


def native_port_cost_events(nodes,bank,ticks,curve):
    interval,through=native_geometry_events(nodes,bank,ticks,curve)
    for n,frame in enumerate(bank['frames']):
        heads=bank['heads'][n]
        present=np.any(heads[:,:,2]>.019,axis=1)
        if present.all():continue
        readable=False
        for side in (0,1):
            visible=heads[side][heads[side,:,2]>.019]
            if not len(visible):continue
            formation=len(visible)>=3 and np.max(visible[:,2])<1.6*np.sort(visible[:,2])[-2]
            readable|=bool(np.max(visible[:,2])>=.055 and not formation)
        # No invented scale/identity error: the same finite missing-port
        # uncertainty used at authored cuts, never an all-source veto.
        uncertainty=DEFAULT_POLICY.missing_port_uncertainty+(2. if readable else 0.)
        at=int(np.searchsorted(nodes,frame))
        if at<len(nodes) and nodes[at]==frame:through[at]+=uncertainty
        elif 0<at<len(nodes):interval[at-1]+=uncertainty
    return interval,through
