"""One-pass transient H/K preparation; no song-sized pair tensor in RAM."""
from pathlib import Path
import json
from time import monotonic
import numpy as np


def prepare(ports,model,*,workspace,check):
    workspace=Path(workspace);q,N=len(ports.states),len(ports.nodes);started=monotonic()
    sources=np.asarray([s.source for s in ports.states]);holds=np.zeros((q,N));written=0
    for n in range(1,N-1):
        check()
        if hasattr(model,'losses_from_ports'):h,k=model.losses_from_ports(ports,n)
        else:
            features=ports.preference_features(n)
            h,k=(model.losses_at(ports,n,features) if hasattr(model,'losses_at') else
                model.losses(features,state_sources=sources))
        holds[:,n]=h
        path=workspace/f'{n}.npz'
        with path.open('xb') as stream:np.savez_compressed(stream,k=k)
        written+=path.stat().st_size
        if written>512*1024**2:raise ValueError('transient preference workspace exceeds512MiB; no silent truncation')
        if n%40==0:print(json.dumps(dict(phase='same_event_cost_preparation',completed_events=n,
            total_events=N-2,states=q,elapsed_seconds=monotonic()-started)),flush=True)
    def cut(n):
        check()
        with np.load(workspace/f'{n}.npz',allow_pickle=False) as saved:values=saved['k']
        if values.shape!=(q,q) or not np.isfinite(values).all():raise ValueError('transient same-event loss domain differs')
        return values
    return holds,cut,dict(event_nodes=N-2,compressed_transient_bytes=written,
        whole_song_pair_tensor_in_memory=False,source_neural_calls=0,
        native_events_outside_authored_grid_measured_motion_still_unknown=True,
        evaluation_only_native_bounds_supplied_separately=True,
        native_event_finite_visibility_costs_retained=True)
