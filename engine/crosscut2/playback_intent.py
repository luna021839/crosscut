"""Immutable source-local intent, bound before a proposed camera is read.

This is a finite observed association window, not a singer plan or a dense
track. Its gaps stay gaps; neither camera visibility nor a competitor can
choose a different target. Content binding is supplied by the real caller.
"""
from dataclasses import dataclass, replace
import hashlib
import numpy as np
from .scene_targets import from_boundary_observation, from_original_view

CONTRACT = 'pre_camera_port_intent_exact_native_window_v1'
ARRAYS=('native_indices','heads','head_observed','body','body_evidence','body_ambiguity',
    'roi_readability','roi_measured','original_heads','original_observed')


@dataclass(frozen=True, eq=False)
class PortIntent:
    source: int
    anchor: int
    side: int
    pinned: bool
    scene: object
    first_native: int
    final_native: int

    def bind(self, binding):
        if not self.scene.binding.startswith('unbound-catalogue-source-') and self.scene.binding!=binding:
            raise ValueError('pre-camera intent physical source binding changed')
        scene = replace(self.scene, binding=binding)
        h = hashlib.sha256(CONTRACT.encode()+binding.encode()+scene.track.encode())
        h.update(np.asarray([scene.role, self.first_native, self.final_native], np.int64).tobytes())
        for name in ARRAYS:
            value = np.asarray(getattr(scene, name))
            h.update(name.encode()+value.dtype.str.encode()+repr(value.shape).encode()+value.tobytes())
        return replace(self, scene=scene), h.hexdigest()

    def same_input(self,other):
        return (isinstance(other,PortIntent)
            and (self.first_native,self.final_native,self.scene.binding,self.scene.track,self.scene.role)==
                (other.first_native,other.final_native,other.scene.binding,other.scene.track,other.scene.role)
            and all(getattr(self.scene,n).dtype==getattr(other.scene,n).dtype
                and np.array_equal(getattr(self.scene,n),getattr(other.scene,n)) for n in ARRAYS))


def before_camera(catalogue, source, node, side, head_index):
    """Only original-image observations can establish the target contract."""
    binding = (catalogue.subjects.binding if catalogue.subjects is not None
        else 'unbound-catalogue-source-'+str(source))
    dense_cuts=()
    if head_index is None:
        scene = from_original_view(catalogue, node, side, source_binding=binding)
    else:
        native = catalogue.observation(node, side, np.tile([1., .5, .5], (7, 1)),
            target_head_index=head_index)
        dense=native.get('dense_subject_context')
        if dense is not None:dense_cuts=dense['window'].native_cuts
        scene = from_boundary_observation(native, frame=int(catalogue.nodes[node]), source_binding=binding)
    # Do not extend an anchor's identity through an actual broadcast cut.
    endpoint = int(catalogue.native[node, 1 if side == 0 else 2])
    cuts = np.unique(np.r_[catalogue.bank.get('native_cut_frames', []),dense_cuts]).astype(np.int64)
    shot = np.searchsorted(cuts, endpoint, side='right')
    same = np.searchsorted(cuts, scene.native_indices, side='right') == shot
    native = scene.native_indices[same]
    if not len(native):
        raise ValueError('pre-camera intent lost its actual source-shot anchor')
    return PortIntent(source, int(catalogue.nodes[node]), side, head_index is not None,
        scene, int(native.min()), int(native.max())+1)
