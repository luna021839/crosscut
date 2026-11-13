"""Pre-camera, physical target observations for contextual editing.

This is an observation producer, not a replacement aesthetic score. Targets
are supplied by the source-clock association before any camera is applied.
Projection never runs a focal argmax, interpolates a missing face, borrows a
competitor's body, or turns a track hypothesis into an ownership certificate.
The arrays are shared immutable inputs, not copies to retain in DP labels.
"""
from dataclasses import dataclass
import numpy as np

UNKNOWN, FEATURE, FORMATION = range(3)
SCHEMA = 'pre_camera_target_scene_v1'


def _frozen(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TargetScene:
    """One source-clock window with already-associated target slots.

    FEATURE has exactly one target slot; its observations may be missing.
    FORMATION is the observed set, not a requirement to show all six people.
    UNKNOWN may retain raw observed faces but asserts no chosen actor/role.
    `head_observed` is independent of ROI/body observation validity. Neither a
    missing ROI nor another visible face can change a FEATURE's target slot.
    Binding/track keys are association provenance, never preference features.
    """
    binding: str
    track: str
    frames: np.ndarray
    native_indices: np.ndarray
    role: int
    heads: np.ndarray
    head_observed: np.ndarray
    body: np.ndarray
    body_evidence: np.ndarray
    body_ambiguity: np.ndarray
    roi_readability: np.ndarray
    roi_measured: np.ndarray
    original_heads: np.ndarray
    original_observed: np.ndarray

    def __post_init__(self):
        frames = np.asarray(self.frames)
        native = np.asarray(self.native_indices)
        heads = np.asarray(self.heads)
        if (not isinstance(self.binding, str) or not self.binding
                or not isinstance(self.track, str) or not self.track
                or type(self.role) is not int or self.role not in (UNKNOWN, FEATURE, FORMATION)
                or frames.ndim != 1 or not len(frames) or frames.dtype.kind not in 'iu'
                or frames.min() < 0 or np.any(np.diff(frames) <= 0)
                or native.shape != frames.shape or native.dtype.kind not in 'iu'
                or native.min() < 0 or np.any(np.diff(native) < 0)
                or heads.ndim != 3 or heads.shape[0] != len(frames) or heads.shape[2] != 4
                or heads.shape[1] < 1
                or (self.role == FEATURE and heads.shape[1] != 1)):
            raise ValueError('pre-camera role/targets and chronological physical clocks required')
        shape = heads.shape[:2]
        specs = dict(head_observed=shape, body=shape+(8, 3), body_evidence=shape+(8,),
            body_ambiguity=shape+(8,), roi_readability=shape, roi_measured=shape)
        for name, expected in specs.items():
            value = np.asarray(getattr(self, name))
            if value.shape != expected:
                raise ValueError('target-bound '+name+' shape differs')
            if name in ('head_observed', 'body_ambiguity', 'roi_measured'):
                if value.dtype != bool:
                    raise ValueError('explicit target validity masks required')
            elif name == 'body_evidence':
                if value.dtype.kind not in 'iu' or np.any((value < 0) | (value > 3)):
                    raise ValueError('actual partial-pose evidence kinds required')
            elif value.dtype.kind not in 'iuf' or not np.isfinite(value).all():
                raise ValueError('finite target observations required')
        original = np.asarray(self.original_heads)
        original_known = np.asarray(self.original_observed)
        if (original.ndim != 3 or original.shape[0] != len(frames) or original.shape[2] != 4
                or original.shape[1] < 1 or not np.isfinite(original).all()
                or original_known.shape != original.shape[:2] or original_known.dtype != bool
                or np.any(original_known & ((original[..., 2] <= .019) | (original[..., 3] < .75)))):
            raise ValueError('the uncropped original observed set must remain separately bound')
        if (heads.dtype.kind not in 'iuf' or not np.isfinite(heads).all()
                or np.any(np.asarray(self.head_observed) & ((heads[..., 2] <= .019) | (heads[..., 3] < .75)))
                or np.any((np.asarray(self.roi_readability) < 0) | (np.asarray(self.roi_readability) > 1))
                or np.any(np.asarray(self.roi_measured) & ~np.asarray(self.head_observed))):
            raise ValueError('actual visible original heads and bounded native ROI required')
        for name in ('frames', 'native_indices', 'heads', *specs, 'original_heads', 'original_observed'):
            object.__setattr__(self, name, _frozen(getattr(self, name)))

    def project(self, cameras):
        """Project the SAME targets; known crop-out does not become UNKNOWN.

        `cameras` is (..., observed_time, 3); arbitrary camera batches share
        this one physical observation. ROI detail is native, never enlarged.
        Returned scene validity does not certify dense or anatomical tracking.
        """
        camera = np.asarray(cameras, float)
        if (camera.ndim < 2 or camera.shape[-2:] != (len(self.frames), 3) or not np.isfinite(camera).all()
                or np.any(camera[..., 0] < 1.)):
            raise ValueError('finite exact camera paths on the target sample clock required')
        z = camera[..., None, :1]
        center = .5 + z * (self.heads[..., :2] - camera[..., None, 1:])
        height = self.heads[..., 2] * camera[..., None, 0]
        half = height[..., None] * [.45*9/16, .5]
        extent = np.maximum(0., np.minimum(1., center+half)-np.maximum(0., center-half))
        fraction = np.prod(extent/np.maximum(1e-12, 2*half), axis=-1)
        observed = np.broadcast_to(self.head_observed, height.shape)
        fraction = np.where(observed, fraction, 0.)
        body_center = .5 + camera[..., None, None, :1]*(self.body[..., :2]-camera[..., None, None, 1:])
        from .physical_display_observations import MODEL_ONLY, PIXEL_SUPPORTED
        raw = self.body
        body_known = ((self.body_evidence == MODEL_ONLY) | (self.body_evidence == PIXEL_SUPPORTED))
        body_known &= (~self.body_ambiguity & (raw[..., 2] >= .3)
            & (raw[..., :2] >= 0).all(-1) & (raw[..., :2] <= 1).all(-1))
        body_known = np.broadcast_to(body_known, body_center.shape[:-1])
        return dict(schema=SCHEMA, role=self.role, centers=center, heights=height,
            head_observed=observed, visible_fraction=fraction,
            known_crop_out=observed & (fraction < 1.-1e-9),
            roi_readability=np.broadcast_to(self.roi_readability, height.shape),
            roi_measured=np.broadcast_to(self.roi_measured, height.shape),
            body_centers=body_center, body_observed=body_known,
            body_known_crop_out=body_known & ((body_center < 0).any(-1) | (body_center > 1).any(-1)),
            original_observed=self.original_observed,
            original_heads=self.original_heads,
            role_known=self.role != UNKNOWN, target_reselected_after_camera=False,
            anatomical_ownership_certified=False, dense_face_track_certified=False)


def from_boundary_observation(observation, *, frame, source_binding):
    """Retain the physical subject already pinned by the boundary producer.

    Dense windows use their own actual face/body and exact native schedule.
    Sparse context has actual face rasters only at -1/0: the repeated head
    proxies at -7/+6 must NOT masquerade as measured face trajectories. The
    existing contemporaneous body observations keep their separate evidence.
    No best-face retrieval or new source NN/decode is performed here.
    """
    if type(frame) is not int or frame < 0:
        raise ValueError('canonical boundary frame required')
    if not observation.get('target_pinned_before_camera'):
        raise ValueError('bind a physical target before proposing its camera')
    if observation.get('source_clock_canonical_frame') != frame:
        raise ValueError('target canonical boundary clock changed')
    dense = observation.get('dense_subject_context')
    if dense is not None:
        window, clock, at = dense['window'], dense['clock'], dense['at']
        if window.binding != source_binding:
            raise ValueError('target physical source binding changed')
        frames = frame + clock.output_frames
        heads = window.heads[at, None]
        head_known = (window.associated[at] & dense['same_shot'])[:, None]
        body = window.body[at, None]
        kinds = np.where(dense['supported'][:, None], 1, 0).astype(np.uint8)
        ambiguity = np.zeros(kinds.shape, bool)
        native = clock.native_indices
        track = window.track
        original = np.zeros((len(at), max(1+len(window.competitors[j]) for j in at), 4))
        for i, j in enumerate(at):
            others = window.competitors[j]
            original[i, 0] = window.heads[j]
            original[i, 1:1+len(others)] = others
        original_known = (original[..., 2] > .019) & (original[..., 3] >= .75)
        quality = np.full(head_known.shape, .5)
        quality_known = np.zeros(head_known.shape, bool)
    else:
        frames = observation.get('source_clock_output_frames')
        if frames is None:
            raise ValueError('sparse target needs the actual output sample clock')
        frames = np.asarray(frames)
        heads = np.asarray(observation.get('source_clock_endpoint_head_proxies', np.zeros((4, 4))))[:, None]
        sample = np.asarray(observation.get('source_clock_observed_sample_mask', np.zeros(4, bool)), bool)
        # These are face proxies at the outer body flanks, not face rasters.
        head_known = (sample & np.array([False, True, True, False]))[:, None]
        body = np.asarray(observation.get('source_clock_body', np.zeros((4, 8, 3))))[:, None]
        kinds = np.asarray(observation.get('source_clock_evidence', np.zeros((4, 8), np.uint8)))[:, None]
        ambiguity = np.asarray(observation.get('source_clock_ambiguity', np.zeros((4, 8), bool)))[:, None]
        if not observation.get('source_clock_subject_associated', False):
            kinds = np.zeros(kinds.shape, np.uint8)
        # The four physical native ordinals must come from the caller's raw
        # observation, not a guessed output/native FPS equivalence.
        native = observation.get('source_clock_native_indices')
        if native is None:
            raise ValueError('sparse target needs all four actual native ordinals')
        subject_key = observation.get('source_clock_subject_key')
        if subject_key is None:
            raise ValueError('sparse context needs its actual endpoint subject association')
        track = 'sparse-associated-boundary-'+repr(subject_key)
        original = np.asarray(observation.get('source_clock_original_heads', np.zeros((4, 8, 4))))
        original_known = (original[..., 2] > .019) & (original[..., 3] >= .75)
        quality = np.asarray(observation.get('source_clock_native_roi_readability', np.full(4, .5)))[:, None]
        quality_known = np.asarray(observation.get('source_clock_native_roi_measured', np.zeros(4, bool)))[:, None]
        quality_known = quality_known & head_known
    final_frame=observation.get('source_clock_final_frame')
    if type(final_frame) is not int or final_frame<1:
        raise ValueError('actual output domain required for target samples')
    inside=(frames>=0)&(frames<final_frame)
    if not inside.all():
        frames=frames[inside];native=np.asarray(native)[inside]
        heads,head_known,body,kinds,ambiguity,original,original_known,quality,quality_known=(v[inside]
            for v in (heads,head_known,body,kinds,ambiguity,original,original_known,quality,quality_known))
    if not len(frames):raise ValueError('target has no observations inside the actual output domain')
    # Clipped endpoint schedules may repeat an output sample. Keep exactly
    # one physical observation. Conflicting duplicate measurements become
    # unknown, not several independent samples or a convenient winner.
    unique = np.unique(frames)
    if len(unique) != len(frames):
        chosen = []
        heads, head_known, kinds = heads.copy(), head_known.copy(), kinds.copy()
        for tick in unique:
            positions = np.flatnonzero(frames == tick)
            first = int(positions[0])
            if len(set(np.asarray(native)[positions].tolist())) != 1:
                raise ValueError('same output sample has conflicting native ordinals')
            actual = positions[head_known[positions, 0]]
            if len(actual):
                first = int(actual[0])
                if not np.all(heads[actual] == heads[first]):
                    head_known[first] = False
            if not np.all(body[positions] == body[first]):
                kinds[first] = 0
            if not np.all(original[positions] == original[first]):
                original_known[first] = False
            chosen.append(first)
        frames = frames[chosen]
        native = np.asarray(native)[chosen]
        heads, head_known, body, kinds, ambiguity, original, original_known, quality, quality_known = (v[chosen]
            for v in (heads, head_known, body, kinds, ambiguity, original, original_known, quality, quality_known))
    return TargetScene(source_binding, track, frames, np.asarray(native), FEATURE,
        heads, head_known, body, kinds, ambiguity, quality, quality_known & head_known, original, original_known)


def from_original_view(catalogue, n, side, *, source_binding):
    """Freeze original FEATURE/FORMATION/UNKNOWN before candidate cameras.

    Reuse the existing native-view scene interpretation, including ambiguous
    detections. An UNKNOWN two-person view is not resolved by a tiny face-size
    tie or by the crop later deleting a competitor. This is not singer truth.
    A paired match may instead explicitly bind its proposed physical subject
    with from_boundary_observation; the two origins must not be conflated.
    """
    if (type(n) is not int or not 0 <= n < len(catalogue.nodes)
            or type(side) is not int or side not in (0, 1)):
        raise ValueError('actual original grid node and side required')
    from .physical_display_observations import focal_metadata
    raw = catalogue.bank['heads'][n:n+1, side:side+1]
    _, indices, _, kinds = focal_metadata(raw, np.array([[[1., .5, .5]]]))
    kind = int(kinds[0, 0])
    frame = int(catalogue.nodes[n])
    camera = np.tile([1., .5, .5], (7, 1))
    if kind == 1:
        observed = catalogue.observation(n, side, camera, target_head_index=int(indices[0, 0]))
        return from_boundary_observation(observed, frame=frame, source_binding=source_binding)
    found = np.flatnonzero((raw[0, 0, :, 2] > .019) & (raw[0, 0, :, 3] >= .75))
    if len(found):
        # Every anchor's own sparse common-clock association is retained.
        # Do not join dense windows with different clocks by array position.
        scenes = [from_boundary_observation(catalogue.observation(n, side, camera,
            target_head_index=int(h), include_subjects=False), frame=frame, source_binding=source_binding)
            for h in found]
        first = scenes[0]
        if any(not np.array_equal(first.frames, s.frames)
                or not np.array_equal(first.native_indices, s.native_indices) for s in scenes[1:]):
            raise ValueError('formation observations disagree on the physical clock')
        target_fields = ('heads', 'head_observed', 'body', 'body_evidence', 'body_ambiguity',
            'roi_readability', 'roi_measured')
        # An observed set is unordered. Actual endpoint correspondence keys
        # can establish the SAME set despite detector-order changes; a side
        # label alone must not make those observations conflict. Missing
        # associations still yield distinct hypotheses, not forced matches.
        set_key = tuple(sorted(s.track for s in scenes))
        return TargetScene(source_binding, 'original-observed-set-'+repr(set_key),
            first.frames, first.native_indices, FORMATION if kind == 2 else UNKNOWN,
            **{name: np.concatenate([getattr(s, name) for s in scenes], axis=1) for name in target_fields},
            original_heads=first.original_heads, original_observed=first.original_observed)
    # A physically observed source without detected faces remains UNKNOWN;
    # it is not a missing source, an empty film, or a fabricated lower-body shot.
    frames = np.clip(frame+np.array([-7, -1, 0, 6]), 0, catalogue.final-1)
    unique, at = np.unique(frames, return_index=True)
    t = len(unique)
    original = np.zeros((4, 8, 4))
    original[[1, 2]] = catalogue.bank['heads'][n]
    original = original[at]
    return TargetScene(source_binding, 'original-unknown-'+str(frame)+'-'+str(side), unique,
        catalogue.native[n, at], UNKNOWN, np.zeros((t, 1, 4)), np.zeros((t, 1), bool),
        np.zeros((t, 1, 8, 3)), np.zeros((t, 1, 8), np.uint8), np.zeros((t, 1, 8), bool),
        np.full((t, 1), .5), np.zeros((t, 1), bool), original,
        (original[..., 2] > .019) & (original[..., 3] >= .75))
