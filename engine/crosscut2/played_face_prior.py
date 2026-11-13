"""Explicit untrained D: played-face geometry, unknown finite, no min(H) gate.

One declared engineering prior, NOT learned editorial preference or a motion
match claim. C is preserved as a rejected comparator. The new CLI is opt-in.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np
from .common_evidence_prior import CommonEvidencePrior
from .played_face_connection import extract
from .native_fragment_integrity import NativeFragmentIntegrity


class PlayedFacePrior(CommonEvidencePrior):
    played_face_prior=True

    def __init__(self):
        super().__init__()
        self.scale_limit=math.log(1.5);self.position_limit=1.;self.unknown_cost=1.
        self.artifact.update(contract='declared_played_face_geometry_D_v1',
            admission='different-source and no measured played-face violation; unknown allowed',
            authored_cost='played_pair_support*max(abs(log(heightA/heightB))/log(1.5),displacement_in_face_heights/1.0); unknown=1',
            maximum_measured_face_height_ratio=1.5,maximum_measured_displacement_in_face_heights=1.,
            unknown_face_cost=1.,hold_baseline_cannot_discount_connection=True,
            authored_native_minimum_fragment_frames=12,original_file_endpoint_fragments_exempt=True,
            actual_played_endpoints_only=True,physical_group_focal_not_selected_actor_label=True,
            measured_body_action_cost_used=False,pair_conditioned_cameras_added=False,
            parameters_predeclared_not_human_calibrated=True)
        for key in ('same_clock_raw_encoder_preserved','unsupported_pose_prior_symmetric',
                'raw_model_only_or_contradicted_pose_is_not_hard_veto','formula','residuals'):
            self.artifact.pop(key,None)
        paths=[Path(__file__),Path(__file__).with_name('played_face_connection.py'),
            Path(__file__).with_name('native_fragment_integrity.py'),Path(__file__).with_name('event_states.py')]
        self.artifact['implementation_sha256']=hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()
        self.artifact.pop('content_sha256',None)
        self.artifact['content_sha256']=hashlib.sha256(json.dumps(self.artifact,sort_keys=True).encode()).hexdigest()

    def bind(self,p,*,native_ports,source_events):
        receipt=super().bind(p,native_ports=native_ports,source_events=source_events)
        self.integrity=NativeFragmentIntegrity(self.support,minimum_frames=12)
        return dict(receipt,minimum_authored_native_fragment_frames=12,
            integrity_applies_only_to_authored_truncation_of_known_native_shots=True)

    def hold_allowed(self,first,final):
        return self.integrity.allowed(int(self.nodes[first]),int(self.nodes[final]),self.sources)

    def costs_and_mask(self,p,n):
        face=extract(p,n);measured=face['measured']
        error=np.maximum(face['log_scale']/self.scale_limit,
            face['face_height_displacement']/self.position_limit)
        allowed=(self.sources[:,None]!=self.sources[None,:])&(~measured|(error<=1.))
        rates=np.where(measured,np.minimum(1.,error),self.unknown_cost)
        self.statistics.append(dict(frame=int(p.nodes[n]),allowed_state_pairs=int(allowed.sum()),
            measured_face_pairs=int(measured.sum()),measured_violation_pairs=int((measured&(error>1.)).sum()),
            unknown_is_not_measured_agreement=True))
        return self.support.cut_weights(int(p.nodes[n]),self.sources)*rates,allowed
