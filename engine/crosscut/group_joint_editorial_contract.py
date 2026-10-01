"""Pre-search shot obligations, separate from boundary composition intent.

This contract describes an edit, not a detector result or quality approval.
Required persistent/semantic references survive missing observations. A
frame-local detection number is never a subject reference. The declaration
API takes no candidate costs, failed edges or projected camera detections.
"""
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json

from .group_joint_source_camera import GroupOutputRole


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


class GroupBoundaryIntent(str, Enum):
    CONTINUITY = 'same_role_continuity'
    COMPOSITION_CHANGE = 'motivated_composition_change'


@dataclass(frozen=True, slots=True)
class GroupSubjectObligation:
    reference: str
    reference_basis: str
    required_parts: tuple[str, ...]

    def __post_init__(self):
        if (not _text(self.reference)
                or self.reference_basis not in {'source_native_persistent_track', 'semantic_subject'}
                or not isinstance(self.required_parts, tuple) or not self.required_parts
                or len(set(self.required_parts)) != len(self.required_parts)
                or any(part not in {'head', 'observed_arms', 'person_extent', 'formation_motion'}
                    for part in self.required_parts)):
            raise ValueError('GROUP obligation needs a persistent/semantic reference and explicit body parts')


@dataclass(frozen=True, slots=True)
class GroupShotRoleDeclaration:
    slot: int
    role: GroupOutputRole
    purpose: str
    obligations: tuple[GroupSubjectObligation, ...]
    incumbent_state_key: str
    native_evidence_sha256: str
    available_modalities: tuple[str, ...]

    def __post_init__(self):
        allowed = {
            GroupOutputRole.FOCAL_HEAD: 'measured_focal_expression_or_head_choreography',
            GroupOutputRole.FOCAL_HEAD_ARMS: 'measured_focal_head_and_arm_choreography',
            GroupOutputRole.FORMATION: 'measured_multi_person_layout_and_collective_choreography',
        }
        if (type(self.slot) is not int or not 0 <= self.slot <= 3
                or not isinstance(self.role, GroupOutputRole) or self.purpose != allowed[self.role]
                or not isinstance(self.obligations, tuple) or not 1 <= len(self.obligations) <= 12
                or any(not isinstance(item, GroupSubjectObligation) for item in self.obligations)
                or len({item.reference for item in self.obligations}) != len(self.obligations)
                or not _sha(self.incumbent_state_key) or not _sha(self.native_evidence_sha256)
                or not isinstance(self.available_modalities, tuple)
                or len(set(self.available_modalities)) != len(self.available_modalities)
                or any(name not in {'source_native_tracks', 'face_head', 'body_joints', 'formation_layout'}
                    for name in self.available_modalities)):
            raise ValueError('GROUP role needs pre-search native evidence and immutable editorial obligations')
        required = {'source_native_tracks'}
        parts = set().union(*(set(item.required_parts) for item in self.obligations))
        if self.role in {GroupOutputRole.FOCAL_HEAD, GroupOutputRole.FOCAL_HEAD_ARMS}:
            required.add('face_head')
            if len(self.obligations) != 1 or 'head' not in parts:
                raise ValueError('focal role requires exactly one retained head obligation')
        if self.role == GroupOutputRole.FOCAL_HEAD_ARMS:
            required.add('body_joints')
            if 'observed_arms' not in parts:
                raise ValueError('observed arms cannot be dropped from a focal action role')
        if self.role == GroupOutputRole.FORMATION:
            required.update(('formation_layout', 'body_joints'))
            if len(self.obligations) < 2 or any(not {'person_extent', 'formation_motion'} <= set(item.required_parts)
                    for item in self.obligations):
                raise ValueError('formation purpose needs a persistent required cohort and collective movement')
        if not required <= set(self.available_modalities):
            raise ValueError('role_unresolved: native track/head/body modalities are missing')


@dataclass(frozen=True, slots=True)
class GroupBoundaryDeclaration:
    boundary_slot: int
    intent: GroupBoundaryIntent
    reason: str
    direction: str | None = None
    subject_handoff: tuple[str, str] | None = None
    motivation_evidence_sha256: str | None = None

    def __post_init__(self):
        if (type(self.boundary_slot) is not int or not 0 <= self.boundary_slot <= 2
                or not isinstance(self.intent, GroupBoundaryIntent)):
            raise ValueError('GROUP boundary intent needs an actual adjacent window boundary')
        if self.intent == GroupBoundaryIntent.CONTINUITY:
            if (self.reason != 'retain_measured_subject_and_choreography'
                    or any(value is not None for value in (
                        self.direction, self.subject_handoff, self.motivation_evidence_sha256))):
                raise ValueError('continuity cannot smuggle a post-search composition waiver')
        elif self.intent == GroupBoundaryIntent.COMPOSITION_CHANGE:
            if (self.reason not in {'predeclared_subject_reveal', 'predeclared_subject_handoff'}
                    or self.direction not in {'focal_to_formation_reveal', 'formation_to_focal_focus', 'subject_handoff'}
                    or not isinstance(self.subject_handoff, tuple) or len(self.subject_handoff) != 2
                    or any(not _text(value) for value in self.subject_handoff)
                    or not _sha(self.motivation_evidence_sha256)):
                raise ValueError('composition change needs independent motivation, direction and subject handoff')
        else:
            raise ValueError('unknown GROUP boundary intent')


@dataclass(frozen=True, slots=True)
class GroupJointEditorialContract:
    shots: tuple[GroupShotRoleDeclaration, ...]
    boundaries: tuple[GroupBoundaryDeclaration, ...]

    def __post_init__(self):
        if (not isinstance(self.shots, tuple) or len(self.shots) != 4
                or any(not isinstance(item, GroupShotRoleDeclaration) or item.slot != index
                    for index, item in enumerate(self.shots))
                or not isinstance(self.boundaries, tuple) or len(self.boundaries) != 3
                or any(not isinstance(item, GroupBoundaryDeclaration) or item.boundary_slot != index
                    for index, item in enumerate(self.boundaries))):
            raise ValueError('GROUP editorial declaration requires four immutable roles and three intents')
        for index, edge in enumerate(self.boundaries):
            left, right = self.shots[index:index+2]
            focal = {GroupOutputRole.FOCAL_HEAD, GroupOutputRole.FOCAL_HEAD_ARMS}
            if (edge.intent == GroupBoundaryIntent.CONTINUITY and left.role != right.role
                    and not {left.role, right.role} <= focal):
                raise ValueError('different shot purposes need a predeclared composition change')
            if edge.subject_handoff is not None:
                if (edge.subject_handoff[0] not in {item.reference for item in left.obligations}
                        or edge.subject_handoff[1] not in {item.reference for item in right.obligations}):
                    raise ValueError('composition handoff references an undeclared required subject')

    def to_dict(self):
        return dict(contract='presearch_group_shot_obligations_and_boundary_intent_v1',
            **asdict(self), candidates_or_failed_edges_used_for_declaration=False,
            named_identity_or_motion_certified=False, quality_approved=False)

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'),
            allow_nan=False).encode()).hexdigest()

    def validate_states(self, states):
        from .group_joint_source_camera import GroupSourceCameraState
        for state in states:
            if (not isinstance(state, GroupSourceCameraState)
                    or state.role != self.shots[state.slot].role
                    or state.editorial_contract_sha256 != self.sha256):
                raise ValueError('joint state changed its pre-search role/subject/boundary declaration')
