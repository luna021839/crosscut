from dataclasses import replace
import hashlib
import json
import unittest

from crosscut.group_joint_editorial_contract import (GroupSubjectObligation, GroupShotRoleDeclaration,
    GroupBoundaryDeclaration, GroupBoundaryIntent, GroupJointEditorialContract)
from crosscut.group_joint_source_camera import GroupOutputRole, select_joint_group_source_camera_path
from engine.tests.test_group_joint_source_camera import JointGroupSourceCameraTests
from engine.tests.test_group_joint_state_factory import JointStateFactoryTests
from crosscut.group_joint_state_factory import candidate_joint_group_state, assemble_joint_group_states


def focal(slot, state_key, *, arms=False):
    return GroupShotRoleDeclaration(slot, GroupOutputRole.FOCAL_HEAD_ARMS if arms else GroupOutputRole.FOCAL_HEAD,
        'measured_focal_head_and_arm_choreography' if arms else 'measured_focal_expression_or_head_choreography',
        (GroupSubjectObligation(f'native-track-{slot}', 'source_native_persistent_track',
            ('head', 'observed_arms') if arms else ('head',)),), state_key, 'b'*64,
        ('source_native_tracks', 'face_head', 'body_joints'))


def declaration(states):
    shots = []
    for i, state in enumerate(states):
        if state.role == GroupOutputRole.FORMATION:
            shots.append(GroupShotRoleDeclaration(i, state.role,
                'measured_multi_person_layout_and_collective_choreography',
                tuple(GroupSubjectObligation(f'native-track-{i}-{n}', 'source_native_persistent_track',
                    ('person_extent', 'formation_motion')) for n in range(3)), state.key, 'b'*64,
                ('source_native_tracks', 'body_joints', 'formation_layout')))
        else:
            shots.append(focal(i, state.key, arms=state.role == GroupOutputRole.FOCAL_HEAD_ARMS))
    return GroupJointEditorialContract(tuple(shots),
        tuple(GroupBoundaryDeclaration(i, GroupBoundaryIntent.CONTINUITY,
            'retain_measured_subject_and_choreography') for i in range(3)))


class GroupJointEditorialContractTests(unittest.TestCase):
    def test_contract_is_immutable_replayable_and_not_a_quality_certificate(self):
        _, states, _, _ = JointGroupSourceCameraTests().fixture()
        contract = declaration(states)
        payload = json.loads(json.dumps(contract.to_dict()))
        self.assertEqual(hashlib.sha256(json.dumps(payload, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(), contract.sha256)
        self.assertFalse(payload['candidates_or_failed_edges_used_for_declaration'])
        self.assertFalse(payload['quality_approved'])
        with self.assertRaises(AttributeError):
            contract.shots[0].purpose = 'search_passed'

    def test_body_only_evidence_cannot_approve_a_head_role_and_arms_cannot_be_downgraded(self):
        role = focal(1, 'a'*64, arms=True)
        for extra in ({'available_modalities': ('source_native_tracks', 'body_joints', 'formation_layout')},
                {'obligations': (replace(role.obligations[0], required_parts=('head',)),)},
                {'purpose': 'chorus_or_failed_formation_match'}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                replace(role, **extra)
        with self.assertRaises(ValueError):
            GroupSubjectObligation(0, 'source_native_persistent_track', ('head',))

    def test_role_change_or_new_hash_rejects_before_any_graph_observation(self):
        args, states, b, c = JointGroupSourceCameraTests().fixture()
        contract = declaration(states)
        bound = tuple(replace(s, editorial_contract_sha256=contract.sha256) for s in states)
        args.update(left=bound[0], right=bound[3],
            states_by_slot=((replace(b, editorial_contract_sha256=contract.sha256),),
                (replace(c, editorial_contract_sha256=contract.sha256),)), editorial_contract=contract)
        calls = []
        original = args['observe_edge']
        args['observe_edge'] = lambda *a: calls.append(a) or original(*a)
        wrong = replace(args['states_by_slot'][0][0], editorial_contract_sha256='c'*64)
        args['states_by_slot'] = ((wrong,), args['states_by_slot'][1])
        with self.assertRaisesRegex(ValueError, 'pre-search role/subject/boundary'):
            select_joint_group_source_camera_path(**args)
        self.assertEqual(calls, [])

    def test_same_purpose_and_motivated_composition_change_are_not_interchangeable(self):
        _, states, _, _ = JointGroupSourceCameraTests().fixture()
        contract = declaration(states)
        subjects = tuple(GroupSubjectObligation(f'formation-track-{i}', 'source_native_persistent_track',
            ('person_extent', 'formation_motion')) for i in range(3))
        formation = GroupShotRoleDeclaration(1, GroupOutputRole.FORMATION,
            'measured_multi_person_layout_and_collective_choreography', subjects, states[1].key, 'c'*64,
            ('source_native_tracks', 'body_joints', 'formation_layout'))
        with self.assertRaisesRegex(ValueError, 'composition change'):
            replace(contract, shots=(contract.shots[0], formation, *contract.shots[2:]))
        reveal = GroupBoundaryDeclaration(0, GroupBoundaryIntent.COMPOSITION_CHANGE,
            'predeclared_subject_reveal', 'focal_to_formation_reveal',
            ('native-track-0', 'formation-track-0'), 'd'*64)
        focus = GroupBoundaryDeclaration(1, GroupBoundaryIntent.COMPOSITION_CHANGE,
            'predeclared_subject_reveal', 'formation_to_focal_focus',
            ('formation-track-0', 'native-track-2'), 'e'*64)
        mixed = replace(contract, shots=(contract.shots[0], formation, *contract.shots[2:]),
            boundaries=(reveal, focus, contract.boundaries[2]))
        self.assertNotEqual(mixed.sha256, contract.sha256)
        with self.assertRaises(ValueError):
            replace(mixed, boundaries=(replace(reveal, subject_handoff=('undeclared', 'formation-track-0')),
                *mixed.boundaries[1:]))

    def test_numeric_factory_and_assembly_keep_the_frozen_contract_and_full_hash_preimage(self):
        plan, domain, states = JointStateFactoryTests().fixture()
        contract = declaration(states)
        bound = tuple(replace(state, editorial_contract_sha256=contract.sha256) for state in states)
        b = candidate_joint_group_state(plan=plan, domain=domain, incumbent_state=bound[1], eligible_source_index=2)
        c = candidate_joint_group_state(plan=plan, domain=domain, incumbent_state=bound[2], eligible_source_index=1)
        proposed, changed = assemble_joint_group_states(plan=plan, domain=domain, path=(bound[0], b, c, bound[3]))
        self.assertEqual(changed, (1, 2))
        contract.validate_states((bound[0], b, c, bound[3]))
        for state in (b, c):
            payload = state.to_dict()
            self.assertEqual(hashlib.sha256(json.dumps(payload['key_preimage'], sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest(), payload['state_key'])
            self.assertEqual(len(payload['key_preimage']['native']), state.end_output_frame-state.first_output_frame)
            self.assertEqual(payload['key_preimage']['segments'][0]['source_index'],
                proposed.segments[state.slot].source_index)
