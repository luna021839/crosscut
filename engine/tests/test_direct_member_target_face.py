from __future__ import annotations

import unittest

from crosscut.direct_member_target_face import (
    FocalFaceCandidate,
    select_focal_face_matching_reference,
    select_focal_face_pair,
)


def _face(x: float, y: float, embedding: tuple[float, ...]):
    return FocalFaceCandidate((x, y, 0.12, 0.10), embedding)


class DirectMemberTargetFaceTests(unittest.TestCase):
    def test_center_prior_resolves_two_valid_same_identity_pairs(self) -> None:
        target_left = _face(0.40, 0.30, (1.0, 0.0, 0.0))
        target_right = _face(0.44, 0.28, (0.96, 0.10, 0.0))
        edge_left = _face(0.82, 0.22, (0.0, 1.0, 0.0))
        edge_right = _face(0.84, 0.24, (0.0, 0.98, 0.05))

        selected = select_focal_face_pair(
            (edge_left, target_left),
            (edge_right, target_right),
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertIs(selected.outgoing, target_left)
        self.assertIs(selected.incoming, target_right)

    def test_unrelated_faces_do_not_create_a_correspondence(self) -> None:
        selected = select_focal_face_pair(
            (_face(0.4, 0.3, (1.0, 0.0)),),
            (_face(0.4, 0.3, (0.0, 1.0)),),
        )

        self.assertIsNone(selected)

    def test_adjacent_frame_extension_uses_verified_reference(self) -> None:
        target = _face(0.42, 0.30, (1.0, 0.0, 0.0))
        selected = select_focal_face_matching_reference(
            (
                _face(0.80, 0.25, (0.0, 1.0, 0.0)),
                _face(0.43, 0.31, (0.98, 0.08, 0.0)),
            ),
            (target,),
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertLess(selected.center_xy[0], 0.60)


if __name__ == "__main__":
    unittest.main()
