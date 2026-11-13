from __future__ import annotations

from types import SimpleNamespace
import unittest

from crosscut.pipeline import _direct_member_highlight_cadence_payload


def _selection(moving_boundary_indices: set[int]) -> SimpleNamespace:
    boundaries = tuple(
        SimpleNamespace(
            cut_applied=True,
            mismatch=SimpleNamespace(
                motion_connection=SimpleNamespace(
                    both_moving=index in moving_boundary_indices,
                )
            ),
        )
        for index in range(9)
    )
    return SimpleNamespace(
        source_indices=tuple(index % 3 for index in range(10)),
        boundaries=boundaries,
    )


def _spans() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            canonical_master_start=float(index * 2),
            canonical_master_end=float((index + 1) * 2),
        )
        for index in range(10)
    )


class DirectMemberHighlightMotionCadenceTests(unittest.TestCase):
    def test_highlight_requires_a_material_share_of_moving_pose_cuts(self) -> None:
        payload = _direct_member_highlight_cadence_payload(
            spans=_spans(),
            visual_selection=_selection({1, 4, 7}),
            highlight_intervals=((0.0, 20.0),),
        )

        self.assertTrue(payload["passed"], payload["failure_reasons"])
        self.assertEqual(payload["actual_highlight_cut_count"], 9)
        self.assertEqual(payload["actual_highlight_moving_cut_count"], 3)
        self.assertAlmostEqual(payload["highlight_moving_cut_fraction"], 1 / 3)
        self.assertEqual(payload["highlight_moving_cuts_per_minute"], 9.0)

    def test_static_highlight_stitching_fails_even_when_cut_rate_is_high(self) -> None:
        payload = _direct_member_highlight_cadence_payload(
            spans=_spans(),
            visual_selection=_selection({4}),
            highlight_intervals=((0.0, 20.0),),
        )

        self.assertFalse(payload["passed"])
        self.assertIn(
            "highlight_moving_cut_fraction_below_minimum",
            payload["failure_reasons"],
        )
        self.assertIn(
            "highlight_moving_cut_density_below_minimum",
            payload["failure_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
