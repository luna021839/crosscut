from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
import unittest

from crosscut.no_good_selector_identity import (
    NO_GOOD_SELECTOR_IDENTITY_CONTRACT,
    NoGoodSelectorIdentity,
    build_no_good_selector_identity,
)
from crosscut.timeline_no_good import (
    TIMELINE_NO_GOOD_CONTRACT,
    TimelineNoGoodEdgeKey,
    TimelineNoGoodSet,
)


def _edge(
    boundary_index: int = 7,
    *,
    incoming_shot_id: str = "5:shot-8",
) -> TimelineNoGoodEdgeKey:
    return TimelineNoGoodEdgeKey(
        boundary_index=boundary_index,
        outgoing_source_id="2",
        outgoing_shot_id="2:shot-3",
        incoming_source_id="5",
        incoming_shot_id=incoming_shot_id,
    )


def _identity(
    *,
    base: str = "a" * 64,
    basis: str = "visual_candidate_restart_input_identity_v1",
    no_goods: TimelineNoGoodSet | None = None,
) -> NoGoodSelectorIdentity:
    return build_no_good_selector_identity(
        base_candidate_input_identity_sha256=base,
        base_identity_basis=basis,
        timeline_no_goods=(
            TimelineNoGoodSet() if no_goods is None else no_goods
        ),
    )


class NoGoodSelectorIdentityTests(unittest.TestCase):
    def test_empty_set_is_explicit_auditable_and_distinct_from_base(self) -> None:
        identity = _identity()
        payload = identity.effective_selector_identity

        self.assertEqual(payload["contract"], NO_GOOD_SELECTOR_IDENTITY_CONTRACT)
        self.assertEqual(
            payload["base_candidate_input_identity_sha256"],
            "a" * 64,
        )
        self.assertEqual(
            payload["timeline_no_goods"],
            {
                "contract": TIMELINE_NO_GOOD_CONTRACT,
                "edge_count": 0,
                "empty": True,
                "fingerprint": TimelineNoGoodSet().fingerprint,
            },
        )
        self.assertNotEqual(identity.effective_selector_identity_sha256, "a" * 64)
        self.assertEqual(len(identity.effective_selector_identity_sha256), 64)

    def test_sha256_is_canonical_and_to_dict_is_self_verifiable(self) -> None:
        no_goods = TimelineNoGoodSet((_edge(8), _edge(2), _edge(8)))
        identity = _identity(no_goods=no_goods)
        encoded = json.dumps(
            identity.effective_selector_identity,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        self.assertEqual(
            identity.effective_selector_identity_sha256,
            hashlib.sha256(encoded).hexdigest(),
        )
        self.assertEqual(
            identity.to_dict(),
            {
                "effective_selector_identity": (
                    identity.effective_selector_identity
                ),
                "timeline_no_goods": no_goods.to_dict(),
                "effective_selector_identity_sha256": (
                    identity.effective_selector_identity_sha256
                ),
            },
        )
        self.assertEqual(
            NoGoodSelectorIdentity.from_dict(identity.to_dict()),
            identity,
        )
        tampered = identity.to_dict()
        tampered["effective_selector_identity_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            NoGoodSelectorIdentity.from_dict(tampered)

    def test_edge_order_and_duplicates_do_not_change_identity(self) -> None:
        edges = (_edge(9), _edge(3))
        forward = _identity(
            no_goods=TimelineNoGoodSet((edges[0], edges[1], edges[0]))
        )
        reverse = _identity(
            no_goods=TimelineNoGoodSet(tuple(reversed(edges)))
        )

        self.assertEqual(
            forward.effective_selector_identity,
            reverse.effective_selector_identity,
        )
        self.assertEqual(
            forward.effective_selector_identity_sha256,
            reverse.effective_selector_identity_sha256,
        )

    def test_edge_basis_and_base_each_change_effective_identity(self) -> None:
        base = _identity(no_goods=TimelineNoGoodSet((_edge(),)))
        variants = (
            _identity(
                no_goods=TimelineNoGoodSet(
                    (_edge(incoming_shot_id="5:shot-9"),)
                )
            ),
            _identity(
                basis="visual_candidate_restart_input_identity_v2",
                no_goods=TimelineNoGoodSet((_edge(),)),
            ),
            _identity(base="b" * 64, no_goods=TimelineNoGoodSet((_edge(),))),
        )

        self.assertEqual(
            len(
                {
                    base.effective_selector_identity_sha256,
                    *(item.effective_selector_identity_sha256 for item in variants),
                }
            ),
            4,
        )

    def test_validation_is_strict_and_instance_is_frozen(self) -> None:
        for invalid_base in (
            "A" * 64,
            "a" * 63,
            "g" * 64,
            " a" * 32,
            7,
            None,
        ):
            with self.subTest(base=invalid_base):
                with self.assertRaises((TypeError, ValueError)):
                    _identity(base=invalid_base)  # type: ignore[arg-type]

        for invalid_basis in ("", "   ", " padded", "padded ", 7, None):
            with self.subTest(basis=invalid_basis):
                with self.assertRaises((TypeError, ValueError)):
                    _identity(basis=invalid_basis)  # type: ignore[arg-type]

        for invalid_no_goods in ((), (_edge(),), object(), None):
            with self.subTest(no_goods=invalid_no_goods):
                with self.assertRaises(TypeError):
                    build_no_good_selector_identity(
                        base_candidate_input_identity_sha256="a" * 64,
                        base_identity_basis="basis-v1",
                        timeline_no_goods=invalid_no_goods,  # type: ignore[arg-type]
                    )

        identity = _identity()
        with self.assertRaises(FrozenInstanceError):
            identity.base_identity_basis = "changed"  # type: ignore[misc]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
