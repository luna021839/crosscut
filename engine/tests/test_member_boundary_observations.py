from dataclasses import replace
import unittest
from unittest.mock import Mock

from crosscut.member_boundary_observations import (
    MemberBoundaryObservationPlan, MemberBoundaryObservationStore,
)


def plan(offset=0):
    cut = 4. + offset/30.
    times = (cut-2/30., cut-1/30., cut, cut+1/30., cut+.2)
    return MemberBoundaryObservationPlan(offset, cut, times,
        tuple(tuple(time+source*.127 for time in times) for source in range(4)),
        ((True,)*5, (True, True, False, True, True), (False, False, True, True, True), (False,)*5))


class MemberBoundaryObservationTests(unittest.TestCase):
    def test_requested_pair_only_keeps_exact_local_clock_and_eligible_overlap_handles(self):
        plans = (plan(), plan(2))
        observer = Mock(side_effect=lambda p, source: tuple((source, a, b, eligible)
            for a, b, eligible in zip(p.master_times, p.local_times_by_source[source],
                                     p.eligibility_by_source[source], strict=True)))
        prepare = Mock()
        store = MemberBoundaryObservationStore(plans, observe=observer, prepare=prepare)
        store.prepare(((0, 1), (0, 2)))
        expected = tuple((source, time) for source in (1, 2)
            for time, eligible in zip(plans[0].local_times_by_source[source],
                                     plans[0].eligibility_by_source[source]) if eligible)
        prepare.assert_called_once_with(expected)
        self.assertEqual(store.observed_demands, frozenset())
        for source in (1, 2):
            samples = store.samples_for(0, source)
            self.assertEqual(tuple(item[2] for item in samples), plans[0].local_times_by_source[source])
        self.assertEqual(store.observed_demands, frozenset(((0, 1), (0, 2))))
        self.assertEqual(observer.call_count, 2)
        for offset, source in ((0, 0), (2, 1), (0, 3)):
            with self.assertRaises(KeyError): store.observed_samples(offset, source)
        self.assertEqual(observer.call_count, 2)

    def test_observed_missing_target_is_cached_but_unobserved_remains_absent(self):
        observer = Mock(return_value=(None,)*5)
        prepare = Mock()
        store = MemberBoundaryObservationStore((plan(),), observe=observer, prepare=prepare)
        self.assertEqual(store.samples_for(0, 0), (None,)*5)
        self.assertIs(store.samples_for(0, 0), store.observed_samples(0, 0))
        store.prepare(((0, 0),))
        observer.assert_called_once()
        prepare.assert_not_called()
        with self.assertRaises(KeyError): store.observed_samples(0, 1)

    def test_failure_or_partial_result_is_not_published(self):
        for failure in (OSError('SSD disconnected'), RuntimeError('cancelled'), (1, 2)):
            observer = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
            store = MemberBoundaryObservationStore((plan(),), observe=observer)
            with self.assertRaises((OSError, RuntimeError)): store.samples_for(0, 0)
            self.assertEqual(store.observed_demands, frozenset())
            observer.assert_called_once()

    def test_invalid_demands_are_validated_before_any_prepare_or_observe(self):
        observer, prepare = Mock(), Mock()
        store = MemberBoundaryObservationStore((plan(),), observe=observer, prepare=prepare)
        for invalid in ((0, True), (True, 0), (0, 4), (7, 1), (0, -1)):
            with self.assertRaises(ValueError): store.prepare(((0, 0), invalid))
            with self.assertRaises(ValueError): store.samples_for(*invalid)
        observer.assert_not_called()
        prepare.assert_not_called()

    def test_plan_validation_rejects_ambiguous_or_changed_clock_domains(self):
        original = plan()
        for change in ({'offset_frames': True}, {'cut_master_seconds': 4.1},
                       {'master_times': (3., 4., 4., 5., 6.)},
                       {'local_times_by_source': ((float('nan'),)*5,)*4},
                       {'eligibility_by_source': ((1,)*5,)*4}):
            with self.assertRaises(ValueError): replace(original, **change)
        for plans in ((), (plan(2),), (original, original),
                      (original, replace(plan(2), local_times_by_source=original.local_times_by_source[:2],
                                         eligibility_by_source=original.eligibility_by_source[:2]))):
            with self.assertRaises(ValueError): MemberBoundaryObservationStore(plans, observe=Mock())


if __name__ == '__main__': unittest.main()
