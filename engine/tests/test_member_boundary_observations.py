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
    def test_slot_demands_preserve_unknowns_then_complete_the_same_source(self):
        observer, prepare = Mock(), Mock()
        slots = Mock(side_effect=lambda p, source, indices:
            {slot: (source, p.master_times[slot], p.local_times_by_source[source][slot]) for slot in indices})
        store = MemberBoundaryObservationStore((plan(), plan(2)), observe=observer,
            observe_samples=slots, prepare=prepare)
        store.prepare_samples(((0, 0, 0), (0, 0, 1)))
        first = store.requested_samples(0, 0, (0, 1))
        self.assertEqual(slots.call_args.args[2], (0, 1))
        self.assertEqual(store.observed_sample_demands, frozenset(((0, 0, 0), (0, 0, 1))))
        self.assertFalse(store.observed_demands)
        with self.assertRaises(KeyError): store.observed_samples(0, 0)
        with self.assertRaises(KeyError): store.observed_sample(0, 0, 3)
        prepare.reset_mock()
        store.prepare(((0, 0),))
        prepare.assert_called_once_with(tuple((0, time) for time in plan().local_times_by_source[0][2:]))
        full = store.samples_for(0, 0)
        self.assertEqual(full[:2], first)
        self.assertEqual(slots.call_args.args[2], (2, 3, 4))
        self.assertEqual(store.observed_demands, frozenset(((0, 0),)))
        self.assertEqual(store.observed_sample(0, 0, 3), full[3])
        self.assertIs(store.samples_for(0, 0), store.observed_samples(0, 0))
        observer.assert_not_called()

    def test_slot_failure_or_wrong_keys_cannot_publish_partial_evidence(self):
        for bad in (OSError('SSD disconnected'), {0: None}, {True: None, 0: None}, {0: None, 1: None, 2: None}):
            callback = Mock(side_effect=bad) if isinstance(bad, Exception) else Mock(return_value=bad)
            store = MemberBoundaryObservationStore((plan(),), observe=Mock(), observe_samples=callback)
            with self.assertRaises((OSError, RuntimeError)):
                store.requested_samples(0, 0, (0, 1))
            self.assertFalse(store.observed_sample_demands)
            self.assertFalse(store.observed_demands)

    def test_slot_missing_is_an_observation_but_unrequested_is_not(self):
        callback = Mock(return_value={1: None})
        store = MemberBoundaryObservationStore((plan(),), observe=Mock(), observe_samples=callback)
        self.assertEqual(store.requested_samples(0, 0, (1,)), (None,))
        self.assertIsNone(store.observed_sample(0, 0, 1))
        with self.assertRaises(KeyError): store.observed_sample(0, 0, 2)
        callback.assert_called_once()

    def test_invalid_slot_domain_fails_before_any_callback(self):
        observer, prepare, slots = Mock(), Mock(), Mock()
        store = MemberBoundaryObservationStore((plan(),), observe=observer, prepare=prepare, observe_samples=slots)
        for invalid in (True, -1, 5, 1.0, '1'):
            with self.assertRaises(ValueError): store.prepare_samples(((0, 0, 0), (0, 0, invalid)))
            with self.assertRaises(ValueError): store.requested_samples(0, 0, (0, invalid))
        observer.assert_not_called()
        prepare.assert_not_called()
        slots.assert_not_called()

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
