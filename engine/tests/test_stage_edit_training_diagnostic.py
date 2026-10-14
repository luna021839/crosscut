"""Small diagnostic-instrument tests, not a substitute for the real model run."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as Item
import unittest
from unittest import mock


MODULE_PATH=Path(__file__).parents[2]/"tools/diagnose_stage_edit_training.py"
spec=importlib.util.spec_from_file_location("training_diagnostic_tool",MODULE_PATH)
diagnostic=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def tracks():
    return [Item(song_id=song,token_index=index,source_id=str(index),
        values=((1.02,.5,.5,0.),)*count, motion_label=motion,
        motion_label_loss_masked=False)
        for song in diagnostic.TRAIN_REFERENCES
        for index,count,motion in ((0,1,"static"),(1,2,"dynamic"),(2,3,"dynamic"),
            (3,2,"static"),(4,4,"dynamic"),(5,2,"static"),(6,1,"static"))]


class TrainingDiagnosticInstrumentTests(unittest.TestCase):
    def test_fixture_is_fixed_label_stratified_and_deterministic(self):
        values=tracks()
        for value in values:
            if value.motion_label=="dynamic":
                value.values=tuple((1.02+i*.01,.5,.5,0.) for i in range(len(value.values)))
        expected=diagnostic.select_fixture(values)
        actual=diagnostic.select_fixture(tuple(reversed(values)))
        self.assertEqual(expected,actual)
        self.assertEqual(len(actual),12)
        self.assertEqual({v.song_id for v in actual},set(diagnostic.TRAIN_REFERENCES))
        self.assertEqual({len(v.values) for v in actual},{1,2,3,4})

    def test_no_motion_range_fails_instead_of_moving_split(self):
        with self.assertRaisesRegex(ValueError,"no_target_trajectory_range"):
            diagnostic.select_fixture(tracks())

    def test_insufficient_real_song_data_fails(self):
        with self.assertRaisesRegex(ValueError,"insufficient real train"):
            diagnostic.select_fixture(tracks()[:4])

    def test_immutable_json_digest_is_not_optional_when_supplied(self):
        with self.assertRaisesRegex(ValueError,"digest mismatch"):
            diagnostic.read_bound_json(MODULE_PATH,"0"*64)

    def test_budget_is_bounded_without_starting_watchdog(self):
        import mlx.core as mx
        budget=diagnostic.DiagnosticBudget(Path("/unwritten"))
        budget.start-=budget.seconds+1
        with self.assertRaisesRegex(RuntimeError,"budget exceeded"):
            budget.check()

    def test_zero_tree_has_finite_norm(self):
        self.assertEqual(diagnostic.tree_norm({}),0.)


if __name__=="__main__":unittest.main()
