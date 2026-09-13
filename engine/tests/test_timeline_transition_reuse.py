"""Compare the memoized DP with its exact pre-change calculation loop."""
from dataclasses import replace
import inspect
import unittest
from unittest.mock import patch

import crosscut.timeline as module
from engine.tests.test_joint_timeline import candidate, isolated_score_config

NEW = '''                transition_key = (previous_index, current_index, previous_shot_run)
                if transition_key not in transition_cache:
                    transition_cache[transition_key] = _transition_score(
                        previous_candidate,
                        current_candidate,
                        previous_shot_run,
                        settings,
                    )
                transition_result = transition_cache[transition_key]'''
OLD = '''                transition_result = _transition_score(
                    previous_candidate,
                    current_candidate,
                    previous_shot_run,
                    settings,
                )'''


class TimelineTransitionReuseTests(unittest.TestCase):
    def test_source_history_states_reuse_exact_pair_and_shot_run_score(self):
        source = inspect.getsource(module.select_timeline)
        self.assertEqual(source.count(NEW), 1)
        original_score = module._transition_score
        for continuous_shots in (False, True):
            lattice = tuple(tuple(replace(
                candidate(i, str(s), .5 + ((i * 3 + s) % 5) * .04),
                shot_id=f'source-{s}' if continuous_shots else f'shot-{i}-{s}',
                pose_embedding=(.1, .2 + s * .01, .4),
                structure_embedding=(.2, .3, .6),
            ) for s in range(5)) for i in range(12))
            config = replace(isolated_score_config(),
                maximum_same_source_hold_segments=4, minimum_hold_segments=2,
                source_hold_penalty=.1, ping_pong_penalty=.1,
                pose_transition_weight=.4)
            counts = {'old': 0, 'new': 0}
            def counted(kind):
                def score(*args):
                    counts[kind] += 1
                    return original_score(*args)
                return score
            namespace = dict(vars(module))
            namespace['_transition_score'] = counted('old')
            exec(source.replace(NEW, OLD), namespace)
            before = namespace['select_timeline'](lattice, config=config)
            with patch.object(module, '_transition_score', counted('new')):
                after = module.select_timeline(lattice, config=config)
            self.assertEqual(before, after)
            self.assertLess(counts['new'], counts['old'])

    def test_separate_calls_do_not_reuse_a_different_configuration(self):
        lattice = tuple(tuple(candidate(i, str(s), .5) for s in range(3)) for i in range(5))
        original_score = module._transition_score
        with patch.object(module, '_transition_score', wraps=original_score) as measured:
            module.select_timeline(lattice, config=isolated_score_config())
            first = measured.call_count
            module.select_timeline(lattice, config=replace(isolated_score_config(), cut_penalty=.8))
        self.assertGreater(measured.call_count, first)
