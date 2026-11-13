from dataclasses import fields,replace
import math
import unittest
from unittest.mock import patch

from crosscut.pose import FormationPersonDescriptor
from crosscut.shot_proximity import observed_shot_proximity,source_path_proximity_summary
from crosscut.timeline import TimelineCandidate,TimelineConfig,TimelineMode,ShotType,_candidate_emission,select_timeline


def person(height):
    return FormationPersonDescriptor((.4,.1,.1,height),(.45,.1+height/2),.9)


def candidate(source,kind=ShotType.GROUP_WIDE,height=None):
    people=() if height is None else (person(height),)
    return TimelineCandidate(0,source,0.,1.,shot_type=kind,entry_shot_type=kind,exit_shot_type=kind,
        formation_people=people,entry_formation_people=people,exit_formation_people=people)


class ShotProximityTests(unittest.TestCase):
    def test_same_wide_label_does_not_hide_stage_distance(self):
        near=observed_shot_proximity(candidate('near',height=.75))
        far=observed_shot_proximity(candidate('far',height=.20))
        self.assertEqual(near.wide_fraction,far.wide_fraction)
        self.assertAlmostEqual(near.distant_wide_cost,.25)
        self.assertAlmostEqual(far.distant_wide_cost,.80)

    def test_missing_person_is_unknown_not_fabricated_far_or_near(self):
        value=observed_shot_proximity(candidate('unknown'))
        self.assertEqual(value.person_height_samples,(None,)*3)
        self.assertEqual(value.distant_wide_cost,0.)
        self.assertEqual(value.wide_height_uncertainty_fraction,1.)
        self.assertFalse(value.to_dict()['missing_height_is_near'])
        summary=source_path_proximity_summary((candidate('unknown'),))
        self.assertEqual(summary['measured_wide_height_duration_seconds'],0.)
        self.assertEqual(summary['missing_wide_height_duration_seconds'],1.)

    def test_missing_height_does_not_gain_free_near_shot_score(self):
        unknown,far=candidate('unknown'),candidate('far',height=.20)
        config=TimelineConfig(group_wide_distance_preference_weight=.65,minimum_hold_segments=1)
        self.assertLess(_candidate_emission(unknown,mode=TimelineMode.GENERAL,target=None,config=config)[0],
            _candidate_emission(far,mode=TimelineMode.GENERAL,target=None,config=config)[0])
        self.assertEqual(select_timeline(((unknown,far),),config=config).candidates[0].source_id,'far')
        self.assertEqual(observed_shot_proximity(unknown).person_height_samples,(None,)*3)

    def test_multiple_people_do_not_use_large_group_union_as_height(self):
        value=candidate('many',height=.20)
        people=(person(.20),person(.25),person(.22))
        value=replace(value,formation_people=people,entry_formation_people=people,exit_formation_people=people)
        self.assertAlmostEqual(observed_shot_proximity(value).distant_wide_cost,.75)

    def test_mixed_shot_and_duration_summary_remain_sampling_not_encoded_qa(self):
        wide=candidate('wide',height=.2)
        mixed=replace(wide,entry_shot_type=ShotType.CLOSE_UP,exit_shot_type=ShotType.MEDIUM)
        self.assertAlmostEqual(observed_shot_proximity(mixed).wide_fraction,1/3)
        close=replace(candidate('close',ShotType.CLOSE_UP),start_seconds=1.,end_seconds=4.)
        report=source_path_proximity_summary((wide,close))
        self.assertEqual(report['three_sample_wide_duration_fraction'],.25)
        self.assertFalse(report['exact_decoded_shot_duration_proven'])

    def test_nearer_wide_can_win_without_forging_quality_or_banning_wide(self):
        near,far=candidate('near',height=.75),candidate('far',height=.20)
        config=TimelineConfig(group_wide_distance_preference_weight=.65,minimum_hold_segments=1)
        result=select_timeline(((far,near),),config=config)
        self.assertEqual(result.candidates[0].source_id,'near')
        self.assertEqual(near.quality_score,1.)
        self.assertEqual(result.candidates[0].shot_type,ShotType.GROUP_WIDE)

    def test_low_quality_closeup_cannot_always_buy_out_good_wide(self):
        wide=candidate('wide',height=.75)
        bad=replace(candidate('bad',ShotType.CLOSE_UP),quality_score=0.)
        config=TimelineConfig(group_wide_frequency_preference_weight=.55,
            group_wide_distance_preference_weight=.65,minimum_hold_segments=1)
        self.assertEqual(select_timeline(((wide,bad),),config=config).candidates[0].source_id,'wide')

    def test_professional_profiles_do_not_trade_away_quality_only_for_closeup_label(self):
        from crosscut.visual_score import _safe_cadence_profiles,VisualDiversityConfig
        far=candidate('sharp_far',height=.20)
        close=candidate('close',ShotType.CLOSE_UP)
        for profile in _safe_cadence_profiles(VisualDiversityConfig()):
            for quality in (0.,.70,.839999):
                soft=replace(close,quality_score=quality)
                with self.subTest(profile=profile.name,quality=quality):
                    self.assertGreater(_candidate_emission(far,mode=TimelineMode.GENERAL,target=None,config=profile.config)[0],
                        _candidate_emission(soft,mode=TimelineMode.GENERAL,target=None,config=profile.config)[0])
            self.assertGreater(_candidate_emission(replace(close,quality_score=.98),mode=TimelineMode.GENERAL,
                target=None,config=profile.config)[0],_candidate_emission(far,mode=TimelineMode.GENERAL,
                target=None,config=profile.config)[0],'high-quality closeup remains favored')
            self.assertGreater(_candidate_emission(candidate('near',height=.75),mode=TimelineMode.GENERAL,
                target=None,config=profile.config)[0],_candidate_emission(far,mode=TimelineMode.GENERAL,
                target=None,config=profile.config)[0],'budget must retain nearer-wide preference')

    def test_quality_floor_is_still_soft_and_original_quality_measurement_is_unchanged(self):
        config=TimelineConfig(quality_soft_floor=.84,group_wide_frequency_preference_weight=.55,
            group_wide_distance_preference_weight=.65,minimum_hold_segments=1)
        soft=replace(candidate('only_soft',ShotType.CLOSE_UP),quality_score=.70)
        self.assertIsNotNone(_candidate_emission(soft,mode=TimelineMode.GENERAL,target=None,config=config))
        self.assertEqual(select_timeline(((soft,),),config=config).candidates[0].source_id,'only_soft')
        self.assertEqual(soft.quality_score,.70)

    def test_member_emission_is_unchanged_by_group_taste(self):
        from crosscut.vision import IVEMember
        wide=replace(candidate('wide'),is_group_wide=True)
        plain=TimelineConfig()
        active=replace(plain,group_wide_frequency_preference_weight=.55,group_wide_distance_preference_weight=.65)
        self.assertEqual(_candidate_emission(wide,mode=TimelineMode.MEMBER,target=IVEMember.GAEUL,config=plain),
            _candidate_emission(wide,mode=TimelineMode.MEMBER,target=IVEMember.GAEUL,config=active))

    def test_preference_weights_are_finite_explicit_and_default_is_neutral(self):
        plain=TimelineConfig();wide=candidate('wide',height=.2)
        self.assertEqual(plain.group_wide_distance_preference_weight,0.)
        self.assertEqual(_candidate_emission(wide,mode=TimelineMode.GENERAL,target=None,config=plain),
            _candidate_emission(replace(wide,shot_type=ShotType.MEDIUM,
                entry_shot_type=ShotType.MEDIUM,exit_shot_type=ShotType.MEDIUM),mode=TimelineMode.GENERAL,target=None,config=plain))
        for name in ('group_wide_frequency_preference_weight','group_wide_distance_preference_weight'):
            for value in (True,-1.,math.nan,math.inf):
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):TimelineConfig(**{name:value})

    def test_optimizer_identity_binds_actual_cadence_and_distance_weights(self):
        import crosscut.visual_score as visual
        args=dict(candidates_by_segment=((candidate('0'),candidate('1')),),
            sources=('0.mp4','1.mp4'),audio_offsets=(0.,0.),source_timewarps=(None,None),
            source_clock_basis='global_scalar_offset',scene_cut_times=((),()),
            professional_montage_policy=visual.ProfessionalMontageSelectionPolicy(**{
                field.name:.4 for field in fields(visual.ProfessionalMontageSelectionPolicy)}))
        original=visual._visual_optimizer_input_identity(**args)
        self.assertEqual(original,visual._visual_optimizer_input_identity(**args))
        profiles=visual._safe_cadence_profiles(visual.VisualDiversityConfig())
        changed=tuple(replace(p,config=replace(p.config,group_wide_distance_preference_weight=.1)) for p in profiles)
        with patch.object(visual,'_safe_cadence_profiles',return_value=changed):
            self.assertNotEqual(original,visual._visual_optimizer_input_identity(**args))


if __name__=='__main__':unittest.main()
