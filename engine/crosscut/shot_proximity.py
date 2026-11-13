"""Soft GROUP shot-distance preference using existing observed person boxes.

Wide is not synonymous with far: a near full-formation camera and an audience
shot can have the same GROUP_WIDE enum. No new detector, identity label, crop,
quality score or hard gate is created here. Missing geometry remains unknown.
"""
from dataclasses import dataclass

PROXIMITY_BASIS = 'group_wide_frequency_observed_person_height_and_explicit_uncertainty_soft_preference_v2'
WIDE_KINDS = frozenset(('group_wide', 'wide'))


@dataclass(frozen=True, slots=True)
class ShotProximity:
    wide_samples: tuple[bool, bool, bool]
    person_height_samples: tuple[float | None, float | None, float | None]

    @property
    def wide_fraction(self):
        return sum(self.wide_samples)/3.

    @property
    def distant_wide_cost(self):
        # This measurement excludes missing height. Uncertainty is scored
        # separately, so a missed detector cannot buy the score of a near shot.
        return sum(1.-height for wide,height in zip(self.wide_samples,self.person_height_samples)
            if wide and height is not None)/3.

    @property
    def wide_height_uncertainty_fraction(self):
        return sum(wide and height is None
            for wide,height in zip(self.wide_samples,self.person_height_samples))/3.

    def to_dict(self):
        return dict(basis=PROXIMITY_BASIS,wide_fraction=self.wide_fraction,
            distant_wide_cost=self.distant_wide_cost,
            wide_height_uncertainty_fraction=self.wide_height_uncertainty_fraction,
            observed_person_height_samples=list(self.person_height_samples),
            missing_height_is_near=False,formation_identity_proven=False,
            hard_visual_gate_relaxed=False,professional_quality_approved=False)


def observed_shot_proximity(candidate):
    kinds=(candidate.entry_shot_type or candidate.shot_type,candidate.shot_type,
        candidate.exit_shot_type or candidate.shot_type)
    people=(candidate.entry_formation_people,candidate.formation_people,candidate.exit_formation_people)
    wide=tuple(kind.value in WIDE_KINDS for kind in kinds)
    # Use an observed person's height, not the bounding union of six people,
    # stage architecture or a guessed face-to-body multiplier. These existing
    # typed boxes have already passed the detector's admission contract.
    heights=tuple(max((person.bbox_xywh[3] for person in group),default=None) for group in people)
    return ShotProximity(wide,heights)


def source_path_proximity_summary(candidates):
    rows=tuple(candidates);duration=sum(c.end_seconds-c.start_seconds for c in rows)
    wide_seconds=known_seconds=unknown_seconds=distant_cost_seconds=0.
    for candidate in rows:
        observed=observed_shot_proximity(candidate);span=candidate.end_seconds-candidate.start_seconds
        wide_seconds+=span*observed.wide_fraction
        distant_cost_seconds+=span*observed.distant_wide_cost
        known_seconds+=span*sum(wide and height is not None
            for wide,height in zip(observed.wide_samples,observed.person_height_samples))/3.
        unknown_seconds+=span*observed.wide_height_uncertainty_fraction
    return dict(basis=PROXIMITY_BASIS,three_sample_wide_duration_fraction=wide_seconds/duration if duration else None,
        measured_wide_height_duration_seconds=known_seconds,
        missing_wide_height_duration_seconds=unknown_seconds,
        duration_weighted_distant_wide_cost=distant_cost_seconds/duration if duration else None,
        duration_weighted_wide_height_uncertainty=unknown_seconds/duration if duration else None,
        exact_decoded_shot_duration_proven=False,professional_quality_approved=False)
