"""Observation-local rejection is not budget or receipt/identity corruption."""


class SceneHypothesisRejection(ValueError):
    """A specific frozen conditional hypothesis fails measured admission.

    Only these failures may advance to another already-discovered hypothesis
    within the SAME session, original duties and remaining work allowance.
    Generic ValueError, source/model identity changes, archive corruption and
    JointGroupBudgetExhausted must remain terminal in that execution.
    """


class SceneRoleObservationRejection(SceneHypothesisRejection):
    """A valid duty cannot be resolved in a particular native observation.

    Invalid duty/ID/reference formats are NOT this outcome. A frame consumer
    may retain separately measured geometry for this observational failure;
    a dense consumer must still reject the candidate without waiving duties.
    """
