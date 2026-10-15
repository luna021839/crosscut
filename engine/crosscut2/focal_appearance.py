"""Anonymous appearance diagnostic of the actual displayed face, not identity."""
import numpy as np


def cosine(left, right):
    """UNKNOWN is NaN for existing report normalization, never a perfect match.

    No threshold, named-person label, admission or editorial reward is added.
    Missing/fourth-face descriptors must not borrow another face's vector.
    """
    values = []
    for observation in (left, right):
        value = np.asarray(observation.get('appearance_vector', np.zeros(128)), float)
        if (value.shape != (128,) or not np.isfinite(value).all()
                or abs(float(np.linalg.norm(value)) - 1.) > .001):
            return float('nan')
        values.append(value)
    return float(np.clip(values[0] @ values[1], -1., 1.))
