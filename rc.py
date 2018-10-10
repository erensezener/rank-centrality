import numpy as np
from scipy.linalg import eig
from typing import List, Tuple, Dict, TypeVar

T = TypeVar('T')


def extract_rc_scores(comparisons: List[Tuple[T, T]], regularized: bool = True) -> Dict[T, float]:
    winners, losers = zip(*comparisons)
    unique_items = np.hstack([np.unique(winners), np.unique(losers)])

    item_to_index = {item: i for i, item in enumerate(unique_items)}

    A = np.ones((len(unique_items), len(unique_items))) * regularized  # Initializing to 1s gives the alpha/beta prior

    for w, l in comparisons:
        A[l, w] += 1

    A_sum = (A[np.triu_indices_from(A, 1)] + A[np.tril_indices_from(A, -1)]) + 1e-6  # to prevent division by zero

    A[np.triu_indices_from(A, 1)] /= A_sum
    A[np.tril_indices_from(A, -1)] /= A_sum

    d_max = np.max(np.sum(A, axis=1))
    A /= d_max

    w, v = eig(A, left=True, right=False)

    max_eigv_i = np.argmax(w)
    scores = np.real(v[:, max_eigv_i])

    return {item: scores[index] for item, index in item_to_index.keys()}

