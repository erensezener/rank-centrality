import numpy as np
import operator
from scipy.linalg import eig
from typing import List, Tuple, Dict, TypeVar

T = TypeVar('T')


def extract_rc_scores(comparisons: List[Tuple[T, T]], regularized: bool = True) -> Dict[T, float]:
    """
    Computes the Rank Centrality scores given a list of pairwise comparisons based on Negahban et al 2016 [1].

    Note it is assumed that the comparisons cannot result in a draw. If you want to include draws, then you can
    treat a draw between `A` and `B` as `A` winning over `B` AND `B` winning over `A`. So for a draw, you can add
    `(A, B)` and `(B, A)` to `comparisons`.

    The regularized version is also implemented. This could be useful when the number of comparisons are small
    with respect to the number of unique items. Note that for properly ranking, number of samples should be in the
    order of n logn, where n is the number of unique items.

    References

    1- Negahban, Sahand et al. “Rank Centrality: Ranking from Pairwise Comparisons.” Operations Research 65 (2017):
    266-287. DOI: https://doi.org/10.1287/opre.2016.1534

    :param comparisons: List of pairs, in `[(winnner, loser)]` format.

    :param regularized: If True, assumes a Beta prior.

    :return: A dictionary of `item -> score`
    """


    winners, losers = zip(*comparisons)
    unique_items = np.hstack([np.unique(winners), np.unique(losers)])

    item_to_index = {item: i for i, item in enumerate(unique_items)}

    A = np.ones((len(unique_items), len(unique_items))) * regularized  # Initializing as ones results in the Beta prior

    for w, l in comparisons:
        A[item_to_index[l], item_to_index[w]] += 1

    A_sum = (A[np.triu_indices_from(A, 1)] + A[np.tril_indices_from(A, -1)]) + 1e-6  # to prevent division by zero

    A[np.triu_indices_from(A, 1)] /= A_sum
    A[np.tril_indices_from(A, -1)] /= A_sum

    d_max = np.max(np.sum(A, axis=1))
    A /= d_max

    w, v = eig(A, left=True, right=False)

    max_eigv_i = np.argmax(w)
    scores = np.real(v[:, max_eigv_i])

    return {item: scores[index] for item, index in item_to_index.items()}


# a simple example below
if __name__ == '__main__':
    matches = [('barcelona', 'man utd'), ('man utd', 'ac milan'), ('barcelona', 'ac milan'), ('ac milan', 'psg'),
               ('psg', 'man utd'), ('barcelona', 'psg'), ('man utd', 'barcelona')]

    team_to_score = extract_rc_scores(matches)
    sorted_teams = sorted(team_to_score.items(), key=operator.itemgetter(1), reverse=True)

    for team, score in sorted_teams:
        print('{} has a score of {!s}'.format(team, round(score, 3)))
