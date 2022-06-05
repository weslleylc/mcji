import scipy.stats
import numpy as np


def combined_ftest_5x2cv(estimator1, estimator2):
    """
    Implements the 5x2cv combined F test proposed
    by Alpaydin 1999,
    to compare the performance of two models.
    Parameters
    ----------
    estimator1 : {array-like, sparse matrix}, shape = [5, 2]
    estimator2 : {array-like, sparse matrix}, shape = [5, 2]
    Returns
    ----------
    f : float
        The F-statistic
    pvalue : float
        Two-tailed p-value.
        If the chosen significance level is larger
        than the p-value, we reject the null hypothesis
        and accept that there are significant differences
        in the two compared models.
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/combined_ftest_5x2cv/
    """

    variances = []
    differences = []

    def score_diff(P1A, P1B, P2A, P2B):

        P1 = P1A - P1B
        P2 = P2A - P2B

        P = (P1 + P2)/2
        s2 = (P1 - P)**2 + (P2 - P)**2

        return P, s2, P1, P2

    if type(estimator1) == list:
        estimator1 = np.array(estimator1)

    if type(estimator2) == list:
        estimator2 = np.array(estimator2)

    if estimator1.shape != (5, 2):
        estimator1 = estimator1.reshape(5, 2)

    if estimator2.shape != (5, 2):
        estimator2 = estimator2.reshape(5, 2)

    for i in range(5):
        Pi, s2i, Pi1, Pi2 = score_diff(estimator1[i, 0],
                                       estimator2[i, 0],
                                       estimator1[i, 1],
                                       estimator2[i, 1])

        differences.extend([Pi1**2, Pi2**2])
        variances.append(s2i)

    numerator = sum(differences)
    denominator = 2 * (sum(variances))
    f_stat = numerator / denominator

    p_value = scipy.stats.f.sf(f_stat, 10, 5)

    return float(f_stat), float(p_value)
