# Python implementation of the Nadeau and Bengio correction of dependent Student's t-test
# using the equation stated in https://www.cs.waikato.ac.nz/~eibe/pubs/bouckaert_and_frank.pdf

from scipy.stats import t
from math import sqrt
from statistics import stdev
import numpy as np

def corrected_dependent_ttest(data1, data2, n_training_samples, n_test_samples, alpha=0.05):
    n = len(data1)
    differences = [(data1[i]-data2[i]) for i in range(n)]
    sd = stdev(differences)
    divisor = 1 / n * sum(differences)
    test_training_ratio = n_test_samples / n_training_samples
    denominator = sqrt(1 / n + test_training_ratio) * sd
    t_stat = divisor / (denominator + np.finfo(float).eps)
    # degrees of freedom
    df = n - 1
    # calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    # return everything
    return t_stat, df, cv, p


def modified_t_student(a_score, b_score, n1, n2):
    # Compute the difference between the results
    diff = [y - x for y, x in zip(a_score, b_score)]
    diff = diff
    # Comopute the mean of differences
    d_bar = np.mean(diff)
    # compute the variance of differences
    sigma2 = np.var(diff)
    # compute the total number of data points
    n = n1 + n2
    # compute the modified variance
    sigma2_mod = sigma2 * (1 / n + n2 / n1)
    # compute the t_static
    t_static = d_bar / np.sqrt(sigma2_mod)
    # Compute p-value and plot the results
    Pvalue = ((1 - t.cdf(t_static, n - 1)) * 200)
    return Pvalue