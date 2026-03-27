import math
import numpy as np


def bernoulli_log_likelihood(data, theta):
    """
    Compute the Bernoulli log-likelihood for binary data.

    Parameters
    ----------
    data : array-like
        Sequence of 0/1 observations.
    theta : float
        Bernoulli parameter, must satisfy 0 < theta < 1.

    Returns
    -------
    float
        Log-likelihood:
            sum_i [x_i log(theta) + (1-x_i) log(1-theta)]

    Requirements
    ------------
    - Raise ValueError if data is empty
    - Raise ValueError if theta is not in (0,1)
    - Raise ValueError if data contains values other than 0 and 1
    """
    data = list(data)
    if len(data) == 0:
        raise ValueError("Data must be non-empty.")
    if not (0 < theta < 1):
        raise ValueError("theta must satisfy 0 < theta < 1.")
    for x in data:
        if x not in (0, 1):
            raise ValueError("Data must contain only 0s and 1s.")
    return sum(x * np.log(theta) + (1 - x) * np.log(1 - theta) for x in data)


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    """
    Estimate the Bernoulli MLE and compare candidate theta values.

    Parameters
    ----------
    data : array-like
        Sequence of 0/1 observations.
    candidate_thetas : array-like or None
        Optional candidate theta values to compare using log-likelihood.
        If None, use [0.2, 0.5, 0.8].

    Returns
    -------
    dict
        A dictionary with:
        - 'mle': float
            The Bernoulli MLE
        - 'num_successes': int
        - 'num_failures': int
        - 'log_likelihoods': dict
            Mapping candidate theta -> log-likelihood
        - 'best_candidate': float
            Candidate theta with highest log-likelihood

    Requirements
    ------------
    - Validate data
    - Compute MLE analytically
    - Compute candidate log-likelihoods using bernoulli_log_likelihood
    - In case of ties in best candidate, return the first one encountered
    """
    data = list(data)
    if len(data) == 0:
        raise ValueError("Data must be non-empty.")
    for x in data:
        if x not in (0, 1):
            raise ValueError("Data must contain only 0s and 1s.")
    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]

    n = len(data)
    num_successes = int(sum(data))
    num_failures = n - num_successes

    # MLE for Bernoulli: theta_hat = (1/n) * sum(x_i)
    mle = num_successes / n

    # Compute log-likelihood for each candidate theta
    log_likelihoods = {}
    for theta in candidate_thetas:
        log_likelihoods[theta] = bernoulli_log_likelihood(data, theta)

    # Pick the candidate with the highest log-likelihood (first on ties)
    best_candidate = max(candidate_thetas, key=lambda t: log_likelihoods[t])

    return {
        "mle": mle,
        "num_successes": num_successes,
        "num_failures": num_failures,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate,
    }


def poisson_log_likelihood(data, lam):
    """
    Compute the Poisson log-likelihood for count data.

    Parameters
    ----------
    data : array-like
        Sequence of nonnegative integer counts.
    lam : float
        Poisson rate, must satisfy lam > 0.

    Returns
    -------
    float
        Log-likelihood:
            sum_i [x_i log(lam) - lam - log(x_i!)]

    Requirements
    ------------
    - Raise ValueError if data is empty
    - Raise ValueError if lam <= 0
    - Raise ValueError if data contains negative or non-integer values

    Notes
    -----
    You may use math.lgamma(x + 1) for log(x!) since log(x!) = lgamma(x+1).
    """
    data = list(data)
    if len(data) == 0:
        raise ValueError("Data must be non-empty.")
    if lam <= 0:
        raise ValueError("lam must be > 0.")
    for x in data:
        if x != int(x) or x < 0:
            raise ValueError("Data must contain nonnegative integers.")

    # sum_i [x_i * log(lam) - lam - log(x_i!)]
    # Use math.lgamma(x + 1) for log(x!) to avoid overflow
    return sum(x * np.log(lam) - lam - math.lgamma(x + 1) for x in data)


def poisson_mle_analysis(data, candidate_lambdas=None):
    """
    Estimate the Poisson MLE and compare candidate lambda values.

    Parameters
    ----------
    data : array-like
        Sequence of nonnegative integer counts.
    candidate_lambdas : array-like or None
        Optional candidate lambdas to compare using log-likelihood.
        If None, use [1.0, 3.0, 5.0].

    Returns
    -------
    dict
        A dictionary with:
        - 'mle': float
            The Poisson MLE
        - 'sample_mean': float
        - 'total_count': int
        - 'n': int
        - 'log_likelihoods': dict
            Mapping candidate lambda -> log-likelihood
        - 'best_candidate': float
            Candidate lambda with highest log-likelihood

    Requirements
    ------------
    - Validate data
    - Compute MLE analytically
    - Compute candidate log-likelihoods using poisson_log_likelihood
    - In case of ties in best candidate, return the first one encountered
    """
    data = list(data)
    if len(data) == 0:
        raise ValueError("Data must be non-empty.")
    for x in data:
        if x != int(x) or x < 0:
            raise ValueError("Data must contain nonnegative integers.")
    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]

    n = len(data)
    total_count = int(sum(data))

    # MLE for Poisson: lambda_hat = sample mean = total_count / n
    mle = total_count / n
    sample_mean = mle

    # Compute log-likelihood for each candidate lambda
    log_likelihoods = {}
    for lam in candidate_lambdas:
        log_likelihoods[lam] = poisson_log_likelihood(data, lam)

    # Pick the candidate with the highest log-likelihood (first on ties)
    best_candidate = max(candidate_lambdas, key=lambda l: log_likelihoods[l])

    return {
        "mle": mle,
        "sample_mean": sample_mean,
        "total_count": total_count,
        "n": n,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate,
    }
