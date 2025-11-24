"""
stats_tests.py

Statistical tests and power analysis.

Contains:
- Independent 2 sample t-test (one-sided)
- Effect size calculation (Cohen's d)
- Theoretical power calculation (Statsmodels)
- Simulated power estimation via simulation
"""

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower


def ttest_smokers_vs_non_smokers(smokers, non_smokers):
    """
    Performs a Welch's t-test comparing smokers and non-smokers.

    The test is one-sided - H1: mean(smokers) > mean(non-smokers)

    Parameters
    ----------
    smokers : array-like
        Sample data for smokers.
    non_smokers : array-like
        Sample data for non-smokers.

    Returns
    -------
    t_stat : float
        T-statistic of the test.
    p_value : float
        P-value corresponding to the t-statistic.
    mean_diff : float
        Difference of means: mean(smokers) - mean(non-smokers).
    """
    t_stat, p_value = stats.ttest_ind(
        smokers,
        non_smokers,
        equal_var=False,
        alternative="greater")

    diff_means = smokers.mean() - non_smokers.mean()
    return t_stat, p_value, diff_means


def effect_calculation(smokers, non_smokers):
    """
    Computes Cohen's d for two independent samples using pooled standard deviation.

    Parameters
    ----------
    smokers : array-like
        Sample data for smokers.
    non_smokers : array-like
        Sample data for non-smokers.

    Returns
    -------
    dict
        Dictionary containing:
        - "effect size": float, Cohen's d.
        - "pooled sd": float, pooled standard deviation.
    """
    n_smokers, n_non_smokers = len(smokers), len(non_smokers)
    sd_smokers, sd_non_smokers = smokers.std(), non_smokers.std()
    pooled_sd = np.sqrt(((n_smokers - 1) * sd_smokers**2 + (n_non_smokers - 1)
                        * sd_non_smokers**2) / (n_smokers + n_non_smokers - 2))
    effect_size = (smokers.mean() - non_smokers.mean()) / pooled_sd
    return {"effect size": effect_size,
            "pooled sd": pooled_sd}


def theoretical_power(effect_size, n_smokers, n_non_smokers, alpha=0.05):
    """
    Computes theoretical statistical power for a one-sided independent t-test.

    Parameters
    ----------
    effect_size : float
        Cohen's d.
    n_smokers : int
        Sample size of smokers.
    n_non_smokers : int
        Sample size of non-smokers.
    alpha : float, optional
        Significance level. Default is 0.05.

    Returns
    -------
    power : float
        Theoretical power of the test.
    """
    
    ratio = n_non_smokers / n_smokers

    solver = TTestIndPower()
    theory_power = solver.power(
        effect_size=effect_size, nobs1=n_smokers, ratio=ratio, alpha=alpha, alternative="larger")

    return theory_power

def simulate_ttest_power(smokers_mean, non_smokers_mean, sd_smokers, sd_non_smokers, n_smokers, n_non_smokers, alpha=0.05, n_sim=5000, seed=42):
    """
    Estimates statistical power via Monte Carlo simulation for a one-sided t-test.

    Parameters
    ----------
    smokers_mean : float
        Mean of smokers population.
    non_smokers_mean : float
        Mean of non-smokers population.
    sd_smokers : float
        Standard deviation of smokers.
    sd_non_smokers : float
        Standard deviation of non-smokers.
    n_smokers : int
        Sample size for smokers.
    n_non_smokers : int
        Sample size for non-smokers.
    alpha : float, optional
        Significance level. Default is 0.05.
    n_sim : int, optional
        Number of Monte Carlo simulations. Default is 5000.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    dict
        Dictionary containing:
        - "hits": float, proportion of simulations rejecting null hypothesis.
        - "t stat simulation": float, last simulated t-statistic.
        - "p simulation": float, last simulated p-value.
    """
    rng = np.random.default_rng(seed)  # Local random number generato
    rejections = 0

    for _ in range(n_sim):
        sample_smokers = rng.normal(smokers_mean, sd_smokers, n_smokers)
        sample_nonsmokers = rng.normal(
            non_smokers_mean, sd_non_smokers, n_non_smokers)

        t_stat_sim, p_val_sim = stats.ttest_ind(
            sample_smokers, sample_nonsmokers, equal_var=False, alternative="greater")

        if p_val_sim < alpha:
            rejections += 1

    return {
        "hits": rejections / n_sim,
        "t stat simulation": t_stat_sim,
        "p simulation": p_val_sim
    }



def power_analysis(n_smokers, n_non_smokers, sd_smokers, sd_non_smokers, alpha=0.05, power_target=0.8):
    """
    Calculates the required mean difference to achieve a target power.

    Uses Cohen's d and pooled standard deviation to determine the mean difference
    needed to reach a specified statistical power.

    Parameters
    ----------
    n_smokers : int
        Sample size of smokers.
    n_non_smokers : int
        Sample size of non-smokers.
    sd_smokers : float
        Standard deviation of smokers.
    sd_non_smokers : float
        Standard deviation of non-smokers.
    alpha : float, optional
        Significance level. Default is 0.05.
    power_target : float, optional
        Desired statistical power. Default is 0.8.

    Returns
    -------
    dict
        Dictionary containing:
        - "required mean difference": float, mean difference required to achieve the target power.
        - "required d": float, Cohen's d corresponding to the required mean difference.
    """
    ratio = n_non_smokers / n_smokers
    s_pooled = np.sqrt(((n_smokers - 1) * sd_smokers**2 + (n_non_smokers - 1)
                       * sd_non_smokers**2) / (n_smokers + n_non_smokers - 2))

    solver = TTestIndPower()
    required_d = solver.solve_power(effect_size=None, nobs1=n_smokers,
                                    ratio=ratio, alpha=alpha, power=power_target, alternative="larger")

    required_mean_diff = required_d * s_pooled

    print(f"""
    För att uppnå {power_target*100}% power skulle testet kräva en medelvärdesskillnad på cirka {required_mean_diff:.2f} mmHg (Cohen's d ≈ {required_d:.2f}).
    """)

    return {
        "required mean difference": required_mean_diff,
        "required d": required_d
    }