"""
stats_tests.py

Statistical tests and power analysis.

Contains:
- Independent 2 sample t-test (one-sided)
- Effect size (Cohen's d)
- Theoretical power (Statsmodels)
- Simulated power estimation
"""

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

# 1. T-test for smokers vs non-smokers
def ttest_smokers_vs_non_smokers(smokers, non_smokers):
    """
    Performs a Welch's t-test (independent samples, unequal variance)
    using a one-sided alternative hypothesis:
    H1: mean(smokers) > mean(non-smokers)
    
    Returns
    -------
    t_stat : float
    p_value : float
    mean_diff : float
    """

# Two indipendent samples, one-sided test (right tail)

    t_stat, p_value = stats.ttest_ind(
        smokers, 
        non_smokers, 
        equal_var=False, 
        alternative="greater")

    diff_means = smokers.mean() - non_smokers.mean()

    return t_stat, p_value, diff_means

# 2. Effect size (Cohen's d)
"""
    Computes Cohen's d for two independent samples using pooled SD.

    Returns
    -------
    d : float
    pooled_sd : float
    """

def effect_calculation(smokers, non_smokers):

    n_smokers, n_non_smokers = len(smokers), len(non_smokers)
    sd_smokers, sd_non_smokers = smokers.std(), non_smokers.std()
    pooled_sd = np.sqrt(((n_smokers - 1) * sd_smokers**2 + (n_non_smokers - 1) * sd_non_smokers**2) / (n_smokers + n_non_smokers - 2))
    effect_size = (smokers.mean() - non_smokers.mean()) / pooled_sd
    return {"effect size": effect_size,
            "pooled sd": pooled_sd}

# 3. Theoretical power (statsmodels)
# Matematisk beräkning av statistisk power 
def theoretical_power(effect_size, n_smokers, n_non_smokers, alpha=0.05):
    """
        Computes theoretical statistical power using Statsmodels.

        Parameters
        ----------
        effect_size : float
            Cohen's d
        n_smokers : int
        n_nonsmokers : int
        alpha : float

        Returns
        -------
        power : float
        """
    ratio = n_non_smokers / n_smokers
    
    solver = TTestIndPower()
    theory_power = solver.power(effect_size=effect_size, nobs1=n_smokers, ratio=ratio, alpha=alpha, alternative="larger")
    
    return theory_power

# 4. Simulation of empirical power
# Beräkning av statistisk power genom simulering
def simulate_ttest_power(smokers_mean, non_smokers_mean, sd_smokers, 
                         sd_non_smokers, n_smokers, n_non_smokers, 
                         alpha=0.05, n_sim=20000, seed=42):
    """
    Monte Carlo simulation to estimate statistical power.

    Parameters
    ----------
    smokers_mean : float
    nonsmokers_mean : float
    sd_smokers : float
    sd_nonsmokers : float
    n_smokers : int
    n_nonsmokers : int
    alpha : float
    n_sim : int

    Returns
    -------
    power : float
        Proportion of simulations where null was rejected.
    """
    rng = np.random.default_rng(seed)  # Local random number generato
    rejections = 0

    for _ in range(n_sim):
        sample_smokers = rng.normal(smokers_mean, sd_smokers, n_smokers)
        sample_nonsmokers = rng.normal(non_smokers_mean, sd_non_smokers, n_non_smokers)
        
        t_stat_sim, p_val_sim = stats.ttest_ind(sample_smokers, sample_nonsmokers, equal_var=False, alternative="greater")
        
        if p_val_sim < alpha:
            rejections += 1

    return {
        "hits": rejections / n_sim,
        "t stat simulation": t_stat_sim,
        "p simulation": p_val_sim
}

# Power analys
def power_analysis(n_smokers, n_non_smokers, sd_smokers, sd_non_smokers, diff_means, alpha=0.05, power_target=0.8):
    """
    Function to calculate the required mean difference to achieve a target power using Cohen's d.
    --------
    summary : str
        A summary string with the required mean difference and observed difference.
    """
    
    # Power analysis setup
    ratio = n_non_smokers / n_smokers
    s_pooled = np.sqrt(((n_smokers - 1) * sd_smokers**2 + (n_non_smokers - 1) * sd_non_smokers**2) / (n_smokers + n_non_smokers - 2))
    
    # Solve for Cohen's d needed to reach the target power
    solver = TTestIndPower()
    required_d = solver.solve_power(effect_size=None, nobs1=n_smokers, ratio=ratio, alpha=alpha, power=power_target, alternative="larger")
    
    # Calculate the required mean difference
    required_mean_diff = required_d * s_pooled
    
    # Create the summary message
    summary = f"""
    För att uppnå {power_target*100}% power skulle testet kräva en medelvärdesskillnad på cirka {required_mean_diff:.2f} mmHg (Cohen's d ≈ {required_d:.2f}). 
    Den observerade skillnaden (≈ {diff_means} mmHg) är mycket mindre, vilket förklarar den låga styrkan.
    """
    
    return summary