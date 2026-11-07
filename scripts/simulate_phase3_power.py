"""
Reproducible Power Simulation for Phase 3 Confirmatory Protocol Audit
File: scripts/simulate_phase3_power.py

Simulates statistical power for:
1. Exact paired binary McNemar test (discrete recording-level discordance).
2. Clustered 3-seed fractional difference permutation test.
3. Slice-level tests (N = 25, 50, 75, 100) under FWER alpha = 0.0125 and nominal alpha = 0.05.
4. Gate 1 pooled confuser test (N = 350) at alpha = 0.05.

Models paired recording-level outcomes with 10,000 simulations per scenario.
"""

import json
import time
import numpy as np
from scipy.stats import binomtest


def simulate_mcnemar_power(n, p10, p01, alpha, n_sims=10000, seed=42):
    """
    Simulate exact McNemar power for paired binary decisions on n recordings.
    p10: P(E3 errors AND E2 correct) -> E2 better
    p01: P(E3 correct AND E2 errors) -> E3 better
    p00: P(both correct or both error) -> concordant
    Delta = p10 - p01
    """
    p00 = max(0.0, 1.0 - p10 - p01)
    probs = [p00, p10, p01]
    rng = np.random.default_rng(seed)
    
    # Multinomial draws for (concordant, b, c)
    counts = rng.multinomial(n, probs, size=n_sims)
    b_arr = counts[:, 1]
    c_arr = counts[:, 2]
    total_disc = b_arr + c_arr
    
    rejections = 0
    p_values = np.ones(n_sims, dtype=float)
    
    for i in range(n_sims):
        k = total_disc[i]
        b = b_arr[i]
        if k == 0:
            p_values[i] = 1.0
            continue
        # Two-sided exact binomial test
        res = binomtest(b, k, p=0.5, alternative='two-sided')
        p_values[i] = res.pvalue
        if res.pvalue < alpha:
            # Check direction: we are testing if E2 is better (b > c)
            if b > c_arr[i]:
                rejections += 1

    power = rejections / n_sims
    return power, p_values


def simulate_clustered_seed_power(n, delta, within_cluster_corr=0.7, alpha=0.0125, n_sims=10000, seed=42):
    """
    Simulate recording-level clustered 3-seed differences.
    Each recording i has latent tendency z_i ~ N(mu, sigma_b^2).
    Across 3 seeds, individual decisions have within-cluster correlation.
    d_i = mean_s (y_E3,s - y_E2,s) in {-1, -2/3, -1/3, 0, 1/3, 2/3, 1}.
    """
    rng = np.random.default_rng(seed)
    rejections = 0
    
    # Pre-generate random signs for 2000 sign flips (Monte Carlo permutation)
    n_perms = 1000
    perm_signs = rng.choice([-1.0, 1.0], size=(n_perms, n))
    
    # Approximating power via paired t-test or sign-flip
    # For speed and precision over 10,000 sims:
    # Under exchangeable sign flip, if mean(d) is far enough from 0:
    for sim in range(n_sims):
        # Latent recording-level propensity difference
        # Delta is expected difference
        # Each seed has probability of discordance
        p_disc = 0.25
        # Simulate fractional differences directly
        # E.g. b_i ~ Binomial(3, p_e3), c_i ~ Binomial(3, p_e2)
        # To match Delta = E[d_i] = 0.15
        p_e3 = 0.35
        p_e2 = 0.35 - delta
        
        # Correlated seeds via latent variable
        u = rng.normal(0, 1, size=n)
        z_e3 = u * np.sqrt(within_cluster_corr) + rng.normal(0, np.sqrt(1 - within_cluster_corr), size=(3, n))
        z_e2 = u * np.sqrt(within_cluster_corr) + rng.normal(0, np.sqrt(1 - within_cluster_corr), size=(3, n))
        
        from scipy.stats import norm
        thresh_e3 = norm.ppf(1 - p_e3)
        thresh_e2 = norm.ppf(1 - p_e2)
        
        y_e3 = (z_e3 > thresh_e3).astype(float)
        y_e2 = (z_e2 > thresh_e2).astype(float)
        
        d_i = np.mean(y_e3 - y_e2, axis=0) # d_i in [-1, 1]
        observed_stat = np.abs(np.sum(d_i))
        
        # Permutation p-value
        perm_stats = np.abs(np.dot(perm_signs, d_i))
        p_val = (np.sum(perm_stats >= observed_stat) + 1.0) / (n_perms + 1.0)
        
        if p_val < alpha and np.sum(d_i) > 0:
            rejections += 1
            
    return rejections / n_sims


def main():
    print("=" * 80)
    print("PHASE 3 CONFIRMATORY PROTOCOL: STATISTICAL POWER SIMULATION AUDIT")
    print("=" * 80)
    
    n_sims = 10000
    sample_sizes = [25, 50, 75, 100]
    
    results = {
        "metadata": {
            "n_simulations": n_sims,
            "target_delta": 0.15,
            "date": "2026-11-02",
        },
        "slice_level_mcnemar": {},
        "pooled_gate1_mcnemar": {},
    }
    
    # 1. Slice-Level Evaluation (Subdomain slices: N = 25, 50, 75, 100)
    print("\n--- 1. Subdomain Slice Comparisons (Delta = 0.15) ---")
    scenarios = [
        ("Clean Unidirectional (p10=0.15, p01=0.00)", 0.15, 0.00),
        ("Moderate Noise (p10=0.20, p01=0.05)", 0.20, 0.05),
        ("High Noise (p10=0.25, p01=0.10)", 0.25, 0.10),
    ]
    
    for name, p10, p01 in scenarios:
        print(f"\nScenario: {name}")
        results["slice_level_mcnemar"][name] = {}
        for n in sample_sizes:
            pwr_fwer, _ = simulate_mcnemar_power(n, p10, p01, alpha=0.0125, n_sims=n_sims)
            pwr_nominal, _ = simulate_mcnemar_power(n, p10, p01, alpha=0.05, n_sims=n_sims)
            results["slice_level_mcnemar"][name][f"N={n}"] = {
                "power_alpha_0.0125": pwr_fwer,
                "power_alpha_0.05": pwr_nominal,
            }
            print(f"  N = {n:3d} | Power (alpha=0.0125): {pwr_fwer*100:5.2f}% | Power (alpha=0.05): {pwr_nominal*100:5.2f}%")
            
    # 2. Gate 1 Pooled Confuser Test (N = 350, alpha = 0.05)
    print("\n--- 2. Gate 1 Primary Confirmatory Endpoint (Pooled N = 350, alpha = 0.05) ---")
    gate1_scenarios = [
        ("Small Effect Delta = 0.05 (p10=0.15, p01=0.10)", 0.15, 0.10),
        ("Moderate Effect Delta = 0.10 (p10=0.20, p01=0.10)", 0.20, 0.10),
        ("Observed Phase 1 Effect Delta = 0.218 (p10=0.32, p01=0.10)", 0.32, 0.10),
    ]
    for name, p10, p01 in gate1_scenarios:
        delta = p10 - p01
        pwr, _ = simulate_mcnemar_power(350, p10, p01, alpha=0.05, n_sims=n_sims)
        results["pooled_gate1_mcnemar"][name] = {
            "N": 350,
            "delta": delta,
            "power_alpha_0.05": pwr,
        }
        print(f"  {name:50s} -> Power: {pwr*100:5.2f}%")

    # Save to JSON
    out_path = "reports/phase3_power_simulation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results written to {out_path}")


if __name__ == "__main__":
    main()
