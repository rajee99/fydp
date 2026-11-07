"""
Comprehensive Statistical Power Simulation for Phase 3 Second-Level Audit
File: scripts/simulate_phase3_power_comprehensive.py

Evaluates statistical power across:
1. Sample sizes: N in [25, 50, 75, 100, 150, 350].
2. Estimands:
   - Discrete Binary Paired McNemar (theta_2 ensemble decisions).
   - Clustered 3-Seed Fractional Permutation Test (theta_1 single-model average).
3. Effect sizes (addressing optimistic bias / Winner's Curse):
   - Conservative: Delta = 0.05 (p10 = 0.15, p01 = 0.10)
   - Moderate:     Delta = 0.08 (p10 = 0.18, p01 = 0.10)
   - Target:       Delta = 0.10 (p10 = 0.20, p01 = 0.10)
   - Strong:       Delta = 0.15 (p10 = 0.20, p01 = 0.05 and p10 = 0.15, p01 = 0.00)
   - Optimistic:   Delta = 0.218 (Phase 1 observed, p10 = 0.32, p01 = 0.10)
4. Multiplicity levels:
   - alpha = 0.05 (Gate 1 Gatekeeper nominal level)
   - alpha = 0.0125 (Subdomain Holm Rank 4)
   - alpha = 0.00714 (Subdomain Holm Rank 1)

Models paired recording clusters with 10,000 Monte Carlo runs per scenario.
"""

import json
import time
import numpy as np
from scipy.stats import binomtest, norm


def simulate_exact_mcnemar_power(n, p10, p01, alpha, n_sims=10000, seed=42):
    """
    Simulate exact McNemar power for paired binary decisions on n recordings.
    b: E3 error & E2 correct (favors E2)
    c: E2 error & E3 correct (favors E3)
    """
    p00 = max(0.0, 1.0 - p10 - p01)
    probs = [p00, p10, p01]
    rng = np.random.default_rng(seed)
    
    counts = rng.multinomial(n, probs, size=n_sims)
    b_arr = counts[:, 1]
    c_arr = counts[:, 2]
    total_disc = b_arr + c_arr
    
    rejections = 0
    for i in range(n_sims):
        k = total_disc[i]
        b = b_arr[i]
        c = c_arr[i]
        if k == 0:
            continue
        res = binomtest(b, k, p=0.5, alternative='two-sided')
        if res.pvalue < alpha and b > c:
            rejections += 1

    return rejections / n_sims


def simulate_clustered_fractional_permutation_power(
    n, p10_base, p01_base, within_cluster_corr=0.65, alpha=0.05, n_sims=10000, n_perms=500, seed=42
):
    """
    Simulates the paired cluster sign-flip permutation test for theta_1.
    For each physical recording i in 1..n:
      Three seeds are evaluated for E3 and E2.
      d_i = 1/3 * sum_s [ I(E3_error) - I(E2_error) ] in {-1, -2/3, -1/3, 0, 1/3, 2/3, 1}.
    """
    rng = np.random.default_rng(seed)
    rejections = 0
    delta = p10_base - p01_base
    
    # Pre-generate permutation sign flip matrix (n_perms, n)
    perm_signs = rng.choice([-1.0, 1.0], size=(n_perms, n))
    
    # Base probabilities of error on confuser
    p_e3 = 0.35
    p_e2 = p_e3 - delta
    thresh_e3 = norm.ppf(1.0 - p_e3)
    thresh_e2 = norm.ppf(1.0 - p_e2)
    
    sigma_b = np.sqrt(within_cluster_corr)
    sigma_w = np.sqrt(1.0 - within_cluster_corr)
    
    for sim in range(n_sims):
        # Latent recording-level acoustic difficulty u_i
        u = rng.normal(0, 1, size=n)
        
        # 3 seeds for E3 and E2
        eps_e3 = rng.normal(0, sigma_w, size=(3, n))
        eps_e2 = rng.normal(0, sigma_w, size=(3, n))
        
        z_e3 = u * sigma_b + eps_e3
        z_e2 = u * sigma_b + eps_e2
        
        y_e3 = (z_e3 > thresh_e3).astype(float)
        y_e2 = (z_e2 > thresh_e2).astype(float)
        
        # Average difference per recording across the 3 seeds
        d_i = np.mean(y_e3 - y_e2, axis=0) # shape (n,)
        observed_stat = np.abs(np.sum(d_i))
        
        # Two-sided permutation test via sign flips
        perm_stats = np.abs(np.dot(perm_signs, d_i))
        p_val = (np.sum(perm_stats >= observed_stat) + 1.0) / (n_perms + 1.0)
        
        if p_val < alpha and np.sum(d_i) > 0:
            rejections += 1
            
    return rejections / n_sims


def main():
    print("=" * 90)
    print("PHASE 3 COMPREHENSIVE STATISTICAL POWER SIMULATION AUDIT")
    print("=" * 90)
    
    sample_sizes = [25, 50, 75, 100, 150, 350]
    n_sims = 10000
    
    scenarios = [
        ("Conservative (Delta=0.05, p10=0.15, p01=0.10)", 0.15, 0.10, 0.05),
        ("Moderate (Delta=0.08, p10=0.18, p01=0.10)",     0.18, 0.10, 0.08),
        ("Target (Delta=0.10, p10=0.20, p01=0.10)",       0.20, 0.10, 0.10),
        ("Strong Clean (Delta=0.15, p10=0.15, p01=0.00)", 0.15, 0.00, 0.15),
        ("Strong Noisy (Delta=0.15, p10=0.20, p01=0.05)", 0.20, 0.05, 0.15),
        ("Optimistic Phase 1 (Delta=0.218, p10=0.32, p01=0.10)", 0.32, 0.10, 0.218),
    ]
    
    results = {
        "metadata": {
            "n_simulations": n_sims,
            "sample_sizes": sample_sizes,
            "date": "2026-11-02",
        },
        "exact_mcnemar_power": {},
        "clustered_permutation_power": {},
    }
    
    print("\n--- 1. Exact Binary McNemar Power (Discrete Decisions / Ensemble theta_2) ---")
    for name, p10, p01, delta in scenarios:
        print(f"\nScenario: {name}")
        results["exact_mcnemar_power"][name] = {}
        for n in sample_sizes:
            pwr_05 = simulate_exact_mcnemar_power(n, p10, p01, alpha=0.05, n_sims=n_sims)
            pwr_0125 = simulate_exact_mcnemar_power(n, p10, p01, alpha=0.0125, n_sims=n_sims)
            pwr_0071 = simulate_exact_mcnemar_power(n, p10, p01, alpha=0.00714, n_sims=n_sims)
            results["exact_mcnemar_power"][name][f"N={n}"] = {
                "alpha_0.05": round(pwr_05, 4),
                "alpha_0.0125": round(pwr_0125, 4),
                "alpha_0.00714": round(pwr_0071, 4),
            }
            print(f"  N = {n:3d} | alpha=0.05: {pwr_05*100:5.2f}% | alpha=0.0125: {pwr_0125*100:5.2f}% | alpha=0.00714: {pwr_0071*100:5.2f}%")
            
    print("\n--- 2. Clustered 3-Seed Permutation Test Power (Single Model Average theta_1) ---")
    # Evaluate key representative sample sizes
    perm_sample_sizes = [25, 50, 75, 100, 150, 350]
    for name, p10, p01, delta in scenarios:
        print(f"\nPermutation Scenario: {name}")
        results["clustered_permutation_power"][name] = {}
        for n in perm_sample_sizes:
            # For N=350, run 5000 sims for speed
            sim_count = 5000 if n >= 150 else 10000
            pwr_perm_05 = simulate_clustered_fractional_permutation_power(
                n, p10, p01, within_cluster_corr=0.65, alpha=0.05, n_sims=sim_count, n_perms=500
            )
            pwr_perm_0125 = simulate_clustered_fractional_permutation_power(
                n, p10, p01, within_cluster_corr=0.65, alpha=0.0125, n_sims=sim_count, n_perms=500
            )
            results["clustered_permutation_power"][name][f"N={n}"] = {
                "alpha_0.05": round(pwr_perm_05, 4),
                "alpha_0.0125": round(pwr_perm_0125, 4),
            }
            print(f"  N = {n:3d} | Permutation alpha=0.05: {pwr_perm_05*100:5.2f}% | alpha=0.0125: {pwr_perm_0125*100:5.2f}%")

    out_file = "reports/phase3_power_simulation_comprehensive.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Comprehensive power results written to {out_file}")


if __name__ == "__main__":
    main()
