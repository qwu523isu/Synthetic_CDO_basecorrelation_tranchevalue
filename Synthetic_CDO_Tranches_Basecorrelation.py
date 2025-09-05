# -*- coding: utf-8 -*-
"""
Created on Wed May  7 08:55:48 2025

@author: Qiong Wu
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ---------------------------
# Model parameters
# ---------------------------
n = 125                         # Number of names in the portfolio
recovery_rate = 0.40            # Recovery rate
risk_free_rate = 0.035          # Constant risk-free rate
cds_index_spread = 50 / 10000   # Annualized index spread
maturity = 5                    # Tranche maturity in years
num_payments_per_year = 4       # Quarterly payments
num_simulations = 100000        # Monte Carlo paths

# Tranche structure includes 0–3%, 3–7%, 7–10%, 10–15%, and 15–30% with market spreads given in bps.
tranches = [
    (0.00, 0.03, 1100),  # Equity tranche: upfront (600bps) + 500bps coupon
    (0.03, 0.07, 180),
    (0.07, 0.10, 100),
    (0.10, 0.15, 50),
    (0.15, 0.30, 25)
]

# Hazard rate approximation from index CDS
hazard_rate = cds_index_spread / (1 - recovery_rate)
print(f"The hazard rate equals to {hazard_rate}")
# The default threshold links between default probability and the Gaussian copula simulation.
# 1. Gaussian Copula Setup
# Each obligor 𝑖 has a latent variable Y_i ~ N(0,1).
# A default occurs if Y_i < threshold. So, if the threshold is chosen correctly, the unconditional probability that 
# Y_i is below the threshold should equal the default probability.
# Formally: P(default) = P(Y_i < threshold) = Phi(threshold) where Phi is the standard normal CDF.
default_prob = 1 - np.exp(-hazard_rate * maturity)
default_threshold = stats.norm.ppf(default_prob)
print(f"The default threshold is: {default_threshold}")
# ---------------------------
# Base correlation solver
# ---------------------------
def objective_base_corr(rho_try, *args):
    n, num_sim, threshold, A, B, recov, r, T, freq, market_spread = args

    # Build correlation matrix
    corr_matrix = rho_try * np.ones((n, n)) + (1 - rho_try) * np.eye(n)
    L = np.linalg.cholesky(corr_matrix)

    # Monte Carlo simulation
    Z = np.random.randn(num_sim, n)
    correlated_values = Z @ L.T
    # correlated_values is a matrix of size (num_sim, n), 
    # each entry is the simulated latent variable for an obligor in a given scenario.
    defaults = (correlated_values < threshold).sum(axis=1)
    # defaults is a vector of length num_sim, giving the number of defaults in each simulated scenario.
    portfolio_losses = (1 - recov) * defaults / n
    # portfolio_losses is the fractional loss of the entire portfolio in each Monte Carlo scenario.

    # Tranche expected loss
    tranche_losses = np.clip(portfolio_losses - A, 0, B - A)
    # tranche_losses is the tranche loss (per unit portfolio notional) for each simulation path.
    # This extracts the portion of the portfolio loss that falls inside the tranche [A,B].
    # Formula: Tranche Loss=min(max(L−A,0),B−A)
    # Explanation:
    # If total portfolio loss 
    # L<A → tranche is unaffected → tranche loss = 0.
    # A<L<B → tranche absorbs loss from A up to L.
    # L≥B → tranche is completely wiped out → tranche loss = B−A.
    # np.clip does exactly this:
    # Subtract A (shift the loss down to tranche level).
    # Clip below 0 (no loss if portfolio loss is less than A).
    # Clip above B−A (cannot lose more than tranche notional).
    expected_tranche_loss = np.mean(tranche_losses)

    # Premium leg
    discount_factors = np.exp(-r * np.arange(1, freq * T + 1) / freq)
    premium_leg = np.sum(discount_factors) * (B - A) / (freq * T)

    # Model-implied spread
    model_spread = expected_tranche_loss / premium_leg
    return model_spread - market_spread

# ---------------------------
# Main calibration loop
# ---------------------------
base_corrs = []
for attach_point, detach_point, market_spread_bps in tranches:
    args = (
        n, num_simulations, default_threshold, attach_point, detach_point,
        recovery_rate, risk_free_rate, maturity,
        num_payments_per_year, market_spread_bps / 10000
    )

    try:
        rho_star = brentq(objective_base_corr, 0.01, 0.99, args=args) 
        # 0.01 and 0.99 represent the bounds for possible correlation values
        base_corrs.append((detach_point, rho_star))
        print(f"Implied base correlation for {int(attach_point*100)}–{int(detach_point*100)}% tranche: {rho_star:.4f}")
    except ValueError:
        base_corrs.append((detach_point, np.nan))
        print(f"Failed to calibrate base correlation for {int(attach_point*100)}–{int(detach_point*100)}% tranche.")

# ---------------------------
# Plot base correlation curve
# ---------------------------
detach_points = [point for point, _ in base_corrs]
corr_values = [val for _, val in base_corrs]

plt.figure(figsize=(8, 5))
plt.plot(detach_points, corr_values, marker='o', label="Implied Base Correlation")
plt.title("Base Correlation Curve")
plt.xlabel("Detachment Point")
plt.ylabel("Base Correlation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
