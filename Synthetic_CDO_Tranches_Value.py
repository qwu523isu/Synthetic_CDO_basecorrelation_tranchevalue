# -*- coding: utf-8 -*-
"""
Created on Wed May  7 21:59:37 2025

@author: Qiong Wu
"""

import numpy as np
import scipy.stats as stats

# ---- Given Parameters ----
n = 125
recovery_rate = 0.40
risk_free_rate = 0.035
cds_index_spread = 50 / 10000
maturity = 5
num_payments_per_year = 4
num_simulations = 100000
attachment_point = 0.03
detachment_point = 0.06

# ---- Hazard Rate ----
hazard_rate = cds_index_spread / (1 - recovery_rate)
default_threshold = stats.norm.ppf(hazard_rate * maturity)

# ---- Base Correlation Curve (Example Values) ----
base_correlation_curve = {
    0.03: 0.08,
    0.06: 0.15  # Substitute your implied base correlation here
}

def expected_loss_at_point(rho, point):
    """Simulate expected portfolio loss capped at detachment point."""
    Z = np.random.randn(num_simulations)
    epsilons = np.random.randn(num_simulations, n)
    firm_values = np.sqrt(rho) * Z[:, None] + np.sqrt(1 - rho) * epsilons
    defaults = (firm_values < default_threshold).sum(axis=1)
    losses = (1 - recovery_rate) * defaults / n
    return np.mean(np.minimum(losses, point))

# ---- Step 1: Compute Expected Loss at A and B ----
rho_A = base_correlation_curve[attachment_point]
rho_B = base_correlation_curve[detachment_point]

EL_A = expected_loss_at_point(rho_A, attachment_point)
EL_B = expected_loss_at_point(rho_B, detachment_point)

expected_tranche_loss = EL_B - EL_A

# ---- Step 2: Premium Leg (Annuity) ----
discount_factors = np.exp(-risk_free_rate * np.arange(1, num_payments_per_year * maturity + 1) / num_payments_per_year)
premium_leg = np.sum(discount_factors) * (detachment_point - attachment_point) / (num_payments_per_year * maturity)

# ---- Step 3: Tranche Spread ----
tranche_spread = expected_tranche_loss / premium_leg
tranche_spread_bps = tranche_spread * 10000

print(f"Fair Tranche Spread (3%–6%): {tranche_spread_bps:.2f} bps")
