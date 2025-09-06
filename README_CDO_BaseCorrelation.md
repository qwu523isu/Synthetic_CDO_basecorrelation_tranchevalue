# Synthetic CDO Tranche Pricing and Base Correlation

This project implements a simplified **Gaussian Copula model** for synthetic CDOs, with two complementary scripts:

1. **`Synthetic_CDO_Tranches_Basecorrelation.py`**  
   - Calibrates **base correlation** from market tranche spreads.  
   - Produces an implied base correlation curve across detachment points.  

2. **`Synthetic_CDO_Tranches_Value.py`**  
   - Uses the calibrated base correlation curve to compute the **fair spread** of a given tranche.  

---

## 1. `Synthetic_CDO_Tranches_Basecorrelation.py`

### Purpose
- Calibrate **base correlations** for equity, mezzanine, and senior tranches based on market spread quotes.  
- Plot the **base correlation curve** (detachment points vs. correlation).  

### Methodology
1. Approximate hazard rate from index CDS spread:
   \[
   \lambda = \frac{s}{1 - R}
   \]
2. Convert to 5-year default probability:
   \[
   p(T) = 1 - e^{-\lambda T}
   \]
   and threshold:
   \[
   \theta = \Phi^{-1}(p(T))
   \]
3. Simulate correlated latent variables using Gaussian copula.  
4. Compute portfolio losses, then **tranche losses** using attachment/detachment points.  
5. Match **model-implied spread** with market spread via root-finding (`brentq`) to solve for base correlation.  

### Output
- Prints implied base correlations for each tranche.  
- Plots the **base correlation curve**.  

---

## 2. `Synthetic_CDO_Tranches_Value.py`

### Purpose
- Compute the **fair spread** of a tranche using the **base correlation framework**.  
- Demonstrates how interval tranches \([A,B]\) are valued as differences of cumulative tranches \([0,B] - [0,A]\).  

### Methodology
1. Load base correlation values at attachment and detachment points.  
2. Simulate expected losses for cumulative tranches \([0,A]\) and \([0,B]\).  
3. Subtract to get expected tranche loss:
   \[
   EL_{[A,B]} = EL_{[0,B]} - EL_{[0,A]}
   \]
4. Approximate **premium leg** using discounted annuity.  
5. Compute **fair spread**:
   \[
   s = \frac{EL_{[A,B]}}{\text{Premium Leg}}
   \]

### Output
- Prints the fair spread of the tranche (in basis points).  

---

## ⚠️ Current Limitations
- **Premium leg** is approximated (constant outstanding notional).  
- Calibration may fail if market spreads are inconsistent with index CDS hazard rate.  
- Monte Carlo simulation (100,000 paths) can be slow.  
- Random seed not fixed → results vary slightly between runs.  

---

## ✅ Possible Extensions
- Normalize tranche loss by notional for more interpretable results.  
- Use path-dependent premium leg with expected outstanding notional.  
- Add seeding for reproducibility.  
- Smooth the base correlation curve using interpolation.  
