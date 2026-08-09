# Horizon Prediction Target Analysis (k = 1, 2, 4, 8, 16)

## Executive Summary
This report analyzes trajectory transition dynamics across prediction horizons $k \in \{1, 2, 4, 8, 16\}$ over 91,800 CybORG transitions (300 episodes) to investigate why $k=8$ achieved peak performance (OOD F1 = 0.7164, AUROC = 0.8223) in Phase 1.

---

## 1. Empirical Trajectory Statistics

For each horizon $k$, we measured the fraction of context-target pairs where the critical server compromise state changes vs. remains unchanged, alongside future compromise prevalence:

| Horizon ($k$) | Total Valid Transition Pairs | State Changed Fraction ($\Delta y \neq 0$) | Persistence / Unchanged Fraction ($\Delta y = 0$) | Future Positive Compromise Prevalence |
|---:|---:|---:|---:|---:|
| **$k=1$** | 91,500 | **3.77%** (3,446) | **96.23%** (88,054) | 45.42% |
| **$k=2$** | 91,200 | **7.45%** (6,794) | **92.55%** (84,406) | 45.57% |
| **$k=4$** | 90,600 | **13.80%** (12,503) | **86.20%** (78,097) | 45.87% |
| **$k=8$** ⭐ | 89,400 | **26.30%** (23,514) | **73.70%** (65,886) | 46.49% |
| **$k=16$** | 87,000 | **46.79%** (40,707) | **53.21%** (46,293) | 47.17% |

---

## 2. Key Observations & Mechanistic Explanations

### 1. Severe Persistence Inertia at Small Horizons ($k=1, 2$)
At $k=1$, **96.23% of transitions experience zero change** in critical server compromise state. A trivial persistence baseline ("predict target = current state") achieves 96.2% accuracy. At $k=2$, 92.55% of transitions remain identical. Short horizons are dominated by static state inertia, providing minimal gradient signal for learning action-conditioned state transitions.

### 2. $k=8$ Matches the Characteristic Attacker Kill-Chain Phase Transition
At $k=8$, **26.30% (~1 in 4) of samples undergo a true state change**. In Scenario1b, 8 timesteps correspond directly to the typical duration required for a Red agent (e.g. B-line or Meander) to perform discovery, subnet lateral movement, and privilege escalation onto operational targets. $k=8$ strikes an optimal balance: it offers sufficient state transition density for the JEPA loss to learn meaningful cyber dynamics without succumbing to the high variance of long horizons.

### 3. State Variance Saturation at $k=16$
At $k=16$, **46.79% of transitions change state**. While state changes are frequent, 16 steps into the future accumulate high stochasticity from unobserved Red policy choices and Blue action interactions, causing downstream linear probe AUROC to degrade from 0.8223 ($k=8$) down to 0.7736 ($k=16$).

---

## Conclusion
The peak performance at $k=8$ is driven by the underlying temporal dynamics of the CybORG environment: $k=8$ is the natural temporal scale of multi-step cyber attack progression.
