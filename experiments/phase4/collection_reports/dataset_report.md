# Dataset Characterization & Diagnostic Report

## Summary
- **Total Transitions**: 90,000
- **Total Episodes**: 1,800
- **Shards Analyzed**: 18
- **Zero-Change Transition Fraction**: 42.29%
- **Acceptance Gates Passed**: **YES**




## Horizon vs. Feature Change Rate (Episode-Bounded)
| Horizon $k$ | Change Rate (%) | Persistence MSE |
|---|---:|---:|
| 1 | 58.31% | 0.024535 |
| 2 | 66.14% | 0.031105 |
| 4 | 71.74% | 0.038800 |
| 8 | 80.02% | 0.055494 |
| 16 | 88.58% | 0.082080 |

## Action Frequency
| Action Type | Count | Percentage |
|---|---:|---:|
| Sleep | 35,552 | 39.50% |
| Restore | 12,294 | 13.66% |
| Misinform | 12,254 | 13.62% |
| Analyse | 12,246 | 13.61% |
| Remove | 12,192 | 13.55% |
| Monitor | 5,462 | 6.07% |
