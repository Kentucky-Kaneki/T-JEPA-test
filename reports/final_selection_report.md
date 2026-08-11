# Final Preregistered Selection & Evaluation Report

## Decision
- **Winning Representation**: **flat**
- **Conclusion**: Validated representation 'flat' selected according to preregistered Section 13 rules.
- **Total Models Evaluated**: 20
- **Surviving (non-collapsed) Models**: 2

## Results at Preregistered Horizon k=4
| Representation | OOD F1 | AUROC | Beats Persistence | Action Sensitive | Collapsed | Eff. Rank |
|---|---:|---:|:---:|:---:|:---:|---:|
| flat | 0.6722 | 0.8167 | YES | YES | YES | 4.7 |
| feature | 0.5777 | 0.6986 | YES | YES | YES | 1.4 |
| host | 0.5353 | 0.6777 | NO | YES | YES | 1.2 |
| hierarchical | 0.4279 | 0.6742 | NO | YES | YES | 1.0 |

## Full Sweep Results
| Run ID | k | OOD F1 | AUROC | JEPA MSE | Pers MSE | Eff. Rank |
|---|---:|---:|---:|---:|---:|---:|
| flat_k1 | 1 | 0.6557 | 0.8269 | 1.082429 | 0.024500 | 6.88 |
| feature_k1 | 1 | 0.5899 | 0.7322 | 0.759019 | 0.024500 | 1.41 |
| host_k1 | 1 | 0.5009 | 0.6574 | 0.959695 | 0.024500 | 1.23 |
| hierarchical_k1 | 1 | 0.4176 | 0.6405 | 1.037079 | 0.024500 | 1.13 |
| flat_k2 | 2 | 0.6838 | 0.8350 | 1.042966 | 0.029385 | 2.03 |
| feature_k2 | 2 | 0.5690 | 0.6939 | 0.852863 | 0.029385 | 1.37 |
| host_k2 | 2 | 0.5411 | 0.6465 | 1.125257 | 0.029385 | 1.11 |
| hierarchical_k2 | 2 | 0.4162 | 0.6241 | 1.121019 | 0.029385 | 1.19 |
| flat_k4 | 4 | 0.6722 | 0.8167 | 1.138182 | 0.039385 | 4.73 |
| feature_k4 | 4 | 0.5777 | 0.6986 | 0.807675 | 0.039385 | 1.45 |
| host_k4 | 4 | 0.5353 | 0.6777 | 1.033436 | 0.039385 | 1.18 |
| hierarchical_k4 | 4 | 0.4279 | 0.6742 | 1.167322 | 0.039385 | 1.03 |
| flat_k8 | 8 | 0.7164 | 0.8223 | 1.027878 | 0.054837 | 9.00 |
| feature_k8 | 8 | 0.6001 | 0.6990 | 0.743731 | 0.054837 | 1.37 |
| host_k8 | 8 | 0.5800 | 0.6894 | 1.105340 | 0.054837 | 1.17 |
| hierarchical_k8 | 8 | 0.4024 | 0.6882 | 1.139739 | 0.054837 | 1.33 |
| flat_k16 | 16 | 0.7033 | 0.7736 | 1.137193 | 0.080519 | 3.74 |
| feature_k16 | 16 | 0.6213 | 0.7115 | 0.949323 | 0.080519 | 1.41 |
| host_k16 | 16 | 0.6071 | 0.6918 | 1.222458 | 0.080519 | 1.22 |
| hierarchical_k16 | 16 | 0.3671 | 0.6978 | 1.164577 | 0.080519 | 1.27 |
