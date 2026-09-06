# Phase 5 Experimental Reports — Summary & Navigation

This directory contains the complete inference and analysis reports for **Phase 5 (Balanced, Regularised Separate-Action Cyber-JEPA)**.

## Reports Overview

1. [**Stage 1: Transition Balancing & Sampling Analysis**](stage1_inference_report.md)
   - *Key takeaway*: 1:1 balanced sampling (static50_dynamic50) outperforms natural and dynamic-heavy distributions, yielding superior downstream linear probe Macro-F1 across validation (0.8810), test (0.8751), and OOD transfer (0.8257).

2. [**Stage 2: Action-Conditioned Contrast & Sensitivity Analysis**](stage2_inference_report.md)
   - *Key takeaway*: The separate action embedding architecture already provides strong action sensitivity (+15.9% loss degradation on zeroed actions). Auxiliary contrastive action losses triggered rarely in sparse CybORG episodes, proving that architectural inductive biases dominate loss penalties.

3. [**Stage 3: One-Factor-At-A-Time (OFAT) Architecture Ablations**](stage3_inference_report.md)
   - *Key takeaway*: 
     - **horizon = 4** is the single most impactful architectural enhancement, lifting probe Macro-F1 to **0.9246 (Val)** and **0.8755 (OOD)** while resolving the persistence prediction deficit.
     - **	arget_normalization = batch_center** expands effective rank from 9.95 to 13.66 and quadruples action sensitivity (+0.1271 shuffled degradation).
     - **hidden_dim = 64** is required (reducing to 32 hurts capacity), and **history_len = 4..8** provides robust temporal context.
