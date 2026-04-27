# T-JEPA on NASim

This repository implements a tabular / host-token JEPA pipeline for NASim and extends it into a frozen-encoder on-policy RL experiment:

1. Static JEPA: masked latent prediction on the current state.
2. Temporal JEPA: action-conditioned latent prediction of the next state.
3. NASim self-supervision: train on rollout transitions instead of isolated rows.
4. Frozen-latent on-policy RL: collect fresh NASim rollouts with the current policy each epoch and train only policy/value heads from NASim task rewards plus optional small one-time progress milestones.

## Implementation Overview

`model.py`
- `TJEPA` core model.
- Context encoder, EMA target encoder, spatial predictor, temporal predictor.
- `encode_context()` for grad-enabled downstream fine-tuning.

`data.py`
- NASim environment creation and rollout collection.
- Exploration policies: `random`, `balanced`.
- Episode-aware train/val splitting.
- Preprocessing and loaders for both JEPA and RL stages.

`trainer.py`
- JEPA pretraining loop and evaluation helpers.

`rl.py`
- `LatentActorCritic` policy/value head on top of JEPA latents.
- Strict on-policy A2C proof-of-concept with a frozen JEPA encoder.
- Random-policy baseline and behavior-focused control metrics.

`run.py`
- End-to-end CLI for JEPA pretraining and frozen-encoder on-policy RL.

`smoke_test.py`
- Lightweight end-to-end validation of JEPA and frozen-encoder on-policy RL.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Validation

Run the smoke test:

```bash
python smoke_test.py
```

## Running the Pipeline

### Step 3: JEPA Pretraining on NASim

```bash
python run.py --scenario tiny --n-transitions 5000 --epochs-pretrain 20 --exploration-policy balanced
```

### Step 4: Frozen-Latent On-Policy A2C

```bash
python run.py --scenario tiny --n-transitions 10000 --epochs-pretrain 20 --epochs-a2c 50 --exploration-policy balanced --max-steps-per-episode 100 --rl-episodes-per-epoch 8 --rl-num-seeds 3 --eval-episodes 20 --random-baseline-episodes 20
```

This stage:

- freezes the JEPA encoder completely
- uses raw NASim task reward as the main learning signal
- optionally adds only small one-time milestone rewards for true attack progress
- collects fresh current-policy rollouts every RL epoch
- updates only policy/value heads from that rollout
- trains separate policy heads across multiple seeds
- compares against a random-policy baseline under identical conditions
- reports success only if behavior improves on actual environment outcomes

## Useful Arguments

`--scenario`
- NASim benchmark scenario such as `tiny`, `small`, `small-linear`.

`--n-transitions`
- Number of rollout transitions collected for the initial dataset.

`--exploration-policy`
- `random` or `balanced`.

`--epochs-pretrain`
- JEPA-only pretraining epochs.

`--epochs-a2c`
- On-policy actor-critic epochs on a frozen encoder.

`--rl-episodes-per-epoch`
- Fresh current-policy episodes collected for each RL epoch.

`--rl-num-seeds`
- Number of independently initialized policy-head seeds to train and evaluate.

`--random-baseline-episodes`
- Random-policy episodes used for the baseline comparison.

`--milestone-reward-scale`
- Optional small one-time bonus for true attack progress events. It is capped at `0.25` and defaults to `0.0`; raw NASim task reward remains the main signal.

`--fail-fast-epochs`
- Stop a seed early if it has no goal success, no return gain over random, and no progress gain over random after this many epochs.

`--min-substantive-rl-epochs`
- Minimum RL epochs required before the run can be labeled a substantive success. Shorter runs are reported as diagnostic only.

`--max-steps-per-episode`
- Cap for rollout length. The RL stage defaults to 100 when enabled.

## RL Verdict

The main report includes:

- goal success rate
- average episode return
- average episode length
- terminal successes
- action diversity
- progress events per episode
- comparison to the random baseline
- whether the training budget was large enough for a substantive claim

Loss curves are not treated as evidence of success. A run is labeled `SUCCESS` only when the learned policy beats random on success rate, return, progress, and remains behaviorally diverse enough to avoid action collapse. Otherwise it is labeled `FAILED` with “no evidence of substantive control learning under a genuine reward signal.”

## Review Notes

- Variance regularization could produce `NaN` on tiny batches; it now uses population std.
- Episode-aware splitting now falls back safely on tiny datasets instead of crashing.
- Encoder trainability controls no longer risk unfreezing the EMA target branch.
- Rollout resets now use deterministic per-episode seeds for reproducibility.
- The pipeline now warns when reward variance is effectively zero.
- The RL stage no longer trains from a fixed offline transition buffer.

## Important Caveat

On short `tiny` runs with capped episode length, reward variance can be effectively constant at `-1`. In that case the RL stage should fail unless the learned policy shows actual improvement over random. Optional milestone rewards are small, one-time signals tied to real attack progress and do not count as task success.

## Example Debug Run

```bash
python run.py --scenario tiny --n-transitions 300 --epochs-pretrain 2 --epochs-a2c 2 --batch-size 64 --device cpu --exploration-policy balanced --max-steps-per-episode 10 --rl-episodes-per-epoch 1 --rl-num-seeds 1 --random-baseline-episodes 1 --eval-episodes 1 --fail-fast-epochs 1
```
