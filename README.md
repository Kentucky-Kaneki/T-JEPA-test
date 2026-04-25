# T-JEPA on NASim

This repository implements a tabular / host-token JEPA pipeline for NASim and extends it into an RL agent in five stages:

1. Static JEPA: masked latent prediction on the current state.
2. Temporal JEPA: action-conditioned latent prediction of the next state.
3. NASim self-supervision: train on rollout transitions instead of isolated rows.
4. Frozen-latent RL: train policy/value heads on top of a pretrained JEPA encoder.
5. Joint JEPA + RL: unfreeze the encoder gradually and train representation + policy together.

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
- Offline A2C training.
- Fresh policy rollout collection.
- Mixed replay and phased joint JEPA + RL training.

`run.py`
- End-to-end CLI for Steps 3, 4, and 5.

`smoke_test.py`
- Lightweight end-to-end validation of JEPA, offline A2C, and joint training.

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

### Step 4: Frozen-Latent Offline A2C

```bash
python run.py --scenario tiny --n-transitions 5000 --epochs-pretrain 20 --epochs-a2c 10 --exploration-policy balanced
```

### Step 5: Joint JEPA + RL

```bash
python run.py --scenario tiny --n-transitions 5000 --epochs-pretrain 20 --epochs-a2c 10 --epochs-joint 10 --fresh-transitions-per-epoch 1000 --exploration-policy balanced
```

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
- Offline actor-critic epochs on a frozen encoder.

`--epochs-joint`
- Joint JEPA + RL epochs with phased unfreezing.

`--fresh-transitions-per-epoch`
- Number of new policy rollouts collected per joint-training epoch.

`--lambda-rl`
- Weight of the RL term inside the joint loss.

`--encoder-lr-scale`
- Scale factor applied to encoder LR relative to the JEPA joint LR.

`--joint-ema-decay`
- EMA decay used during joint training. Default is `0.999`.

`--max-steps-per-episode`
- Optional cap for rollout length. Useful for fast debugging.

## Joint Training Defaults

Step 5 currently uses:

- Unfreezing schedule: `top1 -> top2 -> full`
- Replay mixing: base rollout + recent fresh policy rollouts
- Joint loss: `L_total = L_JEPA + lambda_rl * L_RL`
- Default `lambda_rl = 0.1`
- Default encoder LR scale: `0.1`
- Default joint EMA decay: `0.999`

## Review Notes

The code review fixed a few concrete issues:

- Variance regularization could produce `NaN` on tiny batches; it now uses population std.
- Episode-aware splitting now falls back safely on tiny datasets instead of crashing.
- Encoder trainability controls no longer risk unfreezing the EMA target branch.
- Rollout resets now use deterministic per-episode seeds for reproducibility.
- The pipeline now warns when reward variance is effectively zero.

## Important Caveat

On short `tiny` runs with capped episode length, reward variance can be effectively constant at `-1`. In that case:

- Step 4 and Step 5 still run correctly.
- The RL objective mostly becomes entropy regularization plus a weak value fit.
- You should not expect meaningful policy improvement until you use longer rollouts, a richer scenario, or later reward shaping.

## Example Debug Run

```bash
python run.py --scenario tiny --n-transitions 300 --epochs-pretrain 2 --epochs-a2c 1 --epochs-joint 2 --fresh-transitions-per-epoch 120 --batch-size 64 --device cpu --exploration-policy balanced --max-steps-per-episode 15 --eval-episodes 2
```
