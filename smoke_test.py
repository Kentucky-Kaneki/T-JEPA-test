import sys

sys.path.insert(0, ".")

print("--- data.py ---")
from data import (
    build_nasim_loaders,
    build_nasim_rl_loaders,
    collect_rollout_transitions,
    get_env_dims,
    make_nasim_env,
)

env = make_nasim_env("tiny", fully_obs=False, seed=42)
num_hosts, host_features, num_actions = get_env_dims(env)
print(f"  env: {num_hosts} hosts x {host_features} features, {num_actions} actions")

rollout = collect_rollout_transitions(
    env,
    n_transitions=500,
    exploration_policy="balanced",
    seed=42,
    max_steps_per_episode=20,
)
print(
    "  collected: "
    f"{len(rollout['actions'])} transitions, "
    f"{len(set(rollout['episode_ids'].tolist()))} episodes, "
    f"unique actions: {len(set(rollout['actions'].tolist()))}/{num_actions}, "
    f"reward mean/std: {rollout['rewards'].mean():.4f}/{rollout['rewards'].std():.4f}"
)

train_loader, val_loader, prep = build_nasim_loaders(
    rollout["states"],
    rollout["actions"],
    rollout["next_states"],
    num_hosts,
    host_features,
    episode_ids=rollout["episode_ids"],
    batch_size=64,
    seed=42,
)
print(f"  train batches: {len(train_loader)}, val: {len(val_loader)}")
print(f"  feature_dims: {prep.feature_dims}")

x_t, a_t, x_t1 = next(iter(train_loader))
print(f"  x_t shapes:  {[host.shape for host in x_t]}")
print(f"  a shape:     {a_t.shape}")
print(f"  x_t1 shapes: {[host.shape for host in x_t1]}")

print()
print("--- model.py ---")
from model import TJEPA

tjepa = TJEPA(
    feature_dims=prep.feature_dims,
    hidden_dim=32,
    num_heads=2,
    num_layers=2,
    ffn_dim=64,
    pred_dim=16,
    pred_heads=2,
    pred_layers=1,
    num_actions=num_actions,
    temporal_pred_layers=2,
    temporal_weight=0.5,
)
trainable = sum(param.numel() for param in tjepa.parameters() if param.requires_grad)
print(f"  trainable params: {trainable:,}")
assert tjepa.temporal_predictor is not None, "temporal_predictor missing"
print(f"  temporal_predictor: OK (num_hosts={tjepa.temporal_predictor.num_hosts})")

print()
print("--- forward: spatial + temporal ---")
loss, stats = tjepa(x_t, a_t, x_t1)
print(f"  total loss:     {loss.item():.5f}")
for key, value in stats.items():
    print(f"  {key:<16}: {value:.5f}")

print()
print("--- EMA update ---")
tjepa.update_target_encoder()
print("  OK")

print()
print("--- encode ---")
h = tjepa.encode(x_t)
print(f"  encode shape: {h.shape}  expected (B, {num_hosts}, 32)")
assert h.shape == (x_t[0].shape[0], num_hosts, 32), f"wrong shape: {h.shape}"

print()
print("--- forward: spatial-only (no action/next) ---")
loss2, stats2 = tjepa(x_t)
print(f"  spatial-only loss:  {loss2.item():.5f}")
print(f"  temporal_loss:      {stats2['temporal_loss']}  (should be 0.0)")
assert stats2["temporal_loss"] == 0.0, "temporal_loss should be 0 in spatial-only mode"

print()
print("--- backward pass ---")
loss3, _ = tjepa(x_t, a_t, x_t1)
loss3.backward()
grads = [(name, param.grad is not None) for name, param in tjepa.named_parameters() if param.requires_grad]
no_grad = [name for name, has_grad in grads if not has_grad]
if no_grad:
    print(f"  WARNING: params with no grad: {no_grad[:5]}")
else:
    print(f"  All {len(grads)} trainable params received gradients  OK")

print()
print("--- rl.py ---")
from rl import LatentActorCritic, train_joint_tjepa_a2c, train_offline_a2c

rl_train_loader, rl_val_loader, _, reward_stats = build_nasim_rl_loaders(
    rollout["states"],
    rollout["actions"],
    rollout["rewards"],
    rollout["next_states"],
    rollout["dones"],
    num_hosts,
    host_features,
    episode_ids=rollout["episode_ids"],
    batch_size=64,
    seed=42,
    prep=prep,
)
print(
    "  reward norm stats: "
    f"mean={reward_stats['reward_mean']:.4f}, "
    f"std={reward_stats['reward_std']:.4f}, "
    f"clip={reward_stats['reward_clip']:.1f}"
)

agent = LatentActorCritic(
    encoder=tjepa,
    num_actions=num_actions,
    hidden_dim=32,
    actor_hidden_dim=64,
    freeze_encoder=True,
)
rl_batch = next(iter(rl_train_loader))
x_rl, a_rl, r_rl, x_rl_next, d_rl = rl_batch
logits, values = agent(x_rl)
print(f"  actor logits shape: {logits.shape}  expected ({len(a_rl)}, {num_actions})")
print(f"  value shape:        {values.shape}  expected ({len(a_rl)},)")
assert logits.shape == (len(a_rl), num_actions), f"wrong logits shape: {logits.shape}"
assert values.shape == (len(a_rl),), f"wrong value shape: {values.shape}"

a2c_history = train_offline_a2c(
    agent,
    rl_train_loader,
    rl_val_loader,
    num_epochs=1,
    device="cpu",
    verbose=False,
)
print(f"  offline A2C loss after 1 epoch: {a2c_history[-1]['loss']:.5f}")

print()
print("--- joint rl + jepa ---")
joint_history = train_joint_tjepa_a2c(
    agent,
    base_rollout=rollout,
    preprocessor=prep,
    scenario="tiny",
    fully_obs=False,
    num_hosts=num_hosts,
    host_features=host_features,
    batch_size=64,
    num_epochs=1,
    fresh_transitions_per_epoch=100,
    lambda_rl=0.1,
    encoder_lr=1e-5,
    jepa_lr=1e-4,
    policy_lr=3e-4,
    device="cpu",
    seed=42,
    max_steps_per_episode=20,
    verbose=False,
)
print(
    f"  joint loss after 1 epoch: {joint_history[-1]['total_loss']:.5f} "
    f"(stage={joint_history[-1]['stage']})"
)

print()
print("ALL CHECKS PASSED")
