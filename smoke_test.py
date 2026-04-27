import sys

sys.path.insert(0, ".")

print("--- data.py ---")
from data import (
    build_nasim_loaders,
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
from rl import (
    LatentActorCritic,
    collect_on_policy_latent_rollout,
    evaluate_control_policy,
    train_on_policy_a2c,
)

agent = LatentActorCritic(
    encoder=tjepa,
    num_actions=num_actions,
    hidden_dim=32,
    actor_hidden_dim=64,
    freeze_encoder=True,
)
logits, values = agent(x_t)
print(f"  actor logits shape: {logits.shape}  expected ({len(a_t)}, {num_actions})")
print(f"  value shape:        {values.shape}  expected ({len(a_t)},)")
assert logits.shape == (len(a_t), num_actions), f"wrong logits shape: {logits.shape}"
assert values.shape == (len(a_t),), f"wrong value shape: {values.shape}"

on_policy_env = make_nasim_env("tiny", fully_obs=False, seed=123)
on_policy_rollout = collect_on_policy_latent_rollout(
    agent,
    on_policy_env,
    prep,
    num_episodes=1,
    max_steps_per_episode=5,
    device="cpu",
    seed=123,
)
print(
    "  on-policy rollout: "
    f"{len(on_policy_rollout['actions'])} transitions, "
    f"mean return={on_policy_rollout['stats']['mean_return']:.3f}"
)

random_env = make_nasim_env("tiny", fully_obs=False, seed=456)
random_stats = evaluate_control_policy(
    None,
    random_env,
    prep,
    num_episodes=1,
    max_steps_per_episode=5,
    random_policy=True,
    device="cpu",
    seed=456,
)
print(f"  random baseline return: {random_stats['mean_return']:.3f}")

a2c_history = train_on_policy_a2c(
    agent,
    prep,
    scenario="tiny",
    fully_obs=False,
    num_epochs=1,
    episodes_per_epoch=1,
    max_steps_per_episode=5,
    lr=3e-4,
    device="cpu",
    seed=42,
    random_baseline=random_stats,
    fail_fast_epochs=0,
    eval_episodes=1,
    verbose=False,
)
print(f"  on-policy A2C loss after 1 epoch: {a2c_history[-1]['loss']:.5f}")

print()
print("ALL CHECKS PASSED")
