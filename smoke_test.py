import sys, torch
sys.path.insert(0, '.')

print('--- data.py ---')
from data import make_nasim_env, get_env_dims, collect_transitions, build_nasim_loaders
env = make_nasim_env('tiny', fully_obs=False, seed=42)
H, F, N_A = get_env_dims(env)
print(f'  env: {H} hosts x {F} features, {N_A} actions')

states, actions, next_states = collect_transitions(env, n_transitions=500, seed=42)
print(f'  collected: {len(states)} transitions, unique actions: {len(set(actions.tolist()))}/{N_A}')

train_loader, val_loader, prep = build_nasim_loaders(
    states, actions, next_states, H, F, batch_size=64, seed=42
)
print(f'  train batches: {len(train_loader)}, val: {len(val_loader)}')
print(f'  feature_dims: {prep.feature_dims}')

x_t, a, x_t1 = next(iter(train_loader))
print(f'  x_t shapes:  {[xi.shape for xi in x_t]}')
print(f'  a shape:     {a.shape}')
print(f'  x_t1 shapes: {[xi.shape for xi in x_t1]}')

print()
print('--- model.py ---')
from model import TJEPA, TemporalPredictor
tjepa = TJEPA(
    feature_dims=prep.feature_dims,
    hidden_dim=32, num_heads=2, num_layers=2, ffn_dim=64,
    pred_dim=16, pred_heads=2, pred_layers=1,
    num_actions=N_A, temporal_pred_layers=2, temporal_weight=0.5,
)
trainable = sum(p.numel() for p in tjepa.parameters() if p.requires_grad)
print(f'  trainable params: {trainable:,}')
assert tjepa.temporal_predictor is not None, 'temporal_predictor missing!'
print(f'  temporal_predictor: OK  (num_hosts={tjepa.temporal_predictor.num_hosts})')

print()
print('--- forward: spatial + temporal ---')
loss, stats = tjepa(x_t, a, x_t1)
print(f'  total loss:     {loss.item():.5f}')
for k, v in stats.items():
    print(f'  {k:<16}: {v:.5f}')

print()
print('--- EMA update ---')
tjepa.update_target_encoder()
print('  OK')

print()
print('--- encode ---')
h = tjepa.encode(x_t)
print(f'  encode shape: {h.shape}  expected (B, {H}, 32)')
assert h.shape == (x_t[0].shape[0], H, 32), f'wrong shape: {h.shape}'

print()
print('--- forward: spatial-only (no action/next) ---')
loss2, stats2 = tjepa(x_t)
print(f'  spatial-only loss:  {loss2.item():.5f}')
print(f'  temporal_loss:      {stats2["temporal_loss"]}  (should be 0.0)')
assert stats2['temporal_loss'] == 0.0, 'temporal_loss should be 0 in spatial-only mode'

print()
print('--- backward pass ---')
loss3, _ = tjepa(x_t, a, x_t1)
loss3.backward()
grads = [(n, p.grad is not None) for n, p in tjepa.named_parameters() if p.requires_grad]
no_grad = [n for n, g in grads if not g]
if no_grad:
    print(f'  WARNING: params with no grad: {no_grad[:5]}')
else:
    print(f'  All {len(grads)} trainable params received gradients  OK')

print()
print('ALL CHECKS PASSED')
