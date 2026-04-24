"""
NASim data pipeline for T-JEPA temporal pre-training.

Host-level tokenization: the 2D NASim observation (num_hosts × host_features)
is treated as a sequence of host tokens — each host is ONE token whose full
feature vector is projected to hidden_dim by FeatureEmbedding.

This gives d = num_hosts tokens (e.g. d=4 for 'tiny'), keeping the Transformer
sequence short and hardware-feasible across all scenario sizes.

Yields (s_t, a_t, s_{t+1}) transition triples for temporal JEPA pre-training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import nasim


# ─────────────────────────────────────────────
#  Environment Setup
# ─────────────────────────────────────────────

def make_nasim_env(
    scenario: str = "tiny",
    fully_obs: bool = False,
    seed: int = 42,
):
    """
    Creates a NASim benchmark environment.

    Args:
        scenario  : scenario name, e.g. 'tiny', 'small', 'small-linear'
        fully_obs : False → partial observability (POMDP, realistic)
                    True  → full state visible (useful for debugging)
        seed      : RNG seed passed to env.reset()

    Returns env with:
        observation_space.shape = (num_hosts, host_features)  (flat_obs=False)
        action_space.n          = num discrete actions
    """
    env = nasim.make_benchmark(
        scenario,
        fully_obs=fully_obs,
        flat_obs=False,      # 2D obs → host-level tokenization
        flat_actions=True,   # discrete integer actions
    )
    env.reset(seed=seed)
    return env


def get_env_dims(env) -> tuple[int, int, int]:
    """Returns (num_hosts, host_features, num_actions) from a NASim env."""
    h, f = env.observation_space.shape
    n_a  = env.action_space.n
    return h, f, n_a


# ─────────────────────────────────────────────
#  Transition Collection
# ─────────────────────────────────────────────

def collect_transitions(
    env,
    n_transitions: int = 50_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collects n_transitions of (s_t, a_t, s_{t+1}) via a random policy.

    Skips truncated terminal transitions (time-limit resets) since the
    next state is a reset, not a genuine successor. Includes terminated
    transitions (goal reached) as those are valid causal transitions.

    Returns:
        states      : (N, num_hosts, host_features)  float32
        actions     : (N,)                           int64
        next_states : (N, num_hosts, host_features)  float32
    """
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)

    states, actions, next_states = [], [], []
    collected = 0

    while collected < n_transitions:
        action = int(env.action_space.sample())
        next_obs, _reward, terminated, truncated, _ = env.step(action)

        # Only include non-truncated transitions (truncated = time limit hit)
        if not truncated:
            states.append(obs.copy())
            actions.append(action)
            next_states.append(next_obs.copy())
            collected += 1

        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    return (
        np.stack(states).astype(np.float32),        # (N, H, F)
        np.array(actions, dtype=np.int64),          # (N,)
        np.stack(next_states).astype(np.float32),   # (N, H, F)
    )


# ─────────────────────────────────────────────
#  Preprocessor
# ─────────────────────────────────────────────

class NASIMPreprocessor:
    """
    Per-host StandardScaler. Fitted on training transitions only.

    Converts (N, H, F) arrays into the T-JEPA per-token format:
        list of H arrays, each (N, F)  ← one entry per host token

    feature_dims = [F, F, ..., F]  (H entries, all numerical dim=F)
    """
    def __init__(self, num_hosts: int, host_features: int):
        self.num_hosts     = num_hosts
        self.host_features = host_features
        self.feature_dims  = [host_features] * num_hosts
        self.scalers: list[StandardScaler] = [
            StandardScaler() for _ in range(num_hosts)
        ]

    def fit(self, states: np.ndarray) -> "NASIMPreprocessor":
        """states: (N, H, F)"""
        for h in range(self.num_hosts):
            self.scalers[h].fit(states[:, h, :])
        return self

    def transform(self, states: np.ndarray) -> list[np.ndarray]:
        """
        Returns list of H arrays, each (N, F) — one per host token.
        """
        return [
            self.scalers[h].transform(states[:, h, :]).astype(np.float32)
            for h in range(self.num_hosts)
        ]


# ─────────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────────

class TransitionDataset(Dataset):
    """
    Stores (s_t, a_t, s_{t+1}) in host-token format.
    Yields (x_t_list, action_int, x_t1_list) per sample.
    """
    def __init__(
        self,
        x_t:     list[np.ndarray],   # H arrays, each (N, F)
        actions: np.ndarray,         # (N,) int64
        x_t1:    list[np.ndarray],   # H arrays, each (N, F)
    ):
        self.x_t     = [torch.tensor(f) for f in x_t]
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.x_t1    = [torch.tensor(f) for f in x_t1]
        self.n = self.x_t[0].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x_t  = [f[idx] for f in self.x_t]
        a    = self.actions[idx]
        x_t1 = [f[idx] for f in self.x_t1]
        return x_t, a, x_t1


def transition_collate_fn(batch):
    """
    Collates (x_t_list, action, x_t1_list) samples into batched form:
        x_t_batch  : list of H tensors, each (B, F)
        a_batch    : (B,) int64
        x_t1_batch : list of H tensors, each (B, F)
    """
    x_ts, actions, x_t1s = zip(*batch)
    H = len(x_ts[0])
    B = len(x_ts)

    x_t_batch  = [torch.stack([x_ts[b][h]  for b in range(B)]) for h in range(H)]
    a_batch    = torch.stack(actions)
    x_t1_batch = [torch.stack([x_t1s[b][h] for b in range(B)]) for h in range(H)]

    return x_t_batch, a_batch, x_t1_batch


# ─────────────────────────────────────────────
#  DataLoader Builder
# ─────────────────────────────────────────────

def build_nasim_loaders(
    states:       np.ndarray,   # (N, H, F)
    actions:      np.ndarray,   # (N,)
    next_states:  np.ndarray,   # (N, H, F)
    num_hosts:    int,
    host_features: int,
    batch_size:   int   = 256,
    val_size:     float = 0.10,
    seed:         int   = 42,
) -> tuple[DataLoader, DataLoader, NASIMPreprocessor]:
    """
    Splits transitions into train/val, fits preprocessor on train only.

    Returns:
        train_loader : yields (x_t_batch, a_batch, x_t1_batch)
        val_loader   : same format
        prep         : fitted NASIMPreprocessor (has .feature_dims)
    """
    idx = np.arange(len(states))
    idx_train, idx_val = train_test_split(idx, test_size=val_size, random_state=seed)

    # Fit scaler on training states only
    prep = NASIMPreprocessor(num_hosts, host_features)
    prep.fit(states[idx_train])

    def make_loader(split_idx, shuffle):
        x_t  = prep.transform(states[split_idx])
        a    = actions[split_idx]
        x_t1 = prep.transform(next_states[split_idx])
        ds   = TransitionDataset(x_t, a, x_t1)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            collate_fn=transition_collate_fn, num_workers=0,
        )

    return make_loader(idx_train, True), make_loader(idx_val, False), prep
