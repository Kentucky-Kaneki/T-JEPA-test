"""
NASim data pipeline for T-JEPA temporal pre-training.

Host-level tokenization: the 2D NASim observation (num_hosts x host_features)
is treated as a sequence of host tokens where each host is one token whose full
feature vector is projected to hidden_dim by FeatureEmbedding.

Step 3 focuses on rollout-grounded self-supervision rather than isolated rows.
This module therefore supports:
  - collecting trajectory-aware (s_t, a_t, s_{t+1}) transitions
  - choosing an exploration policy for collection
  - splitting train/val by episode to avoid transition leakage
  - building offline RL datasets with normalized rewards
"""

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import nasim


# --------------------------------------------------------------
# Environment setup
# --------------------------------------------------------------

def make_nasim_env(
    scenario: str = "tiny",
    fully_obs: bool = False,
    seed: int = 42,
):
    """
    Creates a NASim benchmark environment.

    Args:
        scenario  : scenario name, e.g. "tiny", "small", "small-linear"
        fully_obs : False -> partial observability (POMDP, realistic)
                    True  -> full state visible (useful for debugging)
        seed      : RNG seed passed to env.reset()

    Returns env with:
        observation_space.shape = (num_hosts, host_features) when flat_obs=False
        action_space.n          = num discrete actions
    """
    env = nasim.make_benchmark(
        scenario,
        fully_obs=fully_obs,
        flat_obs=False,
        flat_actions=True,
    )
    env.reset(seed=seed)
    return env


def get_env_dims(env) -> tuple[int, int, int]:
    """Returns (num_hosts, host_features, num_actions) from a NASim env."""
    num_hosts, host_features = env.observation_space.shape
    num_actions = env.action_space.n
    return num_hosts, host_features, num_actions


def _split_indices(
    indices: np.ndarray,
    val_size: float,
    seed: int,
    episode_ids: np.ndarray | None = None,
    split_by_episode: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits indices robustly for both large and tiny datasets.

    GroupShuffleSplit requires at least two groups, and train_test_split requires
    at least two samples. This helper falls back gracefully instead of crashing
    on smoke tests or very small rollout collections.
    """
    if len(indices) < 2:
        return indices, indices[:0]

    if split_by_episode and episode_ids is not None:
        unique_episodes = np.unique(episode_ids)
        if len(unique_episodes) >= 2:
            splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
            return next(splitter.split(indices, groups=episode_ids))

    return train_test_split(indices, test_size=val_size, random_state=seed)


# --------------------------------------------------------------
# Exploration policies
# --------------------------------------------------------------

class RandomExplorationPolicy:
    """Uniformly samples NASim actions."""

    def __init__(self, num_actions: int, seed: int = 42):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        return None

    def sample_action(self, _obs: np.ndarray) -> int:
        return int(self.rng.integers(self.num_actions))


class BalancedExplorationPolicy:
    """
    Samples under-used actions more often to improve action coverage.

    This is still exploratory rather than reward-driven, but it reduces the
    chance that a fixed transition budget misses a large chunk of the action
    space.
    """

    def __init__(self, num_actions: int, seed: int = 42, floor_prob: float = 0.01):
        self.num_actions = num_actions
        self.floor_prob = floor_prob
        self.rng = np.random.default_rng(seed)
        self.action_counts = np.ones(num_actions, dtype=np.float64)

    def reset(self) -> None:
        return None

    def sample_action(self, _obs: np.ndarray) -> int:
        probs = 1.0 / self.action_counts
        probs = probs / probs.sum()

        if self.floor_prob > 0.0:
            probs = probs * (1.0 - self.floor_prob * self.num_actions) + self.floor_prob
            probs = probs / probs.sum()

        action = int(self.rng.choice(self.num_actions, p=probs))
        self.action_counts[action] += 1.0
        return action


def make_exploration_policy(policy_name: str, num_actions: int, seed: int = 42):
    """Creates the action policy used during self-supervised rollout collection."""
    if policy_name == "random":
        return RandomExplorationPolicy(num_actions, seed=seed)
    if policy_name == "balanced":
        return BalancedExplorationPolicy(num_actions, seed=seed)
    raise ValueError(
        f"Unknown exploration_policy={policy_name!r}. Expected one of: random, balanced."
    )


# --------------------------------------------------------------
# Transition collection
# --------------------------------------------------------------

def collect_rollout_transitions(
    env,
    n_transitions: int = 50_000,
    exploration_policy: str = "random",
    seed: int = 42,
    max_steps_per_episode: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Collects trajectory-aware (s_t, a_t, s_{t+1}) transitions from NASim.

    Skips truncated terminal transitions because the successor would be a reset
    state rather than a genuine environment transition. Goal-reaching terminal
    transitions are kept.

    Returns:
        states      : (N, num_hosts, host_features) float32
        actions     : (N,)                          int64
        rewards     : (N,)                          float32
        next_states : (N, num_hosts, host_features) float32
        episode_ids : (N,)                          int64
        timesteps   : (N,)                          int64
        dones       : (N,)                          bool
    """
    obs, _ = env.reset(seed=seed)
    policy = make_exploration_policy(exploration_policy, env.action_space.n, seed=seed)

    states, actions, rewards, next_states = [], [], [], []
    episode_ids, timesteps, dones = [], [], []

    collected = 0
    episode_id = 0
    timestep = 0

    while collected < n_transitions:
        action = policy.sample_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        forced_reset = (
            max_steps_per_episode is not None and (timestep + 1) >= max_steps_per_episode
        )

        if not truncated:
            states.append(obs.copy())
            actions.append(action)
            rewards.append(float(reward))
            next_states.append(next_obs.copy())
            episode_ids.append(episode_id)
            timesteps.append(timestep)
            dones.append(bool(terminated))
            collected += 1

        if terminated or truncated or forced_reset:
            obs, _ = env.reset(seed=seed + episode_id + 1)
            policy.reset()
            episode_id += 1
            timestep = 0
        else:
            obs = next_obs
            timestep += 1

    return {
        "states": np.stack(states).astype(np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_states": np.stack(next_states).astype(np.float32),
        "episode_ids": np.array(episode_ids, dtype=np.int64),
        "timesteps": np.array(timesteps, dtype=np.int64),
        "dones": np.array(dones, dtype=bool),
    }


def collect_transitions(
    env,
    n_transitions: int = 50_000,
    exploration_policy: str = "random",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper that returns only arrays expected by older code.
    """
    rollout = collect_rollout_transitions(
        env,
        n_transitions=n_transitions,
        exploration_policy=exploration_policy,
        seed=seed,
    )
    return rollout["states"], rollout["actions"], rollout["next_states"]


# --------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------

class NASIMPreprocessor:
    """
    Per-host StandardScaler fitted on train rollouts only.

    Converts (N, H, F) arrays into the T-JEPA per-token format:
        list of H arrays, each (N, F)

    feature_dims = [F, F, ..., F] with H entries.
    """

    def __init__(self, num_hosts: int, host_features: int):
        self.num_hosts = num_hosts
        self.host_features = host_features
        self.feature_dims = [host_features] * num_hosts
        self.scalers: list[StandardScaler] = [
            StandardScaler() for _ in range(num_hosts)
        ]

    def fit(self, states: np.ndarray) -> "NASIMPreprocessor":
        """Fits one scaler per host. states has shape (N, H, F)."""
        for host_idx in range(self.num_hosts):
            self.scalers[host_idx].fit(states[:, host_idx, :])
        return self

    def transform(self, states: np.ndarray) -> list[np.ndarray]:
        """Returns a list of H arrays, each with shape (N, F)."""
        return [
            self.scalers[host_idx].transform(states[:, host_idx, :]).astype(np.float32)
            for host_idx in range(self.num_hosts)
        ]

    def transform_observation(self, state: np.ndarray) -> list[np.ndarray]:
        """Transforms a single observation with shape (H, F) into per-host token arrays."""
        return [
            self.scalers[host_idx].transform(state[host_idx][None, :]).astype(np.float32)[0]
            for host_idx in range(self.num_hosts)
        ]


# --------------------------------------------------------------
# PyTorch dataset
# --------------------------------------------------------------

class TransitionDataset(Dataset):
    """
    Stores (s_t, a_t, s_{t+1}) in host-token format.
    Yields (x_t_list, action_int, x_t1_list) per sample.
    """

    def __init__(
        self,
        x_t: list[np.ndarray],
        actions: np.ndarray,
        x_t1: list[np.ndarray],
    ):
        self.x_t = [torch.tensor(host_array) for host_array in x_t]
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.x_t1 = [torch.tensor(host_array) for host_array in x_t1]
        self.n = self.x_t[0].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x_t = [host_tensor[idx] for host_tensor in self.x_t]
        action = self.actions[idx]
        x_t1 = [host_tensor[idx] for host_tensor in self.x_t1]
        return x_t, action, x_t1


class RLTransitionDataset(Dataset):
    """
    Stores offline RL tuples (s_t, a_t, r_t, s_{t+1}, done_t) in host-token format.
    """

    def __init__(
        self,
        x_t: list[np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        x_t1: list[np.ndarray],
        dones: np.ndarray,
    ):
        self.x_t = [torch.tensor(host_array) for host_array in x_t]
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)
        self.x_t1 = [torch.tensor(host_array) for host_array in x_t1]
        self.dones = torch.tensor(dones, dtype=torch.float32)
        self.n = self.actions.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x_t = [host_tensor[idx] for host_tensor in self.x_t]
        action = self.actions[idx]
        reward = self.rewards[idx]
        x_t1 = [host_tensor[idx] for host_tensor in self.x_t1]
        done = self.dones[idx]
        return x_t, action, reward, x_t1, done


def transition_collate_fn(batch):
    """
    Collates samples into:
        x_t_batch  : list of H tensors, each (B, F)
        a_batch    : (B,) int64
        x_t1_batch : list of H tensors, each (B, F)
    """
    x_ts, actions, x_t1s = zip(*batch)
    num_hosts = len(x_ts[0])
    batch_size = len(x_ts)

    x_t_batch = [
        torch.stack([x_ts[sample_idx][host_idx] for sample_idx in range(batch_size)])
        for host_idx in range(num_hosts)
    ]
    a_batch = torch.stack(actions)
    x_t1_batch = [
        torch.stack([x_t1s[sample_idx][host_idx] for sample_idx in range(batch_size)])
        for host_idx in range(num_hosts)
    ]

    return x_t_batch, a_batch, x_t1_batch


def rl_transition_collate_fn(batch):
    """
    Collates offline RL samples into:
        x_t_batch  : list of H tensors, each (B, F)
        a_batch    : (B,) int64
        r_batch    : (B,) float32
        x_t1_batch : list of H tensors, each (B, F)
        d_batch    : (B,) float32
    """
    x_ts, actions, rewards, x_t1s, dones = zip(*batch)
    num_hosts = len(x_ts[0])
    batch_size = len(x_ts)

    x_t_batch = [
        torch.stack([x_ts[sample_idx][host_idx] for sample_idx in range(batch_size)])
        for host_idx in range(num_hosts)
    ]
    a_batch = torch.stack(actions)
    r_batch = torch.stack(rewards)
    x_t1_batch = [
        torch.stack([x_t1s[sample_idx][host_idx] for sample_idx in range(batch_size)])
        for host_idx in range(num_hosts)
    ]
    d_batch = torch.stack(dones)

    return x_t_batch, a_batch, r_batch, x_t1_batch, d_batch


# --------------------------------------------------------------
# DataLoader builder
# --------------------------------------------------------------

def build_nasim_loaders(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    num_hosts: int,
    host_features: int,
    episode_ids: np.ndarray | None = None,
    batch_size: int = 256,
    val_size: float = 0.10,
    seed: int = 42,
    split_by_episode: bool = True,
) -> tuple[DataLoader, DataLoader, NASIMPreprocessor]:
    """
    Splits transitions into train/val and fits preprocessing on train only.

    When episode_ids are available, the default split keeps entire episodes on
    one side of the split so validation is not inflated by near-duplicate
    neighboring transitions from the same rollout.
    """
    indices = np.arange(len(states))

    train_idx, val_idx = _split_indices(
        indices,
        val_size=val_size,
        seed=seed,
        episode_ids=episode_ids,
        split_by_episode=split_by_episode,
    )

    prep = NASIMPreprocessor(num_hosts, host_features)
    prep.fit(np.concatenate([states[train_idx], next_states[train_idx]], axis=0))

    def make_loader(split_idx, shuffle):
        x_t = prep.transform(states[split_idx])
        split_actions = actions[split_idx]
        x_t1 = prep.transform(next_states[split_idx])
        dataset = TransitionDataset(x_t, split_actions, x_t1)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=transition_collate_fn,
            num_workers=0,
        )

    return make_loader(train_idx, True), make_loader(val_idx, False), prep


def build_nasim_rl_loaders(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    num_hosts: int,
    host_features: int,
    episode_ids: np.ndarray | None = None,
    batch_size: int = 256,
    val_size: float = 0.10,
    seed: int = 42,
    split_by_episode: bool = True,
    reward_clip: float = 5.0,
    prep: NASIMPreprocessor | None = None,
) -> tuple[DataLoader, DataLoader, NASIMPreprocessor, dict[str, float]]:
    """
    Builds DataLoaders for offline actor-critic training on NASim transitions.

    Rewards are normalized using train-only statistics, then clipped to
    [-reward_clip, reward_clip] for stability.
    """
    indices = np.arange(len(states))

    train_idx, val_idx = _split_indices(
        indices,
        val_size=val_size,
        seed=seed,
        episode_ids=episode_ids,
        split_by_episode=split_by_episode,
    )

    if prep is None:
        prep = NASIMPreprocessor(num_hosts, host_features)
        prep.fit(np.concatenate([states[train_idx], next_states[train_idx]], axis=0))

    reward_mean = float(rewards[train_idx].mean())
    reward_std = float(rewards[train_idx].std())
    reward_std = max(reward_std, 1e-6)

    def normalize_reward(split_rewards: np.ndarray) -> np.ndarray:
        normalized = (split_rewards - reward_mean) / reward_std
        return np.clip(normalized, -reward_clip, reward_clip).astype(np.float32)

    def make_loader(split_idx, shuffle):
        x_t = prep.transform(states[split_idx])
        split_actions = actions[split_idx]
        split_rewards = normalize_reward(rewards[split_idx])
        x_t1 = prep.transform(next_states[split_idx])
        split_dones = dones[split_idx].astype(np.float32)
        dataset = RLTransitionDataset(x_t, split_actions, split_rewards, x_t1, split_dones)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=rl_transition_collate_fn,
            num_workers=0,
        )

    stats = {
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "reward_clip": float(reward_clip),
    }
    return make_loader(train_idx, True), make_loader(val_idx, False), prep, stats
