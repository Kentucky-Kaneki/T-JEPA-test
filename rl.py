"""
On-policy actor-critic utilities for attaching an RL head to T-JEPA latents.

This proof-of-concept keeps the JEPA encoder frozen. RL updates are made only
from fresh NASim interaction collected by the current policy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from data import make_nasim_env


class LatentActorCritic(nn.Module):
    """
    Policy/value heads on top of a T-JEPA encoder.

    The encoder is frozen by default for Step 4. Latents are mean-pooled across
    host tokens to obtain one state embedding per NASim observation.
    """

    def __init__(
        self,
        encoder,
        num_actions: int,
        hidden_dim: int,
        actor_hidden_dim: int = 128,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_actions = num_actions
        self.freeze_encoder = freeze_encoder

        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, actor_hidden_dim),
            nn.LayerNorm(actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(actor_hidden_dim, num_actions)
        self.value_head = nn.Linear(actor_hidden_dim, 1)

        self.set_encoder_trainable(not freeze_encoder)

    def set_encoder_trainable(self, trainable: bool) -> None:
        """
        Allows later transition from frozen-latent RL to joint fine-tuning.

        Only the learnable JEPA path should be toggled here. The EMA target
        branch must remain frozen regardless of RL stage.
        """
        self.freeze_encoder = not trainable
        _set_trainable_flags(self.encoder.embed, trainable)
        _set_trainable_flags(self.encoder.context_encoder, trainable)
        _set_trainable_flags(self.encoder.predictor, trainable)
        _set_trainable_flags(self.encoder.temporal_predictor, trainable)
        _set_trainable_flags(self.encoder.target_embed, False)
        _set_trainable_flags(self.encoder.target_encoder, False)

    def encode_state(self, x_batch: list[torch.Tensor]) -> torch.Tensor:
        """
        Encodes the current state and mean-pools host tokens into one latent.
        """
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                h = self.encoder.encode_context(x_batch)
        else:
            h = self.encoder.encode_context(x_batch)
        return h.mean(dim=1)

    def forward_latent(self, state_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs only the trainable policy/value heads on already-frozen latents.
        """
        hidden = self.trunk(state_latent)
        logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values

    def forward(self, x_batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits : (B, num_actions)
            values : (B,)
        """
        state_latent = self.encode_state(x_batch)
        return self.forward_latent(state_latent)

    def act(
        self,
        x_batch: list[torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples or selects an action from the policy.
        """
        logits, values = self(x_batch)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_probs, values, entropy

    def act_from_latent(
        self,
        state_latent: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples or selects an action from a precomputed latent.
        """
        logits, values = self.forward_latent(state_latent)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_probs, values, entropy

    def policy_value_parameters(self):
        return (
            list(self.trunk.parameters())
            + list(self.policy_head.parameters())
            + list(self.value_head.parameters())
        )


def _move_token_batch(x_batch: list[torch.Tensor], device: str) -> list[torch.Tensor]:
    return [x.to(device) for x in x_batch]


def _obs_to_token_batch(obs: np.ndarray, preprocessor, device: str) -> list[torch.Tensor]:
    token_arrays = preprocessor.transform_observation(obs)
    return [
        torch.tensor(token, dtype=torch.float32, device=device).unsqueeze(0)
        for token in token_arrays
    ]


def _action_space_n(env) -> int:
    return int(env.action_space.n)


def _count_progress_events(
    info: dict,
    episode_tracker: dict,
) -> dict[str, int]:
    """
    Tracks small, explicit proof-of-concept progress events from NASim info.

    These are not task success. They are only visibility signals for whether
    the agent is doing anything more structured than wandering.
    """
    progress = {
        "new_hosts": 0,
        "new_services": 0,
        "exploit_successes": 0,
        "privilege_gains": 0,
    }

    newly_discovered = info.get("newly_discovered", {}) or {}
    for host_key, discovered in newly_discovered.items():
        if bool(discovered) and host_key not in episode_tracker["hosts"]:
            episode_tracker["hosts"].add(host_key)
            progress["new_hosts"] += 1

    services = info.get("services", {}) or {}
    for service_name, value in services.items():
        try:
            seen = float(value) > 0.0
        except (TypeError, ValueError):
            seen = bool(value)
        if seen and service_name not in episode_tracker["services"]:
            episode_tracker["services"].add(service_name)
            progress["new_services"] += 1

    access = info.get("access", None)
    if access is not None:
        try:
            access_value = float(access)
        except (TypeError, ValueError):
            access_value = 0.0
        if access_value > episode_tracker["best_access"]:
            episode_tracker["best_access"] = access_value
            progress["privilege_gains"] += 1

    state_changed = (
        progress["new_hosts"] > 0
        or progress["new_services"] > 0
        or progress["privilege_gains"] > 0
    )
    if bool(info.get("success", False)) and state_changed:
        progress["exploit_successes"] += 1

    return progress


def _empty_rollout_stats(num_actions: int) -> dict:
    return {
        "episode_returns": [],
        "episode_shaped_returns": [],
        "episode_lengths": [],
        "terminal_successes": 0,
        "action_counts": np.zeros(num_actions, dtype=np.int64),
        "new_hosts": 0,
        "new_services": 0,
        "exploit_successes": 0,
        "privilege_gains": 0,
    }


def _finalize_control_stats(stats: dict, num_actions: int) -> dict[str, float]:
    returns = np.asarray(stats["episode_returns"], dtype=np.float32)
    shaped_returns = np.asarray(stats["episode_shaped_returns"], dtype=np.float32)
    lengths = np.asarray(stats["episode_lengths"], dtype=np.float32)
    episodes = max(len(returns), 1)
    action_counts = stats["action_counts"]
    progress_total = (
        int(stats["new_hosts"])
        + int(stats["new_services"])
        + int(stats["exploit_successes"])
        + int(stats["privilege_gains"])
    )

    return {
        "episodes": float(len(returns)),
        "goal_success_rate": float(stats["terminal_successes"] / episodes),
        "terminal_successes": float(stats["terminal_successes"]),
        "mean_return": float(returns.mean()) if len(returns) else 0.0,
        "std_return": float(returns.std()) if len(returns) else 0.0,
        "mean_shaped_return": float(shaped_returns.mean()) if len(shaped_returns) else 0.0,
        "mean_length": float(lengths.mean()) if len(lengths) else 0.0,
        "action_diversity": float(np.count_nonzero(action_counts) / max(num_actions, 1)),
        "unique_actions": float(np.count_nonzero(action_counts)),
        "progress_events": float(progress_total),
        "progress_events_per_episode": float(progress_total / episodes),
        "new_hosts": float(stats["new_hosts"]),
        "new_services": float(stats["new_services"]),
        "exploit_successes": float(stats["exploit_successes"]),
        "privilege_gains": float(stats["privilege_gains"]),
    }


@torch.no_grad()
def collect_on_policy_latent_rollout(
    agent: LatentActorCritic,
    env,
    preprocessor,
    num_episodes: int = 4,
    max_steps_per_episode: int = 30,
    milestone_reward_scale: float = 0.0,
    deterministic: bool = False,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    Collects fresh on-policy tuples:
        latent_state, action, reward, done, next_latent_state

    The JEPA encoder is used under no_grad only. Rewards stored for learning use
    the NASim task reward as the base signal, with optional small one-time
    milestone bonuses for real attack progress. Environment returns remain raw.
    """
    agent = agent.to(device)
    agent.eval()
    agent.set_encoder_trainable(False)

    num_actions = _action_space_n(env)
    stats = _empty_rollout_stats(num_actions)
    latents, next_latents, actions, rewards, dones = [], [], [], [], []

    for episode_idx in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        episode_tracker = {"hosts": set(), "services": set(), "best_access": 0.0}
        ep_return = 0.0
        ep_shaped_return = 0.0
        ep_len = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and ep_len < max_steps_per_episode:
            x_batch = _obs_to_token_batch(obs, preprocessor, device)
            latent = agent.encode_state(x_batch).detach()
            action, _, _, _ = agent.act_from_latent(latent, deterministic=deterministic)
            action_int = int(action.item())

            next_obs, env_reward, terminated, truncated, info = env.step(action_int)
            x_next_batch = _obs_to_token_batch(next_obs, preprocessor, device)
            next_latent = agent.encode_state(x_next_batch).detach()

            progress = _count_progress_events(info, episode_tracker)
            progress_events = sum(progress.values())
            learning_reward = float(env_reward) + milestone_reward_scale * float(progress_events)

            latents.append(latent.squeeze(0).cpu())
            next_latents.append(next_latent.squeeze(0).cpu())
            actions.append(action_int)
            rewards.append(learning_reward)
            dones.append(bool(terminated or truncated or (ep_len + 1) >= max_steps_per_episode))

            ep_return += float(env_reward)
            ep_shaped_return += learning_reward
            ep_len += 1
            stats["action_counts"][action_int] += 1
            for key, value in progress.items():
                stats[key] += int(value)

            obs = next_obs

        stats["episode_returns"].append(ep_return)
        stats["episode_shaped_returns"].append(ep_shaped_return)
        stats["episode_lengths"].append(ep_len)
        if terminated:
            stats["terminal_successes"] += 1

    if not latents:
        raise RuntimeError("On-policy rollout collected no transitions.")

    return {
        "latents": torch.stack(latents).to(device),
        "actions": torch.tensor(actions, dtype=torch.long, device=device),
        "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
        "next_latents": torch.stack(next_latents).to(device),
        "dones": torch.tensor(dones, dtype=torch.float32, device=device),
        "stats": _finalize_control_stats(stats, num_actions),
    }


def _discounted_returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)
    for idx in range(rewards.numel() - 1, -1, -1):
        running = rewards[idx] + gamma * running * (1.0 - dones[idx])
        returns[idx] = running
    return returns


def train_on_policy_a2c(
    agent: LatentActorCritic,
    preprocessor,
    scenario: str = "tiny",
    fully_obs: bool = False,
    num_epochs: int = 10,
    episodes_per_epoch: int = 4,
    max_steps_per_episode: int = 30,
    gamma: float = 0.99,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    grad_clip: float = 1.0,
    milestone_reward_scale: float = 0.0,
    device: str = "cpu",
    seed: int = 42,
    random_baseline: dict[str, float] | None = None,
    fail_fast_epochs: int = 5,
    min_control_margin: float = 0.0,
    eval_episodes: int = 5,
    verbose: bool = True,
) -> list[dict]:
    """
    Strictly on-policy A2C with a frozen JEPA encoder.

    Each epoch creates fresh NASim interaction using the current policy and
    updates policy/value heads only from that rollout.
    """
    agent = agent.to(device)
    agent.set_encoder_trainable(False)
    optimizer = optim.AdamW(agent.policy_value_parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict] = []

    for epoch in range(1, num_epochs + 1):
        rollout_env = make_nasim_env(scenario, fully_obs=fully_obs, seed=seed + 1_000 * epoch)
        rollout = collect_on_policy_latent_rollout(
            agent,
            rollout_env,
            preprocessor,
            num_episodes=episodes_per_epoch,
            max_steps_per_episode=max_steps_per_episode,
            milestone_reward_scale=milestone_reward_scale,
            deterministic=False,
            device=device,
            seed=seed + 10_000 * epoch,
        )

        agent.train()
        latents = rollout["latents"].detach()
        actions = rollout["actions"]
        rewards = rollout["rewards"]
        dones = rollout["dones"]

        logits, values = agent.forward_latent(latents)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        returns = _discounted_returns(rewards, dones, gamma)
        advantages = returns - values

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = 0.5 * advantages.pow(2).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.policy_value_parameters(), grad_clip)
        optimizer.step()

        epoch_stats = {
            "epoch": float(epoch),
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "advantage_abs": float(advantages.detach().abs().mean().item()),
            **rollout["stats"],
        }
        history.append(epoch_stats)

        if verbose:
            print(
                f"  On-policy epoch {epoch:>3}/{num_epochs}  "
                f"return={epoch_stats['mean_return']:.3f}  "
                f"success={epoch_stats['terminal_successes']:.0f}/{epoch_stats['episodes']:.0f}  "
                f"len={epoch_stats['mean_length']:.1f}  "
                f"progress/ep={epoch_stats['progress_events_per_episode']:.2f}  "
                f"actions={epoch_stats['action_diversity']:.2f}"
            )

        if (
            random_baseline is not None
            and fail_fast_epochs > 0
            and epoch == min(fail_fast_epochs, num_epochs)
        ):
            eval_env = make_nasim_env(scenario, fully_obs=fully_obs, seed=seed + 50_000)
            eval_stats = evaluate_control_policy(
                agent,
                eval_env,
                preprocessor,
                num_episodes=eval_episodes,
                max_steps_per_episode=max_steps_per_episode,
                deterministic=True,
                random_policy=False,
                milestone_reward_scale=milestone_reward_scale,
                device=device,
                seed=seed + 60_000,
            )
            no_success = eval_stats["terminal_successes"] <= 0
            no_return_gain = (
                eval_stats["mean_return"]
                <= random_baseline["mean_return"] + min_control_margin
            )
            no_progress_gain = (
                eval_stats["progress_events_per_episode"]
                <= random_baseline["progress_events_per_episode"]
            )
            if no_success and no_return_gain and no_progress_gain:
                epoch_stats["failed_poc"] = 1.0
                epoch_stats["fail_reason"] = (
                    "fail-fast: no goal success, no return gain over random, "
                    "and no progress gain over random"
                )
                if verbose:
                    print(f"  {epoch_stats['fail_reason']}")
                break

    return history


@torch.no_grad()
def evaluate_control_policy(
    agent: LatentActorCritic | None,
    env,
    preprocessor,
    num_episodes: int = 10,
    max_steps_per_episode: int = 30,
    deterministic: bool = True,
    random_policy: bool = False,
    milestone_reward_scale: float = 0.0,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, float]:
    """
    Evaluates real NASim behavior. Losses are intentionally not reported here.
    """
    if agent is not None:
        agent = agent.to(device)
        agent.eval()
        agent.set_encoder_trainable(False)

    num_actions = _action_space_n(env)
    rng = np.random.default_rng(seed)
    stats = _empty_rollout_stats(num_actions)

    for episode_idx in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        episode_tracker = {"hosts": set(), "services": set(), "best_access": 0.0}
        ep_return = 0.0
        ep_shaped_return = 0.0
        ep_len = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and ep_len < max_steps_per_episode:
            if random_policy:
                action_int = int(rng.integers(num_actions))
            else:
                if agent is None:
                    raise ValueError("agent is required unless random_policy=True")
                x_batch = _obs_to_token_batch(obs, preprocessor, device)
                latent = agent.encode_state(x_batch)
                action, _, _, _ = agent.act_from_latent(latent, deterministic=deterministic)
                action_int = int(action.item())

            next_obs, env_reward, terminated, truncated, info = env.step(action_int)
            progress = _count_progress_events(info, episode_tracker)
            learning_reward = float(env_reward) + milestone_reward_scale * float(sum(progress.values()))

            ep_return += float(env_reward)
            ep_shaped_return += learning_reward
            ep_len += 1
            stats["action_counts"][action_int] += 1
            for key, value in progress.items():
                stats[key] += int(value)

            obs = next_obs

        stats["episode_returns"].append(ep_return)
        stats["episode_shaped_returns"].append(ep_shaped_return)
        stats["episode_lengths"].append(ep_len)
        if terminated:
            stats["terminal_successes"] += 1

    return _finalize_control_stats(stats, num_actions)


def train_offline_a2c(
    agent: LatentActorCritic,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 25,
    gamma: float = 0.99,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    grad_clip: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
) -> list[dict]:
    """
    Trains policy/value heads using a TD(0)-style actor-critic objective.

    This is an offline stabilisation stage on fixed trajectories, not a pure
    on-policy A2C implementation. The objective still matches the actor-critic
    structure: policy gradient from advantages plus a value baseline.
    """
    raise RuntimeError(
        "Offline A2C is disabled for this proof-of-concept. "
        "Use train_on_policy_a2c with fresh current-policy rollouts."
    )
    agent = agent.to(device)
    optimizer = optim.AdamW(
        [param for param in agent.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    history: list[dict] = []

    for epoch in range(1, num_epochs + 1):
        agent.train()
        epoch_acc = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "advantage_abs": 0.0,
        }
        n_batches = 0

        for x_batch, a_batch, r_batch, x_next_batch, d_batch in train_loader:
            x_batch = _move_token_batch(x_batch, device)
            x_next_batch = _move_token_batch(x_next_batch, device)
            a_batch = a_batch.to(device)
            r_batch = r_batch.to(device)
            d_batch = d_batch.to(device)

            optimizer.zero_grad()

            logits, values = agent(x_batch)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(a_batch)
            entropy = dist.entropy().mean()

            with torch.no_grad():
                _, next_values = agent(x_next_batch)
                targets = r_batch + gamma * (1.0 - d_batch) * next_values

            advantages = targets - values
            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = 0.5 * advantages.pow(2).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
            optimizer.step()

            epoch_acc["loss"] += loss.item()
            epoch_acc["policy_loss"] += policy_loss.item()
            epoch_acc["value_loss"] += value_loss.item()
            epoch_acc["entropy"] += entropy.item()
            epoch_acc["advantage_abs"] += advantages.detach().abs().mean().item()
            n_batches += 1

        epoch_stats = {key: value / max(n_batches, 1) for key, value in epoch_acc.items()}
        val_stats = evaluate_offline_a2c(
            agent,
            val_loader,
            gamma=gamma,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            device=device,
        )
        epoch_stats.update({f"val_{key}": value for key, value in val_stats.items()})
        history.append(epoch_stats)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  A2C epoch {epoch:>3}/{num_epochs}  "
                f"loss={epoch_stats['loss']:.4f}  "
                f"policy={epoch_stats['policy_loss']:.4f}  "
                f"value={epoch_stats['value_loss']:.4f}  "
                f"entropy={epoch_stats['entropy']:.4f}  "
                f"val={epoch_stats['val_loss']:.4f}"
            )

    return history


def evaluate_offline_a2c(
    agent: LatentActorCritic,
    loader: DataLoader,
    gamma: float = 0.99,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluates the offline actor-critic objective on a held-out loader."""
    agent.eval()
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
    }
    count = 0

    with torch.no_grad():
        for x_batch, a_batch, r_batch, x_next_batch, d_batch in loader:
            x_batch = _move_token_batch(x_batch, device)
            x_next_batch = _move_token_batch(x_next_batch, device)
            a_batch = a_batch.to(device)
            r_batch = r_batch.to(device)
            d_batch = d_batch.to(device)

            logits, values = agent(x_batch)
            _, next_values = agent(x_next_batch)

            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(a_batch)
            entropy = dist.entropy().mean()
            targets = r_batch + gamma * (1.0 - d_batch) * next_values
            advantages = targets - values

            policy_loss = -(log_probs * advantages).mean()
            value_loss = 0.5 * advantages.pow(2).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            totals["loss"] += loss.item()
            totals["policy_loss"] += policy_loss.item()
            totals["value_loss"] += value_loss.item()
            totals["entropy"] += entropy.item()
            count += 1

    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_policy(
    agent: LatentActorCritic,
    env,
    preprocessor,
    num_episodes: int = 10,
    deterministic: bool = True,
    device: str = "cpu",
    seed: int = 42,
    max_steps_per_episode: int | None = None,
) -> dict[str, float]:
    """
    Runs the latent policy online in NASim for evaluation only.
    """
    agent = agent.to(device)
    agent.eval()

    episode_returns = []
    episode_lengths = []

    for episode_idx in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        terminated = False
        truncated = False
        ep_return = 0.0
        ep_len = 0

        while not (terminated or truncated):
            token_arrays = preprocessor.transform_observation(obs)
            x_batch = [
                torch.tensor(token, dtype=torch.float32, device=device).unsqueeze(0)
                for token in token_arrays
            ]
            action, _, _, _ = agent.act(x_batch, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            ep_return += float(reward)
            ep_len += 1

            if max_steps_per_episode is not None and ep_len >= max_steps_per_episode:
                break

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)

    returns_np = np.asarray(episode_returns, dtype=np.float32)
    lengths_np = np.asarray(episode_lengths, dtype=np.float32)
    return {
        "mean_return": float(returns_np.mean()) if len(returns_np) else 0.0,
        "std_return": float(returns_np.std()) if len(returns_np) else 0.0,
        "mean_length": float(lengths_np.mean()) if len(lengths_np) else 0.0,
    }


@torch.no_grad()
def collect_policy_rollout_transitions(
    agent: LatentActorCritic,
    env,
    preprocessor,
    n_transitions: int = 5_000,
    device: str = "cpu",
    seed: int = 42,
    max_steps_per_episode: int | None = None,
    deterministic: bool = False,
) -> dict[str, np.ndarray]:
    """
    Collects fresh transitions using the current policy for mixed replay.
    """
    agent = agent.to(device)
    agent.eval()

    obs, _ = env.reset(seed=seed)
    states, actions, rewards, next_states = [], [], [], []
    episode_ids, timesteps, dones = [], [], []

    collected = 0
    episode_id = 0
    timestep = 0

    while collected < n_transitions:
        token_arrays = preprocessor.transform_observation(obs)
        x_batch = [
            torch.tensor(token, dtype=torch.float32, device=device).unsqueeze(0)
            for token in token_arrays
        ]
        action, _, _, _ = agent.act(x_batch, deterministic=deterministic)
        action_int = int(action.item())

        next_obs, reward, terminated, truncated, _ = env.step(action_int)
        forced_reset = (
            max_steps_per_episode is not None and (timestep + 1) >= max_steps_per_episode
        )

        if not truncated:
            states.append(obs.copy())
            actions.append(action_int)
            rewards.append(float(reward))
            next_states.append(next_obs.copy())
            episode_ids.append(episode_id)
            timesteps.append(timestep)
            dones.append(bool(terminated))
            collected += 1

        if terminated or truncated or forced_reset:
            obs, _ = env.reset(seed=seed + episode_id + 1)
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


def concat_rollout_dicts(*rollouts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Concatenates rollout dicts while re-indexing episode_ids to remain unique.
    """
    valid_rollouts = [rollout for rollout in rollouts if rollout is not None and len(rollout["actions"]) > 0]
    if not valid_rollouts:
        raise ValueError("concat_rollout_dicts requires at least one non-empty rollout.")

    merged: dict[str, list[np.ndarray]] = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "episode_ids": [],
        "timesteps": [],
        "dones": [],
    }
    episode_offset = 0

    for rollout in valid_rollouts:
        merged["states"].append(rollout["states"])
        merged["actions"].append(rollout["actions"])
        merged["rewards"].append(rollout["rewards"])
        merged["next_states"].append(rollout["next_states"])
        merged["episode_ids"].append(rollout["episode_ids"] + episode_offset)
        merged["timesteps"].append(rollout["timesteps"])
        merged["dones"].append(rollout["dones"])
        episode_offset += int(rollout["episode_ids"].max()) + 1 if len(rollout["episode_ids"]) else 0

    return {
        key: np.concatenate(value, axis=0)
        for key, value in merged.items()
    }


def _set_trainable_flags(module: nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(trainable)
        if not trainable:
            param.grad = None


def configure_joint_unfreezing(agent: LatentActorCritic, stage: str) -> None:
    """
    Applies the requested unfreezing stage to the shared JEPA encoder.
    """
    encoder = agent.encoder

    _set_trainable_flags(encoder.embed, False)
    _set_trainable_flags(encoder.context_encoder, False)
    _set_trainable_flags(encoder.target_embed, False)
    _set_trainable_flags(encoder.target_encoder, False)
    _set_trainable_flags(encoder.predictor, True)
    _set_trainable_flags(encoder.temporal_predictor, True)

    layers = list(encoder.context_encoder.encoder.layers)

    if stage == "top1":
        for layer in layers[-1:]:
            _set_trainable_flags(layer, True)
        agent.freeze_encoder = False
        return

    if stage == "top2":
        for layer in layers[-2:]:
            _set_trainable_flags(layer, True)
        agent.freeze_encoder = False
        return

    if stage == "full":
        _set_trainable_flags(encoder.embed, True)
        _set_trainable_flags(encoder.context_encoder, True)
        agent.freeze_encoder = False
        return

    raise ValueError(f"Unknown joint unfreezing stage: {stage!r}")


def _stage_for_epoch(epoch: int, num_epochs: int) -> str:
    """
    Default schedule: top1 -> top2 -> full.
    """
    if num_epochs <= 2:
        return "top1" if epoch == 1 else "full"

    first_boundary = max(1, num_epochs // 3)
    second_boundary = max(first_boundary + 1, (2 * num_epochs) // 3)

    if epoch <= first_boundary:
        return "top1"
    if epoch <= second_boundary:
        return "top2"
    return "full"


def build_joint_optimizer(
    agent: LatentActorCritic,
    encoder_lr: float,
    jepa_lr: float,
    policy_lr: float,
    weight_decay: float = 1e-5,
):
    """
    Builds an optimizer with separate learning rates for encoder, JEPA heads,
    and RL heads.
    """
    encoder = agent.encoder
    param_groups = []

    encoder_params = list(encoder.embed.parameters()) + list(encoder.context_encoder.parameters())
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})

    jepa_params = list(encoder.predictor.parameters())
    if encoder.temporal_predictor is not None:
        jepa_params += list(encoder.temporal_predictor.parameters())
    if jepa_params:
        param_groups.append({"params": jepa_params, "lr": jepa_lr})

    policy_params = (
        list(agent.trunk.parameters())
        + list(agent.policy_head.parameters())
        + list(agent.value_head.parameters())
    )
    if policy_params:
        param_groups.append({"params": policy_params, "lr": policy_lr})

    return optim.AdamW(param_groups, weight_decay=weight_decay)


def evaluate_joint_tjepa_a2c(
    agent: LatentActorCritic,
    loader: DataLoader,
    lambda_rl: float = 0.1,
    gamma: float = 0.99,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Evaluates the combined JEPA + RL objective on a loader.
    """
    agent.eval()
    tjepa = agent.encoder
    totals = {
        "total_loss": 0.0,
        "jepa_loss": 0.0,
        "rl_loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "repr_std": 0.0,
        "temporal_loss": 0.0,
    }
    count = 0

    with torch.no_grad():
        for x_batch, a_batch, r_batch, x_next_batch, d_batch in loader:
            x_batch = _move_token_batch(x_batch, device)
            x_next_batch = _move_token_batch(x_next_batch, device)
            a_batch = a_batch.to(device)
            r_batch = r_batch.to(device)
            d_batch = d_batch.to(device)

            jepa_loss, jepa_stats = tjepa(x_batch, a_batch, x_next_batch)
            logits, values = agent(x_batch)
            _, next_values = agent(x_next_batch)

            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(a_batch)
            entropy = dist.entropy().mean()
            targets = r_batch + gamma * (1.0 - d_batch) * next_values
            advantages = targets - values

            policy_loss = -(log_probs * advantages).mean()
            value_loss = 0.5 * advantages.pow(2).mean()
            rl_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            total_loss = jepa_loss + lambda_rl * rl_loss

            totals["total_loss"] += total_loss.item()
            totals["jepa_loss"] += jepa_loss.item()
            totals["rl_loss"] += rl_loss.item()
            totals["policy_loss"] += policy_loss.item()
            totals["value_loss"] += value_loss.item()
            totals["entropy"] += entropy.item()
            totals["repr_std"] += jepa_stats["repr_std"]
            totals["temporal_loss"] += jepa_stats["temporal_loss"]
            count += 1

    return {key: value / max(count, 1) for key, value in totals.items()}


def train_joint_tjepa_a2c(
    agent: LatentActorCritic,
    base_rollout: dict[str, np.ndarray],
    preprocessor,
    scenario: str,
    fully_obs: bool,
    num_hosts: int,
    host_features: int,
    batch_size: int = 256,
    num_epochs: int = 15,
    fresh_transitions_per_epoch: int = 5_000,
    gamma: float = 0.99,
    lambda_rl: float = 0.1,
    encoder_lr: float = 3e-5,
    jepa_lr: float = 1e-4,
    policy_lr: float = 3e-4,
    weight_decay: float = 1e-5,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    reward_clip: float = 5.0,
    grad_clip: float = 1.0,
    device: str = "cpu",
    seed: int = 42,
    max_steps_per_episode: int | None = None,
    max_recent_policy_rollouts: int = 2,
    verbose: bool = True,
) -> list[dict]:
    """
    Jointly trains JEPA and RL with phased unfreezing and mixed replay.
    """
    raise RuntimeError(
        "Joint JEPA + RL training is disabled for this proof-of-concept. "
        "The JEPA encoder must remain frozen during RL."
    )
    agent = agent.to(device)
    tjepa = agent.encoder
    optimizer = build_joint_optimizer(
        agent,
        encoder_lr=encoder_lr,
        jepa_lr=jepa_lr,
        policy_lr=policy_lr,
        weight_decay=weight_decay,
    )

    history: list[dict] = []
    recent_policy_rollouts: list[dict[str, np.ndarray]] = []

    for epoch in range(1, num_epochs + 1):
        stage = _stage_for_epoch(epoch, num_epochs)
        configure_joint_unfreezing(agent, stage)

        rollout_env = make_nasim_env(
            scenario,
            fully_obs=fully_obs,
            seed=seed + 1_000 * epoch,
        )
        fresh_rollout = collect_policy_rollout_transitions(
            agent,
            rollout_env,
            preprocessor,
            n_transitions=fresh_transitions_per_epoch,
            device=device,
            seed=seed + 10_000 * epoch,
            max_steps_per_episode=max_steps_per_episode,
            deterministic=False,
        )
        recent_policy_rollouts.append(fresh_rollout)
        if len(recent_policy_rollouts) > max_recent_policy_rollouts:
            recent_policy_rollouts.pop(0)

        mixed_rollout = concat_rollout_dicts(base_rollout, *recent_policy_rollouts)
        train_loader, val_loader, _, reward_stats = build_nasim_rl_loaders(
            mixed_rollout["states"],
            mixed_rollout["actions"],
            mixed_rollout["rewards"],
            mixed_rollout["next_states"],
            mixed_rollout["dones"],
            num_hosts=num_hosts,
            host_features=host_features,
            episode_ids=mixed_rollout["episode_ids"],
            batch_size=batch_size,
            seed=seed + epoch,
            split_by_episode=True,
            reward_clip=reward_clip,
            prep=preprocessor,
        )

        agent.train()
        epoch_acc = {
            "total_loss": 0.0,
            "jepa_loss": 0.0,
            "rl_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "repr_std": 0.0,
            "temporal_loss": 0.0,
        }
        batch_count = 0

        for x_batch, a_batch, r_batch, x_next_batch, d_batch in train_loader:
            x_batch = _move_token_batch(x_batch, device)
            x_next_batch = _move_token_batch(x_next_batch, device)
            a_batch = a_batch.to(device)
            r_batch = r_batch.to(device)
            d_batch = d_batch.to(device)

            optimizer.zero_grad()

            jepa_loss, jepa_stats = tjepa(x_batch, a_batch, x_next_batch)
            logits, values = agent(x_batch)
            with torch.no_grad():
                _, next_values = agent(x_next_batch)

            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(a_batch)
            entropy = dist.entropy().mean()
            targets = r_batch + gamma * (1.0 - d_batch) * next_values
            advantages = targets - values

            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = 0.5 * advantages.pow(2).mean()
            rl_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            total_loss = jepa_loss + lambda_rl * rl_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
            optimizer.step()
            tjepa.update_target_encoder()

            epoch_acc["total_loss"] += total_loss.item()
            epoch_acc["jepa_loss"] += jepa_loss.item()
            epoch_acc["rl_loss"] += rl_loss.item()
            epoch_acc["policy_loss"] += policy_loss.item()
            epoch_acc["value_loss"] += value_loss.item()
            epoch_acc["entropy"] += entropy.item()
            epoch_acc["repr_std"] += jepa_stats["repr_std"]
            epoch_acc["temporal_loss"] += jepa_stats["temporal_loss"]
            batch_count += 1

        epoch_stats = {key: value / max(batch_count, 1) for key, value in epoch_acc.items()}
        val_stats = evaluate_joint_tjepa_a2c(
            agent,
            val_loader,
            lambda_rl=lambda_rl,
            gamma=gamma,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            device=device,
        )
        epoch_stats.update({f"val_{key}": value for key, value in val_stats.items()})
        epoch_stats["stage"] = stage
        epoch_stats["fresh_reward_mean"] = float(fresh_rollout["rewards"].mean())
        epoch_stats["fresh_reward_std"] = float(fresh_rollout["rewards"].std())
        epoch_stats["reward_norm_std"] = reward_stats["reward_std"]
        history.append(epoch_stats)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  Joint epoch {epoch:>3}/{num_epochs}  "
                f"stage={stage:<4}  "
                f"total={epoch_stats['total_loss']:.4f}  "
                f"jepa={epoch_stats['jepa_loss']:.4f}  "
                f"rl={epoch_stats['rl_loss']:.4f}  "
                f"val={epoch_stats['val_total_loss']:.4f}"
            )

    return history
