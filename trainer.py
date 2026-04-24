"""
Training loops for T-JEPA pre-training and downstream classifier fine-tuning.

Task 2 update: DataLoaders now yield (x_t_batch, a_batch, x_t1_batch) triples.
All three are unpacked and passed to model.forward() every step.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


COLLAPSE_THRESHOLD = 0.01   # repr_std below this triggers a warning


# ─────────────────────────────────────────────
#  T-JEPA Pre-training
# ─────────────────────────────────────────────

def pretrain_tjepa(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[list[float], list[dict]]:
    """
    Pre-trains T-JEPA.

    DataLoader must yield (x_t_batch, a_batch, x_t1_batch) triples.
    All three are forwarded to model() every step for the combined
    spatial + temporal JEPA objective.

    Returns:
        train_losses : list[float]  — pred_loss (spatial) per epoch
        train_stats  : list[dict]   — full per-epoch stats dict
    """
    model = model.to(device)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0
    )

    train_losses: list[float] = []
    train_stats:  list[dict]  = []

    for epoch in range(1, num_epochs + 1):
        model.train()

        acc = {k: 0.0 for k in
               ["pred_loss", "var_loss", "kl_loss", "temporal_loss", "kl_temporal", "repr_std"]}
        n_batches = 0

        for x_batch, a_batch, x_next_batch in train_loader:
            x_batch      = [x.to(device) for x in x_batch]
            a_batch      = a_batch.to(device)
            x_next_batch = [x.to(device) for x in x_next_batch]

            optimizer.zero_grad()
            loss, stats = model(x_batch, a_batch, x_next_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target_encoder()

            for k in acc:
                acc[k] += stats[k]
            n_batches += 1

        scheduler.step()
        n = max(n_batches, 1)

        epoch_stats = {k: acc[k] / n for k in acc}
        epoch_stats["total_loss"] = sum(
            acc[k] / n for k in ["pred_loss", "var_loss", "kl_loss", "temporal_loss", "kl_temporal"]
        )
        train_stats.append(epoch_stats)
        train_losses.append(epoch_stats["pred_loss"])

        # -- Collapse detection ------------------------------------
        if epoch_stats["repr_std"] < COLLAPSE_THRESHOLD:
            print(
                f"  ⚠️  [COLLAPSE WARNING] epoch {epoch}: "
                f"repr_std={epoch_stats['repr_std']:.5f} < {COLLAPSE_THRESHOLD}. "
                f"Check var_reg_weight / EMA decay."
            )

        if verbose and (epoch % 5 == 0 or epoch == 1):
            val_loss = evaluate_pretrain(model, val_loader, device)
            print(
                f"  Epoch {epoch:>3}/{num_epochs}  "
                f"pred={epoch_stats['pred_loss']:.4f}  "
                f"temp={epoch_stats['temporal_loss']:.4f}  "
                f"var={epoch_stats['var_loss']:.4f}  "
                f"kl={epoch_stats['kl_loss']:.4f}  "
                f"std={epoch_stats['repr_std']:.4f}  "
                f"val={val_loss:.4f}"
            )

    return train_losses, train_stats


def evaluate_pretrain(model, loader: DataLoader, device: str) -> float:
    """Returns average total loss over the loader (no grad)."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x_batch, a_batch, x_next_batch in loader:
            x_batch      = [x.to(device) for x in x_batch]
            a_batch      = a_batch.to(device)
            x_next_batch = [x.to(device) for x in x_next_batch]
            loss, _ = model(x_batch, a_batch, x_next_batch)
            total  += loss.item()
            count  += 1
    return total / max(count, 1)


# ─────────────────────────────────────────────
#  Representation extraction
# ─────────────────────────────────────────────

def extract_representations(
    tjepa_model,
    loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts (representations, labels) using the T-JEPA context encoder.

    Works with transition loaders: (x_t, a, x_t1).
    Returns h = encode(x_t) and a (actions used as pseudo-labels for probing).
    """
    tjepa_model.eval()
    all_h, all_y = [], []
    with torch.no_grad():
        for x_batch, a_batch, _ in loader:
            x_batch = [x.to(device) for x in x_batch]
            h = tjepa_model.encode(x_batch)    # (B, d, h)
            all_h.append(h.cpu())
            all_y.append(a_batch)
    return torch.cat(all_h, dim=0), torch.cat(all_y, dim=0)


# ─────────────────────────────────────────────
#  Downstream fine-tuning (unchanged from Task 1)
# ─────────────────────────────────────────────

def train_downstream(
    classifier,
    train_h: torch.Tensor,
    train_y: torch.Tensor,
    val_h: torch.Tensor,
    val_y: torch.Tensor,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    classifier = classifier.to(device)
    optimizer  = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    criterion  = nn.CrossEntropyLoss()

    from torch.utils.data import TensorDataset
    train_dl = torch.utils.data.DataLoader(
        TensorDataset(train_h, train_y), batch_size=batch_size, shuffle=True
    )

    best_val_acc, best_epoch, best_state = 0.0, 0, None

    for epoch in range(1, num_epochs + 1):
        classifier.train()
        for h_b, y_b in train_dl:
            h_b, y_b = h_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            criterion(classifier(h_b), y_b).backward()
            optimizer.step()
        scheduler.step()

        val_acc = compute_accuracy(classifier, val_h, val_y, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_state   = {k: v.clone() for k, v in classifier.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    epoch {epoch:>3}/{num_epochs}  val_acc={val_acc:.4f}")

    if best_state:
        classifier.load_state_dict(best_state)
    return {"best_val_acc": best_val_acc, "best_epoch": best_epoch}


def compute_accuracy(classifier, h, y, device, batch_size=512) -> float:
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(h), batch_size):
            h_b = h[i:i+batch_size].to(device)
            y_b = y[i:i+batch_size].to(device)
            correct += (classifier(h_b).argmax(dim=1) == y_b).sum().item()
            total   += y_b.size(0)
    return correct / max(total, 1)
