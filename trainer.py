"""
Training loops for T-JEPA pre-training and downstream classifier fine-tuning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


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
) -> list[float]:
    """
    Pre-trains T-JEPA on unlabelled features (labels ignored).
    Returns list of training losses per epoch.
    """
    model = model.to(device)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0
    )

    train_losses = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for x_batch, _ in train_loader:
            x_batch = [x.to(device) for x in x_batch]
            optimizer.zero_grad()
            loss = model(x_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target_encoder()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            val_loss = evaluate_pretrain(model, val_loader, device)
            print(f"  Epoch {epoch:>3}/{num_epochs}  "
                  f"train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}")

    return train_losses


def evaluate_pretrain(model, loader: DataLoader, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = [x.to(device) for x in x_batch]
            loss = model(x_batch)
            total += loss.item()
            count += 1
    return total / max(count, 1)


# ─────────────────────────────────────────────
#  Downstream fine-tuning
# ─────────────────────────────────────────────

def extract_representations(
    tjepa_model,
    loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs the T-JEPA context encoder over the whole loader,
    returns (representations, labels) on CPU.
    """
    tjepa_model.eval()
    all_h, all_y = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = [x.to(device) for x in x_batch]
            h = tjepa_model.encode(x_batch)        # (B, d, h)
            all_h.append(h.cpu())
            all_y.append(y_batch)
    return torch.cat(all_h, dim=0), torch.cat(all_y, dim=0)


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
    """
    Fine-tunes a downstream classifier on the pre-extracted representations.
    Returns dict with best val_acc and corresponding epoch.
    """
    classifier = classifier.to(device)
    optimizer  = optim.AdamW(classifier.parameters(), lr=lr,
                              weight_decay=weight_decay)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0
    )
    criterion = nn.CrossEntropyLoss()

    # Build simple in-memory loaders
    from torch.utils.data import TensorDataset
    train_ds = TensorDataset(train_h, train_y)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    best_val_acc = 0.0
    best_epoch   = 0
    best_state   = None

    for epoch in range(1, num_epochs + 1):
        classifier.train()
        for h_b, y_b in train_dl:
            h_b, y_b = h_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = classifier(h_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_acc = compute_accuracy(classifier, val_h, val_y, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_state   = {k: v.clone() for k, v in classifier.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    epoch {epoch:>3}/{num_epochs}  "
                  f"val_acc={val_acc:.4f}")

    # Restore best
    if best_state is not None:
        classifier.load_state_dict(best_state)

    return {"best_val_acc": best_val_acc, "best_epoch": best_epoch}


def compute_accuracy(
    classifier,
    h: torch.Tensor,
    y: torch.Tensor,
    device: str,
    batch_size: int = 512,
) -> float:
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(h), batch_size):
            h_b = h[i:i+batch_size].to(device)
            y_b = y[i:i+batch_size].to(device)
            preds = classifier(h_b).argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total   += y_b.size(0)
    return correct / max(total, 1)
