import os
import random

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

from src.SFT.cnn import DATA_PATH, SEED, VAL_SPLIT, WEIGHT_DECAY, EPOCHS, LR, CHECKPOINT_DIR, PATIENCE
from src.SFT.res_net.dataset_components import GameDataset, collate_variable_candidates
from src.SFT.cnn.model_components import BoardCNNBot


def train(data_path: str = DATA_PATH):
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load and split dataset ──
    dataset = GameDataset(data_path)
    n = len(dataset)
    n_val = max(1, int(n * VAL_SPLIT))
    n_train = n - n_val

    indices = list(range(n))
    random.shuffle(indices)
    train_set = torch.utils.data.Subset(dataset, indices[:n_train])
    val_set   = torch.utils.data.Subset(dataset, indices[n_train:])

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=32, shuffle=True,
        collate_fn=collate_variable_candidates,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=32, shuffle=False,
        collate_fn=collate_variable_candidates,
    )

    print(f"Train: {n_train}  Val: {n_val}  Total: {n}")

    # ── Model ──
    model = BoardCNNBot().to(device)
    print(f"Parameters: {model.n_params:,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_acc    = 0.0
    patience_counter = 0

    # ── Training loop ──
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for states, candidates, corrects, mask, weights in train_loader:
            states     = states.to(device)
            candidates = candidates.to(device)
            corrects   = corrects.to(device)
            mask       = mask.to(device)
            weights    = weights.to(device)

            scores = model(states, candidates)   # [B, max_N]
            scores[~mask] = float('-inf')

            loss = (F.cross_entropy(scores, corrects, reduction='none') * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * len(corrects)
            train_correct += (scores.argmax(dim=-1) == corrects).sum().item()
            train_total   += len(corrects)

        # ── Validation ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for states, candidates, corrects, mask, weights in val_loader:
                states     = states.to(device)
                candidates = candidates.to(device)
                corrects   = corrects.to(device)
                mask       = mask.to(device)
                weights    = weights.to(device)

                scores = model(states, candidates)
                scores[~mask] = float('-inf')

                loss = (F.cross_entropy(scores, corrects, reduction='none') * weights).mean()

                val_loss    += loss.item() * len(corrects)
                val_correct += (scores.argmax(dim=-1) == corrects).sum().item()
                val_total   += len(corrects)

        scheduler.step()

        t_acc = train_correct / train_total * 100
        v_acc = val_correct   / val_total   * 100
        t_loss = train_loss   / train_total
        v_loss = val_loss     / val_total

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train loss={t_loss:.4f} acc={t_acc:.1f}% | "
              f"val loss={v_loss:.4f} acc={v_acc:.1f}%")

        if v_acc > best_val_acc:
            patience_counter = 0
            best_val_acc = v_acc
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_acc": v_acc,
                "val_loss": v_loss,
            }, f"{CHECKPOINT_DIR}/best.pt")
            print(f"  ✓ New best val acc: {v_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} — no improvement for {PATIENCE} epochs")
                break

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc": v_acc,
            }, f"{CHECKPOINT_DIR}/epoch_{epoch}.pt")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.1f}%")
    print(f"Best model saved to {CHECKPOINT_DIR}/best.pt")


if __name__ == "__main__":
    train(data_path=DATA_PATH)
