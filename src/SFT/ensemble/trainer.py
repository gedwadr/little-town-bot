"""
Ensemble Joint Trainer
======================
Trains ResNet and CNN simultaneously through a single computation graph.
One forward pass → one loss → one backward → both models updated together.

The EnsembleBot wraps both sub-models so their parameters are visible to a
single AdamW optimizer.  The fusion weights are also learnable.

Optional: warm-start either sub-model from an existing checkpoint, or load
the full ensemble from an RL checkpoint to continue fine-tuning.

Usage
-----
    python -m src.SFT.ensemble.trainer

    # warm-start from pretrained sub-models:
    python -m src.SFT.ensemble.trainer \
        --resnet_ckpt nn_all_checkpoints_2p/best.pt \
        --cnn_ckpt    cnn_checkpoints_2p/best.pt

    # continue from RL-trained ensemble:
    python -m src.SFT.ensemble.trainer \
        --rl_ckpt nn_rl_all_checkpoints/ensemble/game_240.pt
"""

import argparse
import os
import random

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.SFT.ensemble import DATA_PATH, SEED, VAL_SPLIT, WEIGHT_DECAY, EPOCHS, LR, CHECKPOINT_DIR, PATIENCE
from src.SFT.res_net.dataset_components import GameDataset, collate_variable_candidates
from src.SFT.res_net.model_components import BoardGameBot
from src.SFT.cnn.model_components import BoardCNNBot
from src.SFT.ensemble.ensemble_model import EnsembleBot


def _load_resnet(ckpt_path: str, device: str) -> BoardGameBot:
    model = BoardGameBot()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  ResNet loaded from {ckpt_path}  (epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc', '?')})")
    return model


def _load_cnn(ckpt_path: str, device: str) -> BoardCNNBot:
    model = BoardCNNBot()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  CNN loaded from {ckpt_path}  (epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc', '?')})")
    return model


def _load_rl_ensemble(ckpt_path: str, device: str) -> EnsembleBot:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ensemble = EnsembleBot(BoardGameBot(), BoardCNNBot(), learnable_weights=True)
    ensemble.load_state_dict(ckpt["state_dict"])
    print(f"  RL ensemble loaded from {ckpt_path}  (game={ckpt.get('game', '?')})")
    return ensemble


def train(
    data_path: str = DATA_PATH,
    resnet_ckpt: str | None = None,
    cnn_ckpt: str | None = None,
    rl_ckpt: str | None = None,
):
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = GameDataset(data_path)
    n = len(dataset)
    n_val   = max(1, int(n * VAL_SPLIT))
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

    # ── Models ────────────────────────────────────────────────────────────────
    print("Loading sub-models...")
    if rl_ckpt:
        if resnet_ckpt or cnn_ckpt:
            raise ValueError("--rl_ckpt cannot be combined with --resnet_ckpt or --cnn_ckpt")
        ensemble = _load_rl_ensemble(rl_ckpt, device).to(device)
    else:
        resnet = _load_resnet(resnet_ckpt, device) if resnet_ckpt else BoardGameBot()
        cnn    = _load_cnn(cnn_ckpt, device)       if cnn_ckpt    else BoardCNNBot()
        ensemble = EnsembleBot(resnet, cnn, learnable_weights=True).to(device)
    print(f"Ensemble parameters: {ensemble.n_params:,}  "
          f"(resnet={sum(p.numel() for p in ensemble.resnet.parameters() if p.requires_grad):,}  "
          f"cnn={sum(p.numel() for p in ensemble.cnn.parameters() if p.requires_grad):,})")

    optimizer = AdamW(ensemble.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_acc     = 0.0
    patience_counter = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        ensemble.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for states, candidates, corrects, mask, weights in train_loader:
            states     = states.to(device)
            candidates = candidates.to(device)
            corrects   = corrects.to(device)
            mask       = mask.to(device)
            weights    = weights.to(device)

            # Single forward pass through both sub-models simultaneously
            scores = ensemble(states, candidates)   # [B, max_N]
            scores[~mask] = float('-inf')

            loss = (F.cross_entropy(scores, corrects, reduction='none') * weights).mean()

            # Single backward — gradients flow into both ResNet and CNN
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * len(corrects)
            train_correct += (scores.argmax(dim=-1) == corrects).sum().item()
            train_total   += len(corrects)

        # ── Validation ───────────────────────────────────────────────────────
        ensemble.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for states, candidates, corrects, mask, weights in val_loader:
                states     = states.to(device)
                candidates = candidates.to(device)
                corrects   = corrects.to(device)
                mask       = mask.to(device)
                weights    = weights.to(device)

                scores = ensemble(states, candidates)
                scores[~mask] = float('-inf')

                loss = (F.cross_entropy(scores, corrects, reduction='none') * weights).mean()

                val_loss    += loss.item() * len(corrects)
                val_correct += (scores.argmax(dim=-1) == corrects).sum().item()
                val_total   += len(corrects)

        scheduler.step()

        t_acc  = train_correct / train_total * 100
        v_acc  = val_correct   / val_total   * 100
        t_loss = train_loss    / train_total
        v_loss = val_loss      / val_total
        w      = ensemble.weights.tolist()

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train loss={t_loss:.4f} acc={t_acc:.1f}% | "
              f"val loss={v_loss:.4f} acc={v_acc:.1f}% | "
              f"w=[{w[0]:.3f},{w[1]:.3f}]")

        if v_acc > best_val_acc:
            patience_counter = 0
            best_val_acc = v_acc

            # Save full ensemble
            torch.save({
                "epoch": epoch,
                "state_dict": ensemble.state_dict(),
                "val_acc": v_acc,
                "val_loss": v_loss,
                "weights": w,
            }, f"{CHECKPOINT_DIR}/best.pt")

            # Save individual sub-models so they can be loaded independently
            torch.save({
                "epoch": epoch,
                "state_dict": ensemble.resnet.state_dict(),
                "val_acc": v_acc,
                "val_loss": v_loss,
            }, f"{CHECKPOINT_DIR}/best_resnet.pt")

            torch.save({
                "epoch": epoch,
                "state_dict": ensemble.cnn.state_dict(),
                "val_acc": v_acc,
                "val_loss": v_loss,
            }, f"{CHECKPOINT_DIR}/best_cnn.pt")

            print(f"  ✓ New best val acc: {v_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} — no improvement for {PATIENCE} epochs")
                break

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "state_dict": ensemble.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc": v_acc,
                "weights": w,
            }, f"{CHECKPOINT_DIR}/epoch_{epoch}.pt")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.1f}%")
    print(f"Checkpoints saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",   default=DATA_PATH)
    parser.add_argument("--resnet_ckpt", default=None, help="Optional pretrained ResNet checkpoint")
    parser.add_argument("--cnn_ckpt",    default=None, help="Optional pretrained CNN checkpoint")
    parser.add_argument("--rl_ckpt",     default=None, help="RL-trained ensemble checkpoint to fine-tune from")
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        resnet_ckpt=args.resnet_ckpt,
        cnn_ckpt=args.cnn_ckpt,
        rl_ckpt=args.rl_ckpt,
    )
