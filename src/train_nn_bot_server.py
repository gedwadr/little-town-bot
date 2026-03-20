"""
train_nn_bot_server.py
================
Flask bot server supporting three model modes:
  resnet   — BoardGameBot (flat ResNet encoder)
  cnn      — BoardCNNBot  (CNN board encoder + MLP global encoder)
  ensemble — EnsembleBot  (ResNet + CNN, score-level fusion)

Select mode via BOT_MODE env var (default: resnet).

Endpoints:
  POST /get-move      — pick an action, record (state, candidates, chosen_idx)
  POST /game-result   — run REINFORCE update, save checkpoint

RL loop:
  - Forward pass → scores [N] → softmax → sample
  - On game end: compute discounted returns, one backward per step
"""

import copy
import json
import os

import requests
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from collections import defaultdict
from datetime import datetime

from flask import Flask, request, jsonify

from src.SFT.res_net.model_components import BoardGameBot
from src.SFT.res_net import STATE_DIM, ACTION_DIM, HIDDEN_DIM, N_BLOCKS
from src.SFT.cnn.model_components import BoardCNNBot
from src.SFT.ensemble.ensemble_model import EnsembleBot
from src.games.game_parser_nn import GameParserNN
import random

# ── Config ────────────────────────────────────────────────────────────────────
BOT_MODE = os.environ.get("BOT_MODE", "resnet").lower()  # resnet | cnn | ensemble
assert BOT_MODE in ("resnet", "cnn", "ensemble"), f"Unknown BOT_MODE: {BOT_MODE}"

RESNET_CHECKPOINT_PATH   = os.environ.get("RESNET_CHECKPOINT",   "./ensemble_checkpoints/v3/best_resnet.pt")
CNN_CHECKPOINT_PATH      = os.environ.get("CNN_CHECKPOINT",      "./ensemble_checkpoints/v3/best_cnn.pt")
ENSEMBLE_CHECKPOINT_PATH = os.environ.get("ENSEMBLE_CHECKPOINT", "./ensemble_checkpoints/v3/best.pt")
RL_CHECKPOINT_DIR      = f"./nn_rl_all_checkpoints/{BOT_MODE}/v3"
RL_CHECKPOINT_PATH     = os.environ.get("RL_CHECKPOINT", f"")
STATS_FILE = "./stats/nn_training_stats.jsonl"

os.makedirs(RL_CHECKPOINT_DIR, exist_ok=True)
os.makedirs("./stats", exist_ok=True)

## continous game
URL = "http://localhost:8000/api/bot-match"
payload = {
    "numPlayers": 2,
    "boardSide": random.choice(["A", "B"]),
    "randomizeTurnOrder": True,
    "cpuSearchProfile": "extreme",
    "bots": [
        {"playerID": "0"},
        {"playerID": "1", "serviceUrl": "http://localhost:9001", "botName": "Ensemble-RL"},
    ],
}

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  BOT_MODE: {BOT_MODE}")


def _load_submodel(cls, path, label, **kwargs):
    """Instantiate cls, load checkpoint if it exists, return model (cpu)."""
    m = cls(**kwargs)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        try:
            m.load_state_dict(ckpt["state_dict"])
            print(f"Loaded {label} from {path} "
                  f"(epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc','?')})")
        except RuntimeError as e:
            print(f"[WARN] {label} checkpoint incompatible, ignoring: {e}")
    else:
        print(f"[WARN] No {label} checkpoint at {path} — using random weights")
    return m


if BOT_MODE == "resnet":
    model = _load_submodel(
        BoardGameBot, RESNET_CHECKPOINT_PATH, "ResNet",
        state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, n_blocks=N_BLOCKS,
    ).to(device)

elif BOT_MODE == "cnn":
    model = _load_submodel(BoardCNNBot, CNN_CHECKPOINT_PATH, "CNN").to(device)

else:  # ensemble
    resnet = _load_submodel(
        BoardGameBot, RESNET_CHECKPOINT_PATH, "ResNet (ensemble)",
        state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, n_blocks=N_BLOCKS,
    )
    cnn = _load_submodel(BoardCNNBot, CNN_CHECKPOINT_PATH, "CNN (ensemble)")

    if os.path.exists(ENSEMBLE_CHECKPOINT_PATH):
        # Load full jointly-trained ensemble (learnable fusion weights)
        ckpt = torch.load(ENSEMBLE_CHECKPOINT_PATH, map_location="cpu")
        model = EnsembleBot(resnet, cnn, learnable_weights=True)
        try:
            model.load_state_dict(ckpt["state_dict"])
            w = ckpt.get("weights", [0.5, 0.5])
            print(f"Loaded ensemble from {ENSEMBLE_CHECKPOINT_PATH} "
                  f"(epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc','?')}, "
                  f"w=[{w[0]:.3f},{w[1]:.3f}])")
        except RuntimeError as e:
            print(f"[WARN] Ensemble checkpoint incompatible, ignoring: {e}")
        model = model.to(device)
    else:
        print(f"[WARN] No ensemble checkpoint at {ENSEMBLE_CHECKPOINT_PATH} — fusing sub-models with equal weights")
        model = EnsembleBot(resnet, cnn, learnable_weights=False).to(device)

# Optionally resume from an RL checkpoint (any mode)
if RL_CHECKPOINT_PATH and os.path.exists(RL_CHECKPOINT_PATH):
    rl_ckpt = torch.load(RL_CHECKPOINT_PATH, map_location=device)
    try:
        model.load_state_dict(rl_ckpt["state_dict"])
        print(f"Loaded RL checkpoint: {RL_CHECKPOINT_PATH} "
              f"(game={rl_ckpt.get('game','?')})")
    except RuntimeError as e:
        print(f"[WARN] RL checkpoint incompatible (arch changed?), ignoring: {e}")

model.eval()  # dropout off during inference; switched to train() only inside _rl_update

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
scaler = GradScaler(enabled=device.type == "cuda")

# ── Server state ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.json.sort_keys = False
game_count = 0
game_stats: list[dict] = []

# Per-game episode data
# steps: list of (state_tensor, candidates_tensor, chosen_idx)
episodes = defaultdict(lambda: {
    "steps":     [],   # (state [D], candidates [N,11], chosen_idx int)
    "rewards":   [],
    "player_id": None,
    "vp_at_last_turn": None,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _temperature(game_idx: int) -> float:
    """Decay from 1.5 → 0.5 over first 500 games, then hold."""
    return max(0.5, 1.5 - game_idx / 500.0)


def _select_action(state_vec: list, candidate_vecs: list, temp: float):
    """
    Run one forward pass and sample a candidate index.
    Returns (chosen_idx, log_prob_float).
    """
    state_t = torch.tensor(state_vec, dtype=torch.float32, device=device)
    cands_t = torch.tensor(candidate_vecs, dtype=torch.float32, device=device)

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        scores = model(state_t.unsqueeze(0), cands_t.unsqueeze(0))[0]
        idx = scores.argmax().item()
        probs = F.softmax(scores.float(), dim=-1)
        log_prob = torch.log(probs[idx] + 1e-8).item()

    return idx, log_prob


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/get-move", methods=["POST"])
def get_move():
    global game_count
    data = request.json
    game_id   = data["game_id"]
    player_id = str(data["player_id"])
    log_data  = {"log": data["log"]}

    # Legal moves come directly from the Majapahit server
    candidate_dicts = data.get("moves", [])
    if not candidate_dicts:
        print(f"[{game_id}] no legal moves")
        return jsonify({"actions": [], "error": "no legal moves"})

    # Build state vector and encode each candidate action
    parser = GameParserNN.from_live_data(log_data)
    state_vec      = parser.build_state_vector(player_id)
    candidate_vecs = [parser.encode_action(m, player_id) for m in candidate_dicts]

    # Sample
    temp = _temperature(game_count)
    chosen_idx, log_prob = _select_action(state_vec, candidate_vecs, temp)
    majapahit_action = candidate_dicts[chosen_idx]

    # Record step for RL
    ep = episodes[game_id]
    ep["player_id"] = player_id

    vp_now = parser.players.states[player_id].get("vp", 0)
    if ep["steps"]:  # not first turn — reward is for the action taken last turn
        vp_delta = vp_now - ep.get("vp_at_last_turn", vp_now)
        step_reward = max(0, vp_delta * 0.1)
        ep["rewards"].append(step_reward)
        print(f"[REWARD] game={game_id} vp_delta={vp_delta} step_reward={step_reward:.3f}")

    ep["steps"].append((state_vec, candidate_vecs, chosen_idx))
    ep["vp_at_last_turn"] = vp_now

    # # Log
    # with open(MOVE_FILE, "a") as f:
    #     f.write(json.dumps({
    #         "game_id":          game_id,
    #         "player_id":        player_id,
    #         "n_candidates":     len(candidate_dicts),
    #         "chosen_idx":       chosen_idx,
    #         "log_prob":         round(log_prob, 4),
    #         "chosen_move":      majapahit_action["move"],
    #     }) + "\n")

    return jsonify({"actions": [majapahit_action]})


@app.route("/game-result", methods=["POST"])
def game_result():
    global game_count
    data             = request.json
    game_id          = data["game_id"]
    position         = data["position"]
    n_players        = data.get("n_players", 4)
    duration_seconds = data.get("duration_seconds", 0)
    final_vp         = data.get("final_vp", 0)

    ep = episodes.get(game_id)
    if ep is None:
        return jsonify({"status": "unknown_game"})

    # Settle last step reward (end-game VP delta)
    vp_at_last_turn    = ep.get("vp_at_last_turn", final_vp)
    last_turn_vp_delta = final_vp - vp_at_last_turn
    last_step_reward   = last_turn_vp_delta * 0.1
    ep["rewards"].append(last_step_reward)

    # Final placement bonus
    placement_bonus = {
        2: {1: +1.0,  2: -1.0},
        3: {1: +1.0,  2: -0.3, 3: -1.0},
        4: {1: +1.0,  2: +0.1, 3: -0.3, 4: -1.0},
    }
    final_reward = placement_bonus.get(n_players, {}).get(position, 0.0)
    ep["rewards"].append(final_reward)

    print(f"Game {game_id} ended | position={position}/{n_players} | "
          f"final_vp={final_vp} | rewards={ep['rewards']}")

    _rl_update(ep)
    game_count += 1

    game_stats.append({
        "game":     game_count,
        "won":      position == 1,
        "position": position,
        "vp":       final_vp,
        "duration": duration_seconds,
    })

    if game_count % 10 == 0:
        ckpt_path = f"{RL_CHECKPOINT_DIR}/game_{game_count}.pt"
        torch.save({"game": game_count, "bot_mode": BOT_MODE, "state_dict": model.state_dict()}, ckpt_path)
        print(f"Saved RL checkpoint: {ckpt_path}")

        # Save INT8 dynamically-quantized model for deployment (CPU, Linear layers only)
        # quantize_dynamic replaces nn.Linear with DynamicQuantizedLinear, so we
        # save the full model object rather than just the state_dict.
        quantized_path = f"{RL_CHECKPOINT_DIR}/game_{game_count}_int8.pt"
        cpu_copy = copy.deepcopy(model).cpu().eval()
        int8_model = torch.quantization.quantize_dynamic(
            cpu_copy, {torch.nn.Linear}, dtype=torch.qint8
        )
        torch.save({"game": game_count, "bot_mode": BOT_MODE, "model": int8_model}, quantized_path)
        print(f"Saved INT8 quantized checkpoint: {quantized_path}")

    if game_count % 10 == 0:
        _write_stats_checkpoint(game_count)

    del episodes[game_id]

    try:
        res = requests.post(URL, json=payload, timeout=10)
        data = res.json()
        print(f"[cron] match started: {data}")
    except Exception as e:
        print(f"[cron] error: {e}")


    return jsonify({"status": "updated", "position": position})


# ── RL update ─────────────────────────────────────────────────────────────────

def _rl_update(ep: dict):
    """REINFORCE update on model weights."""
    steps   = ep["steps"]    # list of (state_vec, candidate_vecs, chosen_idx)
    rewards = ep["rewards"]
    for i in range(len(rewards)):
        if i != len(rewards) - 1:
            rewards[i] = rewards[-1] / 2
    print(rewards)
    if not steps or not rewards:
        print("No steps or rewards — skipping RL update")
        return

    # Align lengths (last step may be missing a reward)
    while len(rewards) < len(steps):
        rewards.append(0.0)

    # Discounted returns
    gamma   = 0.99
    returns = []
    R = 0.0
    for r in reversed(rewards[:len(steps)]):
        R = r + gamma * R
        returns.insert(0, R)

    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    model.train()
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for (state_vec, candidate_vecs, chosen_idx), G in zip(steps, returns_t):
            state_t = torch.tensor(state_vec, dtype=torch.float32, device=device)
            cands_t = torch.tensor(candidate_vecs, dtype=torch.float32, device=device)

            scores = model(state_t.unsqueeze(0), cands_t.unsqueeze(0))[0]
            log_prob = F.log_softmax(scores.float(), dim=-1)[chosen_idx]

            total_loss = total_loss + (-G * log_prob)

    total_loss = total_loss / len(steps)
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
    scaler.step(optimizer)
    scaler.update()
    model.eval()

    print(f"Game Count: {game_count} | "
          f"RL update: {len(steps)} steps | "
          f"mean_return={returns_t.mean():.3f} | "
          f"loss={total_loss.item():.4f}")


# ── Stats ─────────────────────────────────────────────────────────────────────

def _write_stats_checkpoint(up_to_game: int):
    window = game_stats[-10:]
    if not window:
        return
    n       = len(window)
    wins    = sum(1 for g in window if g["won"])
    avg_vp  = sum(g["vp"]       for g in window) / n
    avg_dur = sum(g["duration"] for g in window) / n
    avg_pos = sum(g["position"] for g in window) / n
    record  = {
        "timestamp":        datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "games_so_far":     up_to_game,
        "window":           n,
        "win_rate":         round(wins / n, 4),
        "avg_position":     round(avg_pos, 2),
        "avg_vp":           round(avg_vp, 2),
        "avg_duration_sec": round(avg_dur, 1),
    }
    with open(STATS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[stats] games={up_to_game}  win_rate={record['win_rate']:.2%}  "
          f"avg_vp={record['avg_vp']:.1f}  avg_pos={record['avg_position']:.2f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(port=9001)
