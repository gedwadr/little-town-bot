"""
frozen_nn_bot_server.py
=======================
Read-only bot server using BoardGameBot (ResNet). No RL updates.

Loads the SFT checkpoint first, then overlays an RL checkpoint on top
if one is specified via the RL_CHECKPOINT env var or the default path.

Endpoints:
  POST /get-move      — pick an action (inference only)
  POST /game-result   — log result, no weight update

Port: 9002
"""

import json
import os
import torch
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime

from flask import Flask, request, jsonify

from src.SFT.res_net.model_components import BoardGameBot
from src.SFT.res_net import STATE_DIM, ACTION_DIM, HIDDEN_DIM, N_BLOCKS
from src.games.game_parser_nn import GameParserNN

# ── Paths ─────────────────────────────────────────────────────────────────────
# Base SFT checkpoint
SFT_CHECKPOINT_PATH = "./nn_all_checkpoints/best.pt"
# Optional RL checkpoint to overlay (overrides SFT weights if present)
RL_CHECKPOINT_PATH  = os.environ.get("RL_CHECKPOINT", "./nn_rl_all_checkpoints/heurystic/3p/v1/game_240.pt")

STATS_FILE = "./stats/frozen_nn_training_stats.jsonl"
MOVE_FILE  = "./stats/frozen_nn_move_records.jsonl"

os.makedirs("./stats", exist_ok=True)

# ── Model (frozen — eval mode, no optimizer) ──────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = BoardGameBot(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    hidden_dim=HIDDEN_DIM,
    n_blocks=N_BLOCKS,
).to(device)

# Load SFT base weights
if os.path.exists(SFT_CHECKPOINT_PATH):
    ckpt = torch.load(SFT_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded SFT checkpoint: {SFT_CHECKPOINT_PATH} "
          f"(epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc','?')})")
else:
    print(f"[WARN] No SFT checkpoint at {SFT_CHECKPOINT_PATH} — starting from random weights")

# Overlay RL checkpoint if available
if os.path.exists(RL_CHECKPOINT_PATH):
    rl_ckpt = torch.load(RL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(rl_ckpt["state_dict"])
    print(f"Loaded RL checkpoint: {RL_CHECKPOINT_PATH} "
          f"(game={rl_ckpt.get('game','?')})")
else:
    print(f"[INFO] No RL checkpoint at {RL_CHECKPOINT_PATH} — using SFT weights only")

model.eval()  # frozen: no dropout, no gradient tracking

# ── Server state ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.json.sort_keys = False  # preserve key order from Majapahit server
game_count = 0
game_stats: list[dict] = []

episodes = defaultdict(lambda: {
    "player_id":       None,
    "vp_at_last_turn": None,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _select_action(state_vec: list, candidate_vecs: list):
    """Greedy argmax — no temperature, no sampling, no gradient."""
    state_t = torch.tensor(state_vec,      dtype=torch.float32, device=device)
    cands_t = torch.tensor(candidate_vecs, dtype=torch.float32, device=device)

    with torch.no_grad():
        scores = model(state_t.unsqueeze(0), cands_t.unsqueeze(0))[0]  # [N]
        idx    = scores.argmax().item()
        probs  = F.softmax(scores, dim=-1)
        log_prob = torch.log(probs[idx] + 1e-8).item()

    return idx, log_prob


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/get-move", methods=["POST"])
def get_move():
    global game_count
    data      = request.json
    game_id   = data["game_id"]
    player_id = str(data["player_id"])
    log_data  = {"log": data["log"]}

    # Legal moves come directly from the Majapahit server
    candidate_dicts = data.get("moves", [])
    if not candidate_dicts:
        print(f"[{game_id}] no legal moves")
        return jsonify({"actions": [], "error": "no legal moves"})

    # Build state vector and encode each candidate action
    parser         = GameParserNN.from_live_data(log_data)
    state_vec      = parser.build_state_vector(player_id)
    candidate_vecs = [parser.encode_action(m, player_id) for m in candidate_dicts]

    chosen_idx, log_prob = _select_action(state_vec, candidate_vecs)
    majapahit_action = candidate_dicts[chosen_idx]

    ep = episodes[game_id]
    ep["player_id"] = player_id

    vp_now = parser.players.states[player_id].get("vp", 0)
    if ep["vp_at_last_turn"] is not None:
        vp_delta = vp_now - ep["vp_at_last_turn"]
        print(f"[INFO] game={game_id} vp_delta={vp_delta}")

    ep["vp_at_last_turn"] = vp_now

    with open(MOVE_FILE, "a") as f:
        f.write(json.dumps({
            "game_id":      game_id,
            "player_id":    player_id,
            "n_candidates": len(candidate_dicts),
            "chosen_idx":   chosen_idx,
            "log_prob":     round(log_prob, 4),
            "chosen_move":  majapahit_action["move"],
        }) + "\n")

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

    print(f"Game {game_id} ended | position={position}/{n_players} | final_vp={final_vp}")

    game_count += 1
    game_stats.append({
        "game":     game_count,
        "won":      position == 1,
        "position": position,
        "vp":       final_vp,
        "duration": duration_seconds,
    })

    if game_count % 10 == 0:
        _write_stats_checkpoint(game_count)

    if game_id in episodes:
        del episodes[game_id]

    return jsonify({"status": "ok", "position": position})


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
    app.run(port=9002)
