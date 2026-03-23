"""
frozen_nn_bot_server.py
=======================
Read-only bot server supporting three model modes. No RL updates.

Select mode via BOT_MODE env var (default: resnet).
  resnet   — BoardGameBot (flat ResNet encoder)
  cnn      — BoardCNNBot  (CNN board encoder + MLP global encoder)
  ensemble — EnsembleBot  (ResNet + CNN, score-level fusion)

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

from flask import Flask, request, jsonify

from src.SFT.res_net.model_components import BoardGameBot
from src.SFT.res_net import STATE_DIM, ACTION_DIM, HIDDEN_DIM, N_BLOCKS
from src.SFT.cnn.model_components import BoardCNNBot
from src.SFT.ensemble.ensemble_model import EnsembleBot
from src.games.game_parser_nn import GameParserNN
from src.server_common import load_nn_submodel, write_stats_checkpoint

# ── Config ────────────────────────────────────────────────────────────────────
BOT_MODE = os.environ.get("BOT_MODE", "resnet").lower()
assert BOT_MODE in ("resnet", "cnn", "ensemble"), f"Unknown BOT_MODE: {BOT_MODE}"

RESNET_CHECKPOINT_PATH   = os.environ.get("RESNET_CHECKPOINT",   "./ensemble_checkpoints/best_resnet.pt")
CNN_CHECKPOINT_PATH      = os.environ.get("CNN_CHECKPOINT",      "./ensemble_checkpoints/best_cnn.pt")
ENSEMBLE_CHECKPOINT_PATH = os.environ.get("ENSEMBLE_CHECKPOINT", "./ensemble_checkpoints/best.pt")
RL_CHECKPOINT_PATH       = os.environ.get("RL_CHECKPOINT",       "./nn_rl_all_checkpoints/ensemble/v2/game_940.pt")

STATS_FILE = "./stats/frozen_nn_training_stats.jsonl"
MOVE_FILE  = "./stats/frozen_nn_move_records.jsonl"

os.makedirs("./stats", exist_ok=True)

# ── Model (frozen — eval mode, no optimizer) ──────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  BOT_MODE: {BOT_MODE}")


if BOT_MODE == "resnet":
    model = load_nn_submodel(
        BoardGameBot, RESNET_CHECKPOINT_PATH, "ResNet",
        state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, n_blocks=N_BLOCKS,
    ).to(device)

elif BOT_MODE == "cnn":
    model = load_nn_submodel(BoardCNNBot, CNN_CHECKPOINT_PATH, "CNN").to(device)

else:  # ensemble
    resnet = load_nn_submodel(
        BoardGameBot, RESNET_CHECKPOINT_PATH, "ResNet (ensemble)",
        state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, n_blocks=N_BLOCKS,
    )
    cnn = load_nn_submodel(BoardCNNBot, CNN_CHECKPOINT_PATH, "CNN (ensemble)")

    if os.path.exists(ENSEMBLE_CHECKPOINT_PATH):
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

# Overlay RL checkpoint if available (works for any mode)
if os.path.exists(RL_CHECKPOINT_PATH):
    rl_ckpt = torch.load(RL_CHECKPOINT_PATH, map_location=device)
    try:
        model.load_state_dict(rl_ckpt["state_dict"])
        print(f"Loaded RL checkpoint: {RL_CHECKPOINT_PATH} "
              f"(game={rl_ckpt.get('game','?')})")
    except RuntimeError as e:
        print(f"[WARN] RL checkpoint incompatible (arch mismatch?), ignoring: {e}")
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
        write_stats_checkpoint(game_stats, STATS_FILE, game_count)

    if game_id in episodes:
        del episodes[game_id]

    return jsonify({"status": "ok", "position": position})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(port=9002)
