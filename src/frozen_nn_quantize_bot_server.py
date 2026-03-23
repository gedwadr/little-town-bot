"""
frozen_nn_quantize_bot_server.py
=================================
Read-only bot server that loads a pre-quantized INT8 model (produced by
train_nn_bot_server.py every 10 games as game_N_int8.pt).

Runs entirely on CPU — no CUDA required. The model was saved with
torch.quantization.quantize_dynamic so nn.Linear weights are INT8;
no conversion step is needed here, just load and eval.

Endpoints:
  POST /get-move      — pick an action (greedy argmax, CPU inference)
  POST /game-result   — log result, no weight update

Set QUANTIZED_CHECKPOINT env var to point at the int8 checkpoint file.
Default: ./nn_rl_all_checkpoints/ensemble/v2/game_940_int8.pt

Port: 9003
"""

import json
import os
import torch
import torch.nn.functional as F
from collections import defaultdict

from flask import Flask, request, jsonify

from src.games.game_parser_nn import GameParserNN
from src.server_common import write_stats_checkpoint

# ── Config ────────────────────────────────────────────────────────────────────
QUANTIZED_CHECKPOINT = os.environ.get(
    "QUANTIZED_CHECKPOINT",
    "./nn_rl_all_checkpoints/ensemble/v2/game_940_int8.pt",
)
STATS_FILE = "./stats/frozen_quantize_nn_training_stats.jsonl"
MOVE_FILE  = "./stats/frozen_quantize_nn_move_records.jsonl"

os.makedirs("./stats", exist_ok=True)

# ── Model (CPU-only, INT8) ────────────────────────────────────────────────────
device = torch.device("cpu")
print(f"Device: {device}  |  Quantized checkpoint: {QUANTIZED_CHECKPOINT}")

if not os.path.exists(QUANTIZED_CHECKPOINT):
    raise FileNotFoundError(
        f"Quantized checkpoint not found: {QUANTIZED_CHECKPOINT}\n"
        f"Run train_nn_bot_server.py first to generate int8 checkpoints."
    )

ckpt  = torch.load(QUANTIZED_CHECKPOINT, map_location="cpu", weights_only=False)
model = ckpt["model"]
model.eval()

bot_mode = ckpt.get("bot_mode", "unknown")
print(f"Loaded INT8 model from {QUANTIZED_CHECKPOINT}  "
      f"(game={ckpt.get('game', '?')}, bot_mode={bot_mode})")

# ── Server state ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.json.sort_keys = False
game_count = 0
game_stats: list[dict] = []

episodes = defaultdict(lambda: {
    "player_id":       None,
    "vp_at_last_turn": None,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _select_action(state_vec: list, candidate_vecs: list):
    """Greedy argmax on CPU with the INT8 model."""
    state_t = torch.tensor(state_vec,      dtype=torch.float32)
    cands_t = torch.tensor(candidate_vecs, dtype=torch.float32)

    with torch.no_grad():
        scores   = model(state_t.unsqueeze(0), cands_t.unsqueeze(0))[0]  # [N]
        idx      = scores.argmax().item()
        probs    = F.softmax(scores, dim=-1)
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

    candidate_dicts = data.get("moves", [])
    if not candidate_dicts:
        print(f"[{game_id}] no legal moves")
        return jsonify({"actions": [], "error": "no legal moves"})

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
    app.run(port=9003)
