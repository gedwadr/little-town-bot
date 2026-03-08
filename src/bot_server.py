import json
import os
import time
import torch
from datetime import datetime
from flask import Flask, request, jsonify
from collections import defaultdict

STATS_FILE = "./stats/training_stats.jsonl"
os.makedirs("./stats", exist_ok=True)

# Per-game stats accumulated until the next 50-game checkpoint
game_stats: list[dict] = []

from src.games.utils import sample_action
from src.games.game_parser import GameParser
from src.games.botV1 import BotV1
from src.games.action_validator import ActionValidator

app = Flask(__name__)

bot = BotV1(
        model_path="./models/qwen2.5-1.5b-instruct",
        adapter_path="./checkpoints/RL/checkpoint-1331"
    )
bot.model.train()

for name, param in bot.model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable_parameter = [p for p in bot.model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_parameter, lr=1e-5)
game_count = 0

episodes = defaultdict(lambda: {
    "log_probs":  [],   # log_prob of each action taken
    "rewards":    [],   # step rewards (VP gained per turn)
    "player_id":  None,
})


@app.route("/get-move", methods=["POST"])
def get_move():
    """
    Called by game server each time the bot needs to make a move.
    Body: { game_id, player_id, log: [...], moves: [{move, args}, ...] }
    Returns: { actions: [{move, args}, ...] }
    """
    data      = request.json
    game_id   = data["game_id"]
    player_id = str(data["player_id"])
    log_data  = {"log": data["log"]}
    majapahit_moves = data.get("moves", [])

    # Reconstruct full game state from log
    game_parser = GameParser.from_live_data(log_data)
    vp_before = game_parser.players.states[player_id].get("vp", 0)

    action_validator = ActionValidator(game_parser)
    legal_moves = action_validator.get_legal_moves(player_id)
    if not legal_moves:
        print("no legal moves")
        return jsonify({"actions": [], "error": "no legal moves"})

    # Build prompt
    prompt = game_parser.build_prompt(player_id=player_id)
    full_prompt = prompt["system"] + "\n" + prompt["user"]

    # Sample action (with log_prob for RL)
    action, log_prob = sample_action(
        model       = bot.model,
        tokenizer   = bot.tokenizer,
        prompt      = full_prompt,
        legal_moves = legal_moves,
    )

    # Convert the sampled string action to Majapahit { move, args } format.
    # Falls back to the first legal Majapahit move if conversion fails.
    majapahit_action = action_validator.to_majapahit_move(action, player_id, majapahit_moves)
    if majapahit_action is None:
        if not majapahit_moves:
            return jsonify({"actions": [], "error": "could not convert action and no fallback moves"})
        majapahit_action = majapahit_moves[0]

    # Compute step reward: VP gained since last turn
    # (approximate — we don't know VP after yet, so we store vp_before
    #  and compute the delta on the NEXT call)
    ep = episodes[game_id]
    ep["player_id"] = player_id

    # Settle previous step's reward now that we have the new state
    if ep["log_probs"]:  # not the first turn
        vp_delta = (vp_before - ep.get("vp_at_last_turn", vp_before)) * 0.05
        ep["rewards"].append(vp_delta)

    ep["log_probs"].append(log_prob)
    ep["vp_at_last_turn"] = vp_before

    return jsonify({"actions": [majapahit_action]})

@app.route("/game-result", methods=["POST"])
def game_result():
    """
    Called by game server when game ends.
    Body: { game_id, player_id, position, n_players,
            duration_seconds, final_vp, all_scores }
    """
    global game_count
    data             = request.json
    game_id          = data["game_id"]
    position         = data["position"]          # 1 = winner
    n_players        = data.get("n_players", 4)
    duration_seconds = data.get("duration_seconds", 0)
    final_vp         = data.get("final_vp", 0)

    ep = episodes.get(game_id)
    if not ep or not ep["log_probs"]:
        return jsonify({"status": "no episode data"})

    # Settle the final step reward
    placement_bonus = {
        2: {1: +1.0, 2: -1.0},
        3: {1: +1.0, 2:  0.0, 3: -1.0},
        4: {1: +1.0, 2: +0.3, 3: -0.3, 4: -1.0},
    }
    final_reward = placement_bonus.get(n_players, {}).get(position, 0.0)
    ep["rewards"].append(final_reward)

    # Pad rewards to match log_probs length if needed
    while len(ep["rewards"]) < len(ep["log_probs"]):
        ep["rewards"].append(0.0)

    # Run RL update
    _rl_update(ep["log_probs"], ep["rewards"])
    game_count += 1

    # Record stats for this game
    game_stats.append({
        "game":     game_count,
        "won":      position == 1,
        "position": position,
        "vp":       final_vp,
        "duration": duration_seconds,
    })

    if game_count % 50 == 0:
        bot.model.save_pretrained(f"./rl_checkpoints/game_{game_count}")
        print(f"Saved checkpoint at game {game_count}")
        _write_stats_checkpoint(game_count)

    # Clean up
    del episodes[game_id]

    return jsonify({"status": "updated", "position": position})


def _write_stats_checkpoint(up_to_game: int):
    """Aggregate the last 50 games and append one line to the stats file."""
    window = game_stats[-50:]
    if not window:
        return
    n          = len(window)
    wins       = sum(1 for g in window if g["won"])
    avg_vp     = sum(g["vp"]       for g in window) / n
    avg_dur    = sum(g["duration"] for g in window) / n
    avg_pos    = sum(g["position"] for g in window) / n
    record = {
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
    print(
        f"[stats] games={up_to_game}  win_rate={record['win_rate']:.2%}  "
        f"avg_vp={record['avg_vp']:.1f}  avg_dur={record['avg_duration_sec']:.0f}s  "
        f"avg_pos={record['avg_position']:.2f}"
    )


def _rl_update(log_probs: list, rewards: list):
    """REINFORCE update on the adapter weights."""
    # Discounted returns
    gamma   = 0.99
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, dtype=torch.float32)

    # Normalise (stabilises gradients)
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # REINFORCE loss
    loss = torch.tensor(0.0, requires_grad=True)
    for log_prob, G in zip(log_probs, returns):
        lp = torch.tensor(log_prob, requires_grad=True)
        loss = loss + (-G * lp)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(bot.model.parameters(), max_norm=1.0)
    optimizer.step()

    print(f"RL update: {len(log_probs)} actions, "
          f"mean_return={returns.mean():.3f}, loss={loss.item():.4f}")


if __name__ == "__main__":
    app.run(port=9000)