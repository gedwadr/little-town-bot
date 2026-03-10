import json
import math
import os
import time
import torch
from datetime import datetime

from accelerate.utils import add_model_config_to_megatron_parser
from flask import Flask, request, jsonify
from collections import defaultdict

STATS_FILE = "./stats/training_stats.jsonl"
MOVE_FILE = "./stats/move_records.jsonl"
BOT_LOG_FILE = "./logs/bot_actions.jsonl"
os.makedirs("./stats", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

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

for name, param in bot.model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable_parameter = [p for p in bot.model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_parameter, lr=1e-5)
game_count = 0

episodes = defaultdict(lambda: {
    "actions":  [],   # log_prob of each action taken
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
    global game_count
    data      = request.json
    game_id   = data["game_id"]
    player_id = str(data["player_id"])
    log_data  = {"log": data["log"]}
    majapahit_moves = data.get("moves", [])

    # Reconstruct full game state from log
    game_parser = GameParser.from_live_data(log_data)

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
        game_count  = game_count
    )

    has_activation = "activate" in action
    activation_moves = [m for m in legal_moves if "activate" in m]

    move_record = {
        'legal_moves_count': len(legal_moves),
        'activations_count': len(activation_moves),
        'activating': has_activation,
        'action': action
    }
    with open(MOVE_FILE, "a") as f:
        f.write(json.dumps(move_record) + "\n")

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
    vp_now = game_parser.players.states[player_id].get("vp", 0)

    ep["player_id"] = player_id
    if ep["actions"]:  # not the first turn
        vp_delta = vp_now - ep.get("vp_at_last_turn", vp_now)
        step_reward = max(0, vp_delta * 0.1) if game_count < 100 else vp_delta * 0.1
        step_reward += 0.02
        ep["rewards"].append(step_reward)

        # Debug — should now be non-zero when buildings activated
        print(f"[REWARD] game={game_id} vp_delta={vp_delta} "
              f"step_reward={step_reward:.3f}")

    ep["actions"].append(action)


    ep["vp_at_last_turn"] = vp_now

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

    # ── Settle last turn's step reward ───────────────────────────
    # final_vp includes end-game building bonuses not captured during play
    # vp_at_last_turn is the VP we had at the start of our last action
    vp_at_last_turn = ep.get("vp_at_last_turn", final_vp)
    last_turn_vp_delta = final_vp - vp_at_last_turn

    if last_turn_vp_delta != 0:
        print(f"End-game VP bonus detected: +{last_turn_vp_delta} VP "
              f"(from {vp_at_last_turn} → {final_vp})")

    last_step_reward = max(0, last_turn_vp_delta * 0.1) if game_count < 100 else last_turn_vp_delta * 0.1
    last_step_reward += 0.02
    ep["rewards"].append(last_step_reward)

    # ── Final placement bonus ─────────────────────────────────────
    placement_bonus = {
        2: {1: +0.3,  2: -0.3},
        3: {1: +0.3,  2:  0.0, 3: -0.3},
        4: {1: +0.3,  2: +0.1, 3: -0.1, 4: -0.3},
    }
    final_reward = placement_bonus.get(n_players, {}).get(position, 0.0)
    ep["rewards"].append(final_reward)

    print(f"Game {game_id} ended | position={position}/{n_players} | "
          f"final_vp={final_vp} | vp_last_turn={vp_at_last_turn} | "
          f"end_game_bonus_vp={last_turn_vp_delta} | "
          f"rewards={ep['rewards']}")

    # Run RL update
    _rl_update(ep, bot.model, bot.tokenizer)
    game_count += 1

    game_stats.append({
        "game":     game_count,
        "won":      position == 1,
        "position": position,
        "vp":       final_vp,
        "duration": duration_seconds,
    })

    if game_count % 10 == 0:
        bot.model.save_pretrained(f"./rl_checkpoints/game_{game_count}")
        print(f"Saved checkpoint at game {game_count}")

    if game_count % 10 == 0:
        _write_stats_checkpoint(game_count)

    del episodes[game_id]

    return jsonify({"status": "updated", "position": position})


def compute_log_prob(model, tokenizer, prompt, action, device):
    """Recompute log prob of action given prompt, WITH gradient tracking."""
    # full_text  = prompt + "\n" + action
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    action_ids = tokenizer(
        action,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    prompt_len = prompt_ids.shape[1]

    # Forward pass WITH gradient tracking (no torch.no_grad here)
    outputs = model(prompt_ids)
    logits  = outputs.logits[:, -1, :]   # [1, vocab_size] — last token only

    # Only compute loss over action tokens, not prompt tokens
    log_probs_vocab = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs_vocab[0, action_ids[0]]  # [action_len]

    return token_log_probs.mean()


def _write_stats_checkpoint(up_to_game: int):
    """Aggregate the last 50 games and append one line to the stats file."""
    window = game_stats[-10:]
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


def _rl_update(episodes_data, model, tokenizer):
    """REINFORCE update on the adapter weights."""
    actions = episodes_data["actions"]
    rewards = episodes_data["rewards"]

    if not actions or not rewards:
        print("No actions or rewards, not upodating RL")
        return

    while len(rewards) < len(actions):
        rewards.append(0.0)

    # Discounted returns
    # print(f"actions: {actions}, rewards: {rewards}")
    gamma   = 0.99
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, dtype=torch.float32)
    # print(f"returns: {returns.tolist()}")

    # Normalise (stabilises gradients)
    # if returns.std() > 1e-8:
    #     returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    device = next(bot.model.parameters()).device
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device)

    for action_str, G in zip(actions, returns):
        # Tokenize action only — very short, fits easily
        action_ids = tokenizer(
            action_str,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(device)

        # Forward pass on action tokens only
        outputs = model(action_ids)
        logits = outputs.logits[:, :-1, :]  # [1, len-1, vocab]
        targets = action_ids[:, 1:]  # [1, len-1]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_lp = log_probs[0, range(targets.shape[1]), targets[0]].mean()

        total_loss = total_loss + (-G * token_lp)

        # Free immediately after each action
        del outputs, logits, log_probs, action_ids, targets
        torch.cuda.empty_cache()

    total_loss = total_loss / len(actions)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    print(f"RL update: {len(actions)} actions | "
          f"mean_return={returns.mean():.3f} | "
          f"loss={total_loss.item():.4f}")


if __name__ == "__main__":
    app.run(port=9000)