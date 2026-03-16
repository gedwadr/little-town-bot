import json
import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import torch
import bitsandbytes as bnb
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
        adapter_path="./checkpoints/boardgame-v2/all_player/checkpoint-2066"
    )

for name, param in bot.model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable_parameter = [p for p in bot.model.parameters() if p.requires_grad]
optimizer = bnb.optim.PagedAdamW8bit(trainable_parameter, lr=1e-5)
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
    model_moves, move_map = action_validator.get_legal_moves_v2(player_id)

    illegal_majapahit = data.get("illegal_moves", [])
    if illegal_majapahit:
        illegal_set = {
            (m["move"], json.dumps(m.get("args", []), sort_keys=True))
            for m in illegal_majapahit
        }
        model_moves = [
            mm for mm in model_moves
            if move_map.get(mm) and
               (move_map[mm][0]["move"], json.dumps(move_map[mm][0].get("args", []), sort_keys=True)) not in illegal_set
        ]
        if illegal_majapahit:
            print(f"[illegal_moves] filtered {len(illegal_majapahit)} illegal moves, {len(model_moves)} remaining")

    if not model_moves:
        print("no legal moves")
        return jsonify({"actions": [], "error": "no legal moves"})

    # Build prompt
    prompt = game_parser.build_prompt(player_id=player_id)
    full_prompt = prompt["user"]

    # Sample action (with log_prob for RL)
    action, log_prob = sample_action(
        model       = bot.model,
        tokenizer   = bot.tokenizer,
        prompt      = full_prompt,
        legal_moves = model_moves,
        game_count  = game_count
    )

    has_activation = action.startswith("work ")
    activation_moves = [m for m in model_moves if m.startswith("work ")]

    move_record = {
        'legal_moves_count': len(model_moves),
        'activations_count': len(activation_moves),
        'activating': has_activation,
        'action': action
    }
    with open(MOVE_FILE, "a") as f:
        f.write(json.dumps(move_record) + "\n")

    # Convert sampled action to Majapahit { move, args } format via move_map.
    # Falls back to server-provided moves if not found.
    candidates = move_map.get(action, [])
    if candidates:
        majapahit_action = candidates[0]
    else:
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

        # ── VP reward (always present) ────────────────────────────
        vp_reward = max(0, vp_delta * 0.1)

        # ── Feeding reward (fades out after game 300) ─────────────
        feeding_reward = _compute_feeding_reward(
            state      = game_parser.players.states.get(player_id, {}),
            prev_food  = ep.get("food_at_last_turn", 0),
            game_count = game_count,
        )

        step_reward = vp_reward + feeding_reward
        ep["rewards"].append(step_reward)

        print(f"[REWARD] game={game_id} | vp_delta={vp_delta:.2f} "
              f"vp_reward={vp_reward:.3f} | "
              f"feeding_reward={feeding_reward:.3f} | "
              f"step_reward={step_reward:.3f}")

    ep["actions"].append({"prompt": full_prompt[:-300], "action": action})


    ep["vp_at_last_turn"] = vp_now

    # Track food for feeding reward next turn
    resources = game_parser.players.states.get(player_id, {}).get("resources", {})
    ep["food_at_last_turn"] = resources.get("fish", 0) + resources.get("wheat", 0)

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

    last_step_reward = max(0, last_turn_vp_delta * 0.1)
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
    torch.cuda.empty_cache()
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
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1216,
    ).input_ids.to(device)

    action_ids = tokenizer(
        action,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    outputs = model(prompt_ids)
    logits  = outputs.logits[:, -1, :].clone()  # ← clone to detach from outputs
    del outputs                                  # ← free immediately after clone
    torch.cuda.empty_cache()

    log_probs_vocab = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs_vocab[0, action_ids[0]]

    result = token_log_probs.mean()

    del logits, log_probs_vocab, token_log_probs, prompt_ids, action_ids

    return result


def _compute_feeding_reward(state: dict, prev_food: int, game_count: int) -> float:
    """
    Reward the bot for gathering food (fish + wheat) toward feeding all workers.

    Logic:
      - food needed  = number of workers placed this round
      - food gained  = food now minus food at start of last turn
      - reward ramps up as food approaches the workers threshold
      - no reward once food exceeds workers (already safe — no point hoarding)
      - fades to zero linearly between game 200 and game 300
    """
    # Fade schedule: full reward 0-200 games, linear decay 200-300, zero after 300
    if game_count >= 300:
        return 0.0
    fade = 1.0 if game_count < 200 else (300 - game_count) / 100.0

    # Read current food and workers from player state
    resources     = state.get("resources", {})
    fish_now      = resources.get("fish", 0)
    wheat_now     = resources.get("wheat", 0)
    food_now      = fish_now + wheat_now

    food_needed = state.get("totalWorkers", 5)

    if food_needed == 0:
        return 0.0   # no workers placed yet — nothing to feed

    # Already overfed — no reward for hoarding beyond what's needed
    if prev_food >= food_needed:
        return 0.0

    # Reward = how much of the gap we closed this turn
    # gap_before: how short we were before this action
    food_gained = max(0, food_now - prev_food)
    gap_before = max(0, food_needed - prev_food)
    gap_closed = min(food_gained, gap_before)   # can't close more than existed

    # Scale: closing the last gap (e.g. 1→2 when need 2) is more valuable
    # than the first point (0→1 when need 5) — sigmoid-style scaling
    completion_ratio = (food_now / food_needed)
    importance = min(1.0, completion_ratio)   # 0.0 → 1.0 as food fills up

    feeding_reward = gap_closed * importance * fade

    return feeding_reward


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
    total_loss_val = 0.0

    for action_data, G in zip(actions, returns):
        optimizer.zero_grad()

        token_lp = compute_log_prob(
            model     = model,
            tokenizer = tokenizer,
            prompt    = action_data["prompt"],
            action    = action_data["action"],
            device    = device,
        )
        token_lp = torch.clamp(token_lp, min=-2.0, max=0.0)

        loss = -G * token_lp / len(actions)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
        optimizer.step()

        total_loss_val += loss.item()
        del token_lp, loss
    torch.cuda.empty_cache()

    print(f"RL update: {len(actions)} actions | "
          f"mean_return={returns.mean():.3f} | "
          f"loss={total_loss_val:.4f}")


if __name__ == "__main__":
    app.run(port=9000)