import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import torch
import bitsandbytes as bnb
import requests
from torch.cuda.amp import GradScaler
from flask import Flask, request, jsonify
from collections import defaultdict

from src.games.utils import sample_action
from src.games.game_parser import GameParser
from src.games.botV1 import BotV1
from src.server_common import (
    get_placement_bonus, apply_reward_manipulation,
    discounted_returns, write_stats_checkpoint,
)

STATS_FILE        = "./stats/llm_training_stats_with_1000+_matches.jsonl"
RL_CHECKPOINT_DIR = "./llm_rl_checkpoints"
os.makedirs(RL_CHECKPOINT_DIR, exist_ok=True)
os.makedirs("./stats", exist_ok=True)

URL = "http://localhost:8000/api/bot-match"
payload = {
    "numPlayers": 2,
    "boardSide": random.choice(["A", "B"]),
    "randomizeTurnOrder": True,
    "bots": [
        {"playerID": "0"},
        {"playerID": "1", "serviceUrl": "http://localhost:9000", "botName": "LLM-RL"},
    ],
}

game_stats: list[dict] = []

app = Flask(__name__)
app.json.sort_keys = False

bot = BotV1(
    model_path="./models/qwen2.5-0.5b",
    adapter_path="./checkpoints/llm-qwen/v1/final-adapter/",
)
device = bot.device

for name, param in bot.model.named_parameters():
    param.requires_grad = "lora" in name.lower()

trainable_params = [p for p in bot.model.parameters() if p.requires_grad]
optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=1e-5)
scaler = GradScaler(enabled=device.type == "cuda")
game_count = 0

episodes = defaultdict(lambda: {
    "actions":         [],
    "rewards":         [],
    "player_id":       None,
    "vp_at_last_turn": None,
})


@app.route("/get-move", methods=["POST"])
def get_move():
    global game_count
    data      = request.json
    game_id   = data["game_id"]
    player_id = str(data["player_id"])
    log_data  = {"log": data["log"]}

    # Legal moves come directly from the Majapahit server (same as train_nn_bot_server)
    candidate_dicts = data.get("moves", [])
    if not candidate_dicts:
        print("no legal moves")
        return jsonify({"actions": [], "error": "no legal moves"})

    game_parser = GameParser.from_live_data(log_data)
    prompt      = game_parser.build_prompt(player_id=player_id)
    full_prompt = prompt["user"]

    # Convert majapahit moves to text labels for the LLM to score
    model_moves = [
        f"{m['move']} {json.dumps(m.get('args', []))}"
        for m in candidate_dicts
    ]

    action, log_prob = sample_action(
        model       = bot.model,
        tokenizer   = bot.tokenizer,
        prompt      = full_prompt,
        legal_moves = model_moves,
        game_count  = game_count,
    )

    chosen_idx       = model_moves.index(action)
    majapahit_action = candidate_dicts[chosen_idx]

    ep = episodes[game_id]
    ep["player_id"] = player_id
    vp_now = game_parser.players.states[player_id].get("vp", 0)

    if ep["actions"]:
        vp_delta    = vp_now - ep.get("vp_at_last_turn", vp_now)
        step_reward = max(0, vp_delta * 0.1)
        ep["rewards"].append(step_reward)
        print(f"[REWARD] game={game_id} vp_delta={vp_delta:.2f} step_reward={step_reward:.3f}")

    ep["actions"].append({"prompt": full_prompt[:-300], "action": action})
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

    placement_score = get_placement_bonus(n_players, position)
    final_reward    = 1.0 if placement_score > 0 else -1.0
    ep["rewards"].append(final_reward)

    print(f"Game {game_id} ended | position={position}/{n_players} | "
          f"final_vp={final_vp} | rewards={ep['rewards']}")

    _rl_update(ep)
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
        ckpt_path = f"{RL_CHECKPOINT_DIR}/game_{game_count}"
        bot.model.save_pretrained(ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
        write_stats_checkpoint(game_stats, STATS_FILE, game_count)

    del episodes[game_id]

    try:
        res = requests.post(URL, json=payload, timeout=10)
        print(f"[cron] match started: {res.json()}")
    except Exception as e:
        print(f"[cron] error: {e}")

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
    logits  = outputs.logits[:, -1, :].clone()
    del outputs
    torch.cuda.empty_cache()

    log_probs_vocab = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs_vocab[0, action_ids[0]]
    result = token_log_probs.mean()
    del logits, log_probs_vocab, token_log_probs, prompt_ids, action_ids
    return result


def _rl_update(ep: dict):
    actions = ep["actions"]
    rewards = ep["rewards"]

    apply_reward_manipulation(rewards)

    if not actions or not rewards:
        print("No actions or rewards — skipping RL update")
        return

    while len(rewards) < len(actions):
        rewards.append(0.0)

    returns   = discounted_returns(rewards[:len(actions)])
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    bot.model.train()
    optimizer.zero_grad()
    total_loss_val = 0.0

    for action_data, G in zip(actions, returns_t):
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            log_prob = compute_log_prob(
                model     = bot.model,
                tokenizer = bot.tokenizer,
                prompt    = action_data["prompt"],
                action    = action_data["action"],
                device    = device,
            )
            log_prob = torch.clamp(log_prob, min=-2.0, max=0.0)
            loss = -G * log_prob / len(actions)
        scaler.scale(loss).backward()   # free activations immediately
        total_loss_val += loss.item()
        torch.cuda.empty_cache()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(bot.model.parameters(), max_norm=0.3)
    scaler.step(optimizer)
    scaler.update()
    bot.model.eval()

    print(f"Game Count: {game_count} | RL update: {len(actions)} actions | "
          f"mean_return={returns_t.mean():.3f} | loss={total_loss_val:.4f}")


if __name__ == "__main__":
    app.run(port=9000)
