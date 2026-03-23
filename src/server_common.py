"""Shared utilities for bot_server.py and train_nn_bot_server.py."""

import json
from datetime import datetime

# Placement bonus values shared by all bot servers
PLACEMENT_BONUS = {
    2: {1: +1.0,  2: -1.0},
    3: {1: +1.0,  2: -0.3, 3: -1.0},
    4: {1: +1.0,  2: +0.1, 3: -0.3, 4: -1.0},
}


def get_placement_bonus(n_players: int, position: int) -> float:
    return PLACEMENT_BONUS.get(n_players, {}).get(position, 0.0)


def apply_reward_manipulation(rewards: list) -> None:
    """Set all intermediate rewards to final_reward / 2 (in-place)."""
    for i in range(len(rewards) - 1):
        rewards[i] = rewards[-1] / 2


def discounted_returns(rewards: list, gamma: float = 0.99) -> list:
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def write_stats_checkpoint(game_stats: list, stats_file: str, up_to_game: int) -> None:
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
    with open(stats_file, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(
        f"[stats] games={up_to_game}  win_rate={record['win_rate']:.2%}  "
        f"avg_vp={record['avg_vp']:.1f}  avg_pos={record['avg_position']:.2f}"
    )
