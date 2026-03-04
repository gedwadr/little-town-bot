import re
from typing import Optional

from src.SFT.model import Turn
from src.constant import BUILDINGS


def building_cost_str(bid: int) -> str:
    b = BUILDINGS.get(bid, {})
    cost = b.get("cost", {})
    return ", ".join(f"{v} {k}" for k, v in cost.items())


def building_summary(bid: int) -> str:
    b = BUILDINGS.get(bid, {})
    cost = building_cost_str(bid)
    return f"{b['name']} (cost: {cost} | buildVP: {b['buildVP']} | {b['effect']})"


def format_action(event: dict) -> Optional[str]:
    """Convert a decision event to a clean one-line action string."""
    action  = event["action"]
    details = event.get("details", "")

    if action == "placeWorker":
        m = re.search(r'\((\d+),(\d+)\)', details)
        if m:
            return f"placeWorker {m.group(1)} {m.group(2)}"

    elif action == "buildBuilding":
        m = re.search(r'Built (.+?) at \((\d+),(\d+)\)', details)
        if m:
            return f"buildBuilding {m.group(1).strip()} {m.group(2)} {m.group(3)}"

    elif action == "activate":
        m = re.search(r'^(.+?) at \((\d+),(\d+)\)', details)
        if m:
            return f"activate {m.group(1).strip()} {m.group(2)} {m.group(3)}"

    elif action == "substituteResource":
        # "3 coins -> 1 wood"  →  "substituteResource wood"
        m = re.search(r'-> \d+ (\w+)', details)
        if m:
            return f"substituteResource {m.group(1)}"

    return None


def format_turn_for_history(turn: Turn) -> str:
    """One-line summary of a completed turn for the history section."""
    parts = []
    for e in turn.decision_events:
        a = format_action(e)
        if a:
            parts.append(a)
    if turn.gather_details:
        parts.append("gathered: " + "; ".join(turn.gather_details))
    prefix = f"R{turn.round_num}T{turn.turn_num} P{turn.player_id}"
    body   = " | ".join(parts) if parts else "(no decisions)"
    return f"{prefix}: {body}"