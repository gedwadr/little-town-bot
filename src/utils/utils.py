import re
from typing import Optional

from src.SFT.model import Turn
from src.constant import BUILDINGS, BUILDING_NAME_TO_ID

# ── resource name abbreviations ───────────────────────────────────────────────

RES_ABBREV = {"fish": "fi", "wheat": "wh", "wood": "wd", "stone": "st", "coins": "cn"}
RES_EXPAND = {v: k for k, v in RES_ABBREV.items()}


def abbrev_str(s: str) -> str:
    """Replace full resource/coin names with short tokens in any display string."""
    for full, short in RES_ABBREV.items():
        s = s.replace(full, short)
    return s


def building_cost_str(bid: int) -> str:
    b = BUILDINGS.get(bid, {})
    cost = b.get("cost", {})
    return " ".join(f"{v}{RES_ABBREV.get(k, k)}" for k, v in cost.items())


def building_summary(bid: int) -> str:
    b = BUILDINGS.get(bid, {})
    cost = building_cost_str(bid)
    return f"{b['name']} (cost: {cost} | buildVP: {b['buildVP']} | {b['effect']})"


# ── building effect simulator ─────────────────────────────────────────────────

def apply_effect(effect: str, res: dict, coins: int, vp: int) -> tuple:
    """
    Simulate a worker-trigger building effect.
    Returns (res, coins, vp) — res is a NEW dict (not mutated in-place).
    Skips conversions the player cannot afford.
    """
    res = dict(res)

    def _get(name):
        if name == "coins": return coins
        if name == "vp":    return vp
        return res.get(name, 0)

    def _add(name, amt):
        nonlocal coins, vp
        if name == "coins": coins += amt
        elif name == "vp":  vp += amt
        else: res[name] = res.get(name, 0) + amt

    def _sub(name, amt):
        nonlocal coins, vp
        if name == "coins": coins -= amt
        elif name == "vp":  vp -= amt
        else: res[name] = res.get(name, 0) - amt

    # gain X:N
    m = re.match(r'gain (\w+):(\d+)$', effect)
    if m:
        _add(m.group(1), int(m.group(2)))
        return res, coins, vp

    # convert IN -> OUT  (IN may be "a:N+b:M", OUT may be "a:N + b:M")
    m = re.match(r'convert (.+?) -> (.+)', effect)
    if m:
        in_str, out_str = m.group(1), m.group(2)

        in_parts = [
            (pm.group(1), int(pm.group(2)))
            for part in in_str.split('+')
            for pm in [re.match(r'(\w+):(\d+)', part.strip())]
            if pm
        ]

        if all(_get(name) >= amt for name, amt in in_parts):
            for name, amt in in_parts:
                _sub(name, amt)
            for part in re.split(r'\s*\+\s*', out_str):
                pm = re.match(r'(\w+):(\d+)', part.strip())
                if pm:
                    _add(pm.group(1), int(pm.group(2)))

    return res, coins, vp


# ── state formatting ──────────────────────────────────────────────────────────

def _state_str(ps: dict) -> str:
    r = ps.get("resources", {})
    return (
        f"fi:{r.get('fish', 0)}, wh:{r.get('wheat', 0)}, "
        f"st:{r.get('stone', 0)}, wd:{r.get('wood', 0)}, "
        f"cn:{ps.get('coins', 0)}, vp:{ps.get('vp', 0)}"
    )


def _get_state_after(turn: Turn, last_event: dict) -> Optional[dict]:
    """
    Walk turn.all_events to find the last playerState at or after last_event,
    stopping before the next decision event that follows last_event.
    """
    found = False
    last_ps = None

    # Build set of decision event ids that come AFTER last_event
    after_ids: set = set()
    found_last = False
    for e in turn.decision_events:
        if found_last:
            after_ids.add(id(e))
        if id(e) == id(last_event):
            found_last = True

    for e in turn.all_events:
        if id(e) == id(last_event):
            found = True
        if not found:
            continue
        ps = e.get("playerState")
        if ps:
            last_ps = ps
        if id(e) != id(last_event) and id(e) in after_ids:
            break

    return last_ps


# ── action formatting ─────────────────────────────────────────────────────────

def format_turn_actions(turn: Turn) -> list:
    """
    Return one action string per logical action group in the turn:
      - placeWorker + its consecutive activate events  →  one "work (...)" line
      - buildBuilding                                  →  one "build (...)" line
      - substituteResource                             →  one "sub (...)" line

    Each line embeds the player's state AFTER that group completes.
    """
    lines = []
    events = turn.decision_events
    i = 0

    while i < len(events):
        e = events[i]
        action  = e["action"]
        details = e.get("details", "")

        if action == "placeWorker":
            # Collect all consecutive activate events
            j = i + 1
            while j < len(events) and events[j]["action"] == "activate":
                j += 1
            last_in_group = events[j - 1]  # last activate, or the placeWorker itself
            ps = _get_state_after(turn, last_in_group)

            m = re.search(r'\((\d+),(\d+)\)', details)
            if m:
                r, c = m.group(1), m.group(2)
                state = f", {_state_str(ps)}" if ps else ""
                lines.append(f"work (row:{r}, col:{c}{state})")
            i = j

        elif action == "buildBuilding":
            ps = _get_state_after(turn, e)
            m = re.search(r'Built (.+?) at \((\d+),(\d+)\)', details)
            if m:
                name = m.group(1).strip()
                r, c  = m.group(2), m.group(3)
                bid   = BUILDING_NAME_TO_ID.get(name, -1)
                if ps:
                    build_vp = BUILDINGS.get(bid, {}).get("buildVP", 0)
                    adjusted = dict(ps)
                    adjusted["vp"] = ps.get("vp", 0) + build_vp
                    # Slots after building: check gameState snapshot first,
                    # fall back to playerState, then subtract 1 for this build.
                    gs_pstate = e.get("gameState", {}).get("players", {}).get(turn.player_id, {})
                    pre_slots = gs_pstate.get("housesRemaining", ps.get("housesRemaining"))
                    sl_str = f", sl:{pre_slots - 1}" if pre_slots is not None else ""
                    state = f", {_state_str(adjusted)}{sl_str}"
                else:
                    state = ""
                lines.append(f"build (B{bid}, row:{r}, col:{c}{state})")
            i += 1

        elif action == "substituteResource":
            # Collect all consecutive subs and combine into one line
            gained = []
            while i < len(events) and events[i]["action"] == "substituteResource":
                sub_m = re.search(r'-> \d+ (\w+)', events[i].get("details", ""))
                if sub_m:
                    gained.append(sub_m.group(1))
                i += 1
            if gained:
                lines.append("sub (" + ", ".join(f"{RES_ABBREV.get(r, r)}+1" for r in gained) + ")")

        else:
            # standalone activate or unknown — skip
            i += 1

    return lines


def format_turn_for_history(turn: Turn) -> str:
    """One-line summary of a completed turn using the combined action format."""
    parts  = format_turn_actions(turn)
    prefix = f"R{turn.round_num}T{turn.turn_num} P{turn.player_id}"
    body   = " | ".join(parts) if parts else "(no decisions)"
    return f"{prefix}: {body}"


# kept for any callers that still use it directly
def format_action(event: dict) -> Optional[str]:
    """Compact single-event action string (model output format)."""
    action  = event["action"]
    details = event.get("details", "")

    if action == "placeWorker":
        m = re.search(r'\((\d+),(\d+)\)', details)
        if m:
            return f"work {m.group(1)} {m.group(2)}"

    elif action == "buildBuilding":
        m = re.search(r'Built (.+?) at \((\d+),(\d+)\)', details)
        if m:
            return f"build {m.group(1).strip()} {m.group(2)} {m.group(3)}"

    elif action == "activate":
        m = re.search(r'^(.+?) at \((\d+),(\d+)\)', details)
        if m:
            return f"act {m.group(1).strip()} {m.group(2)} {m.group(3)}"

    elif action == "substituteResource":
        m = re.search(r'-> \d+ (\w+)', details)
        if m:
            return f"sub {m.group(1)}"

    return None
