import re
from itertools import chain, combinations

from src.games.game_parser import GameParser
from src.constant import BUILDING_NAME_TO_ID, BUILDINGS
from src.utils.utils import RES_EXPAND, RES_ABBREV


# ── building effect simulator (abbreviated state keys) ────────────────────────

# Maps full resource names used in BUILDINGS effects → abbreviated state keys
_EFF_KEY = {
    "fish": "fi", "wheat": "wh", "wood": "wd",
    "stone": "st", "coins": "cn", "vp": "vp",
}


def _apply_building_effect(bid: int, current_state: dict) -> dict:
    """
    Apply a worker-trigger building's effect to current_state.

    current_state keys: fi, wh, st, wd, cn, vp
    Returns a NEW dict with same keys.
    Non-worker triggers (endOfRound/endOfGame/none) are skipped.
    Conversions the player cannot afford leave state unchanged.
    """
    bdata = BUILDINGS.get(bid, {})
    if bdata.get("trigger") != "worker":
        return current_state          # endOfRound / endOfGame / none — no activation

    effect = bdata.get("effect", "")
    s = dict(current_state)

    def _k(name: str) -> str:
        return _EFF_KEY.get(name, name)

    # gain X:N
    m = re.match(r'gain (\w+):(\d+)$', effect)
    if m:
        k = _k(m.group(1))
        s[k] = s.get(k, 0) + int(m.group(2))
        return s

    # convert IN -> OUT  (IN may be a:N+b:M; OUT may be a:N + b:M)
    m = re.match(r'convert (.+?) -> (.+)', effect)
    if m:
        in_str, out_str = m.group(1), m.group(2)

        in_parts = [
            (_k(pm.group(1)), int(pm.group(2)))
            for part in in_str.split('+')
            for pm in [re.match(r'(\w+):(\d+)', part.strip())]
            if pm
        ]

        if all(s.get(k, 0) >= v for k, v in in_parts):
            for k, v in in_parts:
                s[k] = s.get(k, 0) - v
            for part in re.split(r'\s*\+\s*', out_str):
                pm = re.match(r'(\w+):(\d+)', part.strip())
                if pm:
                    k = _k(pm.group(1))
                    s[k] = s.get(k, 0) + int(pm.group(2))

    return s


def _powerset(lst: list):
    """All subsets of lst, from empty to full."""
    return chain.from_iterable(combinations(lst, n) for n in range(len(lst) + 1))


# ── action name helpers ───────────────────────────────────────────────────────

_ACTION_MAP = {
    "work":  "placeWorker",
    "build": "buildBuilding",
    "act":   "activate",
    "sub":   "substituteResource",
}


def _normalize(token: str) -> str:
    """Expand short action name to canonical name."""
    return _ACTION_MAP.get(token, token)


def _state_str_from_dict(res: dict, coins: int, vp: int) -> str:
    return (
        f"fi:{res.get('fish', 0)}, wh:{res.get('wheat', 0)}, "
        f"st:{res.get('stone', 0)}, wd:{res.get('wood', 0)}, "
        f"cn:{coins}, vp:{vp}"
    )


# ── action format parsers ─────────────────────────────────────────────────────

def _parse_work(line: str):
    """Parse 'work (row:R, col:C, ...)' → (row, col) or None."""
    m = re.search(r'row:(\d+)[,\s]+col:(\d+)', line)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _parse_build(line: str):
    """Parse 'build (B{id}, row:R, col:C, ...)' → (bid, row, col) or None."""
    m = re.search(r'B(\d+),\s*row:(\d+)[,\s]+col:(\d+)', line)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _parse_sub(line: str):
    """Parse 'sub (wood+1)' or 'sub (wood+1, stone+1)' → list of resources."""
    resources = re.findall(r'(\w+)\+1', line)
    return resources if resources else None


# ── validator / converter class ───────────────────────────────────────────────

class ActionValidator:
    """
    Filters raw model output to only legal actions given the current game state.
    Uses the parser's live state to check what's actually possible.

    Action format (both input and output):
      work (row:R, col:C, fish:F, wheat:W, stone:S, wood:D, gold:G, vp:V)
      build (name:N, row:R, col:C, fish:F, ...)
      sub (resource:R, fish:F, ...)
    """

    def __init__(self, parser: GameParser):
        self.parser = parser

    # ── public API ────────────────────────────────────────────────────────────

    def to_majapahit_move(self, raw_output: str, player_id: str, moves: list) -> dict | None:
        """
        Parse the first valid action from model output and return the matching
        Majapahit { move, args } entry.

        For placeWorker we always activate ALL adjacent buildings (no subset
        selection) — the Majapahit entry that matches the position and has the
        most activations is preferred.

        Returns None if no valid match is found (caller should fall back to moves[0]).
        """
        valid_lines = self.validate(raw_output, player_id)
        if not valid_lines:
            return None

        first = valid_lines[0]
        token = first.split()[0]
        action = _normalize(token)

        if action == "placeWorker":
            parsed = _parse_work(first)
            if parsed is None:
                return None
            row, col = parsed

            candidates = [
                m for m in moves
                if m["move"] == "placeWorker"
                and m["args"][0] == row
                and m["args"][1] == col
            ]
            if not candidates:
                return None

            # Prefer the candidate that activates ALL adjacent buildings
            best = max(
                candidates,
                key=lambda m: len(m["args"][2]) if len(m["args"]) > 2 else 0,
            )
            return best

        elif action == "buildBuilding":
            parsed = _parse_build(first)
            if parsed is None:
                return None
            bid, row, col = parsed
            for m in moves:
                if m["move"] == "buildBuilding" and m["args"] == [bid, row, col]:
                    return m
            return None

        elif action == "substituteResource":
            resources = _parse_sub(first)
            if not resources:
                return None
            resource = RES_EXPAND.get(resources[0], resources[0])  # expand abbrev
            for m in moves:
                if m["move"] == "substituteResource" and m["args"] == [resource]:
                    return m
            return None

        return None

    def validate(self, raw_output: str, player_id: str) -> list[str]:
        """
        Parse raw model output and return only the legal action lines.
        Stops at the first illegal action to preserve action ordering.

        Accepts the combined format:
          work (row:R, col:C, ...)
          build (name:N, row:R, col:C, ...)
          sub (resource:R, ...)
        """
        lines = [l.strip() for l in raw_output.strip().splitlines() if l.strip()]
        valid = []
        state = self.parser.players.states.get(player_id, {})
        board = self.parser.board

        resources = dict(state.get("resources", {}))
        coins     = state.get("coins", 3)
        workers   = state.get("workersRemaining", 0)
        placed_this_turn = []

        for line in lines:
            if not line:
                continue
            token  = line.split()[0]
            action = _normalize(token)

            if action not in ("placeWorker", "buildBuilding", "substituteResource"):
                break  # unknown token — model is hallucinating

            # ── substituteResource ──────────────────────────────
            if action == "substituteResource":
                res_list = _parse_sub(line)
                if not res_list:
                    break
                resource = RES_EXPAND.get(res_list[0], res_list[0])  # expand abbrev
                if resource not in ("wood", "stone", "fish", "wheat"):
                    break
                if coins < 3:
                    break
                coins -= 3
                resources[resource] = resources.get(resource, 0) + 1
                valid.append(line)

            # ── placeWorker ─────────────────────────────────────
            elif action == "placeWorker":
                parsed = _parse_work(line)
                if parsed is None:
                    break
                r, c = parsed
                if workers <= 0:
                    break
                if not board.is_empty_grass(r, c):
                    break
                if (r, c) in placed_this_turn:
                    break
                workers -= 1
                placed_this_turn.append((r, c))
                valid.append(line)

            # ── buildBuilding ───────────────────────────────────
            elif action == "buildBuilding":
                parsed = _parse_build(line)
                if parsed is None:
                    break
                bid, r, c = parsed
                if bid not in self.parser.available_market():
                    break
                if not board.is_empty_grass(r, c):
                    break
                slots = self.parser.players.states[player_id].get("housesRemaining", 0)
                if slots <= 0:
                    break
                cost = BUILDINGS[bid].get("cost", {})
                for res_name, amt in cost.items():
                    if res_name == "coins":
                        if coins < amt:
                            break
                    else:
                        if resources.get(res_name, 0) < amt:
                            break
                else:
                    for res_name, amt in cost.items():
                        if res_name == "coins":
                            coins -= amt
                        else:
                            resources[res_name] = resources.get(res_name, 0) - amt
                    valid.append(line)
                    continue
                break  # couldn't afford

        return valid

    def get_legal_moves(self, player_id: str) -> list[str]:
        """
        Return all legal actions in the combined single-line format:
          work (row:R, col:C, fish:F, wheat:W, stone:S, wood:D, gold:G, vp:V)
          build (name:N, row:R, col:C, fish:F, ...)
          sub (resource:R, fish:F, ...)

        For placeWorker: ONE entry per cell (always activates all adjacent buildings).
        Predicted end-state is embedded so the model can rank moves by outcome.
        """
        moves = []
        state     = self.parser.players.states[player_id]
        board     = self.parser.board
        resources = state.get("resources", {})
        coins     = state.get("coins", 0)
        workers   = state.get("workersRemaining", 0)

        # ── placeWorker ───────────────────────────────────────────────────────
        if workers > 0:
            for r, c in board.empty_grass_cells():
                pred = self.parser.predict_state(player_id, r, c)
                state_str = _state_str_from_dict(
                    pred["resources"], pred["coins"], pred["vp"]
                )
                moves.append(f"work (row:{r}, col:{c}, {state_str})")

        # ── substituteResource ────────────────────────────────────────────────
        if coins >= 3:
            for res_type in ("wood", "stone", "fish", "wheat"):
                moves.append(f"sub ({RES_ABBREV[res_type]}+1)")

        # ── buildBuilding ─────────────────────────────────────────────────────
        slots = state.get("housesRemaining", 0)
        if slots > 0:
            for bid in self.parser.available_market():
                b    = BUILDINGS.get(bid, {})
                cost = b.get("cost", {})
                can_afford = all(
                    (coins >= amt if res_name == "coins"
                     else resources.get(res_name, 0) >= amt)
                    for res_name, amt in cost.items()
                )
                if not can_afford:
                    continue
                pred_res   = dict(resources)
                pred_coins = coins
                pred_vp    = state.get("vp", 0) + b.get("buildVP", 0)
                for res_name, amt in cost.items():
                    if res_name == "coins":
                        pred_coins -= amt
                    else:
                        pred_res[res_name] = pred_res.get(res_name, 0) - amt
                state_str = _state_str_from_dict(pred_res, pred_coins, pred_vp)
                slots_after = slots - 1
                for r, c in board.empty_grass_cells():
                    moves.append(f"build (B{bid}, row:{r}, col:{c}, {state_str}, sl:{slots_after})")

        return moves

    def get_legal_moves_v2(self, player_id: str):
        """
        Returns (model_moves, move_map):
          model_moves : list[str]  — outcome strings (deduplicated)
          move_map    : dict[str, list[dict]]  — outcome → list of Majapahit action dicts

        For placeWorker: enumerates every subset of adjacent worker-trigger buildings
        so the model can choose to skip costly conversions.  Subsets that produce the
        same outcome string are collapsed to one entry (the entry with the most
        activations is kept as the canonical Majapahit dict).
        """
        state     = self.parser.players.states[player_id]
        board     = self.parser.board
        resources = state.get("resources", {})
        coins     = state.get("coins", 0)
        workers   = state.get("workersRemaining", 0)

        move_map: dict[str, list] = {}   # outcome_str -> [majapahit_dict, ...]

        TERRAIN_RESOURCE = {"F": "wood", "M": "stone", "L": "fish", "G": "wheat"}

        # ── placeWorker ───────────────────────────────────────────────────────
        if workers > 0:
            for r, c in board.empty_grass_cells():
                # 1. Terrain resource gains (including own cell)
                terrain_gains = {"fi": 0, "wh": 0, "st": 0, "wd": 0}
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        r2, c2 = r + dr, c + dc
                        if 0 <= r2 < board.rows and 0 <= c2 < board.cols:
                            res = TERRAIN_RESOURCE.get(board.terrain[r2][c2])
                            if res:
                                terrain_gains[_EFF_KEY[res]] += 1

                # 2. Adjacent worker-trigger buildings (exclude own cell)
                adj_buildings = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        r2, c2 = r + dr, c + dc
                        if 0 <= r2 < board.rows and 0 <= c2 < board.cols:
                            b = board.buildings[r2][c2]
                            if b is not None:
                                bid2 = b.get("id", -1)
                                if BUILDINGS.get(bid2, {}).get("trigger") == "worker":
                                    adj_buildings.append((bid2, r2, c2))

                # 3. Base state after terrain gains
                base = {
                    "fi": resources.get("fish",  0) + terrain_gains["fi"],
                    "wh": resources.get("wheat", 0) + terrain_gains["wh"],
                    "st": resources.get("stone", 0) + terrain_gains["st"],
                    "wd": resources.get("wood",  0) + terrain_gains["wd"],
                    "cn": coins,
                    "vp": state.get("vp", 0),
                }

                # 4. Enumerate subsets
                for subset in _powerset(adj_buildings):
                    s = dict(base)
                    for bid2, _, _ in subset:
                        s = _apply_building_effect(bid2, s)

                    outcome = (
                        f"work (row:{r}, col:{c}, "
                        f"fi:{s['fi']}, wh:{s['wh']}, st:{s['st']}, wd:{s['wd']}, "
                        f"cn:{s['cn']}, vp:{s['vp']})"
                    )
                    activations = [{"row": br, "col": bc} for _, br, bc in subset]
                    majapahit   = {"move": "placeWorker", "args": [r, c, activations]}

                    if outcome not in move_map:
                        move_map[outcome] = [majapahit]
                    else:
                        # Keep the candidate with the most activations as canonical
                        existing = move_map[outcome][0]
                        if len(activations) > len(existing["args"][2]):
                            move_map[outcome][0] = majapahit

        # ── substituteResource ────────────────────────────────────────────────
        if coins >= 3:
            for res_type in ("wood", "stone", "fish", "wheat"):
                pred_res   = dict(resources)
                pred_res[res_type] = pred_res.get(res_type, 0) + 1
                pred_coins = coins - 3
                outcome = (
                    f"sub ({RES_ABBREV[res_type]}+1, "
                    f"fi:{pred_res.get('fish',  0)}, wh:{pred_res.get('wheat', 0)}, "
                    f"st:{pred_res.get('stone', 0)}, wd:{pred_res.get('wood',  0)}, "
                    f"cn:{pred_coins}, vp:{state.get('vp', 0)})"
                )
                move_map[outcome] = [{"move": "substituteResource", "args": [res_type]}]

        # ── buildBuilding ─────────────────────────────────────────────────────
        slots = state.get("housesRemaining", 0)
        if slots > 0:
            for bid in self.parser.available_market():
                b    = BUILDINGS.get(bid, {})
                cost = b.get("cost", {})
                can_afford = all(
                    (coins >= amt if res_name == "coins"
                     else resources.get(res_name, 0) >= amt)
                    for res_name, amt in cost.items()
                )
                if not can_afford:
                    continue
                pred_res   = dict(resources)
                pred_coins = coins
                pred_vp    = state.get("vp", 0) + b.get("buildVP", 0)
                for res_name, amt in cost.items():
                    if res_name == "coins":
                        pred_coins -= amt
                    else:
                        pred_res[res_name] = pred_res.get(res_name, 0) - amt
                slots_after = slots - 1
                for r, c in board.empty_grass_cells():
                    outcome = (
                        f"build (B{bid}, row:{r}, col:{c}, "
                        f"fi:{pred_res.get('fish',  0)}, wh:{pred_res.get('wheat', 0)}, "
                        f"st:{pred_res.get('stone', 0)}, wd:{pred_res.get('wood',  0)}, "
                        f"cn:{pred_coins}, vp:{pred_vp}, sl:{slots_after})"
                    )
                    move_map[outcome] = [{"move": "buildBuilding", "args": [bid, r, c]}]

        model_moves = list(move_map.keys())
        return model_moves, move_map

    # ── internal helpers ──────────────────────────────────────────────────────

    def _worker_adjacent(self, br, bc, placed_this_turn, player_id):
        """Check if any worker (placed this turn or already on board) is adjacent to (br, bc)."""
        board = self.parser.board
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = br + dr, bc + dc
                if (r, c) in placed_this_turn:
                    return True
                if board.has_worker_at(r, c, player_id):
                    return True
        return False
