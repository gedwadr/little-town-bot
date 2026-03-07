
from src.games.game_parser import GameParser
from src.constant import BUILDING_NAME_TO_ID

class ActionValidator:
    """
    Filters raw model output to only legal actions given the current game state.
    Uses the parser's live state to check what's actually possible.
    """

    def __init__(self, parser: GameParser):
        self.parser = parser

    def to_majapahit_move(self, raw_output: str, player_id: str, moves: list) -> dict | None:
        """
        Parse model output and return the matching entry from the Majapahit `moves` list.

        Move formats returned:
          placeWorker    -> {"move": "placeWorker",       "args": [row, col, [{row, col}, ...]]}
          buildBuilding  -> {"move": "buildBuilding",     "args": [building_id, row, col]}
          substituteResource -> {"move": "substituteResource", "args": [resource_type]}

        Activate lines following a placeWorker are collapsed into its activations list.
        For exchange buildings (Pawnshop), the exchange resource details are taken from the
        pre-computed moves entry — the model only needs to specify the position.

        Returns None if no valid match is found (caller should fall back to moves[0]).
        """
        valid_lines = self.validate(raw_output, player_id)
        if not valid_lines:
            return None

        parts = valid_lines[0].split()
        action = parts[0]

        if action == "placeWorker":
            try:
                row, col = int(parts[1]), int(parts[2])
            except (ValueError, IndexError):
                return None

            # Collect (row, col) pairs from activate lines that follow this placeWorker
            activate_positions: list[tuple[int, int]] = []
            for line in valid_lines[1:]:
                lparts = line.split()
                if lparts[0] != "activate":
                    break
                try:
                    activate_positions.append((int(lparts[-2]), int(lparts[-1])))
                except (ValueError, IndexError):
                    break

            target_pos_set = set(activate_positions)

            candidates = [
                m for m in moves
                if m["move"] == "placeWorker"
                and m["args"][0] == row
                and m["args"][1] == col
            ]
            if not candidates:
                return None

            # Prefer exact match on activation position set
            for m in candidates:
                m_pos_set = {(a["row"], a["col"]) for a in m["args"][2]} if len(m["args"]) > 2 else set()
                if m_pos_set == target_pos_set:
                    return m

            # Fall back: score by overlap with requested positions minus size difference
            best, best_score = None, -1
            for m in candidates:
                m_pos_set = {(a["row"], a["col"]) for a in m["args"][2]} if len(m["args"]) > 2 else set()
                score = len(m_pos_set & target_pos_set) * 10 - abs(len(m_pos_set) - len(target_pos_set))
                if score > best_score:
                    best_score, best = score, m
            return best

        elif action == "buildBuilding":
            try:
                row, col = int(parts[-2]), int(parts[-1])
                name = " ".join(parts[1:-2])
            except (ValueError, IndexError):
                return None
            bid = BUILDING_NAME_TO_ID.get(name)
            if bid is None:
                return None
            for m in moves:
                if m["move"] == "buildBuilding" and m["args"] == [bid, row, col]:
                    return m
            return None

        elif action == "substituteResource":
            if len(parts) < 2:
                return None
            resource = parts[1]
            for m in moves:
                if m["move"] == "substituteResource" and m["args"] == [resource]:
                    return m
            return None

        return None

    def validate(self, raw_output: str, player_id: str) -> list[str]:
        """
        Parse raw model output and return only the legal action lines.
        Stops at the first illegal action to preserve action ordering.
        """
        lines = [l.strip() for l in raw_output.strip().splitlines() if l.strip()]
        valid  = []
        state  = self.parser.players.states.get(player_id, {})
        board  = self.parser.board

        # Track what the player will have AFTER each action
        # (so substituteResource before placeWorker is handled correctly)
        resources = dict(state.get("resources", {}))
        coins     = state.get("coins", 3)
        workers   = state.get("workersRemaining", 0)
        placed_this_turn = []  # (row, col) placed so far this turn

        for line in lines:
            parts = line.split()
            if not parts:
                continue
            action = parts[0]

            # ── unknown action type ──────────────────────────────
            if action not in ("placeWorker", "buildBuilding",
                              "activate", "substituteResource"):
                break   # stop — model is hallucinating

            # ── substituteResource ───────────────────────────────
            if action == "substituteResource":
                if len(parts) < 2:
                    break
                resource = parts[1]
                if resource not in ("wood", "stone", "fish", "wheat"):
                    break
                if coins < 3:
                    break   # can't afford it
                coins -= 3
                resources[resource] = resources.get(resource, 0) + 1
                valid.append(line)

            # ── placeWorker ──────────────────────────────────────
            elif action == "placeWorker":
                if len(parts) < 3:
                    break
                try:
                    r, c = int(parts[1]), int(parts[2])
                except ValueError:
                    break
                if workers <= 0:
                    break   # no workers left
                if not board.is_empty_grass(r, c):
                    break   # cell occupied or not grass
                if (r, c) in placed_this_turn:
                    break   # already placed here this turn
                workers -= 1
                placed_this_turn.append((r, c))
                valid.append(line)

            # ── buildBuilding ────────────────────────────────────
            elif action == "buildBuilding":
                # buildBuilding <Name parts...> <row> <col>
                if len(parts) < 4:
                    break
                try:
                    r, c = int(parts[-2]), int(parts[-1])
                    name = " ".join(parts[1:-2])
                except (ValueError, IndexError):
                    break
                # Check building is in market
                from src.constant import BUILDINGS
                bid = BUILDING_NAME_TO_ID.get(name)
                if bid is None:
                    break   # unknown building name
                available = self.parser._available_market()
                if bid not in available:
                    break   # not in current market
                # Check cell is empty grass
                if not board.is_empty_grass(r, c):
                    break
                # Check player has slots remaining
                slots = self.parser.players.states[player_id].get("housesRemaining", 0)
                if slots <= 0:
                    break
                # Check player can afford it
                cost = BUILDINGS[bid].get("cost", {})
                for res, amt in cost.items():
                    if res == "coins":
                        if coins < amt:
                            break
                    else:
                        if resources.get(res, 0) < amt:
                            break
                else:
                    # Deduct cost
                    for res, amt in cost.items():
                        if res == "coins":
                            coins -= amt
                        else:
                            resources[res] = resources.get(res, 0) - amt
                    valid.append(line)
                    continue
                break  # couldn't afford

            # ── activate ─────────────────────────────────────────
            elif action == "activate":
                if len(parts) < 4:
                    break
                try:
                    r, c = int(parts[-2]), int(parts[-1])
                    name = " ".join(parts[1:-2])
                except (ValueError, IndexError):
                    break
                # Building must actually exist on the board at (r, c)
                if not board.has_building_at(r, c, name):
                    break
                # Must have a worker adjacent to it (from this turn or prior)
                if not self._worker_adjacent(r, c, placed_this_turn, player_id):
                    break
                valid.append(line)

        return valid

    def _worker_adjacent(self, br, bc, placed_this_turn, player_id):
        """Check if any worker (placed this turn or already on board) is adjacent to (br, bc)."""
        board = self.parser.board
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = br + dr, bc + dc
                # Worker placed this turn
                if (r, c) in placed_this_turn:
                    return True
                # Worker already on board belonging to this player
                if board.has_worker_at(r, c, player_id):
                    return True
        return False