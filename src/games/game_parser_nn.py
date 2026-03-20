import re
from typing import Optional

from src.games.game_parser import GameParser
from src.constant import BUILDINGS, BUILDING_NAME_TO_ID

TERRAIN_RESOURCE  = {"F": "wood", "M": "stone", "L": "fish", "G": "wheat"}
MAX_WORKERS       = {2: 5, 3: 4, 4: 3}
MAX_VP            = {2: 100, 3: 70, 4: 55}
FOOD_BUILDING_IDS = {1, 9, 17}  # Wheat Field, Shop, Pier

# Buildings with direct gains only (no conversion required)
DIRECT_GAIN = {
    1:  {"wheat": 1},   # Wheat Field
    3:  {"vp": 3},      # Bar
    14: {"coins": 3},   # Book Store
    15: {"coins": 2},   # Gold Mine
    17: {"fish": 2},    # Pier
    19: {"vp": 2},      # Well
}


class GameParserNN(GameParser):
    """
    Neural-network-friendly subclass of GameParser.

    Overrides build_prompt() to return numerical feature vectors instead of text:
        {"state": list[float], "actions": list[list[float]]}

    State vector length: 3 + 8*n_players + 4 + 25 + board_rows*board_cols*5 + 8
    Action vector length: 12 floats per move.

    Note: _fresh_replay_parser() returns a plain GameParser, so
    extract_training_examples() will still produce text prompts.
    """

    # ── public API ────────────────────────────────────────────

    def build_prompt(
        self,
        player_id: str,
        legal_moves: Optional[list] = None,
        **kwargs,
    ) -> dict:
        """
        Returns {"state": list[float]} plus optional "actions" key.

        legal_moves: list of Majapahit move dicts {"move": str, "args": list}.
        **kwargs absorbs parent's instruction/show_candidates if called generically.
        """
        result = {"state": self.build_state_vector(player_id)}
        if legal_moves is not None:
            result["actions"] = [self.encode_action(m, player_id) for m in legal_moves]
        return result

    def build_state_vector(self, player_id: str) -> list:
        """
        Flat float vector describing the full game state from player_id's perspective.

        Sections:
          1. Global       — 3 floats
          2. Per-player   — 8 floats × MAX_PLAYERS(4)  (me first, then turn_order)
                           fish/6, wheat/6, stone/6, wood/6, coins/6,
                           vp/max_vp, turn_pos, is_me
          3. Bank         — 5 floats (fish, wheat, stone, wood, coins)
          4. Market       — 25 floats (one-hot over building IDs 1-25)
          5. Board cells  — rows×cols×5 floats (row-major)
          6. Board summary— 8 floats
        Total (6×9 board): 3 + 32 + 5 + 25 + 270 + 8 = 343
        """
        n_players   = len(self.player_ids)
        max_workers = MAX_WORKERS.get(n_players, 3)
        max_vp      = MAX_VP.get(n_players, 55)
        t           = self._current_turn
        round_num   = t.round_num if t else 0
        turn_num    = t.turn_num  if t else 0
        p_state     = self.players.states.get(player_id, {})

        vec = []

        # ── Section 1: Global (3 floats) ──────────────────────
        max_turns = (max_workers * n_players * 4) or 1
        vec += [
            round_num / 4,
            p_state.get("workersRemaining", 0) / max_workers,
            min(1.0, turn_num / max_turns),  # clamped: turn counter can exceed expected max
        ]

        # ── Section 2: Per-player (9 floats × MAX_PLAYERS) ────
        # Slot 9: food_ratio = (fish + wheat) / max_workers  clamped [0, 1]
        MAX_PLAYERS = 4
        ordered_pids = [player_id] + [p for p in self.turn_order if p != player_id]

        for i in range(MAX_PLAYERS):
            if i < len(ordered_pids):
                pid = ordered_pids[i]
                state = self.players.states.get(pid, {})
                resources = state.get("resources", {})
                try:
                    turn_pos = (self.turn_order.index(pid) + 1) / n_players
                except ValueError:
                    turn_pos = 0.0
                vec += [
                    resources.get("fish",  0) / 6,
                    resources.get("wheat", 0) / 6,
                    resources.get("stone", 0) / 6,
                    resources.get("wood",  0) / 6,
                    state.get("coins", 0) / 6,
                    max(0, state.get("vp", 0)) / max_vp,
                    turn_pos,
                    1.0 if pid == player_id else 0.0,
                ]
            else:
                vec += [0.0] * 8

        # ── Section 3: Bank (5 floats) ─────────────────────────
        vec += [
            self.bank.get("fish",  0) / 15,
            self.bank.get("wheat", 0) / 15,
            self.bank.get("stone", 0) / 15,
            self.bank.get("wood",  0) / 15,
            self.bank.get("coins", 0) / 40,
        ]

        # ── Section 4: Market one-hot (25 floats) ─────────────
        available = set(self.available_market())
        vec += [1.0 if bid in available else 0.0 for bid in range(1, 26)]

        # ── Section 5: Board per-cell (rows×cols×5 floats) ────
        TERRAIN_CODE = {"G": 0, "F": 1, "M": 2, "L": 3}
        board = self.board
        for r in range(board.rows):
            for c in range(board.cols):
                t_char   = board.terrain[r][c]
                worker   = board.workers[r][c]
                building = board.buildings[r][c]

                if building is None:
                    bld_owner = 0.0
                    bld_id    = 0.0
                elif building["owner"] == player_id:
                    bld_owner = 0.5   # me  → 1/2
                    bld_id    = building["id"] / 25
                else:
                    bld_owner = 1.0   # opp → 2/2
                    bld_id    = building["id"] / 25

                vec += [
                    TERRAIN_CODE.get(t_char, 0) / 3,
                    1.0 if worker == player_id else 0.0,
                    1.0 if (worker is not None and worker != player_id) else 0.0,
                    bld_owner,
                    bld_id,
                ]

        # ── Section 6: Board building summary (8 floats) ──────
        residence_info   = None  # (r, c, owner_str)
        castle_info      = None
        watchtower_info  = None
        bar_info         = None
        marketplace_info = None
        has_food_bld_me  = False

        for r in range(board.rows):
            for c in range(board.cols):
                b = board.buildings[r][c]
                if b is None:
                    continue
                bid   = b["id"]
                owner = b["owner"]
                if bid in FOOD_BUILDING_IDS and owner == player_id:
                    has_food_bld_me = True
                if bid == 23:
                    residence_info = (r, c, owner)
                elif bid == 24:
                    castle_info = (r, c, owner)
                elif bid == 25:
                    watchtower_info = (r, c, owner)
                elif bid == 3:
                    bar_info = (r, c, owner)
                elif bid == 11:
                    marketplace_info = (r, c, owner)

        def _owner_code(info):
            if info is None:
                return 0.0
            return 0.5 if info[2] == player_id else 1.0

        # Castle: count adjacent buildings I own
        castle_adj = 0
        if castle_info and castle_info[2] == player_id:
            cr, cc = castle_info[0], castle_info[1]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < board.rows and 0 <= nc < board.cols:
                        b2 = board.buildings[nr][nc]
                        if b2 and b2["owner"] == player_id:
                            castle_adj += 1

        # Watchtower: count adjacent empty cells
        wt_adj_empty = 0
        if watchtower_info and watchtower_info[2] == player_id:
            wr, wc = watchtower_info[0], watchtower_info[1]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = wr + dr, wc + dc
                    if 0 <= nr < board.rows and 0 <= nc < board.cols:
                        if board.workers[nr][nc] is None and board.buildings[nr][nc] is None:
                            wt_adj_empty += 1

        vec += [
            1.0 if has_food_bld_me else 0.0,
            _owner_code(residence_info),
            _owner_code(castle_info),
            _owner_code(watchtower_info),
            _owner_code(bar_info),
            _owner_code(marketplace_info),
            castle_adj / 8,
            wt_adj_empty / 8,
        ]

        return vec

    def encode_action(self, move: dict, player_id: str) -> list:
        """
        Encode a Majapahit move dict into an 11-float action vector.

        Move format: {"move": str, "args": list}
          placeWorker:       args = [row, col, activations]
          buildBuilding:     args = [building_id, row, col]
          substituteResource:args = [resource_str]

        Output index 10 is an un-normalised action-type flag: 0=place, 1=build, 2=substitute.
        """
        move_name = move.get("move", "")
        args      = move.get("args", [])

        if move_name == "placeWorker":
            return self._encode_place_worker(args, player_id)
        elif move_name == "buildBuilding":
            return self._encode_build_building(args)
        elif move_name == "substituteResource":
            return self._encode_substitute(args)
        else:
            return [0.0] * 12

    # ── NN training data extraction ───────────────────────────

    def extract_nn_training_examples(self, winner_only: bool = False) -> list:
        """
        Extract NN training examples from a completed game.

        For each player turn returns:
            {
                "state":       list[float],         # state vector before the turn
                "candidates":  list[list[float]],   # one 11-float vector per legal move
                "correct_idx": int,                 # index of the actual move in candidates
            }

        Skips turns where the actual move cannot be matched to any candidate.
        winner_only: only include turns from the winning player.
        """
        from src.games.action_validator import ActionValidator

        self.flush()

        # ── Build weight map: player_id → training weight ──────
        POSITION_WEIGHTS = {
            2: [1.0, 0.1],
            3: [1.0, 0.5, 0.1],
            4: [1.0, 0.7, 0.3, 0.1],
        }
        n_players = len(self.player_ids)
        pos_weights = POSITION_WEIGHTS.get(n_players, [1.0] * n_players)

        # Rank players by final VP (descending) to determine position
        final_vp = {pid: self.players.states.get(pid, {}).get("vp", 0)
                    for pid in self.player_ids}
        ranked = sorted(self.player_ids, key=lambda p: -final_vp[p])

        weight_map: dict[str, float] = {}
        for i, pid in enumerate(ranked):
            w = pos_weights[i] if i < len(pos_weights) else 0.1
            weight_map[pid] = w

        replay = GameParserNN(
            initial_map    = self.initial_map,
            initial_market = self.initial_market,
            player_names   = self.player_names,
            turn_order     = self.turn_order,
            winner_id      = self.winner_id,
            board_side     = self.board_side,
        )
        examples = []

        for turn in self._completed_turns:
            pid = turn.player_id

            if winner_only and pid != self.winner_id:
                self._advance_replay(replay, turn)
                continue

            if not turn.decision_events:
                self._advance_replay(replay, turn)
                continue

            actual_move = GameParserNN._turn_to_majapahit_move(turn)
            if actual_move is None:
                self._advance_replay(replay, turn)
                continue

            # Get all legal moves from the pre-turn state
            _, move_map = ActionValidator(replay).get_legal_moves_v2(pid)
            if not move_map:
                self._advance_replay(replay, turn)
                continue

            # Build parallel lists: Majapahit dicts and encoded action vectors
            state_vec       = replay.build_state_vector(pid)
            candidate_dicts = [dicts[0] for dicts in move_map.values()]
            candidate_vecs  = [replay.encode_action(d, pid) for d in candidate_dicts]

            correct_idx = GameParserNN._find_correct_idx(actual_move, candidate_dicts)
            if correct_idx is None:
                self._advance_replay(replay, turn)
                continue

            examples.append({
                "state":       state_vec,
                "candidates":  candidate_vecs,
                "correct_idx": correct_idx,
                "weight":      weight_map.get(pid, 1.0),
            })

            self._advance_replay(replay, turn)

        return examples

    @staticmethod
    def _find_correct_idx(actual_move: dict, candidate_dicts: list) -> Optional[int]:
        """
        Find the index in candidate_dicts that exactly matches actual_move.
        Returns None if no exact match (caller should skip the example).
        """
        move_name = actual_move["move"]
        args      = actual_move["args"]

        for i, cand in enumerate(candidate_dicts):
            if cand["move"] != move_name:
                continue

            if move_name == "placeWorker":
                if cand["args"][0] != args[0] or cand["args"][1] != args[1]:
                    continue
                actual_acts = frozenset(
                    (a["row"], a["col"]) for a in (args[2] if len(args) > 2 else [])
                )
                cand_acts = frozenset(
                    (a["row"], a["col"]) for a in (cand["args"][2] if len(cand["args"]) > 2 else [])
                )
                if actual_acts == cand_acts:
                    return i

            elif move_name == "buildBuilding":
                if cand["args"] == args:
                    return i

            elif move_name == "substituteResource":
                if cand["args"] == args:
                    return i

        return None

    @staticmethod
    def _turn_to_majapahit_move(turn) -> Optional[dict]:
        """
        Reconstruct a Majapahit move dict from a Turn's decision events.
        Returns the primary move (placeWorker / buildBuilding / substituteResource)
        with activations included for placeWorker.
        """
        place_rc    = None
        build_info  = None
        sub_info    = None
        activations = []

        for event in turn.decision_events:
            action  = event.get("action", "")
            details = event.get("details", "")

            if action == "placeWorker":
                m = re.search(r'\((\d+),(\d+)\)', details)
                if m:
                    place_rc = (int(m.group(1)), int(m.group(2)))

            elif action == "activate":
                m = re.search(r'at \((\d+),(\d+)\)', details)
                if m:
                    activations.append({"row": int(m.group(1)), "col": int(m.group(2))})

            elif action == "buildBuilding":
                m = re.search(r'Built (.+?) at \((\d+),(\d+)\)', details)
                if m:
                    name = m.group(1).strip()
                    bid  = BUILDING_NAME_TO_ID.get(name, -1)
                    build_info = (bid, int(m.group(2)), int(m.group(3)))

            elif action == "substituteResource":
                m = re.search(r'-> \d+ (\w+)', details)
                if m:
                    sub_info = m.group(1)

        if place_rc is not None:
            return {"move": "placeWorker",        "args": [place_rc[0], place_rc[1], activations]}
        if build_info is not None:
            return {"move": "buildBuilding",       "args": list(build_info)}
        if sub_info is not None:
            return {"move": "substituteResource",  "args": [sub_info]}
        return None

    # ── private helpers ───────────────────────────────────────

    def _encode_place_worker(self, args: list, player_id: str) -> list:
        row         = args[0]
        col         = args[1]
        activations = args[2] if len(args) > 2 else []

        terrain = self._terrain_resource_deltas(row, col)
        gains   = self._direct_gain_from_buildings(activations)

        n_players   = len(self.player_ids)
        max_workers = MAX_WORKERS.get(n_players, 3)
        p_state     = self.players.states.get(player_id, {})
        resources   = p_state.get("resources", {})
        workers_rem = p_state.get("workersRemaining", 0)

        workers_placed_after = (max_workers - workers_rem) + 1
        cur_fish  = resources.get("fish",  0)
        cur_wheat = resources.get("wheat", 0)
        food_after = cur_fish + cur_wheat + terrain["fish"] + terrain["wheat"]
        would_feed = 1.0 if food_after >= workers_placed_after else 0.0
        food_ratio = min(1.0, (cur_fish + cur_wheat) / max_workers) if max_workers else 0.0

        return [
            row / 5,
            col / 8,
            (terrain["fish"]  + gains.get("fish",  0)) / 3,
            (terrain["wheat"] + gains.get("wheat", 0)) / 3,
            (terrain["stone"] + gains.get("stone", 0)) / 3,
            (terrain["wood"]  + gains.get("wood",  0)) / 3,
            gains.get("coins", 0) / 5,
            gains.get("vp",    0) / 10,
            len(activations) / 8,
            would_feed,
            0.0 / 2,  # action_type normalised to [0,1]: place=0.0
            food_ratio,
        ]

    def _encode_build_building(self, args: list) -> list:
        bid  = args[0]
        row  = args[1]
        col  = args[2]
        b    = BUILDINGS.get(bid, {})
        cost = b.get("cost", {})
        return [
            row / 5,
            col / 8,
            0.0, 0.0, 0.0, 0.0,
            -cost.get("coins", 0) / 15,
            b.get("buildVP", 0) / 15,
            0.0, 0.0,
            1.0 / 2,  # action_type normalised: build=0.5
            0.0,
        ]

    def _encode_substitute(self, args: list) -> list:
        resource = args[0] if args else ""
        return [
            0.0, 0.0,
            1.0 if resource == "fish"  else 0.0,
            1.0 if resource == "wheat" else 0.0,
            1.0 if resource == "stone" else 0.0,
            1.0 if resource == "wood"  else 0.0,
            -3 / 15,
            0.0, 0.0, 0.0,
            2.0 / 2,  # action_type normalised: substitute=1.0
            0.0,
        ]

    def _terrain_resource_deltas(self, row: int, col: int) -> dict:
        """
        Count terrain-based resource gains for a placement at (row, col).
        Includes all 9 cells (8 neighbours + the cell itself), matching
        the game's adjacency rules (the cell itself is always grass → +1 wheat).
        """
        counts = {"fish": 0, "wheat": 0, "stone": 0, "wood": 0}
        board  = self.board
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r2, c2 = row + dr, col + dc
                if 0 <= r2 < board.rows and 0 <= c2 < board.cols:
                    res = TERRAIN_RESOURCE.get(board.terrain[r2][c2])
                    if res:
                        counts[res] += 1
        return counts

    def _direct_gain_from_buildings(self, activations: list) -> dict:
        """
        Sum direct gains from activated buildings (those in DIRECT_GAIN).
        Conversion buildings (Bakery, Brewery, etc.) are excluded because
        they require spending resources the player may not have.

        activations: list of dicts with "row"/"col" keys (Majapahit format).
        """
        gains = {"fish": 0, "wheat": 0, "stone": 0, "wood": 0, "coins": 0, "vp": 0}
        board = self.board
        for act in activations:
            r = act["row"]
            c = act["col"]
            b = board.buildings[r][c]
            if b is None:
                continue
            bid = b.get("id", -1)
            if bid in DIRECT_GAIN:
                for resource, amount in DIRECT_GAIN[bid].items():
                    gains[resource] = gains.get(resource, 0) + amount
        return gains
