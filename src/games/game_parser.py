import re
from copy import deepcopy
from typing import Optional

from src.SFT.model import Turn, TrainingExample
from src.constant import BUILDINGS, BUILDING_NAME_TO_ID, DECISION_ACTIONS, SFT_PROMPT
from src.games.board_tracker import BoardTracker
from src.games.player_tracker import PlayerTracker

import json

from src.utils.utils import format_turn_for_history, format_action, building_cost_str


class GameParser:
    """
    Stateful parser for a single game session.

    Ingests events one at a time and maintains running state:
      - board (terrain, workers, buildings)
      - player states (resources, coins, VP, workers)
      - market (which buildings are available)
      - action history (list of completed turn summaries)
      - current in-progress turn

    Works identically in offline (batch) and live (streaming) modes.
    """

    def __init__(
        self,
        initial_map:    list,   # ["MGFMGLGMM", "GGGGGGGGG", ...]
        initial_market: list,   # [2, 11, 5, ...] — building IDs
        player_names:   dict,   # {"0": "Greg", "1": "CPU", ...}
        turn_order:     list,   # ["3", "1", "2", "0"]
        winner_id:      Optional[str] = None,
        board_side:     str = "?",
    ):
        self.initial_map    = initial_map
        self.initial_market = initial_market
        self.player_names   = player_names
        self.player_ids     = list(player_names.keys())
        self.turn_order     = turn_order
        self.winner_id      = winner_id
        self.board_side     = board_side

        # Market: always includes Wheat Field (id=1) + the random market pool
        self._always_available = [1]  # Wheat Field is always in market
        self.market = list(initial_market)  # shrinks as buildings are bought

        # Gap 1: Wheat Field supply — only 5 tiles exist
        self.wheat_fields_remaining = 5
        # Gap 2: Resource bank — global supply caps; updated from gameState when available
        self.bank = {"wood": 15, "stone": 15, "fish": 15, "wheat": 15}

        # Pre-populate from catalogue so we never show "B2" unknowns
        self.mkt_names: dict = {bid: BUILDINGS[bid]["name"] for bid in BUILDINGS}

        # State trackers — updated on every ingest_event()
        self.board   = BoardTracker(initial_map)
        self.players = PlayerTracker(self.player_ids, player_names)

        # Gap 3: initialise housesRemaining from player count (before any events)
        _slots = {2: 7, 3: 6, 4: 6}
        initial_slots = _slots.get(len(self.player_ids), 6)
        for pid in self.player_ids:
            self.players.states[pid]["housesRemaining"] = initial_slots

        # History of completed turns (formatted strings)
        self.history: list = []

        # Current in-progress turn
        self._current_turn: Optional[Turn] = None

        # All completed Turn objects (used by extract_training_examples)
        self._completed_turns: list = []

        # Constant setup header for every prompt in this game
        self._setup_header = self._build_setup_header()

    # ── constructors ──────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str) -> "GameParser":
        """
        Load a match JSON file and ingest all events.
        Automatically detects end-game format (has 'match' key)
        or live format (has 'log' key only).
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "match" in data:
            return cls.from_match_data(data)
        elif "log" in data:
            return cls.from_live_data(data)
        else:
            raise ValueError(
                f"Unrecognised format in {path}: "
                f"expected 'match' or 'log' key, got {list(data.keys())}"
            )

    @classmethod
    def from_match_data(cls, data: dict) -> "GameParser":
        """
        Construct from a parsed end-game match dict and ingest all events.
        Use when you already have the JSON in memory.
        End-game format: { match: { player_names, initial_map, initial_market,
                                    full_log, winner_id, board_side, ... } }
        """
        match = data["match"]

        # Extract turn order from first turnOrder event
        turn_order = list(match["player_names"].keys())
        for e in match["full_log"]:
            if e["action"] == "turnOrder":
                gs = e.get("gameState", {})
                turn_order = gs.get("turnOrder", turn_order)
                break

        parser = cls(
            initial_map    = match["initial_map"],
            initial_market = match["initial_market"],
            player_names   = match["player_names"],
            turn_order     = turn_order,
            winner_id      = str(match.get("winner_id", "")),
            board_side     = match.get("board_side", "?"),
        )
        for event in match["full_log"]:
            parser.ingest_event(event)

        return parser

    @classmethod
    def from_live_data(cls, data: dict) -> "GameParser":
        """
        Construct from a live game dict and ingest all events so far.
        Live format: { log: [...events...] }
        Missing fields (player_names, initial_map, board_side, winner_id)
        are reconstructed from the first gameState in the log.

        Player display names default to 'P0', 'P1', ... since the live
        format does not include them. Pass a player_names override if
        you have them from elsewhere.
        """
        log = data["log"]
        if not log:
            raise ValueError("Live game log is empty — cannot reconstruct metadata.")

        meta = cls._extract_meta_from_live_log(log)

        parser = cls(
            initial_map    = meta["initial_map"],
            initial_market = meta["initial_market"],
            player_names   = meta["player_names"],
            turn_order     = meta["turn_order"],
            winner_id      = meta.get("winner_id"),
            board_side     = meta.get("board_side", "?"),
        )
        for event in log:
            parser.ingest_event(event)

        return parser

    @staticmethod
    def _extract_meta_from_live_log(log: list) -> dict:
        """
        Reconstruct the metadata that end-game format stores explicitly
        but live format omits — by reading the first gameState snapshot.

        Returns a dict with:
            initial_map, initial_market, player_names, turn_order,
            winner_id (None for live), board_side ('?')
        """
        # Find first event that has a full gameState with board + market
        first_gs = None
        for event in log:
            gs = event.get("gameState", {})
            if "board" in gs and "market" in gs:
                first_gs = gs
                break

        if first_gs is None:
            raise ValueError(
                "No gameState with board+market found in live log. "
                "Cannot reconstruct map or market."
            )

        # Reconstruct initial_map from board terrain grid
        terrain_map = {
            "mountain": "M",
            "grass":    "G",
            "forest":   "F",
            "lake":     "L",
        }
        initial_map = []
        for row in first_gs["board"]:
            row_str = "".join(terrain_map.get(cell["terrain"], "?") for cell in row)
            initial_map.append(row_str)

        # initial_market from first gameState (excludes Wheat Field id=1)
        initial_market = [bid for bid in first_gs["market"] if bid != 1]

        # Player IDs and default display names (P0, P1, ...)
        player_ids = sorted(first_gs.get("players", {}).keys(), key=lambda x: int(x))
        player_names = {pid: f"P{pid}" for pid in player_ids}

        # Turn order
        turn_order = first_gs.get("turnOrder", player_ids)

        return {
            "initial_map":    initial_map,
            "initial_market": initial_market,
            "player_names":   player_names,
            "turn_order":     turn_order,
            "winner_id":      None,   # not known until game ends
            "board_side":     "?",    # not present in live format
        }

    @classmethod
    def from_match_init(cls, meta: dict) -> "GameParser":
        """
        Construct from live game metadata before any events arrive.
        Call ingest_event() as events stream in from the game server.

        Required meta keys:
            initial_map    : list of row strings e.g. ["MGFMGLGMM", ...]
            initial_market : list of building IDs
            player_names   : dict {player_id: display_name}
            turn_order     : list of player IDs in play order

        Optional:
            board_side     : str label for the map variant
        """
        return cls(
            initial_map    = meta["initial_map"],
            initial_market = meta["initial_market"],
            player_names   = meta["player_names"],
            turn_order     = meta.get("turn_order", list(meta["player_names"].keys())),
            winner_id      = None,
            board_side     = meta.get("board_side", "?"),
        )

    # ── core event ingestion ──────────────────────────────────

    def ingest_event(self, event: dict):
        """
        Process one event from the log or live stream.
        Must be called in sequence — order matters.

        Updates board, player state, market, and the current turn buffer.
        Automatically seals a turn when the next player's event arrives.
        """
        action  = event.get("action", "")
        player  = event.get("player", "system")
        turn_n  = event.get("turn")
        round_n = event.get("round")

        # Always update state trackers first
        self.players.ingest(event)
        self.board.ingest(event)

        # Track market depletion as buildings are bought
        if action == "buildBuilding":
            m = re.search(r'Built (.+?) at', event.get("details", ""))
            if m:
                name = m.group(1).strip()
                bid  = BUILDING_NAME_TO_ID.get(name)
                if bid and bid in self.market:
                    self.market.remove(bid)
                # Gap 1: track Wheat Field supply
                if bid == 1 and self.wheat_fields_remaining > 0:
                    self.wheat_fields_remaining -= 1
                # Gap 3: track building slots locally (works in replay too)
                if player in self.players.states:
                    h = self.players.states[player].get("housesRemaining", 0)
                    if h > 0:
                        self.players.states[player]["housesRemaining"] = h - 1

        # Gap 1 + 2 + 3: sync from gameState snapshot when available
        gs = event.get("gameState", {})
        if "bank" in gs:
            self.bank.update(gs["bank"])
            self.wheat_fields_remaining = gs["wheatFieldsRemaining"]
            for pid, pstate in gs.get("players", {}).items():
                if pid in self.players.states:
                    self.players.update_houses(pid, pstate.get("housesRemaining", 0))

        # Skip system events for turn grouping
        if player == "system" or turn_n is None:
            return

        # Detect turn boundary: new (round, turn, player) triplet
        key         = (round_n, turn_n, player)
        current_key = None
        if self._current_turn is not None:
            t = self._current_turn
            current_key = (t.round_num, t.turn_num, t.player_id)

        if current_key != key:
            if self._current_turn is not None:
                self._seal_current_turn()
            self._current_turn = Turn(
                turn_num  = turn_n,
                round_num = round_n,
                player_id = player,
            )

        # Accumulate into current turn
        t = self._current_turn
        t.all_events.append(event)
        if action in DECISION_ACTIONS:
            t.decision_events.append(event)
        elif action == "gather":
            t.gather_details.append(event.get("details", ""))

    def _seal_current_turn(self):
        """Finalise the current turn and add it to history."""
        t = self._current_turn
        if t is None:
            return
        self._completed_turns.append(t)
        line = format_turn_for_history(t)
        if line.strip():
            self.history.append(line)

    def flush(self):
        """
        Seal the last in-progress turn.
        Call after ingesting all events from a completed game to ensure
        the final turn is not left dangling.
        from_file() and from_match_data() call this automatically.
        """
        if self._current_turn is not None:
            self._seal_current_turn()
            self._current_turn = None

    # ── prompt building ───────────────────────────────────────

    def build_prompt(
        self,
        player_id:   str,
        instruction: str = "",
        show_candidates: bool = False,
    ) -> dict:
        """
        Build an inference prompt for the given player at the current game state.

        Returns {"system": str, "user": str} ready to send to the model.

        player_id       : the player who needs to act
        instruction     : optional strategy hint e.g. "prioritise fish economy"
        show_candidates : if True, include a preview of best placement options
                          with resource/activation summaries (costs tokens)
        """
        t         = self._current_turn
        round_num = t.round_num if t else "?"
        turn_num  = t.turn_num  if t else "?"

        last_action   = self.history[-1] if self.history else "(game start — no moves yet)"
        board_snap    = self.board.compress()
        players_snap  = self.players.compress(current_player=player_id)
        buildings     = self.board.buildings_on_board()
        buildings_str = "\n".join(f"  {b}" for b in buildings) if buildings else "  (none yet)"
        market_snap   = self._market_str(self._available_market())
        instr_block   = f"\nSTRATEGY INSTRUCTION: {instruction}\n" if instruction else ""

        # Optional: show top candidate placements
        candidates_block = ""
        if show_candidates:
            empties = self.board.empty_grass_cells()
            previews = [self.board.adjacency_preview(r, c) for r, c in empties[:12]]
            if previews:
                candidates_block = (
                    "\nCANDIDATE PLACEMENTS (gather + activations per cell):\n"
                    + "\n".join(f"  {p}" for p in previews) + "\n"
                )

        user_content = (
            f"{self._setup_header}\n"
            f"=== LAST ACTION ===\n"
            f"{last_action}\n"
            f"{instr_block}\n"
            f"=== YOUR TURN ===\n"
            f"Round {round_num} of 4  |  Turn {turn_num}  |  "
            f"You are P{player_id} ({self.player_names.get(player_id, '?')})\n\n"
            f"PLAYER STATES:\n{players_snap}\n\n"
            f"RESOURCE BANK:\n{self._bank_str()}\n\n"
            f"MARKET (buildings available to buy):\n{market_snap}\n\n"
            f"BUILDINGS ON BOARD:\n{buildings_str}\n\n"
            f"{board_snap}\n"
            f"{candidates_block}"
        )

        return {"system": SFT_PROMPT, "user": user_content}

    # ── training data extraction ──────────────────────────────

    def extract_training_examples(self, winner_only: bool = False) -> list:
        """
        Extract SFT training examples from a completed game.

        For each player turn:
          user      = full game state + history BEFORE the turn
          assistant = actions actually taken that turn

        winner_only: only include turns from the winning player.
                     Biases training toward winning strategies at the
                     cost of fewer examples per game.

        Call after all events are ingested (from_file handles this).
        """
        self.flush()

        # Replay on a fresh parser to get accurate pre-turn snapshots
        replay = self._fresh_replay_parser()
        examples = []

        for turn in self._completed_turns:
            pid = turn.player_id

            if winner_only and pid != self.winner_id:
                self._advance_replay(replay, turn)
                continue

            if not turn.decision_events:
                self._advance_replay(replay, turn)
                continue

            # Snapshot state BEFORE this turn
            prompt = replay.build_prompt(player_id=pid)

            # Build assistant content
            action_lines = [format_action(e) for e in turn.decision_events]
            action_lines = [a for a in action_lines if a]
            if not action_lines:
                self._advance_replay(replay, turn)
                continue

            examples.append(TrainingExample(
                round_num    = turn.round_num,
                turn_num     = turn.turn_num,
                player_id    = pid,
                user_content = prompt["user"],
                asst_content = "\n".join(action_lines),
            ))

            self._advance_replay(replay, turn)

        return examples

    def _fresh_replay_parser(self) -> "GameParser":
        """Create a blank parser with same metadata for state replay."""
        return GameParser(
            initial_map    = self.initial_map,
            initial_market = self.initial_market,
            player_names   = self.player_names,
            turn_order     = self.turn_order,
            winner_id      = self.winner_id,
            board_side     = self.board_side,
        )

    def _advance_replay(self, replay: "GameParser", turn: Turn):
        """Feed all events from a completed turn into the replay parser."""
        for e in turn.all_events:
            replay.ingest_event(e)

    # ── live helpers ──────────────────────────────────────────

    def needs_decision(self) -> bool:
        """
        True if the current in-progress turn has at least one decision event.
        Use in live mode to know when to call build_prompt().
        """
        return (
            self._current_turn is not None
            and len(self._current_turn.decision_events) > 0
        )

    def current_player(self) -> Optional[str]:
        """Player ID whose turn is currently in progress."""
        return self._current_turn.player_id if self._current_turn else None

    def current_state_summary(self) -> dict:
        """
        Lightweight snapshot of game state.
        Useful for game-server integration or external logging.
        """
        t = self._current_turn
        return {
            "round":          t.round_num if t else None,
            "turn":           t.turn_num  if t else None,
            "current_player": self.current_player(),
            "market":         [self.mkt_names.get(bid, f"B{bid}") for bid in self._available_market()],
            "players":        deepcopy(self.players.states),
            "buildings":      self.board.buildings_on_board(),
            "history_length": len(self.history),
            "empty_grass":    len(self.board.empty_grass_cells()),
        }

    # ── internals ─────────────────────────────────────────────

    def _available_market(self) -> list:
        """Market pool: always-available buildings + remaining random market."""
        always = [bid for bid in self._always_available
                  if bid != 1 or self.wheat_fields_remaining > 0]
        return always + self.market

    def _market_str(self, mkt_ids: list) -> str:
        parts = []
        for bid in mkt_ids:
            b = BUILDINGS.get(bid, {})
            name    = b.get("name", f"B{bid}")
            cost    = building_cost_str(bid)
            build_vp= b.get("buildVP", "?")
            effect  = b.get("effect", "?")
            always  = " [always]" if b.get("alwaysAvailable") else ""
            supply  = f" ({self.wheat_fields_remaining} left)" if bid == 1 else ""
            parts.append(f"  [{bid}] {name}{always}{supply}  cost:{cost}  buildVP:{build_vp}  {effect}")
        return "\n".join(parts) if parts else "  (market empty)"

    def _bank_str(self) -> str:
        parts = []
        for res in ("wood", "stone", "fish", "wheat"):
            v = self.bank.get(res, 0)
            warn = " [LOW]" if v < 3 else ""
            parts.append(f"{res}:{v}{warn}")
        return "  " + "  ".join(parts)

    def _build_setup_header(self) -> str:
        map_str     = " / ".join(self.initial_map)
        order_str   = " -> ".join(f"P{p}" for p in self.turn_order)
        players_str = "  ".join(
            f"P{pid}:{self.player_names.get(pid, '?')}" for pid in self.player_ids
        )
        _worker_slots = {2: (5, 7), 3: (4, 6), 4: (3, 6)}
        workers, slots = _worker_slots.get(len(self.player_ids), (3, 6))
        # Initial market summary
        init_mkt = self._market_str(self._always_available + self.initial_market)
        return (
            f"=== GAME SETUP ===\n"
            f"Map {len(self.initial_map)}x{len(self.initial_map[0])} "
            f"(board {self.board_side}): {map_str}\n"
            f"Players: {players_str}\n"
            f"Turn order: {order_str}\n"
            f"Workers per player: {workers}  Building slots per player: {slots}\n"
            # f"Initial market:\n{init_mkt}\n"
        )

    def __repr__(self) -> str:
        t = self._current_turn
        return (
            f"GameParser("
            f"turns_completed={len(self._completed_turns)}, "
            f"history={len(self.history)}, "
            f"winner={self.winner_id}, "
            f"current={f'R{t.round_num}T{t.turn_num}P{t.player_id}' if t else None}"
            f")"
        )