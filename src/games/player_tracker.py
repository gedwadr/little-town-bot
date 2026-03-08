from typing import Optional


class PlayerTracker:
    """
    Tracks each player's resources, coins, VP, and workers
    from the playerState field present on most log events.
    """

    def __init__(self, player_ids: list, player_names: dict):
        self.player_ids   = player_ids
        self.player_names = player_names
        self.states: dict = {
            pid: {
                "resources":        {"wood": 0, "stone": 0, "fish": 0, "wheat": 0},
                "coins":            3,
                "vp":               0,
                "workersRemaining": 3,
                "housesRemaining":  0,
            }
            for pid in player_ids
        }

    def ingest(self, event: dict):
        player = event.get("player")
        ps     = event.get("playerState")
        if player and ps and player != "system":
            current = self.states.get(player, {})
            self.states[player] = {
                "resources":        ps.get("resources", {}),
                "coins":            ps.get("coins", 0),
                "vp":               ps.get("vp", 0),
                "workersRemaining": ps.get("workersRemaining", 0),
                "housesRemaining":  current.get("housesRemaining", 0),
            }

    def update_houses(self, player_id: str, value: int):
        if player_id in self.states:
            self.states[player_id]["housesRemaining"] = value

    def update_from_game_state(self, player_id: str, pstate: dict):
        """Sync all player fields from a LogGameState.players[pid] snapshot."""
        if player_id not in self.states:
            return
        current = self.states[player_id]
        self.states[player_id] = {
            "resources":        pstate.get("resources", current.get("resources", {})),
            "coins":            pstate.get("coins",     current.get("coins", 0)),
            "vp":               pstate.get("vp",        current.get("vp", 0)),
            "workersRemaining": pstate.get("workersRemaining", current.get("workersRemaining", 0)),
            "housesRemaining":  pstate.get("housesRemaining",  current.get("housesRemaining", 0)),
        }

    def get(self, player_id: str) -> dict:
        return self.states.get(player_id, {})

    def compress(self, current_player: Optional[str] = None) -> str:
        lines = []
        for pid in self.player_ids:
            s    = self.states[pid]
            r    = s["resources"]
            name = self.player_names.get(pid, f"P{pid}")
            you  = "  <-- YOU" if pid == current_player else ""

            # Feeding safety: flag if fish+wheat might not cover workers
            food      = r.get("fish", 0) + r.get("wheat", 0)
            workers   = s["workersRemaining"]
            food_warn = f" [!food:{food}<wk:{workers}]" if food < workers else ""

            lines.append(
                f"  P{pid} {name:<8s}  "
                f"Wd:{r.get('wood',0)} St:{r.get('stone',0)} "
                f"Fi:{r.get('fish',0)} Wh:{r.get('wheat',0)}  "
                f"Coins:{s['coins']}  VP:{s['vp']}  "
                f"Workers:{s['workersRemaining']}  Slots:{s['housesRemaining']}"
                f"{food_warn}{you}"
            )
        return "\n".join(lines)

