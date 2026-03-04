from src.constant import BUILDING_NAME_TO_ID, BUILDINGS
import re


class BoardTracker:
    """
    Incrementally tracks board state from a stream of events.
    Terrain is fixed from initial_map. Workers and buildings are tracked live.
    """

    TERRAIN_FULL = {"G": "grass", "F": "forest", "M": "mountain", "L": "lake"}

    def __init__(self, initial_map: list):
        self.rows      = len(initial_map)
        self.cols      = len(initial_map[0])
        self.terrain   = [list(row) for row in initial_map]
        self.workers   = [[None] * self.cols for _ in range(self.rows)]
        # buildings[r][c] = {"name": str, "owner": str, "id": int}
        self.buildings = [[None] * self.cols for _ in range(self.rows)]

    def ingest(self, event: dict):
        """Update board from a single event."""
        action  = event.get("action", "")
        details = event.get("details", "")
        player  = event.get("player", "")

        if action == "placeWorker":
            m = re.search(r'\((\d+),(\d+)\)', details)
            if m:
                r, c = int(m.group(1)), int(m.group(2))
                self.workers[r][c] = player

        elif action == "buildBuilding":
            # "Built Wheat Field at (2,2), cost: 1 wood, +3 VP"
            m = re.search(r'Built (.+?) at \((\d+),(\d+)\)', details)
            if m:
                name = m.group(1).strip()
                r, c = int(m.group(2)), int(m.group(3))
                bid  = BUILDING_NAME_TO_ID.get(name, -1)
                self.buildings[r][c] = {"name": name, "owner": player, "id": bid}
                self.workers[r][c]   = player  # worker consumed at build site

        elif action == "roundAdvance":
            # All workers go home between rounds
            self.workers = [[None] * self.cols for _ in range(self.rows)]

    def compress(self) -> str:
        """
        Render the board as a compact ASCII grid.
        Each cell: terrain char + optional [Wn] or [Bn:Name4] annotation.
        """
        lines = [
            "BOARD  (G=grass F=forest M=mountain L=lake | "
            "[Wn]=worker of player n | [Bn:Name]=building owned by player n):"
        ]
        for r in range(self.rows):
            cells = []
            for c in range(self.cols):
                t = self.terrain[r][c]
                w = self.workers[r][c]
                b = self.buildings[r][c]
                if b and w:
                    cells.append(f"{t}[W{w}+B{b['owner']}:{b['name'][:5]}]")
                elif b:
                    cells.append(f"{t}[B{b['owner']}:{b['name'][:5]}]")
                elif w:
                    cells.append(f"{t}[W{w}]")
                else:
                    cells.append(t)
            lines.append(f"  r{r}: " + " ".join(cells))
        return "\n".join(lines)

    def buildings_on_board(self) -> list:
        """Return list of human-readable building descriptions."""
        result = []
        for r in range(self.rows):
            for c in range(self.cols):
                b = self.buildings[r][c]
                if b:
                    bid    = b.get("id", -1)
                    effect = BUILDINGS.get(bid, {}).get("effect", "?")
                    trigger= BUILDINGS.get(bid, {}).get("trigger", "worker")
                    result.append(
                        f"({r},{c}) {b['name']} [P{b['owner']}]  "
                        f"trigger:{trigger}  {effect}"
                    )
        return result

    def adjacency_preview(self, row: int, col: int) -> str:
        """
        For a candidate placement at (row, col), summarise what the
        player would gather and which buildings they would activate.
        Useful for building the prompt's 'candidate moves' section.
        """
        resources = {"wood": 0, "stone": 0, "fish": 0, "wheat": 0}
        activates = []
        TERRAIN_RESOURCE = {"F": "wood", "M": "stone", "L": "fish", "G": "wheat"}

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r2, c2 = row + dr, col + dc
                if 0 <= r2 < self.rows and 0 <= c2 < self.cols:
                    t = self.terrain[r2][c2]
                    res = TERRAIN_RESOURCE.get(t)
                    if res:
                        resources[res] += 1
                    b = self.buildings[r2][c2]
                    if b and (dr, dc) != (0, 0):
                        activates.append(f"{b['name']}@({r2},{c2})")

        res_str = " ".join(f"+{v}{k[:2]}" for k, v in resources.items() if v > 0)
        act_str = " + activate: " + ", ".join(activates) if activates else ""
        return f"({row},{col}): gather {res_str}{act_str}"

    def is_empty(self, row: int, col: int) -> bool:
        return self.workers[row][col] is None and self.buildings[row][col] is None

    def empty_grass_cells(self) -> list:
        """All cells where a worker can legally be placed."""
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.terrain[r][c] == "G" and self.is_empty(r, c)
        ]

