# ─────────────────────────────────────────────────────────────
# BUILDING CATALOGUE
# ─────────────────────────────────────────────────────────────

# trigger: 'worker' = activated by adjacent worker placement
#          'endOfRound' = fires automatically at round end
#          'endOfGame'  = scored at game end
#          'none'       = no effect (Statue)

BUILDINGS = {
    1:  {"name": "Wheat Field",  "cost": {"wood": 1},                    "buildVP": 3,  "trigger": "worker",
         "effect": "gain wheat:1",                          "alwaysAvailable": True},
    2:  {"name": "Workshop",     "cost": {"stone": 2},                   "buildVP": 5,  "trigger": "worker",
         "effect": "convert wood:2 -> vp:3"},
    3:  {"name": "Bar",          "cost": {"stone": 2, "wheat": 2},       "buildVP": 7,  "trigger": "worker",
         "effect": "gain vp:3"},
    4:  {"name": "Bakery",       "cost": {"wood": 2},                    "buildVP": 4,  "trigger": "worker",
         "effect": "convert wheat:1 -> coins:4"},
    5:  {"name": "Warehouse",    "cost": {"stone": 4},                   "buildVP": 8,  "trigger": "worker",
         "effect": "convert stone:2 -> vp:5"},
    6:  {"name": "Brewery",      "cost": {"wood": 2},                    "buildVP": 4,  "trigger": "worker",
         "effect": "convert wheat:1 -> vp:3"},
    7:  {"name": "Quarry",       "cost": {"wood": 3},                    "buildVP": 5,  "trigger": "worker",
         "effect": "convert coins:2 -> stone:2"},
    8:  {"name": "Carpenter",    "cost": {"wood": 2},                    "buildVP": 4,  "trigger": "worker",
         "effect": "convert coins:1 -> wood:3"},
    9:  {"name": "Shop",         "cost": {"wood": 2},                    "buildVP": 4,  "trigger": "worker",
         "effect": "convert coins:1 -> wheat:1 + fish:1"},
    10: {"name": "Church",       "cost": {"stone": 4},                   "buildVP": 8,  "trigger": "worker",
         "effect": "convert coins:3 -> vp:5"},
    11: {"name": "Marketplace",  "cost": {"wood": 4},                    "buildVP": 6,  "trigger": "worker",
         "effect": "convert stone:1+wood:1+wheat:1+fish:1 -> vp:7"},
    12: {"name": "Fountain",     "cost": {"stone": 2},                   "buildVP": 5,  "trigger": "worker",
         "effect": "convert coins:1 -> vp:3"},
    13: {"name": "Barn",         "cost": {"wood": 4},                    "buildVP": 6,  "trigger": "worker",
         "effect": "convert wheat:2 -> vp:5"},
    14: {"name": "Book Store",   "cost": {"stone": 4},                   "buildVP": 8,  "trigger": "worker",
         "effect": "gain coins:3"},
    15: {"name": "Gold Mine",    "cost": {"wood": 1, "stone": 1},        "buildVP": 4,  "trigger": "worker",
         "effect": "gain coins:2"},
    16: {"name": "Fishmonger",   "cost": {"wood": 1, "stone": 1},        "buildVP": 4,  "trigger": "worker",
         "effect": "convert fish:1 -> coins:3"},
    17: {"name": "Pier",         "cost": {"wood": 3},                    "buildVP": 5,  "trigger": "worker",
         "effect": "gain fish:2"},
    18: {"name": "Pawnshop",     "cost": {"wood": 3},                    "buildVP": 5,  "trigger": "worker",
         "effect": "exchange 2 resources -> 2 resources (any)"},
    19: {"name": "Well",         "cost": {"wood": 1, "stone": 1},        "buildVP": 4,  "trigger": "worker",
         "effect": "gain vp:2"},
    20: {"name": "Restaurant",   "cost": {"wood": 2, "stone": 2},        "buildVP": 7,  "trigger": "worker",
         "effect": "convert fish:1+wheat:1 -> vp:4"},
    21: {"name": "Statue",       "cost": {"stone": 4},                   "buildVP": 10, "trigger": "none",
         "effect": "none"},
    22: {"name": "Cathedral",    "cost": {"stone": 6},                   "buildVP": 11, "trigger": "endOfRound",
         "effect": "endOfRound: +1 VP per worker adjacent to Cathedral (8 directions)"},
    23: {"name": "Residence",    "cost": {"coins": 6},                   "buildVP": 2,  "trigger": "endOfRound",
         "effect": "endOfRound: activates all worker-buildings adjacent to Residence (opponent buildings cost 1 coin)"},
    24: {"name": "Castle",       "cost": {"stone": 6},                   "buildVP": 0,  "trigger": "endOfGame",
         "effect": "endOfGame: +4 VP per adjacent building YOU own (8 directions)"},
    25: {"name": "Watchtower",   "cost": {"wood": 3, "stone": 3},        "buildVP": 0,  "trigger": "endOfGame",
         "effect": "endOfGame: +2 VP per adjacent EMPTY cell (grass with worker = not empty; 8 directions)"},
}

BUILDING_NAME_TO_ID = {v["name"]: k for k, v in BUILDINGS.items()}

SFT_PROMPT = """You are a strategic agent playing a worker-placement board game. Your goal is to maximise Victory Points (VP) at the end of 4 rounds.

━━━ BOARD ━━━
A grid of terrain cells. Each cell is one of:
  G = grass    (workers are placed here)
  F = forest   → produces wood
  M = mountain → produces stone
  L = lake     → produces fish

━━━ GATHERING ━━━
When you place a worker on a grass cell, you automatically gather from that cell AND all 8 surrounding cells (diagonals included):
  each adjacent F → +1 wood
  each adjacent M → +1 stone
  each adjacent L → +1 fish
Workers can only be placed on empty grass cells (no worker, no building already there).

━━━ BUILDINGS ━━━
Build by spending resources → receive buildVP immediately + place the building on the board.
Activate a building by placing a worker on an adjacent cell (any of 8 surrounding cells):
  YOUR building   → free
  OPPONENT building → pay 1 coin to the owner (they earn rent)

Special trigger rules:
  trigger=endOfRound  → fires automatically at end of each round (Residence, Cathedral)
  trigger=endOfGame   → scored at game end (Castle, Watchtower)
  trigger=none        → no activation (Statue — pure buildVP)

━━━ FULL BUILDING CATALOGUE ━━━
ID | Name         | Cost                 | BuildVP | Effect
---|--------------|----------------------|---------|------------------------------------------
1  | Wheat Field  | 1 wood               | 3       | gain wheat:1  [always in market]
2  | Workshop     | 2 stone              | 5       | convert wood:2 -> vp:3
3  | Bar          | 2 stone + 2 wheat    | 7       | gain vp:3
4  | Bakery       | 2 wood               | 4       | convert wheat:1 -> coins:4
5  | Warehouse    | 4 stone              | 8       | convert stone:2 -> vp:5
6  | Brewery      | 2 wood               | 4       | convert wheat:1 -> vp:3
7  | Quarry       | 3 wood               | 5       | convert coins:2 -> stone:2
8  | Carpenter    | 2 wood               | 4       | convert coins:1 -> wood:3
9  | Shop         | 2 wood               | 4       | convert coins:1 -> wheat:1 + fish:1
10 | Church       | 4 stone              | 8       | convert coins:3 -> vp:5
11 | Marketplace  | 4 wood               | 6       | convert stone+wood+wheat+fish -> vp:7
12 | Fountain     | 2 stone              | 5       | convert coins:1 -> vp:3
13 | Barn         | 4 wood               | 6       | convert wheat:2 -> vp:5
14 | Book Store   | 4 stone              | 8       | gain coins:3
15 | Gold Mine    | 1 wood + 1 stone     | 4       | gain coins:2
16 | Fishmonger   | 1 wood + 1 stone     | 4       | convert fish:1 -> coins:3
17 | Pier         | 3 wood               | 5       | gain fish:2
18 | Pawnshop     | 3 wood               | 5       | exchange any 2 resources -> any 2 resources
19 | Well         | 1 wood + 1 stone     | 4       | gain vp:2
20 | Restaurant   | 2 wood + 2 stone     | 7       | convert fish:1 + wheat:1 -> vp:4
21 | Statue       | 4 stone              | 10      | no effect (pure VP)
22 | Cathedral    | 6 stone              | 11      | END OF ROUND: +1 VP per worker adjacent (8 dirs)
23 | Residence    | 6 coins              | 2       | END OF ROUND: activates all adjacent worker-buildings (opponent costs 1 coin)
24 | Castle       | 6 stone              | 0       | END OF GAME: +4 VP per adjacent building YOU own (8 dirs)
25 | Watchtower   | 3 wood + 3 stone     | 0       | END OF GAME: +2 VP per adjacent empty cell (worker on grass = not empty)

━━━ ROUND STRUCTURE (4 rounds total) ━━━
1. All players take turns placing workers (in turn order) until all workers are placed.
2. End-of-round phase: Residence and Cathedral trigger automatically.
3. Feeding: pay 1 fish OR 1 wheat per worker placed. Shortfall = -3 VP per unfed worker.
   Emergency: spend 3 coins -> 1 fish or wheat during feeding.
4. Round advances. Workers return. Repeat.

━━━ END GAME ━━━
After round 4 feeding:
  Castle  → +4 VP per adjacent building you own
  Watchtower → +2 VP per adjacent empty cell
  Coins   → floor(coins / 3) VP

━━━ ACTIONS YOU CAN TAKE ━━━
substituteResource <resource>
  Spend 3 coins to gain 1 resource (wood/stone/fish/wheat). Use during your turn.

placeWorker <row> <col>
  Place on an empty grass cell. Triggers 8-direction gather + any adjacent building activations.

buildBuilding <BuildingName> <row> <col>
  Spend the listed cost. Place building. Gain buildVP immediately. Consumes 1 worker slot.

activate <BuildingName> <row> <col>
  Happens automatically when you place a worker adjacent. Listed separately for clarity.

━━━ KEY STRATEGY NOTES ━━━
- Choose worker placement to maximise resources gathered AND buildings activated in one turn.
- Buildings with recurring effects (Bar, Church, Brewery) generate compound value over 4 rounds.
- Owning buildings opponents must activate earns you coin rent — passive income.
- Keep enough fish/wheat to feed all workers each round or the -3 VP penalty compounds.
- Coins have end-game VP value: don't spend them all. floor(coins/3) per coin triple.
- Residence + clustered buildings = powerful end-of-round engine.

━━━ OUTPUT FORMAT ━━━
Output ALL actions for your turn, one per line, in the order you take them:
  substituteResource <resource>
  placeWorker <row> <col>
  buildBuilding <BuildingName> <row> <col>
  activate <BuildingName> <row> <col>
Output ONLY the action lines. No explanation."""


DECISION_ACTIONS = {"placeWorker", "buildBuilding", "activate", "substituteResource"}
AUTO_ACTIONS     = {"gather"}
SYSTEM_ACTIONS   = {
    "market", "turnOrder", "roundAdvance", "phase",
    "feedingTimer", "feedWorkers", "moveRejected", "endgame",
}