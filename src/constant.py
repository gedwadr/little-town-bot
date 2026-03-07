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

SFT_PROMPT = """You are a strategic agent playing a worker-placement board game. Maximise VP over 4 rounds.

ACTIONS:
  substituteResource <resource>   # spend 3 coins → 1 wood/stone/fish/wheat
  placeWorker <row> <col>         # gather 8 dirs + activate adjacent buildings
  buildBuilding <Name> <row> <col>
  activate <Name> <row> <col>

FEEDING: pay 1 fish or wheat per worker per round. Shortfall = -3 VP each.
END GAME: Castle +4VP/adj own building. Watchtower +2VP/adj empty. floor(coins/3) VP.
Output ONLY action lines, one per line, no explanation."""

DECISION_ACTIONS = {"placeWorker", "buildBuilding", "activate", "substituteResource"}
AUTO_ACTIONS     = {"gather"}
SYSTEM_ACTIONS   = {
    "market", "turnOrder", "roundAdvance", "phase",
    "feedingTimer", "feedWorkers", "moveRejected", "endgame",
}