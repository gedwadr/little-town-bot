# ── State layout ──────────────────────────────────────────
# STATE_DIM = 343  →  3 + 8*4 + 5 + 25 + 6*9*5 + 8
#   [0   :3  ]  Global      (3)
#   [3   :35 ]  Per-player  (32 = 8 × 4)
#   [35  :40 ]  Bank        (5)
#   [40  :65 ]  Market      (25)
#   [65  :335]  Board       (270 = 6 × 9 × 5)  ← CNN input
#   [335 :343]  Board summary (8)

BOARD_START  = 65
BOARD_END    = 335
BOARD_CH     = 5
BOARD_ROWS   = 6
BOARD_COLS   = 9

GLOBAL_DIM   = 343 - 270        # 73  (everything except the board cells)
ACTION_DIM   = 12
CNN_OUT_DIM  = 64               # CNN board embedding size
GLOBAL_EMB   = 64               # MLP global embedding size
HIDDEN_DIM   = CNN_OUT_DIM + GLOBAL_EMB   # 128, concat of the two streams
N_BLOCKS     = 2

EPOCHS        = 100
LR            = 3e-5
WEIGHT_DECAY  = 1e-3
VAL_SPLIT     = 0.1
SEED          = 42
PATIENCE      = 32
DROPOUT = 0.3

CHECKPOINT_DIR = "./cnn_checkpoints_2p/v2"
DATA_PATH      = "./logs/trainLogs/nn_train_all_2p.jsonl"
