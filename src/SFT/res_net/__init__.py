STATE_DIM = 343  # 3 + 8*4 + 5 + 25 + 6*9*5 + 8
ACTION_DIM = 12  # +food_ratio on placeWorker (0.0 for other action types)
HIDDEN_DIM = 84
N_BLOCKS = 3

EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-3
VAL_SPLIT = 0.1
SEED = 42
DROPOUT = 0.3

CHECKPOINT_DIR = "./nn_all_checkpoints_2p"
DATA_PATH      = "./logs/trainLogs/nn_train_all_2p.jsonl"

PATIENCE = 32  # stop if val acc doesn't improve for 10 epochs