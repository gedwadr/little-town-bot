# Little Town Bot — V1 ResNet + Reinforcement Learning

> **Disclaimer:** This project is an AI bot for the board game **Little Town**. It is intended solely for research and educational purposes. This is not affiliated with, endorsed by, or connected to the publishers or creators of Little Town in any way.

An AI agent for the **Little Town** worker-placement board game, trained with supervised fine-tuning (SFT) followed by online REINFORCE. This README covers the V1 ResNet architecture and the RL training pipeline.

---

## Overview

The bot operates as a Flask server that the game server calls on each turn. After each game, it runs a REINFORCE update on the moves it made, using Victory Points and placement as the reward signal.

Two architectures exist in this repo:
- **NN bot** (this document) — structured ResNet encoder, fast inference, full SFT + RL pipeline
- **LLM bot** — Qwen 2.5-1.5B with LoRA, text-based prompts

---

## Model Architecture

### BoardGameBot (`src/SFT/res_net/model_components.py`)

A two-stage encode-then-score pipeline:

```
Input: state vector (343 floats)  +  candidate actions (N × 12 floats)
         │                                      │
         ▼                                      │
  GameStateEncoder                              │
  ├─ Linear: 343 → 128                          │
  ├─ 4× ResBlock                                │
  │   ├─ LayerNorm → Linear → ReLU              │
  │   ├─ Dropout(0.35) → Linear                 │
  │   └─ Residual skip                          │
  └─ LayerNorm output                           │
         │                                      │
         └──────────────┬───────────────────────┘
                        ▼
                  ActionScorer
                  ├─ Concat(state[128], action[12]) → [140]
                  ├─ Linear: 140 → 64 → 32 → 1
                  └─ Score per candidate → [N] logits
```

The encoder runs once per turn; the scorer runs once per legal move. Softmax over logits gives the action distribution.

### State Representation (343 floats)

| Component | Size | Description |
|-----------|------|-------------|
| Global | 3 | Round, workers remaining, turn progress |
| Per-player (×4) | 32 | Resources (fish/wheat/stone/wood/coins), VP, turn order, is_me flag |
| Bank | 5 | Resource availability |
| Market | 25 | One-hot encoding of available buildings |
| Board cells (6×9) | 270 | Terrain type, worker presence, buildings per cell |
| Board summary | 8 | Special building positions and metrics |

### Action Representation (12 floats)

| Action Type | Features |
|-------------|----------|
| `placeWorker` | row/5, col/8, resource gains, activation count, would_feed, food_ratio |
| `buildBuilding` | row/5, col/8, coin cost, VP value |
| `substituteResource` | resource flags, cost (-3 coins) |

### Hyperparameters (`src/SFT/res_net/__init__.py`)

| Parameter | Value |
|-----------|-------|
| STATE_DIM | 343 |
| ACTION_DIM | 12 |
| HIDDEN_DIM | 128 |
| N_BLOCKS | 4 |
| Dropout | 0.35 |

---

## Supervised Fine-Tuning (SFT)

SFT trains the model to imitate expert moves from recorded games before RL begins.

### Data Generation

```bash
python src/SFT/generate_train_files.py --nn --winner-only
```

Parses game JSON logs and produces `logs/trainLogs/nn_train_all.jsonl`. Each example:
```json
{
  "state": [343 floats],
  "candidates": [[12 floats], ...],
  "correct_idx": 2,
  "weight": 1.1
}
```

Weights by final placement: winner = 1.1, last = −0.1.

### Training

```bash
python src/SFT/res_net/trainer.py
```

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-3 |
| Batch size | 32 |
| LR schedule | CosineAnnealing (T_max=100, eta_min=3e-6) |
| Loss | Weighted cross-entropy (padded candidates masked to −∞) |
| Early stopping | patience=16 |
| Train/val split | 90/10 |

Best checkpoint saved to `nn_all_checkpoints/best.pt`.

---

## Reinforcement Learning (REINFORCE)

Online policy-gradient training that runs during live games.

### Server

```bash
python src/train_nn_bot_server.py   # port 9001
```

For inference only (no gradients):
```bash
python src/frozen_nn_bot_server.py   # port 9002
```

### Training Loop

**On each turn** (`POST /get-move`):
1. Parse the game event log → reconstruct board/player state
2. Enumerate legal moves via `ActionValidator.get_legal_moves_v2()`
3. Encode state + score all candidates
4. Sample action from softmax (temperature-scaled)
5. Store `(log_prob, state)` in episode memory

**After each game** (`POST /game-result`):
1. Compute per-step rewards (see below)
2. Compute discounted returns: `G_t = r_t + γ · G_{t+1}`
3. For each action: `loss = −G_t · log_prob / num_actions`
4. Backward + gradient clip + optimizer step
5. Save checkpoint every 10 games

### Reward Function

| Signal | Value | Notes |
|--------|-------|-------|
| VP delta | `max(0, Δvp) × 0.1` | Per-turn reward |
| Feeding reward | Scaled by food/workers ratio | Fades out games 200–300 (curriculum) |
| End-game VP | `last_vp_delta × 0.1` | Final turn bonus |
| Placement | +0.3 / +0.1 / −0.1 / −0.3 | 1st–4th place |

The feeding reward provides dense early signal while the model is learning basic resource management, then fades so the model focuses on VP maximization.

### RL Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-5 |
| Discount factor (γ) | 0.99 |
| Gradient clip (max_norm) | 1.0 |
| Temperature (start → end) | 1.5 → 0.5 over 500 games |
| Checkpoint frequency | Every 10 games |

---

## Project Structure

```
src/
├── nn_bot_server.py          # RL training server (port 9001)
├── frozen_nn_bot_server.py   # Inference-only server (port 9002)
├── bot_server.py             # LLM RL server (port 9000)
├── constant.py               # Buildings catalogue, game constants
└── SFT/
    ├── generate_train_files.py   # Game log → training data
    ├── res_net/
    │   ├── __init__.py           # Hyperparameters
    │   ├── model_components.py   # BoardGameBot, GameStateEncoder, ActionScorer
    │   └── trainer.py            # SFT training loop
    └── trainer/
        └── fine_tuner.py         # LLM SFT training

src/games/
├── action_validator.py       # Legal move enumeration (get_legal_moves_v2)
├── game_parser_nn.py         # Event log → state/candidate vectors
├── game_parser.py            # Event log → text prompt (LLM)
└── utils.py                  # Action sampling helpers
```

### Checkpoints

| Path | Contents |
|------|----------|
| `nn_all_checkpoints/best.pt` | Best SFT checkpoint |
| `nn_rl_checkpoints/game_N/` | RL checkpoint after game N |
| `stats/training_stats.jsonl` | Win rate, avg VP, placement per 10 games |
| `stats/move_records.jsonl` | Per-move action type and legal move counts |

---

## Game Server Integration

The TypeScript game server at `/home/greg/WebstormProjects/majapahit` calls the bot via:

**`POST /get-move`**
```json
{
  "game_id": "...",
  "player_id": 0,
  "log": [...],
  "moves": [{"move": "placeWorker", "args": [row, col, activations]}, ...],
  "illegal_moves": [...]
}
```
Returns: `{"actions": [{"move": "...", "args": [...]}]}`

**`POST /game-result`**
```json
{
  "game_id": "...",
  "position": 1,
  "n_players": 4,
  "final_vp": 45
}
```
Triggers the REINFORCE update.

---

## Quick Start

**1. Generate training data from game logs:**
```bash
python src/SFT/generate_train_files.py --nn --winner-only
```

**2. Run SFT:**
```bash
python src/SFT/res_net/trainer.py
```

**3. Start the RL bot server:**
```bash
python src/train_nn_bot_server.py
```

**4. Start the frozen inference server (for opponents/evaluation):**
```bash
python src/frozen_nn_bot_server.py
```
