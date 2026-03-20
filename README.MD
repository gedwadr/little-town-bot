# Little Town Bot

> **Disclaimer:** Research project only. Not affiliated with IELLO Games or the creators of Little Town.

A competitive AI bot for the worker-placement board game [Little Town](https://iellogames.com/games/little-town/), built entirely in PyTorch — from raw game logs to a deployed, quantised model that plays in real time.

---

## Pipeline Overview

```
Human game logs (~100 matches)
        │
        ▼
[Stage 1] SFT — imitation learning on winner moves
        │  val acc: 25%
        ▼
[Stage 2] Bot play against heurytic bot benchmark to learn good move(+ ~250 matches)
        │ has 15% win. With this additional data, retrain the SFT to val acc: ~37%
        ▼
[Stage 3] Self-play data generation (bot plays itself)
        │  dataset: 6,746 → 20,912 examples
        ▼
[Stage 4] Online REINFORCE vs frozen opponent (940 games)
        │  win rate: 44% → 65%
        ▼
[Stage 5] SFT retrain on full dataset and INT8 dynamic quantisation → CPU deployment
           inference: ~0.1s/move (vs 35s+ heuristic) 
        │  val acc: 51%
        ▼
[Stage 6] Play agains heurystic model
        │  win rate: 15% → 55%
```

---

## Architecture

Three model variants share a common **encode-then-score** pattern: encode the full game state into a fixed embedding, then score each legal candidate action by concatenating the state embedding with its action vector. This naturally handles the variable action space — sometimes 4 candidates, sometimes 483.

### ResNet Branch

Treats the 343-float state vector as a flat input. Pre-activation residual blocks (LayerNorm before Linear) train more stably than post-activation for small tabular networks.

```
Input: state [343]  +  candidates [N × 12]
          │
    Linear(343→84) + ReLU
    3× ResBlock:
        LayerNorm → Linear → ReLU → Dropout(0.30) → Linear → +skip
    LayerNorm → state_emb [84]
          │
    concat(state_emb[84], action[12]) → [96]
    LayerNorm → Linear(96→64→32→1) → score per candidate
          │
    [N] logits → argmax at inference / softmax+sample during RL
```

### CNN Branch

Splits state into a spatial board tensor `[5×6×9]` and global scalar features `[73]`, encoding them separately before fusing. The CNN learns spatial adjacency naturally — workers harvest from 8 surrounding cells, so local patterns matter.

```
Board [65:335] → reshape [5, 6, 9]
    Conv(5→32, k=3, BN, ReLU)    — local resource patterns
    Conv(32→64, k=3, BN, ReLU)   — mid-range building interactions
    Conv(64→64, k=3, BN, ReLU)
    Dropout2d → GlobalAvgPool → board_emb [64]

Global [0:65 + 335:343] → Linear(73→128→64) → global_emb [64]

concat(board_emb, global_emb) [128]
    2× ResBlock → LayerNorm → state_emb [128]
    ActionScorer → [N] logits
```

### Ensemble (Score-Level Fusion)

Both branches score each candidate independently. Their logits are combined via softmax-normalised learnable weights — the model learns to trust the CNN more when spatial context matters, and the ResNet more for global strategic reasoning.

```python
ensemble_logits = w_resnet * logits_resnet + w_cnn * logits_cnn
# w_resnet, w_cnn: learnable scalars, softmax-normalised (init 0.5/0.5)
```

### State Representation (343 floats, always fixed-size)

| Section | Indices | Size | Description |
|---------|---------|------|-------------|
| Global | `[0:3]` | 3 | Round, workers remaining, turn progress |
| Per-player ×4 | `[3:35]` | 32 | Resources, VP, turn order, is_me flag — **zero-padded to 4 slots** regardless of player count |
| Bank | `[35:40]` | 5 | Resource availability |
| Market | `[40:65]` | 25 | One-hot over building IDs 1–25 |
| Board cells | `[65:335]` | 270 | 6×9 grid, 5 floats/cell: terrain, my worker, opp worker, building owner, building ID |
| Board summary | `[335:343]` | 8 | Special building positions, castle adjacency, watchtower empty cells |

> **Design note:** Variable player count (2–4) caused different state vector sizes per game, breaking DataLoader stacking with `RuntimeError: stack expects each tensor to be equal size`. Fixed by always padding to 4 player slots — a non-obvious bug that took careful debugging to isolate.

### Action Representation (12 floats per candidate)

| Action | Features |
|--------|----------|
| `placeWorker` | `row/5, col/8, Δfish, Δwheat, Δstone, Δwood, Δcoins, Δvp, n_activations, would_feed, type=0.0, food_ratio` |
| `buildBuilding` | `row/5, col/8, 0×4, coin_cost/15, build_vp/15, 0×2, type=0.5, 0` |
| `substituteResource` | `0×2, fish, wheat, stone, wood (one-hot), −3/15, 0×3, type=1.0, 0` |

---

## Supervised Fine-Tuning

Each example: `(state [343], candidates [N×12], correct_idx)` where `correct_idx` is the move the winner chose.

```python
scores = model(state, candidates)     # [B, max_N]
scores[~mask] = float('-inf')         # mask padding for variable N

# VP-weighted loss: winner=1.1, last place=−0.1
loss = (F.cross_entropy(scores, correct_idx, reduction='none')
        * sample_weights).mean()
```

**Data scaling insight:** Winner-only training gives clean signal but limited examples. Using all players with VP-proportional weights tripled dataset size (350 games → ~12k examples) while maintaining signal quality — a player scoring 80% of the winner's VP still made mostly good moves.

**Custom collate** handles variable `N` per example via padding + mask:

```python
def collate_variable_candidates(batch):
    """
    Custom collate: pads candidates to max N in the batch.
    Masked positions get score=-inf before softmax so they never win.
    """
    states, candidates_list, corrects, weights = zip(*batch)

    states = torch.stack(states)
    corrects = torch.stack(corrects)
    weights = torch.stack(weights)

    max_n = max(c.shape[0] for c in candidates_list)
    B = len(candidates_list)
    dim = candidates_list[0].shape[1]

    padded = torch.zeros(B, max_n, dim)
    mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i, c in enumerate(candidates_list):
        n = c.shape[0]
        padded[i, :n] = c
        mask[i, :n] = True

    return states, padded, corrects, mask, weights
```

| Hyperparameter | Value | Note |
|---------------|-------|------|
| Optimizer | AdamW, lr=3e-4, wd=1e-3 | Weight decay critical at small dataset size |
| LR schedule | CosineAnnealing, T_max=100 | eta_min=3e-6 avoids premature stop |
| Batch size | 32 | With variable-N collate |
| Early stopping | patience=32 | Train hit 41% vs val 25% — classic overfit signal |
| Dropout | 0.30 | Tuned up from 0.10 to control overfit |

---

## Reinforcement Learning (REINFORCE)

Online policy-gradient update after each game. The bot plays using `argmax` (greedy) — exploration comes from weight updates changing what's highest-scored, not from random sampling.

```python
# Reward structure
step_reward    = max(0, vp_delta) * 0.1        # positive-only VP signal
feeding_reward = ...                            # curriculum: fades games 200–300
placement      = {1: +0.3, 2: +0.1, 3: -0.1, 4: -0.3}   # primary RL signal

# REINFORCE update
R = 0
for r in reversed(rewards):
    R = r + 0.99 * R
    returns.insert(0, R)

for (state, candidates, chosen_idx), G in zip(steps, returns):
    scores   = model(state, candidates)
    log_prob = F.log_softmax(scores, dim=-1)[chosen_idx]
    loss    += -G * log_prob

loss = loss / len(steps)
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Three failure modes discovered and fixed:**

1. **OOM from full-episode backward** — accumulating loss across all steps held all intermediate tensors. Fixed with per-action forward pass and immediate `del` + `torch.cuda.empty_cache()`.

2. **Return normalisation killed gradients** — when episode reward variance was low, `returns.std() ≈ 0` made all normalised returns ≈ 0 and loss ≈ 0.0014. The model stopped updating entirely. Removed normalisation; gradient clipping provides stability instead.

3. **High LR erased SFT weights** — at `lr=1e-4`, 20 losing games against a stronger opponent degraded avg VP from +34 to −7. Reduced to `lr=1e-5`; the SFT foundation remained intact.

### Self-Play Setup

The RL bot (gradients on) plays against a frozen copy of its own SFT checkpoint. Win rates are perfectly complementary (`RL + Frozen ≈ 1.0`), confirming clean head-to-head comparison.

| Quarter | Games | RL Win Rate | Frozen Win Rate | Trend |
|---------|-------|-------------|-----------------|-------|
| Q1 (10–230) | 230 | 43.9% | 56.1% | RL behind |
| Q2 (240–460) | 220 | 53.0% | 47.0% | Crossover |
| Q3 (470–690) | 220 | 58.7% | 41.3% | RL ahead |
| Q4 (700–940) | 240 | **64.8%** | 35.2% | RL dominant |
| **Overall** | **940** | **55.1%** | **44.9%** | +0.00278/window slope |

The RL bot starts behind its own SFT checkpoint — expected, since early noisy game signals take time to accumulate — then systematically overtakes it. By Q4 it wins nearly 2 in 3 games against the model it started from.

**Observation:** avg VP improved only marginally (+0.41 overall) while win rate climbed +14.8% in Q4. The model learned to win through **consistency** — fewer catastrophic games — rather than higher peak scores. This is visible in the log-prob distribution: 60% of moves are genuinely uncertain (`log_prob < -1.5`), showing the model isn't trivially confident, just more reliably correct.

---

## What the Bot Learned

These behaviours emerged through training, not explicit programming:

**Worker feeding** — The biggest early failure mode was not feeding workers (−3VP each per round). After SFT on winner data, avg VP went from −7 to +25–35. The model learned fish/wheat placements are strategically prioritised when food is short.

**Move type balance** — From 21,483 logged moves: 69% `placeWorker`, 21.5% `buildBuilding`, 7.6% `substituteResource`. This mirrors human winner distributions from the training data.

**Consistency over ceiling** — RL improved win rate without dramatically increasing avg VP. The model learned that avoiding bad games matters more than playing perfectly in good games.

**Spatial building strategy** — The CNN branch learns that building placement relative to existing buildings matters — adjacency bonuses (Castle, Watchtower) depend entirely on spatial positioning that the flat ResNet can't easily model.

---

## Benchmark

| | NN Bot | Heuristic Bot |
|--|--------|--------------|
| Inference | ~0.1s/move | 35s+/move |
| Method | Neural forward pass | Tree search (7,000+ combos) |
| Avg game duration | ~60s | 210s+ |
| GPU at inference | No (INT8 CPU) | No |
| Win rate vs RL bot (early) | ~15% | ~85% |

The heuristic is the stronger player — full evaluation against it is the next planned experiment after ensemble SFT completes.

---

## Deployment

INT8 dynamic quantisation via `torch.quantization.quantize_dynamic` — Linear weights stored as INT8 (4× smaller), LayerNorm/BatchNorm unaffected. No calibration dataset needed.

```python
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
torch.save({'model': model_int8, 'game': game_count}, 'game_N_int8.pt')

# Load — weights_only=False required for quantized models
ckpt  = torch.load('game_N_int8.pt', map_location='cpu', weights_only=False)
model = ckpt['model'].eval()
```

Three-server architecture:

| Server | Port | Purpose |
|--------|------|---------|
| `train_nn_bot_server.py` | 9001 | RL training — gradients on, REINFORCE after each game |
| `frozen_nn_bot_server.py` | 9002 | Inference only — frozen opponent for self-play |
| `frozen_nn_quantize_bot_server.py` | 9003 | INT8 CPU — production, no GPU |

```bash
# Docker — checkpoint mounted at runtime, not baked into image
docker run \
  -v ./nn_rl_all_checkpoints/ensemble/v2/game_940_int8.pt:/checkpoints/model_int8.pt \
  -p 9003:9003 \
  little-town-bot
```

---

## Quick Start

```bash
# 1. Generate training data
python src/SFT/generate_train_files.py --nn --winner-only --player 2

# 2. SFT (run one or all)
python src/SFT/res_net/trainer.py
python src/SFT/cnn/trainer.py
python src/SFT/ensemble/trainer.py

# 3. Self-play RL
python src/train_nn_bot_server.py      # port 9001 — updating bot
python src/frozen_nn_bot_server.py     # port 9002 — frozen opponent

# 4. Quantized deployment
QUANTIZED_CHECKPOINT=./nn_rl_all_checkpoints/ensemble/v2/game_940_int8.pt \
python src/frozen_nn_quantize_bot_server.py
```

---

## Project Structure

```
src/
├── train_nn_bot_server.py             # RL training server (port 9001)
├── frozen_nn_bot_server.py            # Inference-only server (port 9002)
├── frozen_nn_quantize_bot_server.py   # INT8 CPU server (port 9003)
├── bot_server.py                      # LLM RL server (port 9000, alternate approach)
├── constant.py                        # Buildings catalogue, game constants
├── games/
│   ├── action_validator.py            # Legal move enumeration (get_legal_moves_v2)
│   ├── game_parser_nn.py              # Event log → state/candidate vectors
│   └── game_parser.py                 # Event log → text prompt (LLM branch)
└── SFT/
    ├── generate_train_files.py        # Game log → training JSONL
    ├── res_net/                       # ResNet model + trainer
    ├── cnn/                           # CNN model + trainer
    └── ensemble/                      # Ensemble model + trainer

logs/trainLogs/
└── nn_train_all_2p.jsonl              # 20,912 training examples

stats/
├── nn_training_stats.jsonl            # Win rate, VP, placement per 10 games
└── frozen_nn_training_stats.jsonl     # Frozen opponent stats (comparison)
```

---

*Game: https://iellogames.com/games/little-town/*