"""
BoardGame CNN Bot
=================
Architecture: CNN board encoder + MLP global encoder → ResBlocks → ActionScorer

The key insight is that the board is a 6×9 spatial grid with 5 feature channels.
A CNN learns "what's adjacent to what" naturally — exactly how resource gathering
works in Little Town (workers harvest from the 8 surrounding cells).

State split (343 floats total):
  Board   [65:335]  →  reshape [5, 6, 9]  →  CNN      →  board_emb  [64]
  Global  [0:65 + 335:343]  →  MLP        →  global_emb [64]
  concat  [128]  →  ResBlocks × 3  →  state_emb [128]
  concat with action [12]  →  scorer  →  score [1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.SFT.cnn import (
    BOARD_START, BOARD_END, BOARD_CH, BOARD_ROWS, BOARD_COLS,
    GLOBAL_DIM, ACTION_DIM, CNN_OUT_DIM, GLOBAL_EMB, HIDDEN_DIM, N_BLOCKS, DROPOUT,
)


# ── Building blocks ────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    """
    Pre-activation residual block (LayerNorm → Linear → ReLU → Linear + skip).
    Same design as the res_net module for consistency.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ── Encoders ───────────────────────────────────────────────────────────────────

class BoardCNNEncoder(nn.Module):
    """
    Encodes the 6×9 board grid (5 feature channels) into a fixed embedding.

    Input:  [batch, 5, 6, 9]
    Output: [batch, CNN_OUT_DIM]

    Three conv layers slide across the board so the network naturally learns
    spatial relationships — e.g. "forest next to lake" or "building cluster".
    Global average pooling makes the output board-size-agnostic.
    """

    def __init__(self, out_dim: int = CNN_OUT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(BOARD_CH, 32),          # [B, 32, 6, 9]  — local patterns
            ConvBlock(32,       64),          # [B, 64, 6, 9]  — mid-range patterns
            ConvBlock(64,       out_dim),     # [B, out_dim, 6, 9]
            nn.Dropout2d(DROPOUT),
        )
        # Global average pool collapses spatial dims → [B, out_dim]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        """board: [batch, 5, 6, 9]  →  [batch, CNN_OUT_DIM]"""
        x = self.conv(board)
        x = self.pool(x)          # [B, out_dim, 1, 1]
        return x.flatten(1)       # [B, out_dim]


class GlobalMLPEncoder(nn.Module):
    """
    Encodes the non-board global features (players, bank, market, summary)
    into a fixed embedding.

    Input:  [batch, GLOBAL_DIM]   (73 floats)
    Output: [batch, GLOBAL_EMB]
    """

    def __init__(self, in_dim: int = GLOBAL_DIM, out_dim: int = GLOBAL_EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNStateEncoder(nn.Module):
    """
    Full state encoder: splits the state vector, encodes board with CNN and
    global features with MLP, then fuses through ResBlocks.

    Input:  [batch, 343]
    Output: [batch, HIDDEN_DIM]
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, n_blocks: int = N_BLOCKS):
        super().__init__()
        self.board_encoder  = BoardCNNEncoder(CNN_OUT_DIM)
        self.global_encoder = GlobalMLPEncoder(GLOBAL_DIM, GLOBAL_EMB)

        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(n_blocks)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def _split_state(self, state: torch.Tensor):
        """
        Splits state [B, 343] into:
          board   [B, 5, 6, 9]
          global  [B, 73]
        """
        board_flat = state[:, BOARD_START:BOARD_END]                # [B, 270]
        board = board_flat.view(-1, BOARD_CH, BOARD_ROWS, BOARD_COLS)  # [B, 5, 6, 9]

        global_feats = torch.cat(
            [state[:, :BOARD_START], state[:, BOARD_END:]], dim=-1  # [B, 73]
        )
        return board, global_feats

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: [batch, 343]  →  [batch, HIDDEN_DIM]"""
        board, global_feats = self._split_state(state)

        board_emb  = self.board_encoder(board)          # [B, 64]
        global_emb = self.global_encoder(global_feats)  # [B, 64]

        x = torch.cat([board_emb, global_emb], dim=-1)  # [B, 128]
        for block in self.blocks:
            x = block(x)
        return self.output_norm(x)                       # [B, 128]


# ── Scorer (shared with res_net design) ───────────────────────────────────────

class ActionScorer(nn.Module):
    """
    Scores each candidate action given the state embedding.
    Identical structure to the res_net ActionScorer.

    state_emb:  [batch, HIDDEN_DIM]
    candidates: [batch, N, ACTION_DIM]
    returns:    [batch, N]
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        in_dim = hidden_dim + action_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state_emb: torch.Tensor,
                candidates: torch.Tensor) -> torch.Tensor:
        N = candidates.shape[1]
        state_exp = state_emb.unsqueeze(1).expand(-1, N, -1)   # [B, N, HIDDEN]
        combined  = torch.cat([state_exp, candidates], dim=-1)  # [B, N, HIDDEN+ACT]
        return self.net(combined).squeeze(-1)                   # [B, N]


# ── Top-level model ────────────────────────────────────────────────────────────

class BoardCNNBot(nn.Module):
    """
    Full CNN bot: CNNStateEncoder + ActionScorer.

    At inference:
        scores = model(state, candidates)
        chosen_idx = scores.argmax()

    At training:
        scores = model(state, candidates)
        loss = cross_entropy(scores, correct_idx)
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM,
                 n_blocks: int = N_BLOCKS):
        super().__init__()
        self.encoder = CNNStateEncoder(hidden_dim, n_blocks)
        self.scorer  = ActionScorer(hidden_dim, ACTION_DIM)

    def forward(self, state: torch.Tensor,
                candidates: torch.Tensor) -> torch.Tensor:
        """
        state:      [batch, 343]
        candidates: [batch, N, 12]
        returns:    [batch, N]  raw logits
        """
        state_emb = self.encoder(state)
        return self.scorer(state_emb, candidates)

    def select_action(self, state: torch.Tensor,
                      candidates: torch.Tensor,
                      temperature: float = 1.0,
                      greedy: bool = False):
        """
        Inference helper. Returns (chosen_index, log_prob).

        state:      [343]      (no batch dim)
        candidates: [N, 12]    (no batch dim)
        """
        with torch.no_grad():
            scores = self.forward(
                state.unsqueeze(0),
                candidates.unsqueeze(0),
            )[0]  # [N]

            if greedy:
                idx   = scores.argmax().item()
                probs = F.softmax(scores, dim=-1)
            else:
                probs = F.softmax(scores / temperature, dim=-1)
                idx   = torch.multinomial(probs, 1).item()

            log_prob = torch.log(probs[idx] + 1e-8).item()

        return idx, log_prob

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
