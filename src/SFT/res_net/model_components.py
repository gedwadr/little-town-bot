"""
BoardGame ResNet Bot
====================
Architecture: ResNet encoder + ActionScorer head
Training:     Supervised cross-entropy on winner game logs
Input:        state [342] + candidates [N, 12]
Output:       scores [N] → pick argmax at inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.SFT.res_net import STATE_DIM, ACTION_DIM, HIDDEN_DIM, N_BLOCKS, DROPOUT


class ResBlock(nn.Module):
    """
    Pre-activation residual block:
      x → LayerNorm → Linear → ReLU → Linear → + x
    Pre-activation (norm before linear) trains more stably
    than post-activation for small networks.
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
        return x + self.net(x)  # residual skip connection


class GameStateEncoder(nn.Module):
    """
    Encodes game state vector [STATE_DIM] → embedding [HIDDEN_DIM].
    Runs ONCE per turn regardless of how many candidates there are.
    """

    def __init__(self, state_dim: int, hidden_dim: int, n_blocks: int):
        super().__init__()

        # Project raw features into hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        # Stack of residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(n_blocks)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: [batch, STATE_DIM]
        returns: [batch, HIDDEN_DIM]
        """
        x = self.input_proj(state)
        for block in self.blocks:
            x = block(x)
        return self.output_norm(x)


class ActionScorer(nn.Module):
    """
    Scores each candidate action given the state embedding.
    Runs N times (once per candidate) but shares weights.
    Output is always a single scalar per candidate.
    """

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        in_dim = hidden_dim + action_dim  # concat state emb + action vector

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
        """
        state_emb:  [batch, HIDDEN_DIM]
        candidates: [batch, N, ACTION_DIM]   N can be any size
        returns:    [batch, N]               one score per candidate
        """
        N = candidates.shape[1]

        # Expand state to match N candidates
        state_exp = state_emb.unsqueeze(1).expand(-1, N, -1)  # [batch, N, HIDDEN]

        # Concat state embedding with each candidate vector
        combined = torch.cat([state_exp, candidates], dim=-1)  # [batch, N, HIDDEN+ACTION]

        # Score each candidate independently (same weights for all)
        scores = self.net(combined).squeeze(-1)  # [batch, N]
        return scores


class BoardGameBot(nn.Module):
    """
    Full model: encoder + scorer.

    At inference:
        scores = model(state, candidates)
        chosen_idx = scores.argmax()

    At training:
        scores = model(state, candidates)
        loss = cross_entropy(scores, correct_idx)
    """

    def __init__(self, state_dim: int = STATE_DIM,
                 action_dim: int = ACTION_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 n_blocks: int = N_BLOCKS):
        super().__init__()
        self.encoder = GameStateEncoder(state_dim, hidden_dim, n_blocks)
        self.scorer = ActionScorer(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor,
                candidates: torch.Tensor) -> torch.Tensor:
        """
        state:      [batch, STATE_DIM]
        candidates: [batch, N, ACTION_DIM]
        returns:    [batch, N]  raw logits (no softmax — cross_entropy handles it)
        """
        state_emb = self.encoder(state)
        scores = self.scorer(state_emb, candidates)
        return scores

    def select_action(self, state: torch.Tensor,
                      candidates: torch.Tensor,
                      temperature: float = 1.0,
                      greedy: bool = False):
        """
        Inference helper. Returns (chosen_index, log_prob).

        state:      [STATE_DIM]        (no batch dim)
        candidates: [N, ACTION_DIM]    (no batch dim)
        """
        with torch.no_grad():
            scores = self.forward(
                state.unsqueeze(0),  # [1, STATE_DIM]
                candidates.unsqueeze(0),  # [1, N, ACTION_DIM]
            )[0]  # [N]

            if greedy:
                idx = scores.argmax().item()
                probs = F.softmax(scores, dim=-1)
            else:
                probs = F.softmax(scores / temperature, dim=-1)
                idx = torch.multinomial(probs, 1).item()

            log_prob = torch.log(probs[idx] + 1e-8).item()

        return idx, log_prob

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)