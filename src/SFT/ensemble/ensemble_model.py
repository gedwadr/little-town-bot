"""
Ensemble: ResNet + CNN
======================
Combines the ResNet bot (flat state encoder) and the CNN bot (spatial board
encoder) by averaging their output logits.

Both models see the same input — they just encode the state differently:
  - ResNet: treats all 343 features as a flat vector
  - CNN:    separates board [5×6×9] from global features

Averaging logits (score-level fusion) is robust and requires no extra
training — you can also enable learnable weights if you want the ensemble
to adapt.

Usage
-----
    from src.SFT.ensemble.ensemble_model import EnsembleBot

    bot = EnsembleBot.from_checkpoints(
        resnet_ckpt="nn_all_checkpoints/best.pt",
        cnn_ckpt="cnn_checkpoints_2p/best.pt",
        device="cuda",
    )
    idx, log_prob = bot.select_action(state, candidates)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.SFT.res_net.model_components import BoardGameBot
from src.SFT.cnn.model_components import BoardCNNBot


class EnsembleBot(nn.Module):
    """
    Score-level ensemble of ResNet and CNN bots.

    logits_ensemble = w_resnet * logits_resnet + w_cnn * logits_cnn

    By default weights are equal (0.5 / 0.5).
    Set learnable=True to let the weights be trained jointly.
    """

    def __init__(self, resnet: BoardGameBot, cnn: BoardCNNBot,
                 learnable_weights: bool = False):
        super().__init__()
        self.resnet = resnet
        self.cnn    = cnn

        if learnable_weights:
            # Softmax-normalised so they always sum to 1
            self._raw_weights = nn.Parameter(torch.zeros(2))
        else:
            self.register_buffer("_raw_weights", torch.zeros(2))

    @property
    def weights(self) -> torch.Tensor:
        """Returns [w_resnet, w_cnn] normalised to sum to 1."""
        return F.softmax(self._raw_weights, dim=0)

    def forward(self, state: torch.Tensor,
                candidates: torch.Tensor) -> torch.Tensor:
        """
        state:      [batch, 343]
        candidates: [batch, N, 12]
        returns:    [batch, N]  ensemble logits
        """
        w = self.weights
        logits_resnet = self.resnet(state, candidates)  # [B, N]
        logits_cnn    = self.cnn(state, candidates)     # [B, N]
        return w[0] * logits_resnet + w[1] * logits_cnn

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

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoints(cls, resnet_ckpt: str, cnn_ckpt: str,
                         device: str = "cpu",
                         learnable_weights: bool = False) -> "EnsembleBot":
        """Load both sub-models from their best.pt checkpoints."""
        resnet = BoardGameBot()
        resnet.load_state_dict(
            torch.load(resnet_ckpt, map_location=device)["state_dict"]
        )
        resnet.eval()

        cnn = BoardCNNBot()
        cnn.load_state_dict(
            torch.load(cnn_ckpt, map_location=device)["state_dict"]
        )
        cnn.eval()

        model = cls(resnet, cnn, learnable_weights=learnable_weights)
        model.to(device)
        return model
