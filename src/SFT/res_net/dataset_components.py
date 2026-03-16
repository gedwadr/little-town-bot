import json

import torch


class GameDataset(torch.utils.data.Dataset):
    """
    Loads nn_train.jsonl.
    Each example: {state, candidates, correct_idx}
    Returns tensors ready for the model.
    """

    def __init__(self, path: str):
        self.examples = []
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        state = torch.tensor(ex["state"], dtype=torch.float32)
        candidates = torch.tensor(ex["candidates"], dtype=torch.float32)
        correct = torch.tensor(ex["correct_idx"], dtype=torch.long)
        weight = torch.tensor(ex.get("weight", 1.0), dtype=torch.float32)
        return state, candidates, correct, weight


def collate_variable_candidates(batch):
    """
    Custom collate: pads candidates to max N in the batch.
    Masked positions get score=-inf before softmax so they never win.
    """
    states, candidates_list, corrects, weights = zip(*batch)

    states = torch.stack(states)    # [B, STATE_DIM]
    corrects = torch.stack(corrects)  # [B]
    weights = torch.stack(weights)  # [B]

    max_n = max(c.shape[0] for c in candidates_list)
    B = len(candidates_list)
    dim = candidates_list[0].shape[1]

    padded = torch.zeros(B, max_n, dim)
    mask = torch.zeros(B, max_n, dtype=torch.bool)  # True = valid

    for i, c in enumerate(candidates_list):
        n = c.shape[0]
        padded[i, :n] = c
        mask[i, :n] = True

    return states, padded, corrects, mask, weights