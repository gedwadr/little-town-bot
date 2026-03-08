import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_action(model, tokenizer, prompt, legal_moves: list[str]) -> tuple[str, float]:
    """
    Sample one action from the model, constrained to legal moves only.
    Returns (chosen_action, log_prob) for RL loss computation.
    """
    # Get model's probability distribution over all possible next tokens
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]  # shape: [1, vocab_size]

    # Score each legal move by summing log probs of its tokens
    move_scores = {}
    for move in legal_moves:
        move_ids = tokenizer(move, add_special_tokens=False, return_tensors="pt").input_ids[0]
        score = sum(logits[0, token_id].item() for token_id in move_ids)
        move_scores[move] = score / len(move_ids)

    # Convert to probabilities (softmax over legal moves only)
    scores = torch.tensor(list(move_scores.values()))
    probs = torch.softmax(scores, dim=0)

    # Sample from legal distribution
    idx = torch.multinomial(probs, 1).item()
    chosen = list(move_scores.keys())[idx]
    log_prob = torch.log(probs[idx])

    return chosen, log_prob.item()

