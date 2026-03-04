"""
Board Game Log Parser
======================
Class-based parser for both offline SFT training and live inference.

OFFLINE (batch training data generation):
    parser   = GameParser.from_file("10771.txt")
    examples = parser.extract_training_examples(winner_only=True)
    save_jsonl(examples, "training.jsonl")

LIVE (real-time inference during a running game):
    parser = GameParser.from_match_init(meta)
    for event in live_event_stream:
        parser.ingest_event(event)
        if parser.needs_decision():
            prompt = parser.build_prompt(player_id=parser.current_player())
            action = model.predict(prompt)

CLI:
    python log_parser.py --input 10771.txt  --output training.jsonl
    python log_parser.py --input logs/      --output training.jsonl --winner-only
    python log_parser.py --input 10771.txt  --output training.jsonl --preview 2
"""