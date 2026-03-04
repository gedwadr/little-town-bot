import argparse
import json
from pathlib import Path

from src.games.game_parser import GameParser


def parse_folder(folder: str, winner_only: bool = False, verbose: bool = False) -> list:
    """Parse all game files in a folder into training examples."""
    all_examples = []
    paths = sorted(
        list(Path(folder).glob("*.txt")) + list(Path(folder).glob("*.json"))
    )
    for p in paths:
        try:
            parser   = GameParser.from_file(str(p))
            examples = parser.extract_training_examples(winner_only=winner_only)
            all_examples.extend(examples)
            if verbose:
                m     = json.load(open(p))["match"]
                wname = m["player_names"].get(str(m["winner_id"]), "?")
                print(f"  {p.name}: {len(m['full_log'])} events -> "
                      f"{len(examples)} examples  (winner: {wname})")
        except Exception as e:
            print(f"  [SKIP] {p.name}: {e}")
    return all_examples


def save_jsonl(examples: list, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_messages()) + "\n")
    print(f"Saved {len(examples)} examples -> {output_path}")


def print_stats(examples: list):
    from collections import Counter
    print(f"\n{'='*55}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*55}")
    print(f"Total examples : {len(examples)}")

    action_counts = Counter()
    for ex in examples:
        for line in ex.asst_content.splitlines():
            tok = line.split()
            if tok:
                action_counts[tok[0]] += 1

    print("\nAction distribution (assistant tokens):")
    for a, c in action_counts.most_common():
        print(f"  {a:<25s}: {c}")

    player_counts = Counter(ex.player_id for ex in examples)
    print("\nBy player:")
    for pid, c in player_counts.most_common():
        print(f"  P{pid}: {c}")

    if examples:
        avg = sum(len(ex.user_content.split()) for ex in examples) // len(examples)
        print(f"\nAvg user content : ~{avg} words")
        print(f"Avg asst content : ~{sum(len(ex.asst_content.split()) for ex in examples)//len(examples)} words")





def main():
    ap = argparse.ArgumentParser(description="Parse game logs into SFT training data")
    ap.add_argument("--input",       "-i", required=True, help="File or folder")
    ap.add_argument("--output",      "-o", required=True, help="Output .jsonl")
    ap.add_argument("--winner-only",       action="store_true",
                    help="Only include turns from the winning player")
    ap.add_argument("--verbose",     "-v", action="store_true")
    ap.add_argument("--stats-only",        action="store_true",
                    help="Print stats but do not save the file")
    ap.add_argument("--preview",           type=int, default=0,
                    help="Print N full examples to stdout")
    ap.add_argument("--candidates",        action="store_true",
                    help="Include candidate placement previews in prompts")
    args = ap.parse_args()

    import os
    if os.path.isdir(args.input):
        examples = parse_folder(args.input, winner_only=args.winner_only, verbose=args.verbose)
    else:
        parser   = GameParser.from_file(args.input)
        examples = parser.extract_training_examples(winner_only=args.winner_only)
        if args.verbose:
            print(repr(parser))

    print_stats(examples)

    if args.preview:
        for ex in examples[:args.preview]:
            print(f"\n{'─'*60}")
            print(f"R{ex.round_num} T{ex.turn_num} P{ex.player_id}")
            print("─── USER ───")
            print(ex.user_content)
            print("─── ASSISTANT ───")
            print(ex.asst_content)

    if not args.stats_only:
        save_jsonl(examples, args.output)


if __name__ == "__main__":
    main()