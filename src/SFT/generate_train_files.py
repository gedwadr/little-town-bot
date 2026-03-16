import argparse
import json
from pathlib import Path

from src.games.game_parser import GameParser
from src.games.game_parser_nn import GameParserNN


def parse_folder(folder: str, winner_only: bool = False, verbose: bool = False,
                 nn: bool = False) -> list:
    """Parse all game files in a folder into training examples."""
    all_examples = []
    paths = sorted(
        list(Path(folder).glob("*.txt")) + list(Path(folder).glob("*.json"))
    )
    for p in paths:
        try:
            parser = GameParser.from_file(str(p), parser_cls=GameParserNN if nn else None)
            if nn:
                examples = parser.extract_nn_training_examples(winner_only=winner_only)
            else:
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


def save_jsonl(examples: list, output_path: str, nn: bool = False):
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            row = ex if nn else ex.to_messages()
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(examples)} examples -> {output_path}")


def print_stats(examples: list, nn: bool = False):
    from collections import Counter
    print(f"\n{'='*55}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*55}")
    print(f"Total examples : {len(examples)}")

    if nn:
        if examples:
            action_types = {0.0: "placeWorker", 0.5: "buildBuilding", 1.0: "substituteResource"}
            type_counts = Counter(
                action_types.get(ex["candidates"][ex["correct_idx"]][10], "unknown")
                for ex in examples
            )
            print("\nChosen action type distribution:")
            for a, c in type_counts.most_common():
                print(f"  {a:<25s}: {c}")
            avg_cands = sum(len(ex["candidates"]) for ex in examples) // len(examples)
            print(f"\nState vector length  : {len(examples[0]['state'])}")
            print(f"Action vector length : {len(examples[0]['candidates'][0])}")
            print(f"Avg candidates/turn  : {avg_cands}")
        return

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
    ap = argparse.ArgumentParser(description="Parse game logs into SFT or NN training data")
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
                    help="Include candidate placement previews in prompts (SFT only)")
    ap.add_argument("--nn",                action="store_true",
                    help="Generate numerical NN training data instead of SFT text data")
    args = ap.parse_args()

    import os
    if os.path.isdir(args.input):
        examples = parse_folder(args.input, winner_only=args.winner_only,
                                verbose=args.verbose, nn=args.nn)
    else:
        parser = GameParser.from_file(args.input, parser_cls=GameParserNN if args.nn else None)
        if args.nn:
            examples = parser.extract_nn_training_examples(winner_only=args.winner_only)
        else:
            examples = parser.extract_training_examples(winner_only=args.winner_only)
        if args.verbose:
            print(repr(parser))

    print_stats(examples, nn=args.nn)

    if args.preview:
        for ex in examples[:args.preview]:
            print(f"\n{'─'*60}")
            if args.nn:
                print(f"state({len(ex['state'])}): {ex['state'][:6]}...")
                print(f"candidates: {len(ex['candidates'])} moves")
                print(f"correct_idx: {ex['correct_idx']}")
                print(f"chosen: {ex['candidates'][ex['correct_idx']]}")
            else:
                print(f"R{ex.round_num} T{ex.turn_num} P{ex.player_id}")
                print("─── USER ───")
                print(ex.user_content)
                print("─── ASSISTANT ───")
                print(ex.asst_content)

    if not args.stats_only:
        save_jsonl(examples, args.output, nn=args.nn)


if __name__ == "__main__":
    main()