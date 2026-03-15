"""
Extract all scalars from a TensorBoard event file into a single JSON file.
Output: (current work dir)/tmp/all_scalars.json
"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Extract TensorBoard scalars to JSON (one file, keys = metric names)."
    )
    parser.add_argument(
        "event_path",
        type=str,
        nargs="?",
        default="output/events.out.tfevents.0",
        help="Path to TensorBoard event file or directory containing it (default: output/events.out.tfevents.0)",
    )
    args = parser.parse_args()

    event_path = os.path.abspath(args.event_path)
    if os.path.isfile(event_path):
        logdir = os.path.dirname(event_path)
    elif os.path.isdir(event_path):
        logdir = event_path
    else:
        print(f"Error: path not found: {args.event_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as e:
        print(f"Error: tensorboard is required: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        ea = event_accumulator.EventAccumulator(logdir)
        ea.Reload()
    except Exception as e:
        print(f"Error reading TensorBoard data from {logdir}: {e}", file=sys.stderr)
        sys.exit(1)

    scalar_tags = ea.Tags().get("scalars", [])
    if not scalar_tags:
        print("No scalars found in event data.", file=sys.stderr)
        out_data = {}
    else:
        out_data = {}
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            out_data[tag] = [
                {"step": e.step, "value": e.value, "wall_time": e.wall_time}
                for e in events
            ]

    out_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "all_scalars.json")

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
    except OSError as e:
        print(f"Error writing {out_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {len(out_data)} scalar series to {out_path}")


if __name__ == "__main__":
    main()
