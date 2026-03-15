"""
Extract all scalars from a TensorBoard event file into a single JSON file.
Output: (cwd)/tmp/all_scalars.json
Structure: { "metric_name": [{"step": int, "value": float, "wall_time": float}, ...], ... }
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Extract TensorBoard scalars to a single JSON file."
    )
    parser.add_argument(
        "event_file",
        type=str,
        nargs="?",
        default="output/events.out.tfevents.0",
        help="Path to TensorBoard event file (default: output/events.out.tfevents.0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <cwd>/tmp/all_scalars.json)",
    )
    args = parser.parse_args()

    event_path = os.path.abspath(args.event_file)
    if not os.path.isfile(event_path):
        print(f"Error: event file not found: {event_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as e:
        print(f"Error: tensorboard is required: {e}", file=sys.stderr)
        sys.exit(1)

    # EventAccumulator expects a directory; pass the directory containing the event file
    log_dir = os.path.dirname(event_path)
    accumulator = EventAccumulator(log_dir)
    accumulator.Reload()

    scalar_tags = accumulator.Tags().get("scalars", [])
    if not scalar_tags:
        print("No scalars found in the event file.", file=sys.stderr)
        out_data = {}
    else:
        out_data = {}
        for tag in scalar_tags:
            events = accumulator.Scalars(tag)
            out_data[tag] = [
                {"step": e.step, "value": e.value, "wall_time": e.wall_time}
                for e in events
            ]

    out_path = args.output
    if out_path is None:
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        out_path = os.path.join(tmp_dir, "all_scalars.json")
    else:
        out_path = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(out_data)} scalar series to {out_path}")


if __name__ == "__main__":
    main()
