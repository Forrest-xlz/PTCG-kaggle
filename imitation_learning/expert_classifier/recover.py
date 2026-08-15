"""Recover selected complete state-action records from extracted JSONL shards."""
from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_classifier.config import load_settings, project_path


SampleKey = tuple[int, int, int]


def _read_selection(path: Path) -> dict[str, set[SampleKey]]:
    selected: dict[str, set[SampleKey]] = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "episode_id", "player", "step"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"selection CSV is missing columns: {sorted(missing)}")
        for row in reader:
            date = row["date"]
            key = (int(row["episode_id"]), int(row["player"]), int(row["step"]))
            if key in selected[date]:
                raise ValueError(f"duplicate selected sample: {date} {key}")
            selected[date].add(key)
    if not selected:
        raise ValueError("selection CSV contains no samples")
    return dict(selected)


def recover_selected(
    selection_path: Path,
    extracted_root: Path,
    output_root: Path,
) -> dict[str, int]:
    selected = _read_selection(Path(selection_path))
    extracted_root = Path(extracted_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for date, wanted in sorted(selected.items()):
        source = extracted_root / f"{date}.jsonl.gz"
        if not source.is_file():
            raise FileNotFoundError(f"extracted replay shard not found: {source}")
        found: set[SampleKey] = set()
        destination = output_root / f"{date}.jsonl.gz"
        with (
            gzip.open(source, "rt", encoding="utf-8") as reader,
            gzip.open(destination, "wt", encoding="utf-8") as writer,
        ):
            for line_number, line in enumerate(reader, start=1):
                record = json.loads(line)
                try:
                    key = (
                        int(record["episode_id"]),
                        int(record["player"]),
                        int(record["step"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{source.name}:{line_number} has invalid sample identity"
                    ) from exc
                if key in wanted:
                    writer.write(line if line.endswith("\n") else line + "\n")
                    found.add(key)
        missing = wanted - found
        if missing:
            preview = sorted(missing)[:5]
            raise ValueError(
                f"failed to recover {len(missing)} samples for {date}; examples={preview}"
            )
        counts[date] = len(found)
    return counts


def main() -> None:
    settings = load_settings()
    selection = project_path(settings.inference.output) / "selected_loser_samples.csv.gz"
    counts = recover_selected(
        selection,
        project_path(settings.extracted_data),
        project_path(settings.recovery.output),
    )
    for date, count in counts.items():
        print(f"recovered_date={date} samples={count:,}", flush=True)
    print(f"recovered_total={sum(counts.values()):,}", flush=True)


if __name__ == "__main__":
    main()
