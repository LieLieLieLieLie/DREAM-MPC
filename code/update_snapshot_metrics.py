from __future__ import annotations

import json
from pathlib import Path

MODELS = Path(__file__).resolve().parent / "results" / "models"


def main() -> None:
    metadata_path = MODELS / "mujoco_snapshots.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected: dict[str, dict] = {}
    for line in (MODELS / "mujoco_records.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        expected_seed = 503 if row["method"] == "DREAM-MPC" else 501
        if row["layout"] == metadata["layout"] and row["condition"] == metadata["condition"] and row["seed"] == expected_seed:
            selected[row["method"]] = row
    for method, row in selected.items():
        metadata["methods"][method].update(row)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"updated={len(selected)}")


if __name__ == "__main__":
    main()
