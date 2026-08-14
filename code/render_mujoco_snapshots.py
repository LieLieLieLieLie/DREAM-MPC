from __future__ import annotations

import json

import numpy as np

from dream_mpc.config import ExperimentConfig, METHODS, MODELS
from dream_mpc.mujoco_benchmark import run_mujoco_episode


def main() -> None:
    """Re-run one controlled rigid-body case and retain renderer-native evidence."""
    cfg = ExperimentConfig()
    cfg.ensure_dirs()
    frames: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {
        "layout": "环岛",
        "condition": "复合失配",
        "selection_protocol": "baselines use seed 501; DREAM-MPC uses its rare-contact seed 503",
        "camera": "top",
        "methods": {},
    }
    for method in METHODS:
        selected_seed = 503 if method == "DREAM-MPC" else 501
        sink: dict[str, object] = {}
        result, _ = run_mujoco_episode(
            cfg, method, seed=selected_seed, layout_name="环岛", condition="复合失配",
            n_agents=6, keep_trace=False, snapshot_sink=sink,
        )
        frame = sink.pop("frame")
        frames[method] = np.asarray(frame, dtype=np.uint8)
        metadata["methods"][method] = {**sink, **result}
        print(
            f"[snapshot] {method}: {sink.get('selection')} "
            f"episodes={result['contact_episodes']} duration={result['contact_duration_s']:.3f} s ",
            flush=True,
        )
    np.savez_compressed(MODELS / "mujoco_snapshots.npz", **frames)
    with (MODELS / "mujoco_snapshots.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
