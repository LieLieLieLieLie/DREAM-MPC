from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "results" / "models"
STATE_PATH = MODELS / "dream_failure_state.json"


def capture_state() -> None:
    """Run only the documented rare-contact case in the MuJoCo runtime."""
    from dream_mpc.config import ExperimentConfig
    from dream_mpc.mujoco_benchmark import run_mujoco_episode

    cfg = ExperimentConfig()
    sink: dict[str, object] = {}
    result, _ = run_mujoco_episode(
        cfg, "DREAM-MPC", seed=503, layout_name="环岛", condition="复合失配",
        n_agents=6, keep_trace=False, snapshot_sink=sink, snapshot_render=False,
    )
    if sink.get("selection") != "peak_contact_state":
        raise RuntimeError("The selected DREAM-MPC failure run did not yield a contact state")
    payload = {**sink, "result": result}
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"captured DREAM failure state: episodes={result['contact_episodes']} "
        f"duration={result['contact_duration_s']:.2f}s peak={result['peak_contact_force_N']:.1f}N"
    )


def _state_frame(sink: dict[str, object]) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    positions = np.asarray(sink["positions"], dtype=float)
    radii = np.asarray(sink["radii"], dtype=float)
    goals = np.asarray(sink["goals"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=100)
    ax.set_facecolor("#EEEEEE")
    ax.set_xlim(-11.5, 11.5)
    ax.set_ylim(-7.5, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")
    for obstacle in sink["obstacles"]:  # type: ignore[index]
        center = obstacle["center"]  # type: ignore[index]
        radius = obstacle["radius"]  # type: ignore[index]
        ax.add_patch(Circle(center, radius, facecolor="#AFAFAF", edgecolor="#777777", lw=2.2))
    colors = ("#55DDE0", "#FF9999")
    for index, (position, radius, goal) in enumerate(zip(positions, radii, goals)):
        color = colors[index % 2]
        ax.add_patch(Circle(position, radius, facecolor=color, edgecolor="white", lw=1.4, alpha=.98))
        ax.scatter(goal[0], goal[1], s=22, color=color, edgecolor="white", linewidth=.5, zorder=5)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def render_state() -> None:
    """Render the exact saved MuJoCo state with the plotting runtime."""
    sink = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    result = sink.pop("result")
    frames_path = MODELS / "mujoco_snapshots.npz"
    with np.load(frames_path) as existing:
        frames = {key: existing[key] for key in existing.files}
    frames["DREAM-MPC"] = _state_frame(sink)
    np.savez_compressed(frames_path, **frames)

    metadata_path = MODELS / "mujoco_snapshots.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("seed", None)
    metadata["selection_protocol"] = (
        "baselines: roundabout-compound-mismatch seed 501; "
        "DREAM-MPC rare-contact case: seed 503"
    )
    metadata["methods"]["DREAM-MPC"] = {
        **{key: value for key, value in sink.items() if key != "positions"},
        **result,
        "selection": "peak_contact_state_reconstruction",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"rendered DREAM failure state: episodes={result['contact_episodes']} "
        f"duration={result['contact_duration_s']:.2f}s peak={result['peak_contact_force_N']:.1f}N"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "render"))
    args = parser.parse_args()
    capture_state() if args.mode == "capture" else render_state()


if __name__ == "__main__":
    main()
