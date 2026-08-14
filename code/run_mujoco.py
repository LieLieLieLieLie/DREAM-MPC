import os

try:
    os.urandom(1)
except OSError:
    os.urandom = lambda n: bytes((43 + 19 * i) % 256 for i in range(n))  # type: ignore[assignment]

from dream_mpc.mujoco_benchmark import run_mujoco_suite


if __name__ == "__main__":
    records, _ = run_mujoco_suite()
    print(f"MuJoCo trials: {len(records)}")
