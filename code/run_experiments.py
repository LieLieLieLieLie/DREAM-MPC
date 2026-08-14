import os

try:
    os.urandom(1)
except OSError:
    os.urandom = lambda n: bytes((41 + 23 * i) % 256 for i in range(n))  # type: ignore[assignment]

from dream_mpc.experiments import run_all


if __name__ == "__main__":
    records, _ = run_all()
    print(records.groupby(["phase", "method"]).mean(numeric_only=True)[
        ["success", "collision_free", "min_pair_clearance", "completion_time_s", "solve_ms_mean"]
    ].round(4))
