import os

try:
    os.urandom(1)
except OSError:
    os.urandom = lambda n: bytes((11 + 31 * i) % 256 for i in range(n))  # type: ignore[assignment]

from dream_mpc.visualization import generate_all


if __name__ == "__main__":
    generate_all()
