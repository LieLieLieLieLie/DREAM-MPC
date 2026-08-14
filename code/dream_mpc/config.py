from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
TABLES = RESULTS / "tables"

METHODS = (
    "Tube-DMPC",
    "CC-MPC",
    "CVaR-MPC",
    "ECBF-NMPC",
    "Uniform-DRMPC",
    "DREAM-MPC",
)

METHOD_COLORS = {
    "DREAM-MPC": "#FF6666",
    "Tube-DMPC": "#FFAA53",
    "CC-MPC": "#50CC55",
    "CVaR-MPC": "#3399FF",
    "ECBF-NMPC": "#6666FF",
    "Uniform-DRMPC": "#9933FF",
}


@dataclass(frozen=True)
class AgentType:
    name: str
    radius: float
    vmax: float
    amax: float
    mass: float
    actuator_gain: float
    lag: float


AGENT_TYPES = (
    AgentType("AMR", 0.25, 1.45, 1.80, 90.0, 1.00, 0.72),
    AgentType("叉车", 0.34, 1.05, 1.15, 420.0, 0.83, 0.48),
    AgentType("牵引车", 0.30, 1.18, 1.35, 260.0, 0.91, 0.58),
)


@dataclass(frozen=True)
class ControllerConfig:
    dt: float = 0.18
    horizon: int = 7
    scp_iterations: int = 3
    total_risk: float = 0.18
    epsilon_min: float = 0.0012
    epsilon_max: float = 0.055
    covariance_floor: float = 0.0045
    covariance_growth: float = 0.0026
    moment_inflation: float = 1.06
    activation_distance: float = 2.15
    obstacle_activation_distance: float = 1.55
    fixed_tube: float = 0.24
    tracking_weight: float = 2.8
    terminal_weight: float = 9.0
    control_weight: float = 0.075
    smooth_weight: float = 0.22
    slack_linear: float = 2400.0
    slack_quadratic: float = 90.0
    risk_smoothing: float = 0.58
    cbf_gamma: float = 0.34
    cbf_iterations: int = 22


@dataclass(frozen=True)
class ExperimentConfig:
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    steps: int = 240
    goal_tolerance: float = 0.42
    main_seeds: tuple[int, ...] = tuple(range(8))
    auxiliary_seeds: tuple[int, ...] = tuple(range(6))
    mujoco_seeds: tuple[int, ...] = tuple(range(4))
    layouts: tuple[str, ...] = ("交叉口", "环岛", "仓储通道")
    conditions: tuple[str, ...] = ("标称", "重尾扰动", "混合偏移", "突发失配")
    methods: tuple[str, ...] = METHODS

    def ensure_dirs(self) -> None:
        for path in (RESULTS, FIGURES, MODELS, TABLES):
            path.mkdir(parents=True, exist_ok=True)
