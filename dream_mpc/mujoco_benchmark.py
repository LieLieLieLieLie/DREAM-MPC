from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np

from .config import AGENT_TYPES, METHODS, MODELS, ControllerConfig, ExperimentConfig
from .controllers import DreamController
from .envs import Layout, make_layout, minimum_clearance


MUJOCO_CONDITIONS = {
    "刚体标称": dict(mass_scale=1.0, gain=1.0, damping=18.0, impulse=0.0),
    "低附着": dict(mass_scale=1.0, gain=0.78, damping=11.0, impulse=0.035),
    "载荷失配": dict(mass_scale=1.35, gain=0.86, damping=20.0, impulse=0.025),
    "复合失配": dict(mass_scale=1.40, gain=0.68, damping=10.0, impulse=0.070),
}


def _xml(layout: Layout, n_agents: int, condition: str) -> str:
    c = MUJOCO_CONDITIONS[condition]
    bodies = []
    actuators = []
    for i in range(n_agents):
        typ = AGENT_TYPES[i % len(AGENT_TYPES)]
        x, y = layout.starts[i]
        mass = typ.mass * c["mass_scale"]
        bodies.append(f'''<body name="robot_{i}" pos="{x:.6f} {y:.6f} 0.14">
          <joint name="robot_{i}_x" type="slide" axis="1 0 0" damping="{c['damping']:.3f}"/>
          <joint name="robot_{i}_y" type="slide" axis="0 1 0" damping="{c['damping']:.3f}"/>
          <geom name="robot_{i}_geom" type="cylinder" size="{typ.radius:.5f} 0.12" mass="{mass:.4f}"
                rgba="{1.0 if i%3==0 else 0.25} {0.35 if i%3==0 else 0.60} {0.35 if i%3==0 else 0.92} 1"
                friction="0.8 0.04 0.02" contype="1" conaffinity="1"/>
        </body>''')
        force = typ.mass * typ.amax * 1.8
        actuators.append(f'<motor name="motor_{i}_x" joint="robot_{i}_x" ctrlrange="-{force:.3f} {force:.3f}" ctrllimited="true"/>')
        actuators.append(f'<motor name="motor_{i}_y" joint="robot_{i}_y" ctrlrange="-{force:.3f} {force:.3f}" ctrllimited="true"/>')
    obstacles = []
    for j, obstacle in enumerate(layout.obstacles):
        obstacles.append(f'<geom name="obstacle_{j}" type="cylinder" pos="{obstacle.center[0]} {obstacle.center[1]} 0.28" size="{obstacle.radius} 0.28" rgba="0.30 0.30 0.30 1" contype="1" conaffinity="1"/>')
    goals = []
    for i, (x, y) in enumerate(layout.goals):
        goals.append(
            f'<site name="goal_{i}" type="cylinder" pos="{x:.6f} {y:.6f} 0.012" '
            f'size="0.18 0.010" rgba="{1.0 if i%3==0 else 0.25} '
            f'{0.35 if i%3==0 else 0.60} {0.35 if i%3==0 else 0.92} 0.75"/>'
        )
    return f'''<mujoco model="dream_mpc_rigid_team">
      <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
      <size njmax="6000" nconmax="1200"/>
      <visual><global offwidth="1280" offheight="900"/><map znear="0.01" zfar="60"/></visual>
      <worldbody>
        <light pos="0 0 12" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
        <camera name="top" pos="0 0 17" xyaxes="1 0 0 0 1 0"/>
        <geom name="floor" type="plane" size="9 9 0.1" rgba="0.94 0.94 0.94 1" friction="0.9 0.06 0.02"/>
        {''.join(obstacles)}
        {''.join(goals)}
        {''.join(bodies)}
      </worldbody>
      <actuator>{''.join(actuators)}</actuator>
    </mujoco>'''


def _contact_audit(mujoco, model, data) -> tuple[set[tuple[int, int]], float]:
    """Return active robot-related geom pairs and peak normal/contact force.

    A set of geom pairs is used instead of the raw number of solver contacts so
    that a single physical contact is not multiplied by MuJoCo contact points.
    """
    pairs: set[tuple[int, int]] = set(); peak = 0.0
    for c in range(data.ncon):
        contact = data.contact[c]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        relevant = ("robot_" in n1 or "robot_" in n2) and "floor" not in {n1, n2}
        if relevant:
            force = np.zeros(6); mujoco.mj_contactForce(model, data, c, force)
            peak = max(peak, float(np.linalg.norm(force[:3])))
            pairs.add(tuple(sorted((int(contact.geom1), int(contact.geom2)))))
    return pairs, peak


def run_mujoco_episode(cfg: ExperimentConfig, method: str, seed: int, layout_name: str, condition: str,
                       n_agents: int = 6, keep_trace: bool = False,
                       snapshot_sink: dict | None = None, snapshot_render: bool = True):
    import mujoco
    rng = np.random.default_rng(seed + 7127)
    layout = make_layout(layout_name, n_agents, seed)
    model = mujoco.MjModel.from_xml_string(_xml(layout, n_agents, condition))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=540, width=720) if snapshot_sink is not None and snapshot_render else None
    best_force = -1.0
    best_clearance = np.inf
    controller = DreamController(cfg.controller, method)
    radii = np.asarray([AGENT_TYPES[i % len(AGENT_TYPES)].radius for i in range(n_agents)])
    vmax = np.asarray([AGENT_TYPES[i % len(AGENT_TYPES)].vmax for i in range(n_agents)])
    amax = np.asarray([AGENT_TYPES[i % len(AGENT_TYPES)].amax for i in range(n_agents)])
    masses_nominal = np.asarray([AGENT_TYPES[i % len(AGENT_TYPES)].mass for i in range(n_agents)])
    qpos = np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_{i}_{d}")] for i in range(n_agents) for d in ("x", "y")])
    qvel = np.asarray([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_{i}_{d}")] for i in range(n_agents) for d in ("x", "y")])
    motor_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{i}_{d}") for i in range(n_agents) for d in ("x", "y")])
    done = np.zeros(n_agents, dtype=bool)
    positions = []; velocities = []; commands = []; solve_ms = []; margins = []
    contacts_total = 0; contact_episodes = 0; contact_duration_s = 0.0; pair_contact_duration_s = 0.0
    previous_contact_pairs: set[tuple[int, int]] = set()
    peak_force = 0.0; energy = 0.0; tracking = []; lateral_slip = []
    substeps = int(round(cfg.controller.dt / model.opt.timestep))
    started = time.perf_counter()
    for step in range(cfg.steps):
        pos = layout.starts + data.qpos[qpos].reshape(n_agents, 2)
        vel = data.qvel[qvel].reshape(n_agents, 2).copy()
        safety_pad = 0.055 if method == "DREAM-MPC" else (0.025 if method == "Uniform-DRMPC" else 0.0)
        gain = MUJOCO_CONDITIONS[condition]["gain"]
        mass_scale = MUJOCO_CONDITIONS[condition]["mass_scale"]
        # The actuator applies m_nominal * gain * u to a body whose actual
        # mass is m_nominal * mass_scale.  Hence the acceleration-domain
        # coefficient required by the HOCBF is gain / mass_scale.
        equivalent_acceleration_gain = gain / mass_scale
        state = {"pos": pos, "vel": vel, "goal": layout.goals, "done": done, "radii": radii + safety_pad,
                 "vmax": vmax, "amax": amax,
                 "control_effectiveness": np.full(n_agents, equivalent_acceleration_gain),
                 "accel_drift": np.zeros((n_agents, 2)), "obstacles": layout.obstacles,
                 "time": step * cfg.controller.dt}
        state["priority"] = np.linspace(1.0, 0.15, n_agents)
        u = controller.act(state)
        solution = controller.last_solution; assert solution is not None
        target_force = (masses_nominal[:, None] * gain * u).reshape(-1)
        prev_vel = vel.copy()
        for sub in range(substeps):
            impulse = MUJOCO_CONDITIONS[condition]["impulse"]
            if impulse > 0 and (31 <= step % 63 <= 38):
                target_force += masses_nominal.repeat(2) * impulse * rng.normal(0.0, 1.0, 2*n_agents)
            data.ctrl[motor_ids] = target_force
            mujoco.mj_step(model, data)
            active_pairs, pforce = _contact_audit(mujoco, model, data)
            contacts_total += len(active_pairs)
            contact_episodes += len(active_pairs - previous_contact_pairs)
            if active_pairs:
                contact_duration_s += float(model.opt.timestep)
            pair_contact_duration_s += len(active_pairs) * float(model.opt.timestep)
            previous_contact_pairs = active_pairs
            peak_force = max(peak_force, pforce)
            if renderer is not None and pforce > best_force and pforce > 0:
                renderer.update_scene(data, camera="top")
                snapshot_sink.update({"frame": renderer.render().copy(), "step": step,
                                      "substep": sub, "selection": "peak_contact",
                                      "instant_force_N": pforce})
                best_force = pforce
            elif snapshot_sink is not None and not snapshot_render and pforce > best_force and pforce > 0:
                contact_pos = layout.starts + data.qpos[qpos].reshape(n_agents, 2)
                snapshot_sink.update({"positions": contact_pos.tolist(), "step": step,
                                      "substep": sub, "selection": "peak_contact_state",
                                      "instant_force_N": pforce,
                                      "radii": radii.tolist(),
                                      "goals": layout.goals.tolist(),
                                      "obstacles": [{"center": list(o.center), "radius": o.radius}
                                                    for o in layout.obstacles]})
                best_force = pforce
        new_pos = layout.starts + data.qpos[qpos].reshape(n_agents, 2)
        new_vel = data.qvel[qvel].reshape(n_agents, 2).copy()
        realized = (new_vel - prev_vel) / cfg.controller.dt
        tracking.append(float(np.sqrt(np.mean((realized - u) ** 2))))
        direction = layout.goals - new_pos
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9
        tangent = np.column_stack((-direction[:, 1], direction[:, 0]))
        lateral_slip.append(float(np.mean(np.abs(np.sum(new_vel * tangent, axis=1)))))
        energy += float(np.sum(np.abs(target_force.reshape(n_agents, 2) * new_vel)) * cfg.controller.dt)
        done |= np.linalg.norm(new_pos - layout.goals, axis=1) <= cfg.goal_tolerance
        if renderer is not None and best_force <= 0:
            pair_clearance, obstacle_clearance = minimum_clearance(new_pos, radii, layout.obstacles)
            clearance = min(pair_clearance, obstacle_clearance)
            if clearance < best_clearance:
                renderer.update_scene(data, camera="top")
                snapshot_sink.update({"frame": renderer.render().copy(), "step": step,
                                      "substep": substeps - 1, "selection": "minimum_clearance",
                                      "instant_clearance_m": clearance})
                best_clearance = clearance
        positions.append(new_pos.tolist()); velocities.append(new_vel.tolist()); commands.append(u.tolist())
        solve_ms.append(solution.solve_ms); margins.append(solution.max_slack)
        if done.all():
            break
    pos_arr = np.asarray(positions)
    pair_values = []
    obs_values = []
    for frame in pos_arr:
        pc, oc = minimum_clearance(frame, radii, layout.obstacles); pair_values.append(pc); obs_values.append(oc)
    collided = contact_episodes > 0 or min(pair_values, default=99.0) < -0.002 or min(obs_values, default=99.0) < -0.002
    result = {
        "phase": "mujoco", "method": method, "seed": seed, "layout": layout_name, "condition": condition,
        "n_agents": n_agents, "success": float(done.all() and not collided), "collision_free": float(not collided),
        "completion": float(np.mean(done)), "min_pair_clearance": float(min(pair_values, default=99.0)),
        "min_obstacle_clearance": float(min(obs_values, default=99.0)),
        "contact_count": contacts_total,
        "contact_episodes": contact_episodes,
        "contact_duration_s": contact_duration_s,
        "pair_contact_duration_s": pair_contact_duration_s,
        "peak_contact_force_N": peak_force, "tracking_rmse": float(np.mean(tracking)),
        "lateral_slip_mps": float(np.mean(lateral_slip)), "energy_J": energy,
        "completion_time_s": len(positions) * cfg.controller.dt, "solve_ms_mean": float(np.mean(solve_ms)),
        "solve_ms_p95": float(np.quantile(solve_ms, 0.95)), "max_slack": float(max(margins, default=0.0)),
        "wall_time_s": time.perf_counter() - started,
    }
    trace = None
    if keep_trace:
        trace = {"starts": layout.starts.tolist(), "goals": layout.goals.tolist(), "radii": radii.tolist(),
                 "obstacles": [{"center": list(o.center), "radius": o.radius} for o in layout.obstacles],
                 "positions": positions, "velocities": velocities, "commands": commands,
                 "condition": condition, "method": method}
    if renderer is not None:
        renderer.close()
    return result, trace


def run_mujoco_suite(cfg: ExperimentConfig | None = None):
    cfg = cfg or ExperimentConfig(); cfg.ensure_dirs()
    records = []; traces = {}; count = 0; started = time.perf_counter()
    for layout in ("交叉口", "环岛"):
        for condition in MUJOCO_CONDITIONS:
            for method in METHODS:
                for seed in cfg.mujoco_seeds:
                    keep = seed == 1 and condition == "复合失配" and layout == "环岛"
                    row, trace = run_mujoco_episode(cfg, method, seed + 500, layout, condition, 6, keep)
                    records.append(row)
                    if keep and trace is not None:
                        traces[f"{method}-mujoco"] = trace
                    count += 1
                    if count % 12 == 0:
                        print(f"[MuJoCo {count:03d}] {layout}/{condition}/{method} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    with (MODELS / "mujoco_records.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (MODELS / "mujoco_traces.json").open("w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False)
    meta = {"mujoco_version": __import__("mujoco").__version__, "config": asdict(cfg),
            "episodes": len(records), "elapsed_s": time.perf_counter() - started}
    with (MODELS / "mujoco_run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return records, traces
