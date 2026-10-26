"""
NavRL-Optimized Drone Environment  v3 — Three-track optimization
=================================================================

Based on v2 (all bugs fixed).  Adds three optimization tracks targeting
the remaining ~4.6% failure cases (95.4% → target 97-98%):

Track A — Adaptive VO time horizon (τ)
    Fixed τ=3s is too long when obstacles are very close (VO cones become
    huge, LP often infeasible) and too short at high speed with distant
    obstacles.  τ now scales with speed × proximity.

Track B — RRT* safest-direction fallback (enhanced)
    When no goal-directed path exists, the score function now uses
    clearance² × progress instead of linear sum, plus a minimum
    clearance threshold to reject directions that merely graze obstacles.

Track C — Distance-based speed modulation
    A smooth speed reduction layer that activates BEFORE the VO shield
    needs to fire (min_d < 0.15 → 3m).  Gives the system more reaction
    time and handles the "too fast to react" failure mode.  Includes
    emergency brake at min_d < 0.05 (< 1m).

All three tracks are integrated into safe_ruler as sequential layers:
    1. Extract obstacles → 2. Adaptive τ → 3. VO shield →
    4. RRT* blend if VO conservative → 5. Speed modulation →
    6. Distribution-shift guard → 7. Target proximity gate
"""

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList, EvalCallback
)
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import os, threading, time, pickle, sys, random
import vehicle_manager as VM
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import struct
import socket
from scipy.optimize import linprog

MAX_PACKET_SIZE = 62000
global abmatrix
a_b_matrix = np.zeros((7, 24, 2), dtype=np.float64)
x, y = np.arange(-180, 180, 15), np.arange(-45, 50, 15)
xx, yy = np.meshgrid(x, y)
a_b_matrix[:, :, 0] = yy
a_b_matrix[:, :, 1] = xx
abmatrix = a_b_matrix / 180 * np.pi

# ─────────────────────────────────────────────────────────────────────────────
#  RRT* safe-direction planner constants
# ─────────────────────────────────────────────────────────────────────────────

_RRT_HZ           = 2
_RRT_MAX_ITER     = 150
_RRT_STEP         = 0.40
_RRT_GOAL_BIAS    = 0.15
_RRT_REWIRE_R     = 1.5
_OBS_INFLATE_R    = 0.35
_OBS_PROB_THRESH  = 0.50
_PLAN_HORIZON     = 8.0

_D_BLEND_START    = 0.20
_D_BLEND_END      = 0.05
_ALPHA_MAX        = 0.60
_SPEED_FLOOR      = 0.67
_PLAN_AGE_MAX     = 1.5

# ─────────────────────────────────────────────────────────────────────────────
#  VO Safety Shield constants (NavRL-inspired)
# ─────────────────────────────────────────────────────────────────────────────

_VO_TIME_HORIZON   = 3.0       # base τ — Track A makes this adaptive
_VO_TAU_MIN        = 0.5       # Track A: minimum τ (close obstacles, low speed)
_VO_TAU_MAX        = 4.0       # Track A: maximum τ (far obstacles, high speed)
_VO_ROBOT_RADIUS   = 0.50
_VO_SAFE_MARGIN    = 0.425
_VO_MAX_OBSTACLES  = 25
_VO_ACTIVATION_D   = 6.0
_VO_EMERGENCY_D    = 0.8

# Track C — Speed modulation constants
_SPEED_MOD_START   = 0.35      # min_d ratio below which speed reduction begins (=3m)
_SPEED_MOD_FLOOR   = 0.20      # minimum speed fraction at closest approach
_EMERGENCY_SPEED   = 1.5       # m/s hard cap when min_d < _EMERGENCY_DIST
_EMERGENCY_DIST    = 0.1       # min_d ratio for emergency brake (=1m)

# Octomap geometry — must match APF_SeqCalculate3D parameters
_OCTO_ELEV_BINS   = 18
_OCTO_AZI_BINS    = 36
_OCTO_AZI_START   = -180.0
_OCTO_AZI_END     = 180.0
_OCTO_ELEV_START  = -90.0
_OCTO_ELEV_END    = 90.0
_OCTO_AZI_STEP    = 10.0
_OCTO_ELEV_STEP   = 10.0


# ─────────────────────────────────────────────────────────────────────────────
#  Yaw rotation helpers (body ↔ global frame transforms)
# ─────────────────────────────────────────────────────────────────────────────

def _yaw_from_velocity(velocity):
    """Extract yaw angle (radians) from velocity vector. 0 = +x direction."""
    vx, vy = velocity[0], velocity[1]
    if abs(vx) < 1e-9 and abs(vy) < 1e-9:
        return 0.0
    return np.arctan2(vy, vx)


def _rotate_global_to_body(vec_global, yaw_rad):
    """Rotate a 3D vector from global frame to body frame (yaw only, around z)."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    x_body = c * vec_global[0] + s * vec_global[1]
    y_body = -s * vec_global[0] + c * vec_global[1]
    z_body = vec_global[2]
    return np.array([x_body, y_body, z_body], dtype=np.float64)


def _rotate_body_to_global(vec_body, yaw_rad):
    """Rotate a 3D vector from body frame to global frame (yaw only, around z)."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    x_global = c * vec_body[0] - s * vec_body[1]
    y_global = s * vec_body[0] + c * vec_body[1]
    z_global = vec_body[2]
    return np.array([x_global, y_global, z_global], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
#  Precomputed octomap bin geometry (body frame)
# ─────────────────────────────────────────────────────────────────────────────

_azi_centers = np.linspace(
    _OCTO_AZI_START + _OCTO_AZI_STEP / 2,
    _OCTO_AZI_END   - _OCTO_AZI_STEP / 2,
    _OCTO_AZI_BINS
)
_elev_centers = np.linspace(
    _OCTO_ELEV_START + _OCTO_ELEV_STEP / 2,
    _OCTO_ELEV_END   - _OCTO_ELEV_STEP / 2,
    _OCTO_ELEV_BINS
)
_AZI_GRID, _ELEV_GRID = np.meshgrid(
    np.radians(_azi_centers), np.radians(_elev_centers)
)
_DIR_X = np.cos(_ELEV_GRID) * np.cos(_AZI_GRID)
_DIR_Y = np.cos(_ELEV_GRID) * np.sin(_AZI_GRID)
_DIR_Z = np.sin(_ELEV_GRID)
_BIN_DIRS = np.stack([_DIR_X, _DIR_Y, _DIR_Z], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  Octomap → Obstacle list extraction (BODY FRAME)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_obstacles_from_octomap(octomap_body, stat_range=20.0,
                                     activation_dist=_VO_ACTIVATION_D):
    """
    Convert body-frame octomap (18×36 distance ratios) into obstacle descriptors.
    All returned positions are in BODY frame.
    """
    octomap_matrix = octomap_body.reshape(_OCTO_ELEV_BINS, _OCTO_AZI_BINS)
    distances_m = octomap_matrix * stat_range
    close_mask = distances_m < activation_dist

    if not np.any(close_mask):
        return []

    close_dists = distances_m[close_mask]
    close_dirs  = _BIN_DIRS[close_mask]
    points = close_dirs * close_dists[:, np.newaxis]

    # Grid-based clustering
    cell_size = 1.0
    cell_keys = {}
    for i in range(len(points)):
        pt = points[i]
        key = (int(np.floor(pt[0] / cell_size)),
               int(np.floor(pt[1] / cell_size)),
               int(np.floor(pt[2] / cell_size)))
        if key not in cell_keys:
            cell_keys[key] = []
        cell_keys[key].append(i)

    obstacle_list = []
    for key, indices in cell_keys.items():
        cluster_pts = points[indices]
        center = np.mean(cluster_pts, axis=0)

        if len(cluster_pts) > 1:
            spread = np.max(np.linalg.norm(cluster_pts - center, axis=1))
            radius = max(spread * 0.5 + 0.15, 0.25)
        else:
            radius = 0.35

        dist_to_robot = np.linalg.norm(center)
        if 0.1 < dist_to_robot < activation_dist:
            obstacle_list.append({
                'pos_rel': center,
                'vel':     np.zeros(3, dtype=np.float64),
                'radius':  radius,
            })

    obstacle_list.sort(key=lambda o: np.linalg.norm(o['pos_rel']))
    return obstacle_list[:_VO_MAX_OBSTACLES]

_REPULSION_THRESHOLD = 0.25    # distance ratio below which repulsion activates (5m)
_REPULSION_GAIN      = 2.5     # repulsion strength multiplier
 
def _proximity_repulsion(vel_body, octomap_body, stat_range=20.0,
                          danger_threshold=_REPULSION_THRESHOLD,
                          force_gain=_REPULSION_GAIN,
                          max_velocity=7.5):
    """
    Repulsive force field pushing AWAY from close obstacles regardless of
    velocity direction. Catches lateral obstacles that VO misses.
 
    For each octomap cell closer than danger_threshold, computes a repulsive
    force inversely proportional to distance². The net repulsion is added
    to the velocity in body frame.
 
    Parameters
    ----------
    vel_body         : (3,) DRL velocity in body frame
    octomap_body     : (648,) flattened octomap (18×36 distance ratios)
    stat_range       : float, max sensor range in meters
    danger_threshold : float, distance RATIO below which repulsion activates
    force_gain       : float, repulsion strength multiplier
    max_velocity     : float, speed cap after correction
 
    Returns
    -------
    corrected_vel : (3,) corrected velocity in body frame
    was_applied   : bool, whether repulsion was added
    """
    octomap_matrix = octomap_body.reshape(_OCTO_ELEV_BINS, _OCTO_AZI_BINS)
 
    # Find cells with obstacles within danger zone
    danger_mask = octomap_matrix < danger_threshold
    if not np.any(danger_mask):
        return vel_body, False
 
    # Distances in meters and unit direction vectors for danger cells
    danger_ratios = octomap_matrix[danger_mask]
    danger_dists  = danger_ratios * stat_range
    danger_dirs   = _BIN_DIRS[danger_mask]     # (N, 3) unit vectors TOWARD obstacles
 
    # Inverse-square repulsion: stronger when closer
    safe_dists = np.maximum(danger_dists, 0.3)
    weights = 1.0 / (safe_dists ** 2)
 
    # Weighted sum: net repulsion vector (points AWAY from obstacle cluster)
    repulsion_raw = -np.sum(danger_dirs * weights[:, np.newaxis], axis=0)
    repulsion_norm = np.linalg.norm(repulsion_raw)
 
    if repulsion_norm < 1e-9:
        return vel_body, False
 
    repulsion_unit = repulsion_raw / repulsion_norm
 
    # Urgency: 0 at threshold, 1 at contact
    min_d_ratio = np.min(octomap_matrix)
    urgency = np.clip(1.0 - min_d_ratio / danger_threshold, 0.0, 1.0)
 
    # Density boost: wide obstacle / corridor → stronger push
    density_factor = np.clip(np.sum(danger_mask) / 30.0, 0.5, 2.0)
 
    # Final repulsion magnitude
    repulsion_mag = force_gain * urgency * density_factor
 
    # Apply repulsion
    corrected = vel_body + repulsion_unit * repulsion_mag
 
    # Speed cap
    corrected_speed = np.linalg.norm(corrected)
    if corrected_speed > max_velocity:
        corrected = corrected / corrected_speed * max_velocity
 
    # Don't let repulsion reverse the forward direction entirely
    vel_speed = np.linalg.norm(vel_body)
    if vel_speed > 1e-6:
        vel_dir = vel_body / vel_speed
        forward_component = np.dot(corrected, vel_dir)
        if forward_component < -vel_speed * 0.3:
            backward_excess = forward_component + vel_speed * 0.3
            corrected = corrected - backward_excess * vel_dir
 
    return corrected, True

# ─────────────────────────────────────────────────────────────────────────────
#  Track A — Adaptive VO time horizon
# ─────────────────────────────────────────────────────────────────────────────

def _compute_adaptive_tau(current_speed, min_d_ratio, max_velocity, stat_range=20.0):
    """
    Compute adaptive VO time horizon τ based on speed and obstacle proximity.

    At high speed + far obstacles: longer horizon (more conservative look-ahead).
    At low speed + close obstacles: shorter horizon (tighter VO cones, feasible LP).

    min_d_ratio: normalised min obstacle distance ∈ [0, 1] (from octomap).
    """
    # Speed factor: faster → longer look-ahead
    speed_factor = np.clip(current_speed / max_velocity, 0.1, 1.0)

    # Proximity factor: closer obstacles → shorter horizon
    # min_d_ratio=0.2 means 4m → proximity_factor ≈ 1.0
    # min_d_ratio=0.05 means 1m → proximity_factor ≈ 0.33
    proximity_factor = np.clip(min_d_ratio / 0.15, 0.3, 1.5)

    tau = _VO_TIME_HORIZON * speed_factor * proximity_factor
    return float(np.clip(tau, _VO_TAU_MIN, _VO_TAU_MAX))


# ─────────────────────────────────────────────────────────────────────────────
#  VO Safety Shield — NavRL §III-D, body frame
# ─────────────────────────────────────────────────────────────────────────────

def _compute_vo_delta(robot_vel_body, obs_pos_body, obs_vel_body,
                      robot_r, obs_r, safe_margin, tau):
    """
    For a single obstacle (all vectors in BODY frame), check if robot_vel_body
    is inside the Velocity Obstacle cone, and compute the minimum ΔV to exit.
    """
    combined_r = robot_r + obs_r + safe_margin
    dist = np.linalg.norm(obs_pos_body)

    if dist < 1e-6:
        return np.array([-1.0, 0.0, 0.0]) * np.linalg.norm(robot_vel_body), True, np.array([-1, 0, 0.0]), 0.0

    if dist < combined_r:
        away_dir = -obs_pos_body / dist
        push_mag = max(np.linalg.norm(robot_vel_body) * 0.5, 0.5)
        return away_dir * push_mag, True, away_dir, np.dot(away_dir, robot_vel_body + away_dir * push_mag)

    rel_vel = robot_vel_body - obs_vel_body

    rel_speed_sq = np.dot(rel_vel, rel_vel)
    if rel_speed_sq < 1e-12:
        return np.zeros(3), False, np.zeros(3), 0.0

    t_cpa = np.dot(obs_pos_body, rel_vel) / rel_speed_sq
    if t_cpa < 0:
        return np.zeros(3), False, np.zeros(3), 0.0

    if t_cpa > tau:
        pos_at_tau = obs_pos_body - rel_vel * tau
        if np.linalg.norm(pos_at_tau) >= combined_r:
            return np.zeros(3), False, np.zeros(3), 0.0
        t_cpa = tau

    closest_point = obs_pos_body - rel_vel * t_cpa
    closest_dist = np.linalg.norm(closest_point)

    if closest_dist >= combined_r:
        return np.zeros(3), False, np.zeros(3), 0.0

    # Inside VO — compute correction using ORCA formulation
    cap_center = obs_vel_body + obs_pos_body / tau
    cap_radius = combined_r / tau

    w_cap = robot_vel_body - cap_center
    w_cap_len = np.linalg.norm(w_cap)

    if t_cpa >= tau - 0.01 or w_cap_len <= cap_radius + 1e-9:
        # Closest to the cap
        if w_cap_len < 1e-9:
            n = -obs_pos_body / dist
        else:
            n = w_cap / w_cap_len

        if w_cap_len < 1e-9:
            u = n * cap_radius
        else:
            u = (cap_radius - w_cap_len) * n
    else:
        # Closest to a leg
        if closest_dist > 1e-9:
            n = closest_point / closest_dist
        else:
            perp = np.array([-obs_pos_body[1] / dist, obs_pos_body[0] / dist, 0.0])
            perp_norm = np.linalg.norm(perp)
            if perp_norm < 1e-9:
                perp = np.array([0.0, 1.0, 0.0])
                perp_norm = 1.0
            n = perp / perp_norm

        penetration = combined_r - closest_dist
        u = n * penetration / (t_cpa + 1e-9)

    # Half responsibility (ORCA)
    delta_v = u * 0.5
    constraint_point = robot_vel_body + delta_v
    n_norm = np.linalg.norm(n)
    if n_norm > 1e-9:
        n = n / n_norm
    offset = np.dot(n, constraint_point)

    return delta_v, True, n, offset


def _vo_safety_shield(robot_vel_body, obstacle_list, max_velocity,
                      robot_r=_VO_ROBOT_RADIUS,
                      safe_margin=_VO_SAFE_MARGIN,
                      tau=_VO_TIME_HORIZON):
    """
    VO safety shield — all computation in BODY frame.
    Returns (safe_vel_body, was_corrected, n_violations).
    """
    if len(obstacle_list) == 0:
        return robot_vel_body.copy(), False, 0

    constraints_n = []
    constraints_d = []
    total_delta   = np.zeros(3, dtype=np.float64)
    n_violations  = 0
    emergency     = False

    for obs in obstacle_list:
        obs_dist = np.linalg.norm(obs['pos_rel'])

        if obs_dist < _VO_EMERGENCY_D:
            emergency = True
            away = -obs['pos_rel'] / (obs_dist + 1e-9)
            total_delta += away * max_velocity * 0.4
            n_violations += 1
            continue

        delta_v, in_vo, n, d = _compute_vo_delta(
            robot_vel_body, obs['pos_rel'],
            obs.get('vel', np.zeros(3)),
            robot_r, obs['radius'], safe_margin, tau
        )
        if in_vo:
            n_violations += 1
            total_delta += delta_v
            if np.linalg.norm(n) > 1e-9:
                constraints_n.append(n)
                constraints_d.append(d)

    if n_violations == 0:
        return robot_vel_body.copy(), False, 0

    # Try LP projection (NavRL Eq. 13)
    if len(constraints_n) > 0 and not emergency:
        safe_vel = _vo_lp_projection(
            robot_vel_body, constraints_n, constraints_d, max_velocity
        )
        if safe_vel is not None:
            return safe_vel, True, n_violations

    # Geometric fallback
    safe_vel = robot_vel_body + total_delta
    v_norm = np.linalg.norm(safe_vel)
    if v_norm > max_velocity:
        safe_vel = safe_vel * (max_velocity / v_norm)
    elif v_norm < 0.5 and not emergency:
        orig_speed = np.linalg.norm(robot_vel_body)
        if v_norm > 1e-9 and orig_speed > 0.5:
            safe_vel = safe_vel / v_norm * max(orig_speed * 0.4, 0.5)

    return safe_vel, True, n_violations


def _vo_lp_projection(v_rl, normals, offsets, v_max):
    """NavRL Eq. 13: project policy velocity to nearest safe velocity."""
    n_constraints = len(normals)
    if n_constraints == 0:
        return None

    c = np.zeros(9)
    c[3:6] = 1.0
    c[6:9] = 1.0

    A_eq = np.zeros((3, 9))
    A_eq[0:3, 0:3] = np.eye(3)
    A_eq[0:3, 3:6] = -np.eye(3)
    A_eq[0:3, 6:9] = np.eye(3)
    b_eq = v_rl.copy()

    A_ub = np.zeros((n_constraints, 9))
    b_ub = np.zeros(n_constraints)
    for i in range(n_constraints):
        A_ub[i, 0:3] = -np.asarray(normals[i])
        b_ub[i] = -offsets[i]

    bounds = (
        [(-v_max, v_max)] * 3 +
        [(0, None)] * 3 +
        [(0, None)] * 3
    )

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')
        if result.success:
            v_safe = result.x[0:3]
            v_safe_norm = np.linalg.norm(v_safe)
            if v_safe_norm > v_max:
                v_safe = v_safe * (v_max / v_safe_norm)
            return v_safe
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Track C — Distance-based speed modulation
# ─────────────────────────────────────────────────────────────────────────────

def _apply_speed_modulation(velocity, min_d_ratio, max_velocity):
    current_speed = np.linalg.norm(velocity)
    if current_speed < 1e-9:
        return velocity

    if min_d_ratio < _EMERGENCY_DIST:
        if current_speed > _EMERGENCY_SPEED:
            return velocity / current_speed * _EMERGENCY_SPEED
        return velocity

    if min_d_ratio < _SPEED_MOD_START:
        # LINEAR ramp instead of quadratic (faster onset)
        proximity_factor = min_d_ratio / _SPEED_MOD_START
        speed_scale = _SPEED_MOD_FLOOR + (1.0 - _SPEED_MOD_FLOOR) * proximity_factor

        max_allowed = max_velocity * speed_scale
        if current_speed > max_allowed:
            return velocity / current_speed * max_allowed

    return velocity


# ─────────────────────────────────────────────────────────────────────────────
#  RRT* planner — with enhanced safest-direction search (Track B)
# ─────────────────────────────────────────────────────────────────────────────

class _RRTNode:
    __slots__ = ("pos", "parent", "cost")

    def __init__(self, pos, parent=None, cost=0.0):
        self.pos    = np.asarray(pos, dtype=np.float64)
        self.parent = parent
        self.cost   = cost


def _rrt_star_plan(start, goal_dir, obstacle_coords, obstacle_probs,
                   horizon=_PLAN_HORIZON,
                   max_iter=_RRT_MAX_ITER,
                   step=_RRT_STEP,
                   rewire_r=_RRT_REWIRE_R,
                   inflate_r=_OBS_INFLATE_R,
                   prob_thresh=_OBS_PROB_THRESH,
                   goal_bias=_RRT_GOAL_BIAS):
    """
    RRT* planning pass.  Returns (velocity_dir, path_found, clearance).

    Track B enhancement: When no goal path is found, the safest-direction
    search uses clearance² × progress scoring with a minimum clearance
    threshold, rejecting directions that merely graze obstacles.
    """
    if len(obstacle_coords) > 0 and len(obstacle_probs) > 0:
        mask = obstacle_probs >= prob_thresh
        obs  = obstacle_coords[mask]
    else:
        obs = np.zeros((0, 3), dtype=np.float64)

    goal = start + goal_dir * horizon

    def _segment_clear(a, b):
        if len(obs) == 0:
            return True
        ab     = b - a
        ab_len = np.linalg.norm(ab)
        if ab_len < 1e-9:
            return True
        ab_u    = ab / ab_len
        t       = np.clip(np.dot(obs - a, ab_u), 0.0, ab_len)
        closest = a + np.outer(t, ab_u)
        dists   = np.linalg.norm(obs - closest, axis=1)
        return bool(np.all(dists > inflate_r))

    def _min_clearance_at(pos):
        if len(obs) == 0:
            return np.inf
        return float(np.min(np.linalg.norm(obs - pos, axis=1)))

    nodes = [_RRTNode(start, parent=None, cost=0.0)]
    best_goal_node = None
    best_goal_cost = np.inf

    for _ in range(max_iter):
        if random.random() < goal_bias:
            q_rand = goal.copy()
        else:
            v  = np.random.randn(3)
            v /= np.linalg.norm(v) + 1e-12
            r  = horizon * (random.random() ** (1.0 / 3.0))
            q_rand = start + v * r

        dists_sq = np.array([np.dot(q_rand - n.pos, q_rand - n.pos) for n in nodes])
        nn_idx   = int(np.argmin(dists_sq))
        nn       = nodes[nn_idx]

        diff = q_rand - nn.pos
        d    = np.linalg.norm(diff)
        if d < 1e-9:
            continue
        q_new_pos = nn.pos + diff / d * min(d, step)

        if not _segment_clear(nn.pos, q_new_pos):
            continue

        near_indices = [
            i for i, n in enumerate(nodes)
            if np.linalg.norm(n.pos - q_new_pos) <= rewire_r
        ]
        if not near_indices:
            near_indices = [nn_idx]

        best_parent_idx  = nn_idx
        best_parent_cost = nn.cost + np.linalg.norm(q_new_pos - nn.pos)
        for i in near_indices:
            c = nodes[i].cost + np.linalg.norm(q_new_pos - nodes[i].pos)
            if c < best_parent_cost and _segment_clear(nodes[i].pos, q_new_pos):
                best_parent_idx  = i
                best_parent_cost = c

        q_new   = _RRTNode(q_new_pos, parent=best_parent_idx, cost=best_parent_cost)
        new_idx = len(nodes)
        nodes.append(q_new)

        for i in near_indices:
            new_cost = q_new.cost + np.linalg.norm(nodes[i].pos - q_new.pos)
            if new_cost < nodes[i].cost and _segment_clear(q_new.pos, nodes[i].pos):
                nodes[i].parent = new_idx
                nodes[i].cost   = new_cost

        d_goal = np.linalg.norm(q_new.pos - goal)
        if d_goal < step and (q_new.cost + d_goal) < best_goal_cost:
            if _segment_clear(q_new.pos, goal):
                best_goal_cost = q_new.cost + d_goal
                best_goal_node = new_idx

    # ── Goal reached ──
    if best_goal_node is not None:
        idx = best_goal_node
        while nodes[idx].parent not in (None, 0):
            idx = nodes[idx].parent
        first_edge = nodes[idx].pos - start
        fe_norm = np.linalg.norm(first_edge)
        if fe_norm > 1e-9:
            return first_edge / fe_norm, True, _min_clearance_at(nodes[idx].pos)

    # ── Track B: Enhanced safest-direction search ──
    # Score = clearance² × (1 + progress), with minimum clearance threshold.
    # clearance² heavily penalises directions that merely graze obstacles.
    # The +1 in (1+progress) ensures score > 0 even for negative progress
    # (retreating), so a safe retreat beats an unsafe advance.
    if len(nodes) <= 1:
        return goal_dir.copy(), False, 0.0

    min_acceptable_clearance = inflate_r * 1.5  # reject nodes too close to obstacles

    best_score = -np.inf
    best_idx   = 0
    for i in range(1, len(nodes)):
        node_dist = np.linalg.norm(nodes[i].pos - start)
        if node_dist < 0.1:
            continue

        clearance = _min_clearance_at(nodes[i].pos)

        # Reject directions that graze obstacles
        if clearance < min_acceptable_clearance:
            continue

        # Normalised progress toward goal ∈ [-1, 1]
        progress = np.dot(nodes[i].pos - start, goal_dir) / (node_dist + 1e-9)

        # Score: clearance² × (1 + progress) × distance_from_start
        # - clearance² : heavily favours wide-open directions
        # - (1+progress): range [0,2], still prefers goal direction when safe
        # - node_dist   : prefer nodes further from current position (more committed)
        score = clearance * clearance * (1.0 + progress) * min(node_dist, step * 3)

        if score > best_score:
            best_score = score
            best_idx   = i

    # If all candidates were rejected (very dense obstacles), pick max clearance
    if best_score <= 0:
        clearances = [_min_clearance_at(nodes[i].pos) for i in range(1, len(nodes))]
        if clearances:
            best_idx = int(np.argmax(clearances)) + 1

    # Trace back to first edge
    idx = best_idx
    while nodes[idx].parent not in (None, 0):
        idx = nodes[idx].parent
    first_edge = nodes[idx].pos - start
    fe_norm = np.linalg.norm(first_edge)
    if fe_norm > 1e-9:
        return first_edge / fe_norm, False, _min_clearance_at(nodes[idx].pos)

    return goal_dir.copy(), False, 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Reward helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def v_clip_reward_calc(stat_deque, octomap_stat, Seqspace_directions, dt=0.1, mode='hard'):
    if len(stat_deque) < 4:
        return 0.0
    if mode == 'hard':
        history_len = min(10, len(stat_deque))
        stats = list(stat_deque)[-history_len:]
        velocities = []
        for stat in stats:
            if "velocity" in stat:
                velocities.append(stat["velocity"])
        velocities = np.array(velocities)
        if len(velocities) < 4:
            return 0.0
        v_magnitudes = np.linalg.norm(velocities, axis=1)
        accels = np.gradient(v_magnitudes, dt)
        jerks  = np.gradient(accels, dt) if len(accels) > 1 else np.zeros_like(accels)
        if len(velocities) >= 3:
            v_norms    = np.linalg.norm(velocities, axis=1)
            valid_mask = v_norms > 0.1
            if np.sum(valid_mask) >= 2:
                valid_velocities = velocities[valid_mask]
                valid_v_norms    = v_norms[valid_mask]
                directions       = valid_velocities / valid_v_norms[:, np.newaxis]
                if len(directions) >= 2:
                    crosses            = np.cross(directions[:-1], directions[1:])
                    sin_theta          = np.linalg.norm(crosses, axis=1)
                    angular_velocities = np.arcsin(np.clip(sin_theta, 0, 1)) / dt
                    angular_penalty    = np.mean(angular_velocities ** 2)
                else:
                    angular_penalty = 0.0
            else:
                angular_penalty = 0.0
        else:
            angular_penalty = 0.0
        current_speed = v_magnitudes[-1] if len(v_magnitudes) > 0 else 0.0
        max_accel = 4.0 + 6.0 * current_speed / 10.0
        max_jerk  = 8.0 + 4.0 * current_speed / 10.0
        accel_excess  = np.maximum(np.abs(accels) - max_accel, 0)
        accel_penalty = np.percentile(accel_excess, 75) if len(accel_excess) > 0 else 0.0
        jerk_excess  = np.maximum(np.abs(jerks) - max_jerk, 0)
        jerk_penalty = np.max(jerk_excess) if len(jerk_excess) > 0 else 0.0
        smoothness_score = (np.exp(-0.1 * accel_penalty) *
                            np.exp(-0.05 * jerk_penalty) *
                            np.exp(-0.05 * angular_penalty))
        return float(np.clip(smoothness_score - 0.5, -0.5, 0.5))
    elif mode == 'easy':
        history_len = min(10, len(stat_deque))
        stats = list(stat_deque)[-history_len:]
        velocities = []
        for stat in stats:
            if "velocity" in stat:
                velocities.append(stat["velocity"])
        current_speed = stat_deque[-1]["velocity"]
        velocities    = np.array(velocities)
        if len(velocities) < 2:
            return 0.0
        a_xyz     = np.gradient(velocities, dt, axis=0)
        a_xyz_std = np.std(a_xyz, axis=0)
        v_norm        = np.linalg.norm(current_speed) / 10.0
        v_norm_reward = np.exp(-(v_norm - 0.2) * (v_norm - 0.8) / 0.09 - 1) - 1
        accel_reward  = np.clip(0.5 - 0.05 * np.sum(a_xyz_std), -0.5, 0.5)
        return accel_reward + v_norm_reward
    else:
        return 0.0


def Avoid_Reward_Optimized_BodyFrame(octomap_body, q, prev_min_dist=None, D=0.3334):
    octomap_matrix = octomap_body.reshape(18, 36)
    mask_2D = octomap_matrix <= 2 * D
    if not np.any(mask_2D):
        min_dist = np.min(octomap_matrix)
        R_safe   = (1 - np.exp(-(min_dist - 2 * D) / D))
        return R_safe, 1.0, 0.0
    H, W = octomap_matrix.shape
    row_idx, col_idx = np.where(mask_2D)
    azimuth   = (col_idx / (W - 1) - 0.5) * 2 * np.pi
    elevation = (row_idx / (H - 1) - 0.5) * np.pi
    azimuth_abs = np.abs(azimuth)
    h_counts    = np.zeros(4, dtype=np.int32)
    h_counts[0] = np.sum(azimuth_abs <= np.pi / 6)
    h_counts[1] = np.sum((azimuth_abs > np.pi / 6)  & (azimuth_abs <= np.pi / 3))
    h_counts[2] = np.sum((azimuth_abs > np.pi / 3)  & (azimuth_abs <= np.pi / 2))
    h_counts[3] = np.sum(azimuth_abs > np.pi / 2)
    elevation_abs = np.abs(elevation)
    v_counts      = np.zeros(3, dtype=np.int32)
    v_counts[0]   = np.sum(elevation_abs <= np.pi / 6)
    v_counts[1]   = np.sum((elevation_abs > np.pi / 6) & (elevation_abs <= np.pi / 3))
    v_counts[2]   = np.sum(elevation_abs > np.pi / 3)
    h_weights = np.array([1.75, 1.5, 1.25, 1.0], dtype=np.float32)
    v_weights = np.array([1.25, 1.0, 0.9],        dtype=np.float32)
    w_h = np.sum(h_counts * h_weights) / np.sum(h_counts) if np.sum(h_counts) > 0 else 1.0
    w_v = np.sum(v_counts * v_weights) / np.sum(v_counts) if np.sum(v_counts) > 0 else 1.0
    W_dir = w_h * w_v
    octomap_in_2D         = octomap_matrix[mask_2D]
    octomap_in_2D_clipped = np.clip(octomap_in_2D, 0, 2 * D)
    exp_distances         = 1 / np.e - np.exp(1 - octomap_in_2D_clipped / D)
    avg_exp_distance      = np.mean(exp_distances)
    R_distribution        = avg_exp_distance * W_dir
    current_min_d = np.min(octomap_in_2D)
    R_change, delta_d = 0.0, 0.0
    if current_min_d < 2 * D and prev_min_dist is not None:
        delta_d = current_min_d - np.clip(prev_min_dist, 0, 2 * D)
        if delta_d < 0:
            R_change = max(1 * delta_d / D, -1)
        else:
            R_change = min(0.5 * delta_d / D, 0.5)
    R_trend = R_distribution + R_change
    R_trend = np.clip(R_trend, -4, 0.5)
    return float(R_trend), float(W_dir), float(delta_d)


def softmax(x, axis=None):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x   = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def get_real_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        print("错误代码: 01", "无法获取IP地址，请检查网络连接")
        return '127.0.0.1'


# ─────────────────────────────────────────────────────────────────────────────
#  Main environment
# ─────────────────────────────────────────────────────────────────────────────

class AirSimDroneEnv(gym.Env):
    def __init__(self, uavname='UAV01', datamanager_ip='', datamanager_port=5812,
                 trainmanager_ip='', start_point=np.array([0, 0, -0.5], dtype=np.float64),
                 safe_rule=False):
        self.vehicle = VM.VehicleThread(uavname, stat_range=20, start_point=start_point,
                                        max_task_steps=600)
        octomap_shape = (36, 18)
        stat_shape    = (8,)
        self.safe_rule = safe_rule

        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(36 * 18 + 8,), dtype=np.float64)
        self.action_space = spaces.Box(low=-3.75, high=3.75,
                                       shape=(3,), dtype=np.float64)
        self.octomap_shape   = octomap_shape
        self.stat_shape      = stat_shape
        self.observation_deque         = deque(maxlen=10)
        self.obstacle_deque            = deque(maxlen=10)
        self.last_success_times        = deque(maxlen=300)
        self.last_success_episode_time = deque(maxlen=300)
        self.target_closer    = 1.0
        self.uavname          = uavname
        self.max_velocity     = 7.5
        self.k_avoid          = 0.0
        self.uav_id           = int(uavname[3:]) - 1
        self.datamanager_ip   = datamanager_ip  if datamanager_ip  else get_real_ip()
        self.datamanager_port = datamanager_port
        self.trainmanager_ip  = trainmanager_ip if trainmanager_ip else get_real_ip()

        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.bind((trainmanager_ip, 2000 + self.uav_id))
        self.udp_deque          = deque(maxlen=25)
        self.training_progress  = 1.0
        self.pause_enable       = True
        self.collision_occurred = False
        self.collision_occurred_reset = False
        self.target_reached     = False
        self.outtime            = False
        self.episode_steps      = 0
        self.episode_length     = self.vehicle.max_task_steps
        self.total_steps        = 0
        self.episode_counter    = 0
        self.episode_start_time = time.time()
        self.pause_time         = self.episode_start_time
        self.lastupdatetime     = time.time()
        self.min_d              = None
        self.delta_d            = 0.0
        self.last_distance      = 1

        # VO safety shield state
        self._vo_correction_count = 0
        self._current_yaw_rad     = 0.0

        # RRT* background thread
        self._rrt_lock        = threading.RLock()
        self._rrt_result      = {
            "velocity_dir":  None,
            "path_found":    False,
            "timestamp":     0.0,
            "min_d_at_plan": np.inf,
            "clearance":     0.0,
        }
        self._rrt_thread_stop = threading.Event()
        self._rrt_thread      = None

        if self.safe_rule:
            self._rrt_thread = threading.Thread(
                target=self._rrt_worker_loop,
                daemon=True,
                name=f"rrt_star_{self.uavname}"
            )
            self._rrt_thread.start()

        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()

    def set_training_progress(self, progress: float) -> None:
        self.training_progress = progress

    # =========================================================================
    #  RRT* background thread
    # =========================================================================

    def _rrt_worker_loop(self):
        period = 1.0 / _RRT_HZ
        while not self._rrt_thread_stop.is_set():
            t0 = time.time()
            try:
                self._rrt_plan_once()
            except Exception as e:
                if self.uavname == 'UAV01':
                    print(f"[RRT* worker] exception: {e}")
            elapsed = time.time() - t0
            time.sleep(max(0.0, period - elapsed))

    def _rrt_plan_once(self):
        """One RRT* planning cycle. Direction returned in GLOBAL frame."""
        coords_arr, prob_arr = self.vehicle.windowstat

        if len(self.vehicle.stat_deque) == 0:
            return

        latest_stat   = self.vehicle.stat_deque[-1]
        uav_pos_world = latest_stat["position"] + self.vehicle.pos

        target_vec  = self.vehicle.target - uav_pos_world
        target_dist = np.linalg.norm(target_vec)
        if target_dist < 1e-9:
            return
        goal_dir_world = target_vec / target_dist

        if (isinstance(coords_arr, np.ndarray) and
                coords_arr.ndim == 2 and coords_arr.shape[0] > 0):
            obs_local = coords_arr - uav_pos_world
            prob_arr_ = np.asarray(prob_arr, dtype=np.float64)
        else:
            obs_local = np.zeros((0, 3), dtype=np.float64)
            prob_arr_ = np.zeros((0,),   dtype=np.float64)

        current_min_d = self.min_d if self.min_d is not None else np.inf

        taper_start     = 30.0
        arrival_r       = 10.0
        if target_dist >= taper_start:
            dynamic_horizon = _PLAN_HORIZON
        else:
            scale           = max(0.0, (target_dist - arrival_r) / (taper_start - arrival_r))
            dynamic_horizon = _PLAN_HORIZON * scale

        if dynamic_horizon < 0.1:
            return

        start_local = np.zeros(3, dtype=np.float64)
        vel_dir, path_found, clearance = _rrt_star_plan(
            start           = start_local,
            goal_dir        = goal_dir_world,
            obstacle_coords = obs_local,
            obstacle_probs  = prob_arr_,
            horizon         = dynamic_horizon,
        )

        with self._rrt_lock:
            self._rrt_result["velocity_dir"]  = vel_dir
            self._rrt_result["path_found"]    = path_found
            self._rrt_result["timestamp"]     = time.time()
            self._rrt_result["min_d_at_plan"] = current_min_d
            self._rrt_result["clearance"]     = clearance

        if self.uavname == 'UAV01':
            print(
                f"[RRT*] min_d={current_min_d:.4f}  "
                f"path_found={path_found}  clearance={clearance:.2f}m  "
                f"horizon={dynamic_horizon:.2f}m  "
                f"dir={np.round(vel_dir, 3)}"
            )

    # =========================================================================
    #  Safe ruler — VO + RRT* + speed modulation (all three tracks)
    # =========================================================================

    def safe_ruler(self, min_d, delta_d, target_speed_global, octomap_body=None, enable=False):
        """
        Four-layer safety shield:
 
        Layer 0 (NEW):   Proximity repulsion — pushes away from lateral obstacles
        Layer 1 (Track A): Adaptive τ for VO
        Layer 2 (VO + RRT*): Direction correction in body frame
        Layer 3 (Track C): Speed modulation based on min_d
        + Distribution-shift guard + Target proximity gate
        """
        if not enable:
            v_norm = np.linalg.norm(target_speed_global)
            if v_norm > self.max_velocity:
                return target_speed_global * (self.max_velocity / v_norm)
            return target_speed_global
 
        if min_d is None:
            min_d = np.inf
 
        drl_speed = np.linalg.norm(target_speed_global)
        if drl_speed < 1e-9:
            return target_speed_global
 
        # ── Layer 1 (Track A): Adaptive τ ──
        stat_range = self.vehicle.stat_range if hasattr(self.vehicle, 'stat_range') else 20.0
        tau_adaptive = _compute_adaptive_tau(
            current_speed=drl_speed,
            min_d_ratio=min_d,
            max_velocity=self.max_velocity,
            stat_range=stat_range
        )
 
        # ── Transform DRL velocity to body frame ──
        yaw_rad = self._current_yaw_rad
        drl_vel_body = _rotate_global_to_body(target_speed_global, yaw_rad)
 
        # ── Layer 0 (NEW): Proximity repulsion in body frame ──
        # This fires BEFORE VO, pushing the velocity away from nearby lateral
        # obstacles that VO would miss (velocity parallel to obstacle surface).
        if octomap_body is not None:
            drl_vel_body, repulsion_applied = _proximity_repulsion(
                drl_vel_body, octomap_body,
                stat_range=stat_range,
                danger_threshold=_REPULSION_THRESHOLD,
                force_gain=_REPULSION_GAIN,
                max_velocity=self.max_velocity,
            )
            if repulsion_applied and self.uavname == 'UAV01' and self.total_steps % 5 == 0:
                print(
                    f"[Repulsion] min_d={min_d:.4f}  "
                    f"speed: {drl_speed:.2f}→{np.linalg.norm(drl_vel_body):.2f}"
                )
        else:
            repulsion_applied = False
 
        # ── Extract obstacles from octomap (body frame) ──
        obstacle_list = []
        if octomap_body is not None:
            obstacle_list = _extract_obstacles_from_octomap(
                octomap_body, stat_range=stat_range, activation_dist=_VO_ACTIVATION_D
            )
 
        # ── Layer 2: VO safety shield (body frame) with adaptive τ ──
        vo_safe_body, was_corrected, n_violations = _vo_safety_shield(
            robot_vel_body = drl_vel_body,
            obstacle_list  = obstacle_list,
            max_velocity   = self.max_velocity,
            robot_r        = _VO_ROBOT_RADIUS,
            safe_margin    = _VO_SAFE_MARGIN,
            tau            = tau_adaptive,
        )
 
        if was_corrected or repulsion_applied:
            if was_corrected:
                self._vo_correction_count += 1
                result_body = vo_safe_body
            else:
                # Repulsion fired but VO didn't — use the repulsion-adjusted velocity
                result_body = drl_vel_body
 
            # If VO is too conservative, blend with RRT* safe direction
            result_speed = np.linalg.norm(result_body)
            if result_speed < drl_speed * 0.3 and result_speed < 1.5:
                with self._rrt_lock:
                    rrt_dir_global = self._rrt_result.get("velocity_dir", None)
                    path_found     = self._rrt_result.get("path_found", False)
                    plan_age       = time.time() - self._rrt_result.get("timestamp", 0.0)
                    rrt_clearance  = self._rrt_result.get("clearance", 0.0)
 
                if rrt_dir_global is not None and plan_age < _PLAN_AGE_MAX:
                    rrt_dir_body = _rotate_global_to_body(rrt_dir_global, yaw_rad)
                    rrt_speed = drl_speed * _SPEED_FLOOR
 
                    blend_w = 0.6 if path_found else 0.4
 
                    blended_body = (1.0 - blend_w) * result_body + blend_w * rrt_dir_body * rrt_speed
                    bl_norm = np.linalg.norm(blended_body)
                    if bl_norm > self.max_velocity:
                        blended_body = blended_body * (self.max_velocity / bl_norm)
                    elif bl_norm > 1e-9:
                        min_speed = max(drl_speed * _SPEED_FLOOR, 1.0)
                        if bl_norm < min_speed:
                            blended_body = blended_body / bl_norm * min_speed
 
                    result_body = blended_body
 
                    if self.uavname == 'UAV01' and self.total_steps % 5 == 0:
                        print(
                            f"[VO+RRT*] blended  "
                            f"speed: {drl_speed:.2f}→{np.linalg.norm(result_body):.2f}  "
                            f"rrt_clearance={rrt_clearance:.2f}m  rrt_found={path_found}"
                        )
 
            # Transform back to global frame
            safe_vel_global = _rotate_body_to_global(result_body, yaw_rad)
 
            # Distribution-shift guard
            drl_dir = target_speed_global / drl_speed
            safe_dir = safe_vel_global / (np.linalg.norm(safe_vel_global) + 1e-9)
            cos_sim = float(np.clip(np.dot(drl_dir, safe_dir), -1.0, 1.0))
 
            if cos_sim < 0.0:
                correction = safe_vel_global - target_speed_global
                safe_vel_global = target_speed_global + _ALPHA_MAX * correction
                v_norm = np.linalg.norm(safe_vel_global)
                if v_norm > self.max_velocity:
                    safe_vel_global = safe_vel_global * (self.max_velocity / v_norm)
 
            safe_dir_final = safe_vel_global / (np.linalg.norm(safe_vel_global) + 1e-9)
            cos_sim_final = float(np.clip(np.dot(drl_dir, safe_dir_final), -1.0, 1.0))
 
            if self.uavname == 'UAV01' and self.total_steps % 5 == 0:
                angle_deg = float(np.degrees(np.arccos(np.clip(cos_sim_final, -1.0, 1.0))))
                print(
                    f"[SafeShield] min_d={min_d:.4f}  τ={tau_adaptive:.2f}  "
                    f"n_obs={len(obstacle_list)}  n_vio={n_violations}  "
                    f"repulsion={'Y' if repulsion_applied else 'N'}  "
                    f"corr={angle_deg:.1f}°  "
                    f"speed: {drl_speed:.2f}→{np.linalg.norm(safe_vel_global):.2f}  "
                    f"total_vo={self._vo_correction_count}"
                )
 
            # Target proximity gate
            d_to_target = self.vehicle.current_target_d_norm
            if d_to_target < 30.0:
                proximity_scale = max(0.0, (d_to_target - 10.0) / 20.0)
                safe_vel_global = (1.0 - proximity_scale) * target_speed_global + proximity_scale * safe_vel_global
 
            # ── Layer 3 (Track C): Speed modulation ──
            safe_vel_global = _apply_speed_modulation(safe_vel_global, min_d, self.max_velocity)
 
            return safe_vel_global
 
        # No correction at all — still apply speed modulation (Track C)
        result = target_speed_global.copy()
        v_norm = np.linalg.norm(result)
        if v_norm > self.max_velocity:
            result = result * (self.max_velocity / v_norm)
 
        result = _apply_speed_modulation(result, min_d, self.max_velocity)
 
        return result

    # =========================================================================
    #  Statistics helpers
    # =========================================================================

    def success_rate_cal(self):
        if len(self.last_success_times) == 0:
            return 0.0
        return sum(self.last_success_times), len(self.last_success_times)

    def success_episode_time_hook(self):
        if len(self.last_success_episode_time) == 0:
            return 0.0
        return self.last_success_episode_time, len(self.last_success_episode_time)

    # =========================================================================
    #  UDP receive thread
    # =========================================================================

    def _recv_loop(self):
        buffers = {}
        while True:
            packet, _ = self.udp_sock.recvfrom(MAX_PACKET_SIZE + 16)
            pkt_id, total_pkts = struct.unpack('!II', packet[:8])
            seq   = struct.unpack('!I', packet[8:12])[0]
            chunk = packet[12:]
            if pkt_id not in buffers:
                buffers[pkt_id] = {seq: chunk}
            else:
                buffers[pkt_id][seq] = chunk
            if len(buffers[pkt_id]) == total_pkts:
                full_data = b''.join([buffers[pkt_id][i] for i in sorted(buffers[pkt_id])])
                new_stat  = pickle.loads(full_data)
                del buffers[pkt_id]
                if new_stat['timestamp'] > self.lastupdatetime:
                    self.observation_deque.append(new_stat)
                    self.collision_occurred = new_stat['collision']

    def send_command(self, number, command, msg):
        try:
            data        = pickle.dumps([command, msg])
            total_size  = len(data)
            num_packets = (total_size + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
            header      = struct.pack('!II', number, num_packets)
            for seq in range(num_packets):
                start = seq * MAX_PACKET_SIZE
                chunk = data[start:start + MAX_PACKET_SIZE]
                self.udp_sock.sendto(
                    header + struct.pack('!I', seq) + chunk,
                    (self.datamanager_ip, self.datamanager_port + 1)
                )
                time.sleep(0.01)
        except Exception as e:
            print(f"发送指令失败: {e}")

    # =========================================================================
    #  Gym interface
    # =========================================================================

    def reset(self, seed=None):
        self.outtime               = False
        self.min_d                 = None
        self.k_avoid               = 0.0
        self.pause_enable          = True
        self.target_closer         = 1.0
        self.collision_occurred    = False
        self.delta_d               = 0.0
        self.collision_occurred_reset = False
        self.target_reached        = False
        self.episode_steps         = 0
        self.episode_counter      += 1
        self.episode_start_time    = time.time()
        self.pause_time            = self.episode_start_time
        self.obstacle_deque.clear()
        self._vo_correction_count  = 0
        self._current_yaw_rad      = 0.0
        try:
            self.send_command(
                self.episode_counter % 25 + int(self.uavname[3:]) * 1000,
                "reset", [self.uavname]
            )
        except Exception as e:
            print(f"发送reset命令失败: {e}")
        time.sleep(0.2)
        self.observation_deque.clear()
        self.vehicle.start_time = time.time()
        self._reset_settings()
        self.vehicle.environment_reset(modesetting='easy')
        obs = self._get_obs()
        self.last_distance = 1
        info = {
            'timeout':            self.outtime,
            'collision_occurred': self.collision_occurred,
            'target_reached':     self.target_reached,
        }
        return obs, info

    def _reset_settings(self):
        while len(self.observation_deque) == 0:
            time.sleep(0.01)
        init_obs = self.observation_deque.popleft()
        position, velocity, orientation, eular_angle, lidar_data, collision = (
            init_obs['position'], init_obs['velocity'], init_obs['orientation'],
            init_obs['eular_angle'], init_obs['Lidar'], init_obs['collision']
        )
        self.vehicle.stat_deque.append({
            "position":     position,
            "orientation":  orientation,
            "eular_angles": eular_angle,
            "velocity":     velocity,
        })

    def _get_obs(self):
        while len(self.observation_deque) == 0 or not self.vehicle.running:
            time.sleep(0.01)
        try:
            obs = self.observation_deque.popleft()
            self.lastupdatetime = obs['timestamp']
            position, velocity, orientation, eular_angle, lidar_data, collision = (
                obs['position'], obs['velocity'], obs['orientation'],
                obs['eular_angle'], obs['Lidar'], obs['collision']
            )
            stat, octomap = self.vehicle._Lidar_RLstat_update(
                position, orientation, eular_angle, velocity, lidar_data
            )
            self._current_yaw_rad = _yaw_from_velocity(velocity)
            return np.concatenate([octomap.reshape(-1), stat])
        except Exception as e:
            print(f'_get_obs error: {e}')
            octomap = np.ones(self.octomap_shape).flatten()
            stat    = np.zeros(self.stat_shape)
            return np.concatenate([octomap, stat]).astype(np.float64)

    def _execute_action(self, action_original, octomap_stat, override_min_d=None):
        """Execute DRL action with safety shield. Lever 2 preserved."""
        action   = action_original.copy()
        speed    = self.vehicle.stat_deque[-1]["velocity"]

        if self.uavname == 'UAV01' and self.total_steps % 5 == 0:
            print(f'{self.uavname}动作输出: {action}')

        avoid_speed      = np.zeros(3, dtype=np.float64)
        avoid_speed[0:3] = action[0:3]
        action_global    = avoid_speed.copy()
        yaw              = self.vehicle.expect_yaw * 180

        # Lever 2: DRL-intended velocity (before correction)
        drl_intended = action_global + speed
        drl_v_norm   = np.linalg.norm(drl_intended)
        if drl_v_norm > self.max_velocity:
            drl_intended = drl_intended / drl_v_norm * self.max_velocity

        if len(self.vehicle.stat_deque) > 0:
            self.vehicle.stat_deque[-1]["velocity"] = drl_intended.copy()

        # Apply three-layer safety shield
        target_speed = drl_intended.copy()
        if self.safe_rule:
            min_d_for_ruler = override_min_d if override_min_d is not None else self.min_d
            target_speed = self.safe_ruler(
                min_d_for_ruler, self.delta_d, target_speed,
                octomap_body=octomap_stat, enable=True
            )

        ctrl_speed = target_speed
        try:
            msg = [self.uavname,
                   np.array([ctrl_speed[0], ctrl_speed[1], ctrl_speed[2], yaw],
                             dtype=np.float64)]
            self.send_command(
                self.total_steps % 5 + int(self.uavname[3:]) * 10, "action", msg
            )
        except Exception as e:
            print(f"发送动作命令失败: {e}")

    def _compute_reward(self, octomap_stat, stat_vector):
        reward, sparse_reward = 0.0, 0
        current_time   = time.time()
        survive_reward = -0.1 - 0.1 * self.training_progress

        r_obstacle, W_dir, delta_d = Avoid_Reward_Optimized_BodyFrame(
            octomap_stat,
            self.vehicle.stat_deque[-1]["orientation"],
            prev_min_dist=self.min_d,
            D=0.2
        )
        self.min_d   = np.min(octomap_stat)
        self.delta_d = delta_d
        self.k_avoid = (np.exp(-self.min_d * 3) *
                        np.clip(np.exp(W_dir - 1.5), 0, 1))

        if self.uavname == 'UAV01' and self.total_steps % 5 == 0:
            print(f'{self.uavname}避障奖励: r_obstacle:{r_obstacle:.3f}, '
                  f'W_dir:{W_dir:.3f}, k_avoid:{self.k_avoid:.3f}')

        if self.training_progress > 0.25:
            v_clip_reward = v_clip_reward_calc(
                self.vehicle.stat_deque, octomap_stat,
                self.vehicle.Seqspace_directions, 0.1, 'hard'
            )
        else:
            v_clip_reward = 0.0

        target_dir, v_dir, v_norm = stat_vector[0:3], stat_vector[4:7], stat_vector[7]
        delta_l        = np.dot(target_dir, v_dir) * v_norm * self.max_velocity
        forward_reward = np.clip(
            delta_l / self.vehicle.max_distance * 50,
            -1 - self.training_progress, 1 + self.training_progress
        )

        distance_reward    = 0.0
        normalize_distance = (self.vehicle.current_target_d_norm /
                               self.vehicle.max_distance)
        if (normalize_distance > self.last_distance) and (self.k_avoid < 0.25):
            distance_reward -= 1.0
        if normalize_distance < 1 and normalize_distance < self.target_closer - 0.01:
            distance_reward    = 1.0
            self.target_closer = normalize_distance
        self.last_distance = normalize_distance

        dense_reward = (
            1.0 * r_obstacle +
            (self.training_progress - 0.25) / 0.75 * v_clip_reward +
            (1.0 - self.k_avoid) * forward_reward +
            1.0 * distance_reward +
            1.0 * survive_reward
        )
        reward += dense_reward

        d_norm = self.vehicle.current_target_d_norm
        if self.uavname == 'UAV01' and self.total_steps % 5 == 0:
            print(f'{self.uavname}当前目标点距离: {d_norm:.3f} m')

        if d_norm < 5:
            print(f'{self.uavname}到达目标点!')
            sparse_reward      += 50
            self.target_reached = True
            reward += sparse_reward
            return reward

        if self.collision_occurred and current_time - self.episode_start_time > 3.0:
            reward -= 80
            self.collision_occurred_reset = True
            return reward

        if self.episode_steps >= self.episode_length:
            reward -= 80
            self.outtime = True
            return reward

        return reward

    def _check_done(self):
        truncated, terminated = False, False
        if self.target_reached:
            terminated = True
            self.last_success_times.append(1)
            self.last_success_episode_time.append(time.time() - self.episode_start_time)
            return terminated, truncated
        if self.outtime:
            truncated = True
            self.last_success_times.append(0)
            return terminated, truncated
        if self.collision_occurred_reset:
            terminated = True
            self.last_success_times.append(0)
            return terminated, truncated
        return terminated, truncated

    def step(self, action):
        self.episode_steps += 1
        self.total_steps   += 1
        time_step_start     = time.time()

        obs          = self._get_obs()
        stat_vector  = obs[-8:]
        octomap_stat = obs[:36 * 18]

        current_min_d = float(np.min(octomap_stat))

        self._execute_action(action, octomap_stat, override_min_d=current_min_d)

        reward = self._compute_reward(octomap_stat, stat_vector)

        terminated, truncated = self._check_done()
        info = {
            'timeout':            self.outtime,
            'collision_occurred': self.collision_occurred,
            'target_reached':     self.target_reached,
            'vo_corrections':     self._vo_correction_count,
        }
        time_step_end = time.time()
        time.sleep(max(0.0, 0.05 - (time_step_end - time_step_start)))
        return obs, reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
#  Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def linear_schedule(initial_value: float, final_value: float = 0):
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return func


class LinearEntropyScheduler(BaseCallback):
    def __init__(self, start_ent_coef, end_ent_coef, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.start_ent_coef  = start_ent_coef
        self.end_ent_coef    = end_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress         = min(self.num_timesteps / self.total_timesteps, 1.0)
        current_ent_coef = (self.start_ent_coef -
                            (self.start_ent_coef - self.end_ent_coef) * progress)
        self.model.ent_coef = current_ent_coef
        self.logger.record("train/ent_coef", current_ent_coef)
        return True


class SuccessRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        rates = self.training_env.env_method('success_rate_cal')
        if len(rates) > 0:
            success_rate = np.mean(rates)
            self.logger.record("train/success_rate", float(success_rate))
        return True


class TrainProgresserCal(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        self.training_env.env_method('set_training_progress', progress)
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point — evaluation only (model loaded from file)
# ─────────────────────────────────────────────────────────────────────────────

def make_env(uavname):
    def _init():
        env = AirSimDroneEnv(
            uavname=uavname,
            datamanager_ip='172.21.236.121',
            trainmanager_ip='172.21.236.121',
            start_point=np.array([0, 0, -0.5], dtype=np.float64),
            safe_rule=False
        )
        env = Monitor(env)
        return env
    return _init


if __name__ == "__main__":
    print("正在初始化环境...")
    from collision_diagnostics import CollisionDiagnostics
    env = SubprocVecEnv([make_env(f"UAV{i+1:02d}") for i in range(1)])
    diag = CollisionDiagnostics(env, window=20)
    print("环境初始化完成。")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("当前工作目录:", os.getcwd())

    model_path = "../results/SFT_avoid_modelv3.zip"
    if os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        model = PPO.load(model_path, env=env)
        target_std  = 0.25
        new_log_std = np.log(target_std)
        model.policy.log_std.data = torch.tensor(
            [new_log_std] * 3,
            device=model.device,
            dtype=torch.float32
        )
    else:
        print(f"模型文件不存在: {model_path}")
        sys.exit(1)

    from stable_baselines3.common.evaluation import evaluate_policy
    print("开始评估...")
    mean_reward, std_reward = evaluate_policy(
        model, env,
        n_eval_episodes=525,
        deterministic=True,
        render=False,
    )

    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    success_msg       = env.env_method('success_rate_cal')
    success_time_used = env.env_method('success_episode_time_hook')

    success_times, total_times = 0.0, 0
    total_success_time         = np.array([])
    for i in range(len(success_msg)):
        success_times     += success_msg[i][0]
        total_times       += success_msg[i][1]
        total_success_time = np.concatenate((total_success_time,
                                              success_time_used[i][0]))

    print(f"Overall Success Rate: {success_times}/{total_times} "
          f"= {success_times / total_times:.2%}")
    print(f"Average Time for Successful Episodes: "
          f"{np.average(total_success_time):.4f} seconds")
    print(f"Std Time for Successful Episodes: "
          f"{np.std(total_success_time):.4f} seconds")
    diag.print_summary()
    diag.dump_report("failure_analysis.txt")