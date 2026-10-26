"""
Collision Diagnostic Logger
============================
Drop-in module that wraps AirSimDroneEnv.step() to record the trajectory
of key variables in the N steps before each failure (collision or timeout).

After evaluation, call `dump_report()` to get a structured analysis of
WHY each failure happened, categorised into failure modes.

Usage:
    from collision_diagnostics import CollisionDiagnostics

    # After creating env but before evaluation:
    diag = CollisionDiagnostics(env, window=20)

    # Run evaluation as normal...
    evaluate_policy(model, env, ...)

    # After evaluation:
    diag.dump_report("failure_analysis.txt")
    diag.print_summary()
"""

import numpy as np
from collections import deque
import time


class EpisodeRecorder:
    """Records per-step data for a single episode."""

    def __init__(self, window=20):
        self.window = window
        self.steps = deque(maxlen=window)
        self.episode_start = time.time()
        self.outcome = None  # 'collision', 'timeout', 'success'

    def record_step(self, step_data):
        self.steps.append(step_data)

    def finalise(self, outcome):
        self.outcome = outcome
        self.duration = time.time() - self.episode_start
        # Freeze the deque into a list for later analysis
        self.history = list(self.steps)


class CollisionDiagnostics:
    """
    Wraps the environment's step() to collect pre-failure trajectories.

    Analyses each failure and classifies it into one of these modes:
      A. Sudden appearance  — min_d drops from >0.15 to <0.05 in ≤2 steps
      B. Gradual approach   — min_d declines over 5+ steps, shield fires but fails
      C. High-speed impact  — speed > 5 m/s when min_d first drops below 0.10
      D. Dead-end trap      — UAV oscillates (direction reversal) for 5+ steps
      E. Timeout stall      — UAV stopped making progress for final 30%+ of episode
      F. VO overcorrection  — VO correction angle > 60° in 3+ consecutive steps
    """

    def __init__(self, vec_env, window=20):
        """
        Parameters
        ----------
        vec_env : SubprocVecEnv or similar
            The vectorised environment used for evaluation.
        window : int
            Number of steps to record before each failure.
        """
        self.vec_env = vec_env
        self.window = window
        self.n_envs = vec_env.num_envs

        # Per-env state
        self.recorders = [EpisodeRecorder(window) for _ in range(self.n_envs)]
        self.completed_episodes = []

        # Monkey-patch step to intercept data
        self._original_step = vec_env.step
        vec_env.step = self._instrumented_step

        # Track step count per env
        self._env_step_counts = [0] * self.n_envs

    def _instrumented_step(self, actions):
        """Wrapper around vec_env.step() that records diagnostics."""
        obs, rewards, dones, infos = self._original_step(actions)

        for i in range(self.n_envs):
            self._env_step_counts[i] += 1

            # Extract diagnostic data from observation
            octomap_stat = obs[i][:36 * 18]
            stat_vector = obs[i][-8:]

            min_d = float(np.min(octomap_stat))
            target_dir = stat_vector[0:3]
            v_dir = stat_vector[4:7]
            v_norm = float(stat_vector[7])

            # Compute derived metrics
            speed_ms = v_norm * 7.5  # max_velocity
            forward_component = float(np.dot(target_dir, v_dir)) * speed_ms
            # How many octomap cells are dangerously close
            n_close_cells = int(np.sum(octomap_stat[:648] < 0.10))
            n_medium_cells = int(np.sum(octomap_stat[:648] < 0.20))

            # Front-sector min_d (azimuth bins 13-22 ≈ ±50° forward)
            octo_matrix = octomap_stat[:648].reshape(18, 36)
            front_sector = octo_matrix[:, 13:23]  # ±50° from forward
            front_min_d = float(np.min(front_sector))

            step_data = {
                'step':              self._env_step_counts[i],
                'min_d':             min_d,
                'front_min_d':       front_min_d,
                'speed_ms':          speed_ms,
                'forward_component': forward_component,
                'n_close_cells':     n_close_cells,
                'n_medium_cells':    n_medium_cells,
                'reward':            float(rewards[i]),
                'action':            actions[i].copy() if hasattr(actions[i], 'copy') else np.array(actions[i]),
                'target_dir':        target_dir.copy(),
                'v_dir':             v_dir.copy(),
                'vo_corrections':    infos[i].get('vo_corrections', 0),
            }

            self.recorders[i].record_step(step_data)

            # Check if episode ended
            if dones[i]:
                if infos[i].get('target_reached', False):
                    outcome = 'success'
                elif infos[i].get('collision_occurred', False):
                    outcome = 'collision'
                elif infos[i].get('timeout', False):
                    outcome = 'timeout'
                else:
                    outcome = 'unknown'

                self.recorders[i].finalise(outcome)

                if outcome != 'success':
                    self.completed_episodes.append(self.recorders[i])

                # Reset recorder for next episode
                self.recorders[i] = EpisodeRecorder(self.window)
                self._env_step_counts[i] = 0

        return obs, rewards, dones, infos

    def _classify_failure(self, episode):
        """Classify a failed episode into a failure mode."""
        history = episode.history
        if len(history) < 3:
            return 'F_UNKNOWN', 'Too few steps recorded'

        min_ds = [s['min_d'] for s in history]
        speeds = [s['speed_ms'] for s in history]
        forwards = [s['forward_component'] for s in history]
        front_min_ds = [s['front_min_d'] for s in history]

        # ── Mode A: Sudden appearance ──
        # min_d drops from safe to danger in ≤2 steps
        for j in range(1, len(min_ds)):
            if min_ds[j] < 0.05 and min_ds[j-1] > 0.15:
                return 'A_SUDDEN', (
                    f'min_d dropped {min_ds[j-1]:.3f}→{min_ds[j]:.3f} in 1 step '
                    f'(speed={speeds[j]:.1f}m/s)'
                )

        # ── Mode C: High-speed impact ──
        # Speed > 5 m/s when min_d first enters danger zone
        danger_entry = None
        for j in range(len(min_ds)):
            if min_ds[j] < 0.10:
                danger_entry = j
                break
        if danger_entry is not None and speeds[danger_entry] > 5.0:
            return 'C_HIGHSPEED', (
                f'Speed={speeds[danger_entry]:.1f}m/s when min_d={min_ds[danger_entry]:.3f} '
                f'(entered danger zone at step {history[danger_entry]["step"]})'
            )

        # ── Mode B: Gradual approach ──
        # min_d declining over 5+ steps
        declining_count = 0
        for j in range(1, len(min_ds)):
            if min_ds[j] < min_ds[j-1]:
                declining_count += 1
            else:
                declining_count = 0
            if declining_count >= 5:
                return 'B_GRADUAL', (
                    f'min_d declined for {declining_count} consecutive steps: '
                    f'{min_ds[j-declining_count]:.3f}→{min_ds[j]:.3f}'
                )

        # ── Mode D: Dead-end trap (oscillation) ──
        # Check for direction reversals in forward_component
        if len(forwards) >= 6:
            sign_changes = 0
            for j in range(1, len(forwards)):
                if forwards[j] * forwards[j-1] < 0:
                    sign_changes += 1
            if sign_changes >= 4:
                return 'D_DEADEND', (
                    f'{sign_changes} direction reversals in {len(forwards)} steps '
                    f'(oscillating, final min_d={min_ds[-1]:.3f})'
                )

        # ── Mode E: Timeout stall ──
        if episode.outcome == 'timeout':
            # Check if forward component was near-zero for extended period
            stall_count = sum(1 for f in forwards if abs(f) < 0.5)
            if stall_count > len(forwards) * 0.5:
                return 'E_STALL', (
                    f'Forward speed < 0.5m/s for {stall_count}/{len(forwards)} '
                    f'of final recorded steps'
                )
            return 'E_TIMEOUT', f'Timed out, avg forward speed={np.mean(forwards):.2f}m/s'

        # ── Mode F: VO overcorrection ──
        # Would need correction angle data — use action magnitude as proxy
        action_norms = [np.linalg.norm(s['action']) for s in history]
        if len(action_norms) >= 3:
            large_actions = sum(1 for a in action_norms[-5:] if a > 3.0)
            if large_actions >= 3:
                return 'F_OVERCORRECT', (
                    f'{large_actions}/5 final actions at max magnitude '
                    f'(policy saturated, min_d={min_ds[-1]:.3f})'
                )

        return 'G_OTHER', (
            f'Unclassified {episode.outcome}: '
            f'final min_d={min_ds[-1]:.3f}, speed={speeds[-1]:.1f}m/s'
        )

    def print_summary(self):
        """Print a concise summary of failure modes."""
        if not self.completed_episodes:
            print("No failures recorded.")
            return

        # Classify all failures
        classifications = {}
        for ep in self.completed_episodes:
            mode, detail = self._classify_failure(ep)
            if mode not in classifications:
                classifications[mode] = []
            classifications[mode].append(detail)

        total = len(self.completed_episodes)
        print(f"\n{'='*70}")
        print(f"FAILURE ANALYSIS: {total} failures recorded")
        print(f"{'='*70}")

        # Sort by frequency
        sorted_modes = sorted(classifications.items(), key=lambda x: -len(x[1]))

        for mode, details in sorted_modes:
            count = len(details)
            pct = count / total * 100
            print(f"\n  {mode}: {count} failures ({pct:.1f}%)")
            print(f"  {'─'*60}")
            # Show first 3 examples
            for d in details[:3]:
                print(f"    • {d}")
            if count > 3:
                print(f"    ... and {count-3} more")

        # Aggregate statistics
        print(f"\n{'='*70}")
        print("AGGREGATE STATISTICS (across all failures):")
        print(f"{'='*70}")

        all_final_min_d = [ep.history[-1]['min_d'] for ep in self.completed_episodes if ep.history]
        all_final_speed = [ep.history[-1]['speed_ms'] for ep in self.completed_episodes if ep.history]
        all_durations = [ep.duration for ep in self.completed_episodes]

        if all_final_min_d:
            print(f"  Final min_d:  mean={np.mean(all_final_min_d):.4f}  "
                  f"median={np.median(all_final_min_d):.4f}  "
                  f"min={np.min(all_final_min_d):.4f}")
        if all_final_speed:
            print(f"  Final speed:  mean={np.mean(all_final_speed):.2f}m/s  "
                  f"median={np.median(all_final_speed):.2f}m/s")
        if all_durations:
            print(f"  Episode time: mean={np.mean(all_durations):.1f}s  "
                  f"median={np.median(all_durations):.1f}s")

        # Collision vs timeout breakdown
        collisions = sum(1 for ep in self.completed_episodes if ep.outcome == 'collision')
        timeouts = sum(1 for ep in self.completed_episodes if ep.outcome == 'timeout')
        print(f"\n  Collisions: {collisions}  |  Timeouts: {timeouts}  |  Other: {total-collisions-timeouts}")

        # Actionable recommendations
        print(f"\n{'='*70}")
        print("RECOMMENDATIONS:")
        print(f"{'='*70}")

        if sorted_modes:
            top_mode = sorted_modes[0][0]
            top_pct = len(sorted_modes[0][1]) / total * 100

            recommendations = {
                'A_SUDDEN':
                    "Obstacle appears too fast for detection. Consider:\n"
                    "    - Increase LiDAR scan rate or reduce step interval\n"
                    "    - Add temporal filtering to octomap (predict obstacle motion)\n"
                    "    - Lower speed in areas with poor visibility",
                'B_GRADUAL':
                    "Shield fires but correction insufficient. Consider:\n"
                    "    - Increase _VO_SAFE_MARGIN (currently 0.20m)\n"
                    "    - Raise _SPEED_MOD_START (start slowing earlier)\n"
                    "    - Increase RRT* frequency (_RRT_HZ) for faster replanning",
                'C_HIGHSPEED':
                    "UAV enters danger zone too fast for recovery. Consider:\n"
                    "    - Lower _SPEED_MOD_START from 0.15 to 0.20 (start slowing at 4m)\n"
                    "    - Raise _SPEED_MOD_FLOOR from 0.30 to 0.20 (slower in danger zone)\n"
                    "    - Lower _EMERGENCY_SPEED from 2.0 to 1.5 m/s",
                'D_DEADEND':
                    "UAV trapped in oscillation between obstacles. Consider:\n"
                    "    - Increase RRT* _RRT_MAX_ITER for better path search\n"
                    "    - Add oscillation detection → force RRT* direction for N steps\n"
                    "    - Widen _OBS_INFLATE_R to give more clearance in tight spaces",
                'E_STALL':
                    "UAV stalled, not making progress. Consider:\n"
                    "    - Check if VO overcorrection is cancelling forward motion\n"
                    "    - Reduce _ALPHA_MAX if VO corrections are too aggressive\n"
                    "    - Add a 'stuck detector' that temporarily disables shield",
                'E_TIMEOUT':
                    "UAV moving but too slowly. Consider:\n"
                    "    - Check if speed modulation is too aggressive\n"
                    "    - Verify forward_reward is still incentivising progress\n"
                    "    - Reduce _SPEED_MOD_START if UAV is unnecessarily slow",
                'F_OVERCORRECT':
                    "Policy actions saturated (fighting the shield). Consider:\n"
                    "    - Fine-tune DRL with safe_rule=True so policy cooperates\n"
                    "    - Reduce _ALPHA_MAX to limit shield correction magnitude\n"
                    "    - Check if obstacle extraction is producing phantom obstacles",
                'G_OTHER':
                    "Unclassified failures. Consider:\n"
                    "    - Add more logging to identify the pattern\n"
                    "    - Check for edge cases in the simulator",
            }

            if top_mode in recommendations:
                print(f"\n  Top failure mode ({top_mode}, {top_pct:.0f}%):")
                print(f"  {recommendations[top_mode]}")

    def dump_report(self, filepath):
        """Write detailed per-failure data to file."""
        with open(filepath, 'w') as f:
            f.write(f"Failure Analysis Report\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total failures: {len(self.completed_episodes)}\n")
            f.write(f"{'='*70}\n\n")

            for idx, ep in enumerate(self.completed_episodes):
                mode, detail = self._classify_failure(ep)
                f.write(f"--- Failure #{idx+1}: {ep.outcome} ({mode}) ---\n")
                f.write(f"  Classification: {detail}\n")
                f.write(f"  Duration: {ep.duration:.1f}s\n")
                f.write(f"  Steps recorded: {len(ep.history)}\n\n")

                f.write(f"  {'Step':>5} {'min_d':>7} {'front_d':>7} "
                        f"{'speed':>7} {'fwd_v':>7} {'close':>5} {'reward':>8}\n")
                f.write(f"  {'-'*55}\n")

                for s in ep.history:
                    f.write(
                        f"  {s['step']:5d} {s['min_d']:7.4f} {s['front_min_d']:7.4f} "
                        f"{s['speed_ms']:7.2f} {s['forward_component']:7.2f} "
                        f"{s['n_close_cells']:5d} {s['reward']:8.3f}\n"
                    )
                f.write(f"\n")

        print(f"Report written to {filepath}")