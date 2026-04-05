"""
grader.py

Trajectory-based grader for the Adaptive Human Productivity & Burnout Environment.
Evaluates agent performance across easy, medium, and hard tasks.

Tasks:
  - Easy:   progress > 0.8
  - Medium: progress > 0.8 AND stress < 0.6
  - Hard:   progress > 0.9 AND stress NEVER exceeds 0.85 during episode
"""

import numpy as np
from environment import ProductivityEnvironment


# =========================================================================
# TRAJECTORY RECORDER
# =========================================================================
class TrajectoryRecorder:
    """Records full episode history for post-hoc grading"""

    def __init__(self):
        self.states = []       # list of (energy, stress, motivation, progress)
        self.actions = []      # list of action ints
        self.rewards = []      # list of step rewards
        self.max_stress = 0.0  # peak stress seen during episode
        self.burnout_steps = 0 # steps where burnout was active

    def record(self, state, action, reward, info=None):
        """Record a single step of the trajectory"""
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        # Track peak stress (index 1 = stress)
        self.max_stress = max(self.max_stress, float(state[1]))
        # Track burnout events from info dict
        if info and info.get("burnout_counter", 0) >= 3:
            self.burnout_steps += 1

    @property
    def final_state(self):
        """Return the last recorded state"""
        if not self.states:
            return np.array([0.0, 0.0, 0.0, 0.0])
        return self.states[-1]

    @property
    def final_progress(self):
        """Final progress value (index 3)"""
        return float(self.final_state[3])

    @property
    def final_stress(self):
        """Final stress value (index 1)"""
        return float(self.final_state[1])

    @property
    def final_energy(self):
        """Final energy value (index 0)"""
        return float(self.final_state[0])

    @property
    def final_motivation(self):
        """Final motivation value (index 2)"""
        return float(self.final_state[2])

    @property
    def total_reward(self):
        """Sum of all step rewards"""
        return sum(self.rewards)

    @property
    def num_steps(self):
        """Number of steps taken"""
        return len(self.actions)

    @property
    def action_diversity(self):
        """Ratio of unique actions used (0.0 to 1.0)"""
        if not self.actions:
            return 0.0
        return len(set(self.actions)) / 5.0  # 5 possible actions

    @property
    def avg_stress(self):
        """Average stress across the trajectory"""
        if not self.states:
            return 0.0
        return float(np.mean([s[1] for s in self.states]))

    @property
    def progress_stability(self):
        """Measure how steadily progress increased (1.0 = perfectly monotonic)"""
        if len(self.states) < 2:
            return 0.0
        progress_vals = [float(s[3]) for s in self.states]
        increases = sum(
            1 for i in range(1, len(progress_vals))
            if progress_vals[i] >= progress_vals[i - 1]
        )
        return increases / (len(progress_vals) - 1)


# =========================================================================
# SCORING FUNCTIONS
# =========================================================================
def _compute_progress_score(trajectory: TrajectoryRecorder, threshold: float) -> float:
    """
    Score based on how close final progress is to the threshold.
    Returns 1.0 if progress >= threshold, partial credit otherwise.
    """
    progress = trajectory.final_progress
    if progress >= threshold:
        return 1.0
    # Partial credit: linear scale from 0 to threshold
    return min(1.0, progress / threshold)


def _compute_stress_penalty(trajectory: TrajectoryRecorder) -> float:
    """
    Penalty for high stress. Returns value between 0.0 (no penalty) and 0.4.
    Penalizes both final stress and average stress across the episode.
    """
    avg_penalty = max(0.0, trajectory.avg_stress - 0.4) * 0.5
    final_penalty = max(0.0, trajectory.final_stress - 0.5) * 0.3
    peak_penalty = max(0.0, trajectory.max_stress - 0.7) * 0.2
    return min(0.4, avg_penalty + final_penalty + peak_penalty)


def _compute_stability_bonus(trajectory: TrajectoryRecorder) -> float:
    """
    Bonus for stable, long-term progress and action diversity.
    Returns value between 0.0 and 0.2.
    """
    stability = trajectory.progress_stability * 0.1
    diversity = trajectory.action_diversity * 0.1
    return min(0.2, stability + diversity)


def _compute_efficiency_bonus(trajectory: TrajectoryRecorder) -> float:
    """
    Bonus for achieving goals in fewer steps.
    Returns value between 0.0 and 0.1.
    """
    if trajectory.num_steps == 0:
        return 0.0
    # Fewer steps = more efficient (baseline: 200 max steps)
    efficiency = max(0.0, 1.0 - (trajectory.num_steps / 200.0))
    return efficiency * 0.1


# =========================================================================
# TASK GRADING
# =========================================================================
def grade_easy(trajectory: TrajectoryRecorder) -> dict:
    """
    Easy task: progress > 0.8

    Scoring breakdown:
      - Progress score (0.0-0.7): how close to 0.8
      - Stability bonus (0.0-0.2): steady progress + action diversity
      - Efficiency bonus (0.0-0.1): fewer steps = better
    """
    passed = trajectory.final_progress > 0.8

    progress_score = _compute_progress_score(trajectory, 0.8) * 0.7
    stability = _compute_stability_bonus(trajectory)
    efficiency = _compute_efficiency_bonus(trajectory)

    score = min(1.0, progress_score + stability + efficiency)

    return {
        "task": "easy",
        "passed": passed,
        "score": round(score, 4),
        "details": {
            "final_progress": round(trajectory.final_progress, 4),
            "threshold": 0.8,
            "steps_taken": trajectory.num_steps,
            "progress_component": round(progress_score, 4),
            "stability_bonus": round(stability, 4),
            "efficiency_bonus": round(efficiency, 4),
        }
    }


def grade_medium(trajectory: TrajectoryRecorder) -> dict:
    """
    Medium task: progress > 0.8 AND stress < 0.6

    Scoring breakdown:
      - Progress score (0.0-0.5): how close to 0.8
      - Stress penalty (0.0-0.4): penalize high stress
      - Stability bonus (0.0-0.2): steady progress + diversity
      - Efficiency bonus (0.0-0.1): fewer steps = better
    """
    passed = trajectory.final_progress > 0.8 and trajectory.final_stress < 0.6

    progress_score = _compute_progress_score(trajectory, 0.8) * 0.5
    stress_penalty = _compute_stress_penalty(trajectory)
    stability = _compute_stability_bonus(trajectory)
    efficiency = _compute_efficiency_bonus(trajectory)

    # Bonus for keeping stress well below 0.6
    stress_bonus = max(0.0, 0.6 - trajectory.final_stress) * 0.3

    score = max(0.0, min(1.0,
        progress_score + stress_bonus + stability + efficiency - stress_penalty
    ))

    return {
        "task": "medium",
        "passed": passed,
        "score": round(score, 4),
        "details": {
            "final_progress": round(trajectory.final_progress, 4),
            "final_stress": round(trajectory.final_stress, 4),
            "progress_threshold": 0.8,
            "stress_threshold": 0.6,
            "steps_taken": trajectory.num_steps,
            "progress_component": round(progress_score, 4),
            "stress_bonus": round(stress_bonus, 4),
            "stress_penalty": round(stress_penalty, 4),
            "stability_bonus": round(stability, 4),
            "efficiency_bonus": round(efficiency, 4),
        }
    }


def grade_hard(trajectory: TrajectoryRecorder) -> dict:
    """
    Hard task: progress > 0.9 AND stress NEVER exceeds 0.85

    Scoring breakdown:
      - Progress score (0.0-0.4): how close to 0.9
      - Stress management (0.0-0.3): bonus for keeping peak stress low
      - Stress penalty (0.0-0.4): penalize high average/peak stress
      - Stability bonus (0.0-0.2): steady progress + diversity
      - Efficiency bonus (0.0-0.1): fewer steps = better
    """
    stress_never_exceeded = trajectory.max_stress <= 0.85
    passed = trajectory.final_progress > 0.9 and stress_never_exceeded

    progress_score = _compute_progress_score(trajectory, 0.9) * 0.4
    stress_penalty = _compute_stress_penalty(trajectory)
    stability = _compute_stability_bonus(trajectory)
    efficiency = _compute_efficiency_bonus(trajectory)

    # Bonus for keeping peak stress well below 0.85
    stress_mgmt_bonus = max(0.0, 0.85 - trajectory.max_stress) * 0.35

    # Extra penalty if stress ever exceeded 0.85
    if not stress_never_exceeded:
        stress_penalty += 0.15

    # Burnout penalty — penalize strategies that relied on burnout
    burnout_penalty = min(0.1, trajectory.burnout_steps * 0.01)

    score = max(0.0, min(1.0,
        progress_score + stress_mgmt_bonus + stability + efficiency
        - stress_penalty - burnout_penalty
    ))

    return {
        "task": "hard",
        "passed": passed,
        "score": round(score, 4),
        "details": {
            "final_progress": round(trajectory.final_progress, 4),
            "max_stress": round(trajectory.max_stress, 4),
            "progress_threshold": 0.9,
            "stress_ceiling": 0.85,
            "stress_never_exceeded": stress_never_exceeded,
            "burnout_steps": trajectory.burnout_steps,
            "steps_taken": trajectory.num_steps,
            "progress_component": round(progress_score, 4),
            "stress_mgmt_bonus": round(stress_mgmt_bonus, 4),
            "stress_penalty": round(stress_penalty, 4),
            "burnout_penalty": round(burnout_penalty, 4),
            "stability_bonus": round(stability, 4),
            "efficiency_bonus": round(efficiency, 4),
        }
    }


# =========================================================================
# MAIN GRADING ENTRY POINT
# =========================================================================
def grade_trajectory(trajectory: TrajectoryRecorder, difficulty: str = "easy") -> dict:
    """
    Grade a full trajectory based on difficulty level.

    Args:
        trajectory: TrajectoryRecorder with full episode history
        difficulty: one of 'easy', 'medium', 'hard'

    Returns:
        dict with 'task', 'passed', 'score', and 'details'
    """
    graders = {
        "easy": grade_easy,
        "medium": grade_medium,
        "hard": grade_hard,
    }

    grader_fn = graders.get(difficulty)
    if grader_fn is None:
        return {
            "task": difficulty,
            "passed": False,
            "score": 0.0,
            "details": {"error": f"Unknown difficulty: {difficulty}. Use easy/medium/hard."}
        }

    return grader_fn(trajectory)


# =========================================================================
# RUN A FULL EPISODE AND GRADE IT
# =========================================================================
def run_and_grade(action_sequence: list, difficulty: str = "easy") -> dict:
    """
    Run a full episode with a given action sequence and grade the result.

    Args:
        action_sequence: list of action ints (0-4)
        difficulty: 'easy', 'medium', or 'hard'

    Returns:
        dict with grading result
    """
    env = ProductivityEnvironment()
    recorder = TrajectoryRecorder()

    state = env.reset()

    for action in action_sequence:
        # Clamp action to valid range
        action = max(0, min(4, action))
        next_state, reward, done, info = env.step(action)
        recorder.record(next_state, action, reward, info)
        if done:
            break

    return grade_trajectory(recorder, difficulty)


# =========================================================================
# SELF-TEST
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GRADER SELF-TEST")
    print("=" * 60)

    # Test with a simple strategy: alternate study and rest
    test_actions = [0, 1, 0, 1, 4, 2, 0, 1, 4, 3] * 20  # 200 actions

    for diff in ["easy", "medium", "hard"]:
        result = run_and_grade(test_actions, diff)
        status = "PASS ✅" if result["passed"] else "FAIL ❌"
        print(f"\n[{diff.upper()}] {status}")
        print(f"  Score: {result['score']}")
        for k, v in result["details"].items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("GRADER SELF-TEST COMPLETE")
    print("=" * 60)
