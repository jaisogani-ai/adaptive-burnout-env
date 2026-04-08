import numpy as np


# =========================================================================
# OPENENV GRADER — ProductivityGrader (used by inference.py)
# =========================================================================
class ProductivityGrader:
    def grade_easy(self, trajectory):
        progress = trajectory[-1]['obs']['progress']
        passed = progress > 0.8
        score = max(0.01, min(0.99, progress))
        return {"pass": bool(passed), "score": float(score)}

    def grade_medium(self, trajectory):
        progress = trajectory[-1]['obs']['progress']
        stress = trajectory[-1]['obs']['stress']
        passed = progress > 0.8 and stress < 0.6
        score = max(0.01, min(0.99, (progress + (1 - stress)) / 2))
        return {"pass": bool(passed), "score": float(score)}

    def grade_hard(self, trajectory):
        progress = trajectory[-1]['obs']['progress']
        max_stress = max(step['obs']['stress'] for step in trajectory)
        passed = progress > 0.9 and max_stress <= 0.85
        score = max(0.01, min(0.99, progress * (1 - max_stress)))
        return {"pass": bool(passed), "score": float(score)}


# =========================================================================
# BACKWARD COMPAT — TrajectoryRecorder + grade_trajectory used by app.py
# =========================================================================
class TrajectoryRecorder:
    """Records full episode history for post-hoc grading (app.py compat)."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.max_stress = 0.0
        self.burnout_steps = 0

    def record(self, state, action, reward, info=None):
        if hasattr(state, 'copy'):
            self.states.append(state.copy())
        else:
            self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        # Track peak stress (index 1 = stress)
        try:
            self.max_stress = max(self.max_stress, float(state[1]))
        except (IndexError, TypeError, KeyError):
            pass
        if info and info.get("burnout_counter", 0) >= 3:
            self.burnout_steps += 1

    @property
    def final_state(self):
        if not self.states:
            return np.array([0.0, 0.0, 0.0, 0.0])
        return self.states[-1]

    @property
    def final_progress(self):
        return float(self.final_state[3])

    @property
    def final_stress(self):
        return float(self.final_state[1])

    @property
    def final_energy(self):
        return float(self.final_state[0])

    @property
    def final_motivation(self):
        return float(self.final_state[2])

    @property
    def total_reward(self):
        return sum(self.rewards)

    @property
    def num_steps(self):
        return len(self.actions)

    @property
    def action_diversity(self):
        if not self.actions:
            return 0.0
        return len(set(self.actions)) / 5.0

    @property
    def avg_stress(self):
        if not self.states:
            return 0.0
        return float(np.mean([float(s[1]) for s in self.states]))

    @property
    def progress_stability(self):
        if len(self.states) < 2:
            return 0.0
        progress_vals = [float(s[3]) for s in self.states]
        increases = sum(
            1 for i in range(1, len(progress_vals))
            if progress_vals[i] >= progress_vals[i - 1]
        )
        return increases / (len(progress_vals) - 1)


def _compute_progress_score(trajectory, threshold):
    progress = trajectory.final_progress
    if progress >= threshold:
        return 1.0
    return min(1.0, progress / threshold)


def _compute_stress_penalty(trajectory):
    avg_penalty = max(0.0, trajectory.avg_stress - 0.4) * 0.5
    final_penalty = max(0.0, trajectory.final_stress - 0.5) * 0.3
    peak_penalty = max(0.0, trajectory.max_stress - 0.7) * 0.2
    return min(0.4, avg_penalty + final_penalty + peak_penalty)


def _compute_stability_bonus(trajectory):
    stability = trajectory.progress_stability * 0.1
    diversity = trajectory.action_diversity * 0.1
    return min(0.2, stability + diversity)


def _compute_efficiency_bonus(trajectory):
    if trajectory.num_steps == 0:
        return 0.0
    efficiency = max(0.0, 1.0 - (trajectory.num_steps / 200.0))
    return efficiency * 0.1


def grade_easy(trajectory):
    passed = trajectory.final_progress > 0.8
    progress_score = _compute_progress_score(trajectory, 0.8) * 0.7
    stability = _compute_stability_bonus(trajectory)
    efficiency = _compute_efficiency_bonus(trajectory)
    score = max(0.01, min(0.99, progress_score + stability + efficiency))
    return {
        "task": "easy", "passed": passed, "score": round(score, 4),
        "details": {"final_progress": round(trajectory.final_progress, 4), "threshold": 0.8, "steps_taken": trajectory.num_steps}
    }


def grade_medium(trajectory):
    passed = trajectory.final_progress > 0.8 and trajectory.final_stress < 0.6
    progress_score = _compute_progress_score(trajectory, 0.8) * 0.5
    stress_penalty = _compute_stress_penalty(trajectory)
    stability = _compute_stability_bonus(trajectory)
    efficiency = _compute_efficiency_bonus(trajectory)
    stress_bonus = max(0.0, 0.6 - trajectory.final_stress) * 0.3
    score = max(0.01, min(0.99, progress_score + stress_bonus + stability + efficiency - stress_penalty))
    return {
        "task": "medium", "passed": passed, "score": round(score, 4),
        "details": {"final_progress": round(trajectory.final_progress, 4), "final_stress": round(trajectory.final_stress, 4)}
    }


def grade_hard(trajectory):
    stress_never_exceeded = trajectory.max_stress <= 0.85
    passed = trajectory.final_progress > 0.9 and stress_never_exceeded
    progress_score = _compute_progress_score(trajectory, 0.9) * 0.4
    stress_penalty = _compute_stress_penalty(trajectory)
    stability = _compute_stability_bonus(trajectory)
    efficiency = _compute_efficiency_bonus(trajectory)
    stress_mgmt_bonus = max(0.0, 0.85 - trajectory.max_stress) * 0.35
    if not stress_never_exceeded:
        stress_penalty += 0.15
    burnout_penalty = min(0.1, trajectory.burnout_steps * 0.01)
    score = max(0.01, min(0.99, progress_score + stress_mgmt_bonus + stability + efficiency - stress_penalty - burnout_penalty))
    return {
        "task": "hard", "passed": passed, "score": round(score, 4),
        "details": {"final_progress": round(trajectory.final_progress, 4), "max_stress": round(trajectory.max_stress, 4)}
    }


def grade_trajectory(trajectory, difficulty="easy"):
    """Grade a TrajectoryRecorder by difficulty. Used by app.py."""
    graders = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}
    grader_fn = graders.get(difficulty)
    if grader_fn is None:
        return {"task": difficulty, "passed": False, "score": 0.01, "details": {"error": f"Unknown: {difficulty}"}}
    return grader_fn(trajectory)
