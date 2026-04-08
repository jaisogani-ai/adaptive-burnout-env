import random
from typing import Tuple, List
from pydantic import BaseModel, Field


# =========================================================================
# OPENENV PYDANTIC SCHEMAS
# =========================================================================
class Action(BaseModel):
    action: int = Field(..., ge=0, le=4)

class Observation(BaseModel):
    energy: float = Field(..., ge=0.0, le=1.0)
    stress: float = Field(..., ge=0.0, le=1.0)
    motivation: float = Field(..., ge=0.0, le=1.0)
    progress: float = Field(..., ge=0.0, le=1.0)

class Reward(BaseModel):
    reward: float = Field(..., ge=0.0, le=1.0)


# =========================================================================
# OBSERVATION RESULT — dict that also supports integer indexing for app.py
# =========================================================================
class ObservationResult(dict):
    """Dict subclass that supports [0],[1],[2],[3] indexing for backward compat."""
    _KEYS = ['energy', 'stress', 'motivation', 'progress']

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(self._KEYS[key])
        return super().__getitem__(key)

    def copy(self):
        return ObservationResult(super().copy())


# =========================================================================
# CORE ENVIRONMENT
# =========================================================================
class ProductivityEnv:
    """
    Adaptive Human Productivity & Burnout Environment.
    OpenEnv-compatible: reset(), step(), state(), render()
    """
    def __init__(self):
        self.reset()

    def reset(self) -> dict:
        self.energy = 0.8
        self.stress = 0.2
        self.motivation = 0.7
        self.progress = 0.0
        self.step_count = 0
        self.action_history: List[int] = []
        self.burnout_counter = 0
        self.fatigue_level = 0.0
        return self.state()

    def state(self) -> dict:
        return ObservationResult({
            "energy": float(round(self.energy, 4)),
            "stress": float(round(self.stress, 4)),
            "motivation": float(round(self.motivation, 4)),
            "progress": float(round(self.progress, 4))
        })

    def render(self):
        pass

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        # Accept Action pydantic model, dict, or raw int
        if isinstance(action, dict):
            action = action.get("action", 0)
        elif hasattr(action, "action") and not isinstance(action, int):
            action = action.action

        dp, ds, de, dm = 0.0, 0.0, 0.0, 0.0

        if action == 0:    # study
            dp, ds, de = 0.1, 0.1, -0.1
        elif action == 1:  # rest
            de, ds = 0.15, -0.1
        elif action == 2:  # exercise
            de, dm = 0.1, 0.1
        elif action == 3:  # social
            dm, ds = 0.1, -0.1
        elif action == 4:  # work_hard
            dp, ds, de = 0.2, 0.2, -0.2

        self.progress += dp
        self.stress += ds
        self.energy += de
        self.motivation += dm

        # Track action history (last 5)
        self.action_history.append(int(action))
        if len(self.action_history) > 5:
            self.action_history.pop(0)

        # Delayed burnout: stress > 0.7 for 3 steps triggers penalty
        if self.stress > 0.7:
            self.burnout_counter += 1
        else:
            self.burnout_counter = max(0, self.burnout_counter - 1)

        if self.burnout_counter >= 3:
            self.motivation -= 0.1
            self.energy -= 0.1

        # Track motivation penalty before clipping
        motivation_penalty_active = self.motivation < 0.3

        # Clip all state values to [0, 1]
        self.energy = float(max(0.0, min(1.0, self.energy)))
        self.stress = float(max(0.0, min(1.0, self.stress)))
        self.motivation = float(max(0.0, min(1.0, self.motivation)))
        self.progress = float(max(0.0, min(1.0, self.progress)))

        self.step_count += 1

        # Reward: Non-linear with balanced state bonus
        is_balanced = (0.3 < self.energy < 0.8) and (self.stress < 0.5)
        reward = dp
        if is_balanced:
            reward += 0.2
        if motivation_penalty_active:
            reward *= 0.7

        # Clip reward to [0.0, 1.0]
        reward = float(max(0.0, min(1.0, reward)))

        # Done conditions
        done = False
        if self.progress >= 1.0 or self.stress >= 1.0 or self.step_count > 50:
            done = True

        # Info dict — include BOTH 'step' and 'step_count' for compat with app.py
        info = {
            "burnout_counter": int(self.burnout_counter),
            "fatigue_level": float(round(self.fatigue_level, 4)),
            "action_history": self.action_history.copy(),
            "step": int(self.step_count),
            "step_count": int(self.step_count),
            "progress_gain": float(round(dp, 4)),
            "reward": float(round(reward, 4)),
        }

        return self.state(), reward, done, info


# =========================================================================
# BACKWARD COMPAT ALIAS — app.py uses environment.ProductivityEnvironment
# =========================================================================
ProductivityEnvironment = ProductivityEnv
