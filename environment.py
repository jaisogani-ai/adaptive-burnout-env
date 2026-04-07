"""
environment.py

Adaptive Human Productivity & Burnout Environment
OpenEnv-compatible RL Environment
"""

import random
import numpy as np
from pydantic import BaseModel, Field

# =========================================================================
# OPENENV SCHEMAS
# =========================================================================
class Observation(BaseModel):
    """Pydantic model representing environment observation per OpenEnv standard"""
    energy: float = Field(..., ge=0.0, le=1.0)
    stress: float = Field(..., ge=0.0, le=1.0)
    motivation: float = Field(..., ge=0.0, le=1.0)
    progress: float = Field(..., ge=0.0, le=1.0)

    # Legacy syntax compatibility (app.py and grader.py expect indexing/copying)
    def __getitem__(self, item: int) -> float:
        return [self.energy, self.stress, self.motivation, self.progress][item]
        
    def copy(self):
        """Support numpy .copy() used by grader"""
        return Observation(
            energy=self.energy, stress=self.stress, 
            motivation=self.motivation, progress=self.progress
        )

class StepResult(BaseModel):
    """Full step return payload"""
    observation: Observation
    reward: float
    done: bool
    info: dict

# Define discrete Action Space schema representation
ActionSpace = {
    0: "study",
    1: "rest", 
    2: "exercise",
    3: "social",
    4: "work_hard"
}


class ProductivityEnvironment:
    """
    RL Environment simulating human productivity and burnout dynamics
    """

    def __init__(self):
        self.reset()


    # -------------------------
    # RESET ENVIRONMENT
    # -------------------------
    def reset(self):
        """Initialize environment state"""
        self.energy = 0.8
        self.stress = 0.2
        self.motivation = 0.7
        self.progress = 0.0

        self.step_count = 0
        self.action_history = []
        self.burnout_counter = 0
        self.fatigue_level = 0.0  # cumulative fatigue from sustained work
        self.max_steps = 200

        return self._get_state()

    # -------------------------
    # GET STATE (Standardized)
    # -------------------------
    def state(self) -> Observation:
        """OpenEnv compliant state getter"""
        return Observation(
            energy=self.energy,
            stress=self.stress,
            motivation=self.motivation,
            progress=self.progress
        )

    def _get_state(self) -> Observation:
        """Legacy internal wrapper"""
        return self.state()

    # -------------------------
    # APPLY ACTION EFFECTS
    # -------------------------
    def _apply_action(self, action):
        """
        Actions:
        0=study, 1=rest, 2=exercise, 3=social, 4=work_hard
        """
        effects = {
            0: (-0.1, +0.08, -0.02, +0.12),
            1: (+0.15, -0.1, +0.05, 0.0),
            2: (+0.1, -0.12, +0.1, +0.02),
            3: (+0.05, -0.08, +0.12, +0.01),
            4: (-0.2, +0.15, -0.05, +0.2),
        }

        de, ds, dm, dp = effects.get(action, (0, 0, 0, 0))

        self.energy += de

        # Change 2: Non-linear stress compounding — stress spirals when already high
        if ds > 0:
            stress_multiplier = 1.0 + (self.stress * 0.5)  # up to 1.5x at max stress
            self.stress += ds * stress_multiplier
        else:
            self.stress += ds  # stress relief stays linear

        self.motivation += dm

        # Change 3: Diminishing returns on progress — last 10% is hardest
        if dp > 0:
            diminishing = 1.0 - (self.progress * 0.4)  # harder as progress increases
            self.progress += dp * max(0.3, diminishing)
        else:
            self.progress += dp

    # -------------------------
    # RANDOM EVENTS
    # -------------------------
    def _random_events(self):
        """Simulate good/bad days"""
        r = random.random()

        if r < 0.1:
            # bad day
            self.stress += 0.1
            self.motivation -= 0.1
        elif r > 0.9:
            # good day
            self.energy += 0.1
            self.motivation += 0.1

    # -------------------------
    # BURNOUT SYSTEM
    # -------------------------
    def _burnout(self):
        """Delayed burnout effect"""
        if self.stress > 0.75:
            self.burnout_counter += 1
        else:
            self.burnout_counter = max(0, self.burnout_counter - 1)

        if self.burnout_counter >= 3:
            self.motivation -= 0.2
            self.energy -= 0.1

    # -------------------------
    # MOTIVATION FEEDBACK
    # -------------------------
    def _motivation_feedback(self):
        if self.progress > 0.5:
            self.motivation += 0.05

        if self.energy < 0.3 and self.stress > 0.6:
            self.motivation -= 0.1

    # -------------------------
    # ACTION REPETITION PENALTY
    # -------------------------
    def _action_penalty(self, action):
        if len(self.action_history) >= 3:
            if all(a == action for a in self.action_history[-3:]):
                return -0.1
        return 0.0

    # -------------------------
    # CLIP VALUES
    # -------------------------
    def _clip(self):
        self.energy = np.clip(self.energy, 0, 1)
        self.stress = np.clip(self.stress, 0, 1)
        self.motivation = np.clip(self.motivation, 0, 1)
        self.progress = np.clip(self.progress, 0, 1)

    # -------------------------
    # FATIGUE SYSTEM
    # -------------------------
    def _fatigue(self):
        """Cumulative fatigue — sustained work without rest degrades performance"""
        last_action = self.action_history[-1] if self.action_history else None
        # Work/study increase fatigue
        if last_action in (0, 4):
            self.fatigue_level = min(1.0, self.fatigue_level + 0.05)
        # Rest/exercise reduce fatigue
        elif last_action in (1, 2):
            self.fatigue_level = max(0.0, self.fatigue_level - 0.08)
        # Social slightly reduces fatigue
        elif last_action == 3:
            self.fatigue_level = max(0.0, self.fatigue_level - 0.03)

        # High fatigue reduces effectiveness
        if self.fatigue_level > 0.6:
            self.progress -= 0.02  # diminished returns
            self.energy -= 0.02    # extra energy drain

    # -------------------------
    # REWARD FUNCTION
    # -------------------------
    def _compute_reward(self, progress_gain, penalty):
        # Change 4: Wellbeing-weighted reward — balanced states are valuable
        wellbeing = (
            (self.energy * 0.3)
            + ((1.0 - self.stress) * 0.3)
            + (self.motivation * 0.2)
        )

        return (
            progress_gain * 1.5          # progress matters most
            - (self.stress ** 2)          # quadratic stress penalty
            + wellbeing * 0.15            # reward balanced states
            + (self.motivation * 0.1)     # motivation bonus
            + penalty                     # repetition penalty
        )

    # -------------------------
    # STEP FUNCTION (FIXED)
    # -------------------------
    def step(self, action):
        prev_progress = self.progress

        # Apply action
        self._apply_action(action)

        # Random events
        self._random_events()

        # Burnout
        self._burnout()

        # Fatigue (Change 1)
        self._fatigue()

        # Motivation
        self._motivation_feedback()

        # Clip values
        self._clip()

        # Track actions
        self.action_history.append(action)

        # Compute reward
        progress_gain = self.progress - prev_progress
        penalty = self._action_penalty(action)
        reward = self._compute_reward(progress_gain, penalty)

        # FIX: increment BEFORE done check
        self.step_count += 1

        # Done condition
        done = (
            self.progress >= 1.0
            or self.energy <= 0
            or self.step_count >= self.max_steps
        )

        # Info dict for grader and observability
        info = {
            "burnout_counter": self.burnout_counter,
            "fatigue_level": round(float(self.fatigue_level), 4),
            "step_count": self.step_count,
            "progress_gain": round(float(progress_gain), 4),
            "reward": round(float(reward), 4),
            "action_history": self.action_history
        }

        # Return OpenEnv format natively (App.py safely unpacks the tuple, grader uses it)
        # Note: step() intentionally returns a tuple (obs, reward, done, info) to remain functionally valid 
        # for gym and app.py unpacks while obs is strongly typed.
        return self.state(), float(reward), done, info
