"""
server.py
OpenEnv server entrypoint.
Wraps ProductivityEnvironment for OpenEnv compatibility.
"""

from typing import Any, Optional
from uuid import uuid4
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from environment import ProductivityEnvironment


# -------------------------
# CUSTOM ACTION MODEL
# -------------------------
class BurnoutAction(Action):
    """Action for the Burnout environment (0-4)"""
    action: int = 0


# -------------------------
# CUSTOM OBSERVATION MODEL
# -------------------------
class BurnoutObservation(Observation):
    """Observation returned by the Burnout environment"""
    energy: float = 0.0
    stress: float = 0.0
    motivation: float = 0.0
    progress: float = 0.0


# -------------------------
# OPENENV WRAPPER
# -------------------------
class BurnoutEnvironment(Environment):
    """Wraps ProductivityEnvironment for OpenEnv compatibility"""

    def __init__(self):
        super().__init__()
        self.env = ProductivityEnvironment()
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, seed=None, episode_id=None, **kwargs):
        """Reset environment, return initial observation"""
        state_array = self.env.reset()
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return BurnoutObservation(
            energy=float(state_array[0]),
            stress=float(state_array[1]),
            motivation=float(state_array[2]),
            progress=float(state_array[3]),
            done=False, reward=0.0)

    def step(self, action: BurnoutAction, **kwargs):
        """Execute action, return observation"""
        state_array, reward, done, info = self.env.step(action.action)
        self._state.step_count += 1
        return BurnoutObservation(
            energy=float(state_array[0]),
            stress=float(state_array[1]),
            motivation=float(state_array[2]),
            progress=float(state_array[3]),
            done=done, reward=reward, metadata=info)

    @property
    def state(self):
        """Return current environment state"""
        return self._state


# -------------------------
# CREATE APP
# -------------------------
app = create_app(BurnoutEnvironment, BurnoutAction, BurnoutObservation, env_name="burnout_env")
@app.get("/")
def home():
    return {"message": "Adaptive Burnout Environment Running"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
