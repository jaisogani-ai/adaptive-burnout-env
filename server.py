"""
server.py — OpenEnv API + Gradio UI on the same port (7860).
Premium dark theme is applied via mount_gradio_app() parameters.
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from environment import ProductivityEnv, Action
import uvicorn
import gradio as gr

# =========================================================================
# FASTAPI APP WITH OPENENV ENDPOINTS
# =========================================================================
api = FastAPI(title="Adaptive AI Productivity Engine", version="2.0.0")
env = ProductivityEnv()


@api.get("/api/health")
def health():
    return {"status": "ok", "env": "productivity-burnout-env"}


@api.post("/reset")
@api.get("/reset")
def reset():
    obs = env.reset()
    return JSONResponse(content=dict(obs))


@api.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action.action)
    return JSONResponse(content={
        "observation": dict(obs),
        "reward": float(reward),
        "done": done,
        "info": {k: v for k, v in info.items() if k != "action_history"}
    })


@api.get("/state")
def state():
    return JSONResponse(content=dict(env.state()))


@api.get("/info")
def info():
    return {
        "name": "Adaptive AI Productivity Engine",
        "version": "2.0.0",
        "action_space": {"type": "int", "min": 0, "max": 4},
        "observation_space": {
            "type": "object",
            "fields": ["energy", "stress", "motivation", "progress"]
        }
    }


# =========================================================================
# MOUNT GRADIO UI AT ROOT / WITH PREMIUM DARK THEME
# =========================================================================
from app import demo as gradio_app, premium_css, THEME_JS, dark_theme

app = gr.mount_gradio_app(
    api,
    gradio_app,
    path="/",
    theme=dark_theme,
    css=premium_css,
    js=THEME_JS,
    footer_links=[]
)


# =========================================================================
# ENTRYPOINT
# =========================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
