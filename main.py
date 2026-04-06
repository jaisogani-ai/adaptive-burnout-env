import gradio as gr
from server import app as fastapi_app
from app import demo as gradio_app

# Mount the Gradio UI onto the FastAPI OpenEnv app at the root path '/'
# This makes both the API endpoints AND the User Interface available on the same port!
app = gr.mount_gradio_app(fastapi_app, gradio_app, path="/")
