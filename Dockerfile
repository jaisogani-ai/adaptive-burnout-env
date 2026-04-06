FROM python:3.11-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy requirements FIRST
COPY --chown=user requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# THEN copy the rest of the code
COPY --chown=user . .

EXPOSE 7860

CMD ["bash", "-c", "python app.py & sleep 5 && uvicorn server:app --host 0.0.0.0 --port 8000"]

