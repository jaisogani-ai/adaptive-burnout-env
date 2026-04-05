<div align="center">

# 🧠 Adaptive Human Productivity & Burnout Environment

### AI-Powered Reinforcement Learning Environment for Human Wellbeing

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenEnv](https://img.shields.io/badge/OpenEnv-Framework-FF6B6B?style=for-the-badge&logo=meta&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Gemini AI](https://img.shields.io/badge/Gemini_AI-2.0_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Hackathon](https://img.shields.io/badge/Meta_OpenEnv-Hackathon_2026-FF4500?style=for-the-badge)

---

**A research-grade reinforcement learning environment that simulates the complex dynamics of human productivity, energy management, and burnout — where AI agents learn that sustainable performance beats short-term grinding.**

</div>

---

## 🎬 Demo

<div align="center">

![Demo](demo.gif)

</div>

<table align="center">
<tr>
<td align="center"><b>⚡ 5 Actions</b><br>Study · Rest · Exercise<br>Social · Work Hard</td>
<td align="center"><b>📊 4 State Variables</b><br>Energy · Stress<br>Motivation · Progress</td>
<td align="center"><b>🎯 3 Difficulty Levels</b><br>Easy · Medium · Hard<br>Trajectory Grading</td>
<td align="center"><b>🤖 Gemini AI Powered</b><br>LLM Decisions +<br>Rule-Based Fallback</td>
</tr>
</table>

---

## 📖 Project Overview

### The Problem

Modern AI focuses on maximizing output — but ignores the **human cost**. In the real world, sustained high performance without recovery leads to burnout, crashes productivity, and destroys long-term outcomes. Current RL benchmarks don't model this critical dynamic.

### Why It Matters

This environment teaches AI agents a fundamental truth: **sustainable performance always beats short-term grinding.** An agent that pushes too hard will trigger burnout spirals — crashing motivation and energy in ways that are expensive to recover from. The optimal policy must balance progress with wellbeing.

### What Makes It Unique

| Feature | Standard RL Envs | This Environment |
|---------|-----------------|------------------|
| Burnout modeling | ❌ | ✅ Delayed burnout with compounding stress |
| Fatigue accumulation | ❌ | ✅ Cumulative fatigue degrades performance |
| Diminishing returns | ❌ | ✅ Last 10% of progress is hardest |
| Stochastic events | Rare | ✅ Good/bad days affect state randomly |
| Motivation feedback | ❌ | ✅ Progress boosts motivation, burnout destroys it |
| Multi-level grading | ❌ | ✅ Easy/Medium/Hard trajectory evaluation |
| LLM-powered agent | ❌ | ✅ Gemini AI with triple fallback safety |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SYSTEM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐    action     ┌──────────────────┐              │
│   │          │──────────────▶│                  │              │
│   │  AGENT   │               │   ENVIRONMENT    │              │
│   │          │◀──────────────│                  │              │
│   └────┬─────┘  state,reward └────────┬─────────┘              │
│        │                              │                         │
│        │                              ▼                         │
│        │                     ┌──────────────────┐              │
│        │                     │  INTERNAL SYSTEMS │              │
│        │                     │  · Burnout Engine │              │
│        │                     │  · Fatigue System │              │
│        │                     │  · Stress Compound│              │
│        │                     │  · Random Events  │              │
│        │                     │  · Motivation Loop│              │
│        │                     └──────────────────┘              │
│        │                                                        │
│   ┌────▼──────────────────────────────────────────┐            │
│   │              INFERENCE ENGINE                  │            │
│   │                                                │            │
│   │   ┌────────────┐    fail    ┌──────────────┐  │            │
│   │   │ Gemini API │──────────▶│  Rule-Based  │  │            │
│   │   │ (SDK/REST) │           │   Fallback   │  │            │
│   │   └────────────┘           └──────────────┘  │            │
│   └───────────────────────────────────────────────┘            │
│                                                                 │
│   ┌───────────────┐          ┌───────────────────┐             │
│   │    GRADER     │          │    SERVER (API)    │             │
│   │ Easy/Med/Hard │          │  FastAPI + OpenEnv │             │
│   └───────────────┘          └───────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🧠 Reinforcement Learning Environment
Gymnasium-compatible environment with `reset()` → `step()` → `(state, reward, done, info)` interface. Standard RL loop with rich internal dynamics.

### ⚡ Delayed Burnout System
Stress above 0.75 for 3+ consecutive steps triggers a burnout cascade — motivation drops 0.2 and energy drops 0.1 per step. Recovery is slow and costly.

### 🔁 Motivation Feedback Loop
Progress above 50% boosts motivation. But low energy + high stress crushes it. Creates a realistic virtuous/vicious cycle.

### 🎲 Stochastic Events
10% chance of a bad day (stress spike + motivation drop). 10% chance of a good day (energy + motivation boost). Agents must handle uncertainty.

</td>
<td width="50%">

### 🤖 Gemini AI Decision System
Google Gemini 2.0 Flash analyzes state and picks optimal actions. Understands burnout risks and balances short-term vs long-term gains.

### 🛡️ Triple Fallback Safety
Gemini SDK → REST API → Rule-Based Logic. System **never crashes** on API failure. Graceful degradation with decision reasoning.

### 📊 Non-Linear Reward Shaping
Stress compounds non-linearly (up to 1.5x). Progress has diminishing returns. Cumulative fatigue degrades work output. Wellbeing is rewarded.

### 🏆 3-Level Trajectory Grading
Full-episode evaluation with pass/fail and 0.0–1.0 scoring. Penalizes burnout-reliant strategies. Rewards stability and action diversity.

</td>
</tr>
</table>

---

## 🎮 Environment Design

### Actions

| Code | Action | Energy | Stress | Motivation | Progress | Best Used When |
|:----:|--------|:------:|:------:|:----------:|:--------:|----------------|
| `0` | **Study** | -0.10 | +0.08 | -0.02 | +0.12 | Healthy state, need steady progress |
| `1` | **Rest** | +0.15 | -0.10 | +0.05 | 0.00 | Energy low, need recovery |
| `2` | **Exercise** | +0.10 | -0.12 | +0.10 | +0.02 | Stress high, need de-stress |
| `3` | **Social** | +0.05 | -0.08 | +0.12 | +0.01 | Motivation low, need boost |
| `4` | **Work Hard** | -0.20 | +0.15 | -0.05 | +0.20 | High energy & low stress only |

### State Variables

| Variable | Range | Initial | Description |
|----------|:-----:|:-------:|-------------|
| **Energy** | 0.0 – 1.0 | 0.8 | Physical/mental capacity. Drops below 0 = collapse |
| **Stress** | 0.0 – 1.0 | 0.2 | Accumulated pressure. Above 0.75 triggers burnout |
| **Motivation** | 0.0 – 1.0 | 0.7 | Drive to work. Destroyed by burnout, boosted by progress |
| **Progress** | 0.0 – 1.0 | 0.0 | Goal completion. Reaching 1.0 = success |

### Hidden Systems

| System | Trigger | Effect |
|--------|---------|--------|
| **Burnout** | Stress > 0.75 for 3+ steps | Motivation -0.2, Energy -0.1 per step |
| **Fatigue** | Consecutive study/work | Diminished progress, extra energy drain |
| **Stress Compounding** | High stress + stress-causing action | Stress gains multiply up to 1.5x |
| **Diminishing Returns** | Progress approaching 1.0 | Progress gains scale down by up to 70% |

---

## 🏆 Grading System

| Level | Pass Condition | Scoring Breakdown |
|-------|---------------|-------------------|
| 🟢 **Easy** | Progress > 0.8 | 70% progress + 20% stability + 10% efficiency |
| 🟡 **Medium** | Progress > 0.8 **AND** final stress < 0.6 | 50% progress + 30% stress management + 20% bonuses |
| 🔴 **Hard** | Progress > 0.9 **AND** stress **never** exceeds 0.85 | 40% progress + 35% stress control + 25% bonuses − burnout penalty |

> **Note:** The Hard task requires mastery of work-rest cycling. Brute-force grinding strategies will fail due to stress compounding, fatigue accumulation, and the burnout penalty in scoring.

---

## 🔌 API Endpoints

| Method | Endpoint | Description | Example |
|:------:|----------|-------------|---------|
| `POST` | `/reset` | Reset environment, start new episode | Returns initial observation |
| `POST` | `/step` | Execute an action (0–4) | `{"action": {"action": 0}}` |
| `GET` | `/health` | Server health check | `{"status": "healthy"}` |
| `GET` | `/docs` | Interactive Swagger UI | Auto-generated by FastAPI |
| `WS` | `/ws` | WebSocket for real-time sessions | OpenEnv standard protocol |

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|:----------:|:-------:|---------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) | 3.11 | Core language |
| ![OpenEnv](https://img.shields.io/badge/-OpenEnv-FF6B6B?style=flat-square&logo=meta&logoColor=white) | 0.1.0+ | RL environment framework |
| ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) | 0.110+ | Auto-generated API server |
| ![Gemini](https://img.shields.io/badge/-Gemini_AI-4285F4?style=flat-square&logo=google&logoColor=white) | 2.0 Flash | AI-powered decision making |
| ![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white) | Latest | Containerized deployment |
| ![Uvicorn](https://img.shields.io/badge/-Uvicorn-2D2D2D?style=flat-square) | 0.29+ | ASGI server |
| ![Pydantic](https://img.shields.io/badge/-Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white) | 2.6+ | Data validation |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | 1.26+ | Numerical computing |

---

## 📁 Project Structure

```
adaptive-burnout-env/
│
├── 🧠 environment.py    # Core RL environment with burnout, fatigue, stress dynamics
├── 🌐 server.py          # OpenEnv API server (FastAPI + create_app wrapper)
├── 🏆 grader.py           # Trajectory-based grading system (Easy/Medium/Hard)
├── 🤖 inference.py        # Gemini AI agent + rule-based fallback + verbose runner
├── 📊 visualize.py        # Matplotlib state trajectory visualization
│
├── 📋 requirements.txt    # Python dependencies with version bounds
├── 🐳 dockerfile           # Docker container configuration
├── 🔐 .env                 # API keys (git-ignored)
├── 🚫 .gitignore           # Git exclusion rules
├── 📄 LICENSE              # MIT License
└── 📖 README.md            # You are here
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/jaisogani-ai/adaptive-burnout-env.git
cd adaptive-burnout-env
```

### 2. Set Up Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Start the Server

```bash
python server.py
```

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### 4. Run the AI Agent (Verbose Mode)

```bash
python inference.py
```

---

## 🧪 Quick API Test

```bash
# Health check
curl http://localhost:8000/health

# Reset environment
curl -X POST http://localhost:8000/reset

# Take an action (study)
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action": 0}}'
```

**Expected response from `/step`:**

```json
{
  "observation": {
    "energy": 0.700,
    "stress": 0.288,
    "motivation": 0.680,
    "progress": 0.120
  },
  "reward": 0.249,
  "done": false
}
```

---

## 🤖 Gemini AI Setup

### Option A: Environment Variable

```bash
export GEMINI_API_KEY="your-gemini-api-key"
python inference.py
```

### Option B: `.env` File (Recommended)

```bash
# Create .env file in project root
echo 'GEMINI_API_KEY=your-gemini-api-key' > .env
python inference.py
```

> **Note:** If Gemini is unavailable or the API key is not set, the system automatically falls back to the rule-based decision engine. The system **never crashes** on API failure.

### Decision Flow

```
┌─────────────────┐     success     ┌──────────────┐
│  Gemini SDK     │────────────────▶│  action +    │
│  (google.genai) │                 │  reasoning   │
└────────┬────────┘                 └──────────────┘
         │ fail
         ▼
┌─────────────────┐     success     ┌──────────────┐
│  Gemini REST    │────────────────▶│  action +    │
│  (urllib)       │                 │  reasoning   │
└────────┬────────┘                 └──────────────┘
         │ fail
         ▼
┌─────────────────┐     always      ┌──────────────┐
│  Rule-Based     │────────────────▶│  action +    │
│  Fallback       │                 │  reasoning   │
└─────────────────┘                 └──────────────┘
```

---

## 🐳 Docker Deployment

```bash
# Build the container
docker build -t burnout-env .

# Run with API key
docker run -p 8000:8000 -e GEMINI_API_KEY="your-key" burnout-env

# Run without Gemini (fallback mode)
docker run -p 8000:8000 burnout-env
```

---

## 📊 Visualization

```bash
python visualize.py
```

Generates `burnout_plot.png` with:
- State trajectories (energy, stress, motivation, progress)
- Burnout zones highlighted in red
- Low energy zones highlighted in orange
- Episode summary with risk detection

---

## 🧑‍💻 Author & Hackathon

<table>
<tr>
<td>

| | |
|---|---|
| **Built for** | Meta OpenEnv Hackathon 2026 |
| **Author** | Jai Sogani |
| **GitHub** | [github.com/jaisogani-ai](https://github.com/jaisogani-ai) |
| **Framework** | [OpenEnv](https://github.com/meta-pytorch/OpenEnv) by Meta & Hugging Face |
| **Submitted** | April 2026 |

</td>
</tr>
</table>

---

## 📄 License

```
MIT License

Copyright (c) 2026 Jai Sogani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
```

---

<div align="center">

**Built with 💡 for the Meta OpenEnv Hackathon 2026**

*Teaching AI that sustainable performance beats short-term grinding.*

</div>
<<<<<<< HEAD
=======

>>>>>>> e95651d4d7a7c6c53d154d7d4f0fc2618bf229f2
