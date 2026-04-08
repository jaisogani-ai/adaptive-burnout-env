import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import concurrent.futures
import time
import numpy as np
import random
import os

# Use non-interactive backend for peak performance
matplotlib.use('agg')
from matplotlib.figure import Figure

# ==========================================
# IMPORTS & GLOBAL REGISTRY
# ==========================================
import environment
import grader
from inference import get_action, rule_based_action

# Initialization
# env is now session-isolated via gr.State (see sys_env below)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ==========================================
# DESIGN TOKENS (ELITE PRODUCT DESIGN)
# ==========================================
COLORS = {
    "background": "#0d1117",
    "surface": "#161b22",
    "border": "#30363d",
    "primary": "#58a6ff",
    "secondary": "#2ea043",
    "warning": "#d29922",
    "danger": "#f85149",
    "text": "#c9d1d9",
    "muted": "#8b949e",
    "highlight": "#21262d",
    "glow": "rgba(88, 166, 255, 0.15)"
}

ACTION_MAP = {
    0: "📚 Study", 1: "💤 Rest", 2: "🏃 Exercise", 3: "💬 Social", 4: "🚀 Work Hard"
}

# ==========================================
# PREMIUM FUTURISTIC CSS (CSS Custom Properties)
# ==========================================
premium_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

/* === THEME VARIABLES (data-attribute based for Gradio compat) === */
:root, :root[data-theme="dark"] {
    --bg-primary: #0d1117;
    --bg-surface: #161b22;
    --bg-highlight: #21262d;
    --border-color: #30363d;
    --text-primary: #c9d1d9;
    --text-muted: #8b949e;
    --accent-primary: #58a6ff;
    --accent-green: #2ea043;
    --accent-warning: #d29922;
    --accent-danger: #f85149;
    --shadow-card: 0 8px 32px rgba(0,0,0,0.4);
    --shadow-hover: 0 0 25px rgba(88, 166, 255, 0.35);
    --sweep-color: rgba(88, 166, 255, 0.25);
    --dropdown-shadow: 0 10px 40px rgba(0,0,0,0.8);
    --input-bg: #0d1117;
    --input-border: #30363d;
    --bar-track: #0d1117;
}

:root[data-theme="light"] {
    --bg-primary: #f0f2f5;
    --bg-surface: #ffffff;
    --bg-highlight: #f6f8fa;
    --border-color: #d0d7de;
    --text-primary: #1f2328;
    --text-muted: #57606a;
    --accent-primary: #0969da;
    --accent-green: #1a7f37;
    --accent-warning: #9a6700;
    --accent-danger: #cf222e;
    --shadow-card: 0 8px 32px rgba(0,0,0,0.08);
    --shadow-hover: 0 0 25px rgba(9, 105, 218, 0.2);
    --sweep-color: rgba(9, 105, 218, 0.12);
    --dropdown-shadow: 0 10px 40px rgba(0,0,0,0.15);
    --input-bg: #ffffff;
    --input-border: #d0d7de;
    --bar-track: #e1e4e8;
}

* { font-family: 'Inter', -apple-system, sans-serif; }
.glass-card, .metric-box, button, a, input, select, textarea { transition: all 0.3s ease-in-out; }
footer { display: none !important; }

/* === GLOBAL BODY === */
body, .gradio-container {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.main-container { max-width: 1200px; margin: auto; padding: 20px; }

.glass-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 24px;
    box-shadow: var(--shadow-card);
    margin-bottom: 20px;
    overflow: visible !important;
    z-index: 10;
    transition: all 0.35s ease;
    color: var(--text-primary);
}

.glass-card > * {
    position: relative;
    z-index: 2;
}

.glass-card .form, .glass-card .gr-form, .glass-card .gr-box, .glass-card .gr-row, .glass-card .gr-col {
    overflow: visible !important;
}

.gr-dropdown-menu, .gradio-dropdown-menu {
    z-index: 9999 !important;
    overflow: visible !important;
}

#dropdown-priority-row {
    z-index: 500 !important;
    position: relative !important;
}

.gr-dropdown-menu, .gradio-dropdown-menu, .dropdown-menu {
    z-index: 9999 !important;
    position: absolute !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--accent-primary) !important;
    box-shadow: var(--dropdown-shadow) !important;
    display: block !important;
}

.gr-dropdown-arrow, .gradio-dropdown-arrow, .dropdown-arrow-container, .gr-dropdown-button::after {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
}

.gr-dropdown, .gradio-dropdown {
    cursor: pointer !important;
}

.glass-card:hover {
    border-color: var(--accent-primary) !important;
    box-shadow: var(--shadow-hover);
}

.glass-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 60%;
    height: 100%;
    background: linear-gradient(120deg, transparent, var(--sweep-color), transparent);
    transform: skewX(-25deg);
    pointer-events: none !important;
    z-index: 0 !important;
    clip-path: inset(0 round 16px);
}

/* FIX: Ensure interactive elements are clickable without trapping z-index for dropdowns */
.glass-card > * {
    position: relative !important;
    z-index: 10 !important;
}




.glass-card:hover::before {
    left: 130%;
    transition: 0.8s;
}

.metric-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
.metric-box {
    background: var(--bg-highlight);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid var(--border-color);
    transition: all 0.35s ease;
}

.metric-box:hover {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 18px rgba(88, 166, 255, 0.25);
}

.metric-label { font-size: 0.75rem; font-weight: 800; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 2.2rem; font-weight: 800; color: var(--text-primary); margin: 4px 0; }

.status-indicator {
    display: inline-flex; align-items: center; padding: 6px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
}
.status-green { background: rgba(46, 160, 67, 0.1); color: var(--accent-green); border: 1px solid var(--accent-green); }
.status-yellow { background: rgba(210, 153, 34, 0.1); color: var(--accent-warning); border: 1px solid var(--accent-warning); }
.status-red { background: rgba(248, 81, 73, 0.1); color: var(--accent-danger); border: 1px solid var(--accent-danger); }

.pulse { animation: animate-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
@keyframes animate-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }

.gradient-btn {
    background: linear-gradient(135deg, var(--accent-primary) 0%, #2f81f7 100%);
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}
.gradient-btn:hover { filter: brightness(1.1); transform: translateY(-1px); }

.reward-item { display: grid; grid-template-columns: 1fr auto; font-size: 0.9rem; padding: 4px 0; color: var(--text-primary); }

/* FIX: Global pointer-events for all interactive elements */
button, input, select, textarea {
    pointer-events: auto !important;
}

/* === DEEP GRADIO ELEMENT THEMING === */
input:not([type="checkbox"]):not([type="radio"]), textarea, select {
    background: var(--input-bg) !important;
    color: var(--text-primary) !important;
    border-color: var(--input-border) !important;
}
label, .gr-label, span.svelte-1gfkn6j {
    color: var(--text-muted) !important;
}
h1, h2, h3, h4, h5, h6, p, span, li {
    color: var(--text-primary) !important;
}

/* Gradio component internals */
.gradio-container, .gradio-container *:not([style*="color:var(--accent"]) {
    --body-text-color: var(--text-primary) !important;
    --block-title-text-color: var(--text-primary) !important;
    --block-label-text-color: var(--text-muted) !important;
    --input-background-fill: var(--input-bg) !important;
    --background-fill-primary: var(--bg-primary) !important;
    --background-fill-secondary: var(--bg-surface) !important;
    --block-background-fill: var(--bg-surface) !important;
    --border-color-primary: var(--border-color) !important;
    --body-background-fill: var(--bg-primary) !important;
}

/* Buttons */
.gradio-container button.secondary,
.gradio-container button.stop,
.gradio-container button.sm {
    background: var(--bg-highlight) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
}
button, .gr-button {
    color: var(--text-primary) !important;
}
button[variant="primary"], .gr-button-primary {
    background: linear-gradient(135deg, var(--accent-primary) 0%, #2f81f7 100%) !important;
    color: white !important;
    border: none !important;
}
.gradient-btn {
    color: white !important;
}

/* Checkbox */
.gr-check-radio, input[type="checkbox"] {
    accent-color: var(--accent-primary) !important;
    background: transparent !important;
    border: none !important;
    appearance: auto !important;
    -webkit-appearance: auto !important;
}
.gr-checkbox-label, .checkbox-label span {
    color: var(--text-primary) !important;
}

/* Dropdown */
.gr-dropdown, .gradio-dropdown, select, option {
    background: var(--input-bg) !important;
    color: var(--text-primary) !important;
}

/* Textbox */
.gr-textbox, .gr-text-input, .gr-input {
    background: var(--input-bg) !important;
    color: var(--text-primary) !important;
    border-color: var(--input-border) !important;
}
.gr-textbox label, .gr-text-input label {
    color: var(--text-muted) !important;
}

/* Slider */
.gr-slider input[type="range"] {
    accent-color: var(--accent-primary) !important;
}

/* Markdown */
.gr-markdown, .gr-markdown *, .markdown-text, .prose * {
    color: var(--text-primary) !important;
}

/* Plot background */
.gr-plot {
    background: var(--bg-surface) !important;
}

/* Grade panel styling */
.grade-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 8px; font-weight: 700; font-size: 0.85rem;
}
.grade-pass { background: rgba(46,160,67,0.12); color: var(--accent-green) !important; border: 1px solid var(--accent-green); }
.grade-fail { background: rgba(248,81,73,0.12); color: var(--accent-danger) !important; border: 1px solid var(--accent-danger); }
.grade-score { font-size: 1.4rem; font-weight: 900; color: var(--accent-primary) !important; }

/* Responsive */
@media (max-width: 768px) {
    .main-container { padding: 10px; }
    .metric-grid { grid-template-columns: 1fr; }
    .metric-value { font-size: 1.6rem; }
}

/* === NUCLEAR TEXT VISIBILITY FIX === */
/* Override Gradio's deep internal specificity for ALL themes */
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container p,
.gradio-container span,
.gradio-container label,
.gradio-container li,
.gradio-container td,
.gradio-container th,
.gradio-container div:not(.glass-card):not(.metric-box):not(.grade-badge) {
    color: var(--text-primary) !important;
}

/* Preserve accent colors for elements that explicitly set them */
[style*="color:var(--accent"],
[style*="color: var(--accent"],
.grade-pass, .grade-fail, .grade-score,
.status-green, .status-yellow, .status-red,
.gradient-btn {
    color: unset;
}
.gradient-btn { color: white !important; }

/* Fix Gradio's .prose override that kills heading color */
.prose h1, .prose h2, .prose h3, .prose h4, .prose p {
    color: var(--text-primary) !important;
}

/* Gradio separator line */
.gradio-container hr {
    border-color: var(--border-color) !important;
}

/* Fix checkbox label specificity */
.gradio-container label span,
.gradio-container .gr-checkbox label,
.gradio-container input[type="checkbox"] + span {
    color: var(--text-primary) !important;
}
"""

# ==========================================
# THEME MANAGEMENT (data-attribute on :root)
# ==========================================
THEME_JS = """
() => {
    document.documentElement.setAttribute('data-theme', 'dark');
    document.body.style.backgroundColor = '#0d1117';
    document.body.style.color = '#c9d1d9';
    const style = document.createElement('style');
    style.textContent = `
        .gradio-container, .main, .contain, .wrap, body, html {
            background-color: #0d1117 !important;
            color: #c9d1d9 !important;
        }
        .block, .gr-block, .gr-box, .gr-panel, .panel {
            background-color: #161b22 !important;
            border-color: #30363d !important;
        }
    `;
    document.head.appendChild(style);
}
"""

def toggle_theme(mode):
    """Toggle between dark and light themes. Returns [new_mode, unused_html, button_label]."""
    if mode == "light":
        return "dark", "", "☀️"
    else:
        return "light", "", "🌙"

# ==========================================
# CORE LOGIC: SIMULATION & ENVIRONMENT
# ==========================================
# ==========================================
# INTELLIGENCE ENGINE (PREDICTIVE & ANALYTIC)
# ==========================================
def compute_performance_score(state):
    if not state: return 0
    e, s, m, p = [float(state.get(k, 0)) for k in ['energy', 'stress', 'motivation', 'progress']]
    # Performance = Balanced Output (Progress + Health)
    score = (p * 0.45 + (1 - s) * 0.35 + e * 0.1 + m * 0.1) * 100
    return int(max(0, min(100, score)))

def predict_burnout_forecast(state):
    if not state: return "STABLE", "var(--accent-green)"
    s = float(state.get('stress', 0))
    e = float(state.get('energy', 0))
    future_s = s + (0.04 * 8)
    future_e = e - (0.03 * 8)
    if future_s > 0.85 or future_e < 0.15: return "CRITICAL RISK", "var(--accent-danger)"
    if future_s > 0.7: return "HIGH RISK", "var(--accent-warning)"
    return "LOW RISK", "var(--accent-green)"

def get_ai_coach_advice(state, action_idx):
    e, s, m = [float(state.get(k, 0)) for k in ['energy', 'stress', 'motivation']]
    
    # 🧠 Dynamic Actionable Coaching logic
    recs = []
    if s > 0.75:
        recs = ["🔴 Take immediate 10 min break", "🔴 Reduce current workload by 50%", "🔴 Switch to low-intensity manual task"]
    elif e < 0.35:
        recs = ["🟠 Pause for hydration/nutrition", "🟠 Delegate high-energy tasks", "🟠 Focus on 15-min recovery window"]
    elif m < 0.45:
        recs = ["🟡 Change environment/setting", "🟡 Switch to Social or Exercise mode", "🟡 Resume with one small focused task"]
    else:
        recs = ["🟢 Maintain high-performance flow", "🟢 Document current progress", "🟢 Scope next 3 objectives"]
    
    coach_items = "".join([f"<li style='margin-bottom:4px;'>{r}</li>" for r in recs])
    
    coach_text = f"""
    <div style="border-top:1px solid var(--border-color); margin-top:15px; padding-top:10px;">
        <div style="font-size:0.75rem; color:var(--accent-primary); font-weight:800; margin-bottom:8px;">🧠 AI PERFORMANCE COACH</div>
        <ul style="font-size:0.85rem; color:var(--text-primary); padding-left:18px; margin:0; line-height:1.4;">
            {coach_items}
        </ul>
    </div>
    """
    return coach_text

def fetch_telemetry(state):
    if not state: state = {}
    try:
        e, s, m, p = [float(state.get(k, 0)) for k in ['energy', 'stress', 'motivation', 'progress']]
    except (TypeError, ValueError, AttributeError): e, s, m, p = 0, 0, 0, 0
    
    score = compute_performance_score(state)
    forecast_label, forecast_color = predict_burnout_forecast(state)
    stress_alert = "border-color: var(--accent-danger); box-shadow: 0 0 15px rgba(248, 81, 73, 0.2);" if s > 0.8 else ""
    energy_warn = "color: var(--accent-warning);" if e < 0.3 else ""
    
    def bar(val, color):
        w = max(2, min(100, int(val * 100)))
        return f'<div style="width:100%;height:6px;background:var(--bar-track);border-radius:3px;margin-top:8px;overflow:hidden;"><div style="width:{w}%;height:100%;background:{color};border-radius:3px;"></div></div>'

    stress_val_color = "var(--accent-danger)" if s > 0.7 else "var(--text-primary)"
    stress_bar_color = "var(--accent-danger)" if s > 0.7 else "var(--accent-primary)"

    return f"""
    <div style="margin-bottom:15px; display:flex; gap:10px;">
        <div style="flex:1; background:var(--bg-highlight); padding:10px; border-radius:10px; border-left:4px solid var(--accent-primary);">
            <div style="font-size:0.65rem; font-weight:800; color:var(--text-muted); text-transform:uppercase;">Performance Index</div>
            <div style="font-size:1.5rem; font-weight:900; color:var(--text-primary);">{score}</div>
        </div>
        <div style="flex:1; background:var(--bg-highlight); padding:10px; border-radius:10px; border-left:4px solid {forecast_color};">
            <div style="font-size:0.65rem; font-weight:800; color:var(--text-muted); text-transform:uppercase;">Burnout Forecast</div>
            <div style="font-size:0.9rem; font-weight:900; color:{forecast_color};">{forecast_label}</div>
        </div>
    </div>
    <div class="metric-grid" style="{stress_alert}">
        <div class="metric-box">
            <div class="metric-label">Vital Energy</div>
            <div class="metric-value" style="{energy_warn}">{e:.2f}</div>
            {bar(e, 'var(--accent-green)')}
        </div>
        <div class="metric-box">
            <div class="metric-label">Stress Load</div>
            <div class="metric-value" style="color:{stress_val_color}">{s:.2f}</div>
            {bar(s, stress_bar_color)}
        </div>
        <div class="metric-box">
            <div class="metric-label">Motivation</div>
            <div class="metric-value">{m:.2f}</div>
            {bar(m, 'var(--accent-primary)')}
        </div>
        <div class="metric-box">
            <div class="metric-label">Episode Progress</div>
            <div class="metric-value" style="color:var(--accent-warning)">{p:.2f}</div>
            {bar(p, 'var(--accent-warning)')}
        </div>
    </div>
    """

def upgrade_plot(history):
    """
    High-fidelity RL Trajectory Visualization.
    Displays Raw exploration noise + Smoothed trend lines.
    """
    # 1. Performance constraint: Limit history
    max_history = 120
    if len(history) > max_history:
        history = history[-max_history:]

    # 2. Startup Logic: Generate realistic initial trajectory if empty
    if not history:
        fig = Figure(figsize=(7, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor(COLORS['background'])
        ax.set_facecolor(COLORS['surface'])
        ax.text(0.5, 0.5, 'Run simulation to see trajectory',
                ha='center', va='center', color=COLORS['muted'],
                fontsize=12, fontstyle='italic')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        fig.tight_layout()
        return fig

    try:
        fig = Figure(figsize=(7, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor(COLORS['background'])
        ax.set_facecolor(COLORS['surface'])
        
        steps = np.arange(len(history))
        raw_data = np.array([[float(h[k]) for k in ['energy', 'stress', 'motivation', 'progress']] for h in history])
        
        # 3. Add Stochastic Noise to simulate real-time exploration (Gaussian)
        # Apply more noise to earlier steps to simulate instability
        noise_scale = np.linspace(0.02, 0.008, len(steps)) 
        noisy_data = raw_data + np.random.normal(0, 1, raw_data.shape) * noise_scale[:, np.newaxis]
        noisy_data = np.clip(noisy_data, 0, 1)

        # 4. Moving Average Smoothing
        window = 5
        def smooth(y):
            if len(y) < window: return y
            return np.convolve(y, np.ones(window)/window, mode='same')

        labels = ["Energy", "Stress", "Motivation", "Progress"]
        colors = [COLORS['secondary'], COLORS['danger'], COLORS['primary'], COLORS['warning']]
        
        for i in range(4):
            y_raw = noisy_data[:, i]
            y_smooth = smooth(raw_data[:, i])
            
            # Plot Raw Data (exploration noise)
            ax.plot(steps, y_raw, color=colors[i], alpha=0.15, linewidth=1)
            
            # Plot Smoothed Trend (Policy Convergence)
            if i == 3: # Progress is distinct (dashed)
                ax.plot(steps, y_smooth, label=labels[i], color=colors[i], linewidth=3, linestyle='--', alpha=0.9)
            else:
                ax.plot(steps, y_smooth, label=labels[i], color=colors[i], linewidth=2, alpha=0.8)
            
            # 5. Confidence Band for Energy (Environment Stochasticity)
            if i == 0:
                ax.fill_between(steps, y_smooth - 0.04, y_smooth + 0.04, color=colors[i], alpha=0.1)

        # 6. Burnout Event Visualization (Markers)
        spikes = np.where(raw_data[:, 1] > 0.85)[0]
        if len(spikes) > 0:
            ax.scatter(spikes, raw_data[spikes, 1], color=COLORS['danger'], s=50, 
                       marker='x', label="Burnout Risk", zorder=10)

        # 7. Axes & HUD Styling
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.03, color=COLORS['text'])
        ax.legend(loc='upper right', facecolor=COLORS['background'], edgecolor=COLORS['border'], 
                  labelcolor=COLORS['text'], fontsize=6, ncol=2)
        ax.tick_params(colors=COLORS['muted'], labelsize=7)
        for s in ax.spines.values(): s.set_edgecolor(COLORS['border'])
        
        fig.tight_layout()
        return fig
    except Exception as ex:
        # Fallback for plotting errors
        empty_fig = Figure(figsize=(7, 3.5))
        ax = empty_fig.add_subplot(111)
        ax.text(0.5, 0.5, 'PLOT INITIALIZING...', ha='center', color='#8b949e')
        ax.axis('off')
        return empty_fig

def get_reward_explanation(state, prev_p, reward, mode="step"):
    try:
        if mode == "injection":
            quality = (state['energy'] * 0.4 + (1 - state['stress']) * 0.4 + state['motivation'] * 0.2) * 100
            diag = "HEALTHY" if quality > 60 else "STRESSFUL" if quality > 30 else "CRITICAL"
            explain_text = f"""
            <div style="margin-top:15px; padding:12px; background:rgba(88,166,255,0.05); border-radius:8px; border:1px solid rgba(88,166,255,0.2);">
                <div style="font-weight:bold; color:var(--accent-primary); margin-bottom:5px; font-size:0.75rem;">STATE QUALITY DIAGNOSTIC</div>
                <p style="margin:0; font-size:0.8rem; color:var(--text-muted); line-height:1.4;">
                    Injected logic: System is currently in a <b>{diag}</b> configuration. 
                    Calculated State Health: <b>{quality:.1f}%</b>. High-fidelity analytics restored for new trajectory.
                </p>
            </div>
            """
            return explain_text
            
        p_delta = (state['progress'] - prev_p)
        p_gain = p_delta * 1.5
        s_pen = -(state['stress'] ** 2)
        wellbeing = (state['energy'] * 0.3 + (1.0 - state['stress']) * 0.3 + state['motivation'] * 0.2)
        wb_bonus = wellbeing * 0.15
        mot_bonus = state['motivation'] * 0.1
        
        def row(label, val, color, detail=""):
            sign = "+" if val >= 0 else ""
            val_color = 'var(--accent-green)' if val >= 0 else 'var(--accent-danger)'
            det_html = f'<div style="font-size:0.65rem; color:var(--text-muted);">{detail}</div>' if detail else ""
            return f'''
            <div style="margin-bottom:8px;">
                <div class="reward-item">
                    <span style="color:var(--text-primary);">{label}</span>
                    <span style="color:{val_color};font-weight:bold;font-size:1.05rem;">{sign}{val:.4f}</span>
                </div>
                {det_html}
            </div>
            '''

        reward_color = 'var(--accent-green)' if reward > 0 else 'var(--accent-danger)'

        explain_text = f"""
        <div style="margin-top:15px; padding:12px; background:rgba(88,166,255,0.05); border-radius:8px; border:1px solid rgba(88,166,255,0.2);">
            <div style="font-weight:bold; color:var(--accent-primary); margin-bottom:5px; font-size:0.75rem;">RL STRATEGY DIAGNOSTIC</div>
            <p style="margin:0; font-size:0.8rem; color:var(--text-muted); line-height:1.4;">
                State transition analysis shows <b>{'Optimal Progress' if p_delta > 0.02 else 'Resource Recovery'}</b> bias. 
                Agent is currently penalizing <b>Stress Accumulation</b> at a quadratic (s²) scale to prevent long-term trajectory collapse.
            </p>
        </div>
        """

        explain_header = f"""
        <div style="background:var(--bg-highlight);border:1px solid var(--border-color);border-radius:10px;padding:16px;">
            <div style="font-size:0.75rem;font-weight:800;color:var(--text-muted);margin-bottom:12px;border-bottom:1px solid var(--border-color);padding-bottom:10px;">HIGH-FIDELITY RL REWARD DECOMPOSITION</div>
"""
        explain_body = f"""
            {row("Progress Velocity (Δp)", p_gain, 'var(--accent-green)', f"Δp: {p_delta:+.3f} | Weight: 1.5x")}
            {row("Stress Penalty (s²)", s_pen, 'var(--accent-danger)', f"Stress: {state['stress']:.3f} | Quadratic")}
            {row("Wellbeing Equilibrium", wb_bonus, 'var(--accent-primary)', f"E·0.3 + (1-S)·0.3 + M·0.2 = {wellbeing:.2f} | Coeff: 0.15")}
            {row("Motivation Bonus", mot_bonus, 'var(--accent-warning)', f"Motivation: {state['motivation']:.3f} | Weight: 0.1")}
            
            <div style="margin-top:10px;padding-top:10px;border-top:2px solid var(--border-color);display:flex;justify-content:space-between;align-items:center;">
                <div style="display:flex; flex-direction:column;">
                    <span style="font-size:0.6rem; color:var(--text-muted); font-weight:800;">TOTAL SCALAR REWARD</span>
                    <span style="font-weight:800;color:var(--text-primary)">AGGREGATED SUM</span>
                </div>
                <span style="font-size:1.4rem;font-weight:900;color:{reward_color}">{reward:+.4f}</span>
            </div>
        </div>
        """
        return explain_header + explain_body + explain_text
    except Exception: return "<div style='padding:20px; color:var(--text-muted);'>Diagnostic Engine Initializing...</div>"

# ==========================================
# HANDLERS: ACTIONS & AI
# ==========================================
def reset_core(env_obj=None):
    try:
        if env_obj is None:
            env_obj = environment.ProductivityEnvironment()
        obs = env_obj.reset()
        state = {'energy': float(obs[0]), 'stress': float(obs[1]), 'motivation': float(obs[2]), 'progress': float(obs[3]), 'done': False, 'step': 0, 'reward': 0.0}
        history = [state]
        return env_obj, history, state, upgrade_plot(history), fetch_telemetry(state), "ENV INITIALIZED", "", "🟡 STANDBY", "Ready for instruction."
    except Exception as e: return env_obj or environment.ProductivityEnvironment(), [], {}, upgrade_plot([]), fetch_telemetry(None), f"ERROR: {str(e)}", "", "🔴 OFFLINE", ""

def step_core(action_idx, env_obj, history, state):
    try:
        if not history or not isinstance(history, list): 
            return reset_core(env_obj)
        
        if state.get('done', False): 
            return env_obj, history, state, upgrade_plot(history), fetch_telemetry(state), "TERMINAL REACHED. RESET CORE.", "", "🟡 STANDBY", ""
        
        prev_p = float(state.get('progress', 0.0))
        obs, reward, done, info = env_obj.step(int(action_idx))
        
        new_state = {
            'energy': float(obs[0]), 
            'stress': float(obs[1]), 
            'motivation': float(obs[2]), 
            'progress': float(obs[3]), 
            'done': bool(done), 
            'step': int(info.get('step_count', 0)), 
            'reward': float(reward or 0.0)
        }
        
        history.append(new_state)
        act_name = ACTION_MAP.get(int(action_idx), "Unknown")
        status = f"Step {new_state['step']} | {act_name} | Rwd: {new_state['reward']:.3f} | Done: {new_state['done']}"
        
        return env_obj, history, new_state, upgrade_plot(history), fetch_telemetry(new_state), status, get_reward_explanation(new_state, prev_p, new_state['reward']), "🟢 ACTIVE", ""
    except Exception as e: 
        return env_obj, history, state, upgrade_plot(history), fetch_telemetry(state), f"SYSTEM EXCEPTION: {str(e)}", "", "🔴 ERR", ""

def handle_ai_intelligence(use_gemini, env_obj, history, state):
    """
    Non-blocking AI inference handler. Yields 'Thinking' state before execution.
    """
    try:
        if env_obj is None:
            env_obj = environment.ProductivityEnvironment()
        
        # Surgical boolean normalization against Gradio serialization bugs
        if not isinstance(use_gemini, bool):
            use_gemini = str(use_gemini).strip().lower() in ("true", "1", "t", "y", "yes") if use_gemini not in (None, "") else False

        # === GEMINI DEBUG BLOCK ===
        api_key = os.environ.get("GEMINI_API_KEY", "")
        print("KEY EXISTS:", bool(api_key))
        print(f"[AI DEBUG] Gemini checkbox: {use_gemini} (type={type(use_gemini).__name__})")
        print(f"[AI DEBUG] API key present: {bool(api_key and api_key != 'your_key_here')} (len={len(api_key)})")
        
        if not history or state.get('done', False): 
            ev, h, s, p, t, st, ex, tag, res = step_core(0, env_obj, history, state)
            yield ev, h, s, p, t, st, ex, tag, res
            return

        yield env_obj, history, state, upgrade_plot(history), fetch_telemetry(state), "AI AGENT IS ANALYZING STATE...", "", "🟠 THINKING", ""

        e, s, m, p = [state[k] for k in ['energy', 'stress', 'motivation', 'progress']]
        
        if use_gemini is False:
            print("[AI DEBUG] → Rule-based path (checkbox OFF)")
            action_val, reason = rule_based_action(e, s, m, p)
            decision = {'action': action_val, 'reason': reason, 'source': 'fast_rule'}
            act_idx = int(decision['action'])
            ev, h, s, p, t, st, ex, tag, res = step_core(act_idx, env_obj, history, state)
            yield ev, h, s, p, t, st, ex, tag, res
            return
        
        # Gemini path — validate API key first
        elapsed = 0.0
        if not api_key or api_key == "your_key_here":
            print("[AI DEBUG] ⚠️ No valid GEMINI_API_KEY — falling back to rules")
            action_val, reason = rule_based_action(e, s, m, p)
            decision = {'action': action_val, 'reason': f"⚠️ No API key set. {reason}", 'source': 'no_key_fallback'}
            status_tag = "🟡 No API Key"
        else:
            print(f"[AI DEBUG] → Gemini path (checkbox ON, key present)")
            future = executor.submit(get_action, e, s, m, p, use_gemini)
            
            start = time.time()
            try:
                decision = future.result(timeout=4.0)
                source = decision.get('source', 'unknown')
                print(f"[AI DEBUG] ✅ Inference returned: source={source}, action={decision.get('action')}")
                if source == 'gemini':
                    status_tag = "🟢 Gemini Active"
                elif source == 'rule_based':
                    status_tag = "🟡 Fallback (API issue)"
                else:
                    status_tag = f"🟡 {source}"
            except Exception as timeout_ex:
                print(f"[AI DEBUG] ❌ Timeout/Error: {timeout_ex}")
                act_val, rsn = rule_based_action(e, s, m, p)
                decision = {'action': act_val, 'reason': rsn, 'source': 'timeout_fallback'}
                status_tag = "🟡 Timeout Fallback"
            elapsed = time.time() - start
        act_idx = int(decision['action'])
        
        env_obj, h, n_s, plot, tele, status, expl, _, _ = step_core(act_idx, env_obj, history, state)
        coach_tips = get_ai_coach_advice(n_s, act_idx)
        
        reasoning_html = f"""
        <div style="background:var(--bg-surface);border:1px solid var(--border-color);border-radius:10px;padding:12px;margin-top:10px;">
            <div style="font-size:0.7rem;color:var(--text-muted);font-weight:700;margin-bottom:4px;">INFERENCE ORIGIN: {decision['source'].upper()} (Latency: {elapsed:.2f}s)</div>
            <div style="font-weight:bold;color:var(--accent-primary);margin-bottom:4px;">ACTION: {ACTION_MAP[act_idx].upper()}</div>
            <div style="font-size:0.85rem;color:var(--text-primary);line-height:1.4;margin-bottom:10px;">{decision.get('reason', 'Optimizing trajectory for long-term progress.')}</div>
            {coach_tips}
        </div>
        """
        yield env_obj, h, n_s, plot, tele, status, expl, status_tag, reasoning_html

    except Exception as ex: 
        yield env_obj, history, state, upgrade_plot(history), fetch_telemetry(state), f"AI ERROR: {str(ex)}", "", "🔴 ERR", ""

def simulate_batch(mode, steps, use_gemini, env_obj, history, state):
    """
    Streaming simulation loop. Yields intermediate states for zero-lag UI feedback.
    """
    if env_obj is None:
        env_obj = environment.ProductivityEnvironment()
    if not history: 
        env_obj, history, state, *_ = reset_core(env_obj)
        yield env_obj, history, state, upgrade_plot(history), fetch_telemetry(state), "STREAMING SIMULATION STARTED...", "", "🟡 BUSY", ""

    try:
        current_h = history.copy()
        current_s = state.copy()
        
        total_steps = int(steps)
        for i in range(total_steps):
            if current_s.get('done', False): break
            
            if mode == "AI":
                e, s, m, p = [current_s[k] for k in ['energy', 'stress', 'motivation', 'progress']]
                future = executor.submit(get_action, e, s, m, p, use_gemini)
                try: d = future.result(timeout=4.5)
                except Exception:
                    act_val, rsn = rule_based_action(e, s, m, p)
                    d = {'action': act_val, 'reason': rsn}
                action = int(d['action'])
            else:
                action = random.randint(0, 4)
            
            env_obj, current_h, current_s, plot, telemetry, status, expl, tag, reasoning = step_core(action, env_obj, current_h, current_s)
            
            if total_steps < 10 or i % 2 == 0 or i == total_steps - 1:
                display_status = f"BATCH {mode} | Processing Step {current_s['step']}..."
                yield env_obj, current_h, current_s, plot, telemetry, display_status, expl, "🟠 BUSY", reasoning
            
        yield env_obj, current_h, current_s, upgrade_plot(current_h), fetch_telemetry(current_s), f"BATCH {mode} COMPLETED ({total_steps} steps)", get_reward_explanation(current_s, history[-1]['progress'] if history else 0, current_s['reward']), "🟢 IDLE", ""
    except Exception as ex: 
        yield env_obj, history, state, upgrade_plot(history), fetch_telemetry(state), f"BATCH FAIL: {str(ex)}", "", "🔴 ERR", ""

def handle_chat(message, env_obj, history, state):
    """
    Deterministic Command Console.
    """
    if env_obj is None:
        env_obj = environment.ProductivityEnvironment()
    if not message: 
        return env_obj, history, state, upgrade_plot(history), fetch_telemetry(state), "🟢 LOGGING", ""
    
    msg = message.lower().strip()
    
    if msg == "reset":
        ev, h, s, p, t, _, _, status, reasoning = reset_core(env_obj)
        return ev, h, s, p, t, status, reasoning
    
    if msg == "increase energy":
        env_obj.energy = float(np.clip(env_obj.energy + 0.2, 0, 1))
    elif msg == "reduce stress":
        env_obj.stress = float(np.clip(env_obj.stress - 0.2, 0, 1))
    
    new_state = {
        'energy': env_obj.energy, 
        'stress': env_obj.stress, 
        'motivation': env_obj.motivation, 
        'progress': env_obj.progress, 
        'done': state.get('done', False), 
        'step': env_obj.step_count, 
        'reward': state.get('reward', 0.0)
    }
    
    if history:
        history[-1] = new_state 
    else:
        history = [new_state]
    
    return env_obj, history, new_state, upgrade_plot(history), fetch_telemetry(new_state), "🟢 LOGGING", ""

def apply_custom_scenario(e, s, m, p, env_obj, history, state):
    try:
        if env_obj is None:
            env_obj = environment.ProductivityEnvironment()
        env_obj.energy, env_obj.stress, env_obj.motivation, env_obj.progress = e, s, m, p
        new_state = {'energy': e, 'stress': s, 'motivation': m, 'progress': p, 'done': False, 'step': env_obj.step_count, 'reward': 0.0, 'is_injection': True}
        history.append(new_state)
        
        coach_advice = get_ai_coach_advice(new_state, 0)
        reward_diag = get_reward_explanation(new_state, 0, 0, mode="injection")
        
        reasoning_html = f"""
        <div style="background:var(--bg-surface);border:1px solid var(--border-color);border-radius:10px;padding:12px;margin-top:10px;">
            <div style="font-size:0.7rem;color:var(--text-muted);font-weight:700;margin-bottom:4px;">EVENT: STOCHASTIC INJECTION</div>
            <div style="font-size:0.85rem;color:var(--text-primary);line-height:1.4;margin-bottom:10px;">New state vector applied. Re-analyzing future risk and performance coaching strategy.</div>
            {coach_advice}
        </div>
        """
        
        return env_obj, history, new_state, upgrade_plot(history), fetch_telemetry(new_state), "CUSTOM SCENARIO INJECTED.", reward_diag, "🟡 SYNCED", reasoning_html
    except Exception as ex: return env_obj, history, state, upgrade_plot(history), fetch_telemetry(state), f"INJECTION ERROR: {str(ex)}", "", "🔴 ERR", ""

# ==========================================
# EPISODE GRADING (Exposes grader.py in UI)
# ==========================================
def grade_current_episode(history):
    """Grade the current trajectory using the grader."""
    try:
        if not history or len(history) < 2:
            return "<div style='color:var(--text-muted); padding:10px;'>Run at least 2 steps to grade the episode.</div>"
        
        recorder = grader.TrajectoryRecorder()
        for i, h in enumerate(history):
            state_arr = np.array([h['energy'], h['stress'], h['motivation'], h['progress']], dtype=np.float32)
            action = 0
            reward = h.get('reward', 0.0)
            info = {'burnout_counter': 0, 'fatigue_level': 0}
            if i > 0:
                recorder.record(state_arr, action, reward, info)
        
        results = {}
        for diff in ['easy', 'medium', 'hard']:
            results[diff] = grader.grade_trajectory(recorder, diff)
        
        def grade_row(label, result):
            passed = result['passed']
            score = result['score']
            badge_class = 'grade-pass' if passed else 'grade-fail'
            icon = '✅' if passed else '❌'
            pct = int(score * 100)
            bar_color = 'var(--accent-green)' if passed else 'var(--accent-danger)'
            return f"""
            <div style="display:flex; justify-content:space-between; align-items:center; padding:10px; margin-bottom:8px; background:var(--bg-highlight); border-radius:8px; border:1px solid var(--border-color);">
                <div>
                    <div style="font-weight:800; font-size:0.85rem; color:var(--text-primary);">{label}</div>
                    <div class="grade-badge {badge_class}" style="margin-top:4px;">{icon} {'PASS' if passed else 'FAIL'}</div>
                </div>
                <div style="text-align:right;">
                    <div class="grade-score">{pct}%</div>
                    <div style="width:80px; height:6px; background:var(--bar-track); border-radius:3px; margin-top:4px; overflow:hidden;">
                        <div style="width:{pct}%; height:100%; background:{bar_color}; border-radius:3px;"></div>
                    </div>
                </div>
            </div>
            """
        
        html = f"""
        <div style="background:var(--bg-surface); border-radius:10px; padding:4px;">
            {grade_row('🟢 EASY — Progress > 0.8', results['easy'])}
            {grade_row('🟡 MEDIUM — Progress > 0.8 & Stress < 0.6', results['medium'])}
            {grade_row('🔴 HARD — Progress > 0.9 & Stress never > 0.85', results['hard'])}
            <div style="margin-top:8px; padding:8px; font-size:0.7rem; color:var(--text-muted); border-top:1px solid var(--border-color);">
                Steps: {recorder.num_steps} | Total Reward: {recorder.total_reward:.2f} | Peak Stress: {recorder.max_stress:.2f}
            </div>
        </div>
        """
        return html
    except Exception as ex:
        return f"<div style='color:var(--accent-danger); padding:10px;'>Grading Error: {str(ex)}</div>"

# BONUS: MATH AI
def handle_math_ai(problem):
    try:
        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        p = f"Solve this math problem: {problem}. Be concise. Show only the answer and one sentence of explanation."
        r = client.models.generate_content(model="gemini-2.0-flash", contents=p)
        res = r.text
        math_diag_html = "<div style='background:var(--bg-highlight); border:1px solid var(--border-color); border-radius:10px; padding:12px;'><div style='font-size:0.7rem; color:var(--text-muted); font-weight:800; margin-bottom:4px;'>COGNITIVE LOAD ANALYSIS</div><div style='font-size:1.1rem; font-weight:700; color:var(--accent-primary);'>HIGH LOAD</div><div style='font-size:0.8rem; color:var(--text-muted);'>Processing logic-heavy mathematical primitives.</div></div>"
        return res, math_diag_html
    except Exception:
        try:
            import math
            import numpy as np
            res_val = eval(problem, {"__builtins__": None, "math": math, "np": np})
            res = f"Answer: {res_val} (Calculated via Python Fallback Engine)"
            math_diag_html = "<div style='background:var(--bg-highlight); border:1px solid var(--border-color); border-radius:10px; padding:12px;'><div style='font-size:0.7rem; color:var(--text-muted); font-weight:800; margin-bottom:4px;'>COGNITIVE LOAD ANALYSIS</div><div style='font-size:1.1rem; font-weight:700; color:var(--accent-green);'>SYSTEMIC EVAL</div><div style='font-size:0.8rem; color:var(--text-muted);'>Local compute utilized for arithmetic resolution.</div></div>"
            return res, math_diag_html
        except Exception: return "Error: Unable to process mathematical query.", "Computation Error"

# ==========================================
# UI ASSEMBLY (ELITE DASHBOARD)
# ==========================================
dark_theme = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="gray",
).set(
    body_background_fill="#0d1117",
    body_background_fill_dark="#0d1117",
    body_text_color="#c9d1d9",
    body_text_color_dark="#c9d1d9",
    block_background_fill="#161b22",
    block_background_fill_dark="#161b22",
    block_border_color="#30363d",
    block_border_color_dark="#30363d",
    block_title_text_color="#c9d1d9",
    block_title_text_color_dark="#c9d1d9",
    block_label_text_color="#8b949e",
    block_label_text_color_dark="#8b949e",
    input_background_fill="#0d1117",
    input_background_fill_dark="#0d1117",
    input_border_color="#30363d",
    input_border_color_dark="#30363d",
    button_primary_background_fill="#58a6ff",
    button_primary_background_fill_dark="#58a6ff",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    background_fill_primary="#0d1117",
    background_fill_primary_dark="#0d1117",
    background_fill_secondary="#161b22",
    background_fill_secondary_dark="#161b22",
    border_color_primary="#30363d",
    border_color_primary_dark="#30363d",
)
with gr.Blocks(title="Adaptive AI Productivity Engine") as demo:
    
    with gr.Column(elem_classes="main-container"):
        # --- HEADER SECTION ---
        with gr.Row():
            with gr.Column(scale=8):
                gr.HTML("""
                <div style='margin-bottom:20px;'>
                    <h1 style='margin:0; font-size:2.4rem; font-weight:800; letter-spacing:-0.04em; color:var(--text-primary);'>Adaptive AI Productivity Engine</h1>
                    <p style='margin:0; color:var(--text-muted); font-weight:600;'>Advanced Human Performance Intelligence System v2.0</p>
                </div>
                """)
            with gr.Column(scale=2):
                with gr.Row():
                    btn_theme = gr.Button("☀️", scale=1, visible=False) # Hidden to preserve dark premium UI
                    ai_status = gr.HTML("<div class='status-indicator status-yellow pulse'>🟡 STANDBY</div>")
                theme_html = gr.HTML("")

        gr.Markdown("---")
        
        sys_history = gr.State([])
        sys_state = gr.State({})
        sys_env = gr.State(None)
        theme_state = gr.State("dark")

        with gr.Row():
            # --- LEFT: REAL-TIME TELEMETRY ---
            with gr.Column(scale=5):
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("### 📡 REAL-TIME TELEMETRY")
                    telemetry_view = gr.HTML(fetch_telemetry(None))
                
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("### 🧬 SCENARIO CONTROL (STOCHASTIC INJECTION)")
                    with gr.Row():
                        sl_e = gr.Slider(0, 1, 0.8, step=0.05, label="Vital Energy")
                        sl_s = gr.Slider(0, 1, 0.2, step=0.05, label="Stress Load")
                    with gr.Row():
                        sl_m = gr.Slider(0, 1, 0.7, step=0.05, label="Motivation")
                        sl_p = gr.Slider(0, 1, 0.0, step=0.05, label="Progress")
                    btn_apply = gr.Button("APPLY SCENARIO", elem_classes="gradient-btn")

            # --- RIGHT: PERFORMANCE TRAJECTORY ---
            with gr.Column(scale=7):
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("### 📈 PERFORMANCE TRAJECTORY")
                    plot_view = gr.Plot()
                
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("### ⚙️ SIMULATION ENGINE")
                    with gr.Row(elem_id="dropdown-priority-row"):
                        act_drop = gr.Dropdown(list(ACTION_MAP.values()), label="Select Manual Action", value="📚 Study", scale=2)
                        btn_step = gr.Button("EXECUTE STEP", variant="primary", scale=1)
                    with gr.Row():
                        btn_sim_ai = gr.Button("🤖 AUTO AI (10)", variant="secondary")
                        btn_sim_rnd = gr.Button("🎲 AUTO RND (10)", variant="secondary")
                        btn_reset = gr.Button("🔄 REBOOT SYSTEM", variant="stop")

        with gr.Row():
            # --- BOTTOM LEFT: AI INTELLIGENCE ---
            with gr.Column(scale=6):
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("### 🧠 AI DECISION INTELLIGENCE")
                    with gr.Row():
                        check_gemini = gr.Checkbox(label="Neural Gemini Mode", value=False, interactive=True)
                        btn_ai_single = gr.Button("EXECUTE AI INFERENCE", variant="primary")
                    ai_reasoning = gr.HTML("<div style='color:#8b949e; padding:10px;'>Awaiting neural decision...</div>")
                
                with gr.Row():
                    with gr.Column(elem_classes="glass-card"):
                        gr.Markdown("### 📊 SYSTEM LOG")
                        sys_log = gr.Textbox(label="", value="READY.", interactive=True, placeholder="Ask assistant or command (e.g., 'reduce stress')")
                    
                    with gr.Column(elem_classes="glass-card"):
                        gr.Markdown("### 🧮 MATH AI (BONUS)")
                        math_input = gr.Textbox(label="Math Problem", placeholder="e.g. 52 * 12 + sqrt(144)")
                        math_btn = gr.Button("SOLVE", size="sm")
                        math_out = gr.Textbox(label="Result", interactive=False)
                        math_diag = gr.HTML("<div style='color:#8b949e; padding:10px;'>Solve a problem to see analysis...</div>")

            # --- BOTTOM RIGHT: REWARD DIAGNOSTICS ---
            with gr.Column(scale=4):
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("### 💰 REWARD EXPLAINER")
                    reward_expl = gr.HTML("<div style='color:#8b949e; padding:10px;'>Execute action to analyze rewards...</div>")
                    gr.Markdown("<p style='font-size:0.7rem; color:#8b949e; margin-top:10px;'>Reinforcement Learning Reward Formula: <i>Gain*1.5 - Stress² + Wellbeing×0.15 + Motivation×0.1</i></p>")
                
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("### 🏆 EPISODE GRADING")
                    btn_grade = gr.Button("GRADE EPISODE", elem_classes="gradient-btn")
                    grade_view = gr.HTML("<div style='color:#8b949e; padding:10px;'>Run a simulation, then grade your trajectory.</div>")

    # ==========================
    # EVENT BINDINGS (Session-Isolated)
    # ==========================
    btns = [btn_apply, btn_step, btn_sim_ai, btn_sim_rnd, btn_reset, btn_ai_single, math_btn]
    
    def ui_busy(): return [gr.update(interactive=False)] * len(btns)
    def ui_ready(): return [gr.update(interactive=True)] * len(btns)
    
    # All handlers now return env_obj as first value
    out_all = [sys_env, sys_history, sys_state, plot_view, telemetry_view, sys_log, reward_expl, ai_status, ai_reasoning]
    out_chat = [sys_env, sys_history, sys_state, plot_view, telemetry_view, ai_status, ai_reasoning]

    btn_reset.click(fn=ui_busy, outputs=btns).then(fn=reset_core, inputs=[sys_env], outputs=out_all).then(fn=ui_ready, outputs=btns)
    
    btn_apply.click(fn=ui_busy, outputs=btns).then(fn=apply_custom_scenario, inputs=[sl_e, sl_s, sl_m, sl_p, sys_env, sys_history, sys_state], outputs=out_all).then(fn=ui_ready, outputs=btns)
    
    act_rev = {v: k for k, v in ACTION_MAP.items()}
    btn_step.click(fn=ui_busy, outputs=btns).then(fn=lambda a, ev, h, s: step_core(act_rev[a], ev, h, s), inputs=[act_drop, sys_env, sys_history, sys_state], outputs=out_all).then(fn=ui_ready, outputs=btns)
    
    btn_sim_ai.click(fn=simulate_batch, inputs=[gr.State("AI"), gr.State(10), check_gemini, sys_env, sys_history, sys_state], outputs=out_all)
    btn_sim_rnd.click(fn=simulate_batch, inputs=[gr.State("RND"), gr.State(10), check_gemini, sys_env, sys_history, sys_state], outputs=out_all)
    
    btn_ai_single.click(fn=handle_ai_intelligence, inputs=[check_gemini, sys_env, sys_history, sys_state], outputs=out_all)
    
    TOGGLE_JS = "(mode, html, label) => { document.documentElement.setAttribute('data-theme', mode); return [mode, html, label]; }"
    btn_theme.click(fn=toggle_theme, inputs=theme_state, outputs=[theme_state, theme_html, btn_theme], js=TOGGLE_JS, queue=False)

    math_btn.click(fn=handle_math_ai, inputs=math_input, outputs=[math_out, math_diag])
    
    btn_grade.click(fn=grade_current_episode, inputs=[sys_history], outputs=[grade_view])
    
    sys_log.submit(fn=handle_chat, inputs=[sys_log, sys_env, sys_history, sys_state], outputs=out_chat)

    demo.load(fn=reset_core, inputs=[sys_env], outputs=out_all)

# ==========================
# LAUNCH GRADIO 6.0 READY
# ==========================
if __name__ == "__main__":
    demo.queue(default_concurrency_limit=20).launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=dark_theme,
        css=premium_css,
        js=THEME_JS
    )
