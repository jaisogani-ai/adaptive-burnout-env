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

# ==========================================
# IMPORTS & GLOBAL REGISTRY
# ==========================================
import environment
import grader
from inference import get_action, rule_based_action

# Initialization
env = environment.ProductivityEnvironment()
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
# PREMIUM FUTURISTIC CSS
# ==========================================
premium_css = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

* {{ font-family: 'Inter', -apple-system, sans-serif; }}
.glass-card, .metric-box, button, a, input, select {{ transition: all 0.25s ease-in-out; }}
body {{ background-color: {COLORS['background']} !important; }}

.main-container {{ max-width: 1200px; margin: auto; padding: 20px; }}

.glass-card {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 20px;
    /* Critical: Ensure dropdowns and menus can expand outside the card */
    overflow: visible !important;
    z-index: 10;
    transition: all 0.35s ease;
}}

/* Elevate all content inside cards above the scanning sweep (z-index: 1) */
.glass-card > * {{
    position: relative;
    z-index: 2;
}}

/* Deep force visibility for Gradio internal layouts */
.glass-card .form, .glass-card .gr-form, .glass-card .gr-box, .glass-card .gr-row, .glass-card .gr-col {{
    overflow: visible !important;
}}

/* Specifically target Gradio dropdown menu overflow */
.gr-dropdown-menu, .gradio-dropdown-menu {{
    z-index: 9999 !important;
    overflow: visible !important;
}}

/* Force the priority row to be on top of other rows */
#dropdown-priority-row {{
    z-index: 500 !important;
    position: relative !important;
}}

/* ENSURE DROPDOWN MENU FLOATS ABOVE ALL BUTTONS */
.gr-dropdown-menu, .gradio-dropdown-menu, .dropdown-menu {{
    z-index: 9999 !important;
    position: absolute !important;
    background: {COLORS['surface']} !important;
    border: 1px solid {COLORS['primary']} !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.8) !important;
    display: block !important;
}}

/* HIDE THE ANNOYING SMALL SELECTION SYMBOL AS REQUESTED */
.gr-dropdown-arrow, .gradio-dropdown-arrow, .dropdown-arrow-container, .gr-dropdown-button::after {{
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
}}

/* Ensure the entire selection box is the trigger */
.gr-dropdown, .gradio-dropdown {{
    cursor: pointer !important;
}}

/* 🔥 PREMIUM HOVER EFFECT (Displacement removed for selection stability) */
.glass-card:hover {{
    border-color: {COLORS['primary']} !important;
    box-shadow: 0 0 25px rgba(88, 166, 255, 0.35);
}}

/* 🔮 Moving blue light sweep */
.glass-card::before {{
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 60%;
    height: 100%;
    background: linear-gradient(
        120deg,
        transparent,
        rgba(88, 166, 255, 0.25),
        transparent
    );
    transform: skewX(-25deg);
    pointer-events: none;
    clip-path: inset(0 round 16px);
    z-index: 1;
}}

.glass-card:hover::before {{
    left: 130%;
    transition: 0.8s;
}}

.metric-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
.metric-box {{
    background: {COLORS['highlight']};
    border-radius: 12px;
    padding: 16px;
    border: 1px solid {COLORS['border']};
    transition: all 0.35s ease;
}}

.metric-box:hover {{
    border-color: {COLORS['primary']} !important;
    box-shadow: 0 0 18px rgba(88, 166, 255, 0.25);
}}

.metric-label {{ font-size: 0.75rem; font-weight: 800; color: {COLORS['muted']}; text-transform: uppercase; letter-spacing: 0.05em; }}
.metric-value {{ font-size: 2.2rem; font-weight: 800; color: {COLORS['text']}; margin: 4px 0; }}

.status-indicator {{
    display: inline-flex; align-items: center; padding: 6px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
}}
.status-green {{ background: rgba(46, 160, 67, 0.1); color: {COLORS['secondary']}; border: 1px solid {COLORS['secondary']}; }}
.status-yellow {{ background: rgba(210, 153, 34, 0.1); color: {COLORS['warning']}; border: 1px solid {COLORS['warning']}; }}
.status-red {{ background: rgba(248, 81, 73, 0.1); color: {COLORS['danger']}; border: 1px solid {COLORS['danger']}; }}

.pulse {{ animation: animate-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }}
@keyframes animate-pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: .5; }} }}

.gradient-btn {{
    background: linear-gradient(135deg, {COLORS['primary']} 0%, #2f81f7 100%);
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}}
.gradient-btn:hover {{ filter: brightness(1.1); transform: translateY(-1px); }}

.reward-item {{ display: grid; grid-template-columns: 1fr auto; font-size: 0.9rem; padding: 4px 0; }}
"""

# ==========================================
# CORE LOGIC: SIMULATION & ENVIRONMENT
# ==========================================
def fetch_telemetry(state):
    if not state: state = {}
    try:
        e, s, m, p = [float(state.get(k, 0)) for k in ['energy', 'stress', 'motivation', 'progress']]
    except (TypeError, ValueError, AttributeError): e, s, m, p = 0, 0, 0, 0
    
    stress_alert = "border-color: #f85149; box-shadow: 0 0 15px rgba(248, 81, 73, 0.2);" if s > 0.8 else ""
    energy_warn = f"color: {COLORS['warning']};" if e < 0.3 else ""
    
    def bar(val, color):
        w = max(2, min(100, int(val * 100)))
        return f'<div style="width:100%;height:6px;background:#0d1117;border-radius:3px;margin-top:8px;overflow:hidden;"><div style="width:{w}%;height:100%;background:{color};border-radius:3px;"></div></div>'

    return f"""
    <div class="metric-grid" style="{stress_alert}">
        <div class="metric-box">
            <div class="metric-label">Vital Energy</div>
            <div class="metric-value" style="{energy_warn}">{e:.2f}</div>
            {bar(e, COLORS['secondary'])}
        </div>
        <div class="metric-box">
            <div class="metric-label">Stress Load</div>
            <div class="metric-value" style="color:{COLORS['danger'] if s > 0.7 else COLORS['text']}">{s:.2f}</div>
            {bar(s, COLORS['danger'] if s > 0.7 else COLORS['primary'])}
        </div>
        <div class="metric-box">
            <div class="metric-label">Motivation</div>
            <div class="metric-value">{m:.2f}</div>
            {bar(m, COLORS['primary'])}
        </div>
        <div class="metric-box">
            <div class="metric-label">Episode Progress</div>
            <div class="metric-value" style="color:{COLORS['warning']}">{p:.2f}</div>
            {bar(p, COLORS['warning'])}
        </div>
    </div>
    """

def upgrade_plot(history):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=100)
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['surface'])
    
    if not history:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "AWAITING CORE INITIALIZATION", color=COLORS['muted'], ha='center', va='center', weight='bold')
        return fig
    
    try:
        steps = np.arange(len(history))
        data = np.array([[float(h[k]) for k in ['energy', 'stress', 'motivation', 'progress']] for h in history])
        
        # Plot with thicker lines and better visibility
        ax.plot(steps, data[:, 0], label="Energy", color=COLORS['secondary'], linewidth=2.5, alpha=0.9)
        ax.plot(steps, data[:, 1], label="Stress", color=COLORS['danger'], linewidth=2.5, alpha=0.9)
        ax.plot(steps, data[:, 2], label="Motivation", color=COLORS['primary'], linewidth=2.5, alpha=0.9)
        ax.plot(steps, data[:, 3], label="Progress", color=COLORS['warning'], linewidth=3.5, linestyle='--')
        
        # Highlight stress spikes
        spikes = np.where(data[:, 1] > 0.8)[0]
        if len(spikes) > 0:
            ax.scatter(spikes, data[spikes, 1], color=COLORS['danger'], s=80, edgecolors='white', zorder=5, label="Burnout Risk")

        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.05, color=COLORS['text'])
        ax.legend(loc='upper left', facecolor=COLORS['background'], edgecolor=COLORS['border'], labelcolor=COLORS['text'], fontsize=8)
        ax.tick_params(colors=COLORS['muted'], labelsize=9)
        for s in ax.spines.values(): s.set_edgecolor(COLORS['border'])
        plt.tight_layout()
    except Exception: ax.axis('off')
    return fig

def get_reward_explanation(state, prev_p, reward):
    try:
        p_gain = (state['progress'] - prev_p) * 1.5
        s_pen = -(state['stress'] ** 2)
        wellbeing = (state['energy'] * 0.3 + (1.0 - state['stress']) * 0.3 + state['motivation'] * 0.2)
        wb_bonus = wellbeing * 0.15 + (state['motivation'] * 0.1)
        
        def row(label, val, color):
            sign = "+" if val >= 0 else ""
            return f'<div class="reward-item"><span>{label}</span><span style="color:{color};font-weight:bold;">{sign}{val:.4f}</span></div>'

        return f"""
        <div style="background:{COLORS['highlight']};border:1px solid {COLORS['border']};border-radius:10px;padding:16px;">
            <div style="font-size:0.75rem;font-weight:800;color:{COLORS['muted']};margin-bottom:12px;border-bottom:1px solid {COLORS['border']};padding-bottom:10px;">REWARD SCALAR EXPLANATION</div>
            {row("Progress Contribution", p_gain, COLORS['secondary'])}
            {row("Stress Impact Penalty", s_pen, COLORS['danger'])}
            {row("State Balance Bonus", wb_bonus, COLORS['primary'])}
            <div style="margin-top:10px;padding-top:10px;border-top:2px solid {COLORS['border']};display:flex;justify-content:space-between;align-items:center;">
                <span style="font-weight:800;color:{COLORS['text']}">FINAL REWARD</span>
                <span style="font-size:1.4rem;font-weight:900;color:{COLORS['warning'] if reward > 0 else COLORS['danger']}">{reward:.4f}</span>
            </div>
        </div>
        """
    except Exception: return "Breakdown Unavailable"

# ==========================================
# HANDLERS: ACTIONS & AI
# ==========================================
def reset_core():
    try:
        obs = env.reset()
        state = {'energy': float(obs[0]), 'stress': float(obs[1]), 'motivation': float(obs[2]), 'progress': float(obs[3]), 'done': False, 'step': 0, 'reward': 0.0}
        history = [state]
        return history, state, upgrade_plot(history), fetch_telemetry(state), "ENV INITIALIZED", "", "🟡 STANDBY", "Ready for instruction."
    except Exception as e: return [], {}, upgrade_plot([]), fetch_telemetry(None), f"ERROR: {str(e)}", "", "🔴 OFFLINE", ""

def step_core(action_idx, history, state):
    try:
        if not history: return reset_core()
        if state.get('done', False): return history, state, upgrade_plot(history), fetch_telemetry(state), "TERMINAL REACHED. RESET CORE.", "", "🟡 STANDBY", ""
        
        prev_p = state['progress']
        obs, reward, done, info = env.step(int(action_idx))
        new_state = {'energy': float(obs[0]), 'stress': float(obs[1]), 'motivation': float(obs[2]), 'progress': float(obs[3]), 'done': done, 'step': info['step_count'], 'reward': float(reward)}
        history.append(new_state)
        
        act_name = ACTION_MAP.get(int(action_idx), "Unknown")
        status = f"Step {info['step_count']} | {act_name} | Rwd: {reward:.3f} | Done: {done}"
        return history, new_state, upgrade_plot(history), fetch_telemetry(new_state), status, get_reward_explanation(new_state, prev_p, reward), "🟢 ACTIVE", ""
    except Exception as e: return history, state, upgrade_plot(history), fetch_telemetry(state), f"EXCEPTION: {str(e)}", "", "🔴 ERR", ""

def handle_ai_intelligence(use_gemini, history, state):
    try:
        if not history or state.get('done', False): return step_core(0, history, state)
        
        # Hard constraint: 5s timeout
        e, s, m, p = [state[k] for k in ['energy', 'stress', 'motivation', 'progress']]
        future = executor.submit(get_action, e, s, m, p, use_gemini)
        
        start = time.time()
        try:
            decision = future.result(timeout=4.8)
            status_tag = "🟢 Gemini Active" if decision.get('source') != 'fallback' else "🟡 Fallback Mode"
        except Exception:
            decision = rule_based_action(e, s, m, p)
            decision['source'] = 'timeout_fallback'
            status_tag = "🟡 Fallback Mode"
            
        elapsed = time.time() - start
        act_idx = int(decision['action'])
        reason = f"Decision Logic (Latency: {elapsed:.2fs}): {decision.get('reason', 'N/A')}"
        
        h, n_s, plot, tele, status, expl, _, _ = step_core(act_idx, history, state)
        
        reasoning_html = f"""
        <div style="background:{COLORS['surface']};border:1px solid {COLORS['border']};border-radius:10px;padding:12px;margin-top:10px;">
            <div style="font-size:0.7rem;color:{COLORS['muted']};font-weight:700;margin-bottom:4px;">INFERENCE ORIGIN: {decision['source'].upper()}</div>
            <div style="font-weight:bold;color:{COLORS['primary']};margin-bottom:4px;">ACTION: {ACTION_MAP[act_idx].upper()}</div>
            <div style="font-size:0.85rem;line-height:1.4;">{decision.get('reason', 'Optimizing trajectory for long-term progress.')}</div>
        </div>
        """
        return h, n_s, plot, tele, status, expl, status_tag, reasoning_html
    except Exception as ex: return history, state, upgrade_plot(history), fetch_telemetry(state), f"AI ERROR: {str(ex)}", "", "🔴 ERR", ""

def simulate_batch(mode, steps, use_gemini, history, state):
    """Zero-lag simulation loop without yield"""
    try:
        temp_h, temp_s = list(history), dict(state)
        for _ in range(int(steps)):
            if temp_s.get('done', False): break
            
            if mode == "AI":
                e, s, m, p = [temp_s[k] for k in ['energy', 'stress', 'motivation', 'progress']]
                future = executor.submit(get_action, e, s, m, p, use_gemini)
                try: d = future.result(timeout=4.5)
                except Exception: d = rule_based_action(e, s, m, p)
                action = int(d['action'])
            else: action = random.randint(0, 4)
            
            temp_h, temp_s, _, _, _, _, _, _ = step_core(action, temp_h, temp_s)
            
        return temp_h, temp_s, upgrade_plot(temp_h), fetch_telemetry(temp_s), f"BATCH {mode} SIMULATION COMPLETED ({steps} steps)", get_reward_explanation(temp_s, history[-1]['progress'], temp_s['reward']), "🟢 IDLE", ""
    except Exception as ex: return history, state, upgrade_plot(history), fetch_telemetry(state), f"BATCH FAIL: {str(ex)}", "", "🔴 ERR", ""

def apply_custom_scenario(e, s, m, p, history, state):
    try:
        # Direct attribute injection - do NOT reset
        env.energy, env.stress, env.motivation, env.progress = e, s, m, p
        new_state = {'energy': e, 'stress': s, 'motivation': m, 'progress': p, 'done': False, 'step': env.step_count, 'reward': 0.0}
        if history: history[-1] = new_state
        else: history = [new_state]
        return history, new_state, upgrade_plot(history), fetch_telemetry(new_state), "CUSTOM SCENARIO INJECTED.", "", "🟡 SYNCED", ""
    except Exception as ex: return history, state, upgrade_plot(history), fetch_telemetry(state), f"INJECTION ERROR: {str(ex)}", "", "🔴 ERR", ""

# BONUS: MATH AI
def handle_math_ai(problem):
    try:
        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        p = f"Solve this math problem: {problem}. Be concise. Show only the answer and one sentence of explanation."
        r = client.models.generate_content(model="gemini-2.0-flash", contents=p)
        return r.text
    except Exception:
        try: # Python fallback
            import math
            res = eval(problem, {"__builtins__": None, "math": math})
            return f"Answer: {res} (Calculated via Python Fallback Engine)"
        except Exception: return "Error: Unable to process mathematical query."

# ==========================================
# UI ASSEMBLY (ELITE DASHBOARD)
# ==========================================
with gr.Blocks(title="Adaptive AI Productivity Engine") as demo:
    
    with gr.Column(elem_classes="main-container"):
        # --- HEADER SECTION ---
        with gr.Row():
            with gr.Column(scale=8):
                gr.HTML(f"""
                <div style='margin-bottom:20px;'>
                    <h1 style='margin:0; font-size:2.4rem; font-weight:800; letter-spacing:-0.04em;'>Adaptive AI Productivity Engine</h1>
                    <p style='margin:0; color:{COLORS['muted']}; font-weight:600;'>Advanced Human Performance Intelligence System v2.0</p>
                </div>
                """)
            with gr.Column(scale=2):
                ai_status = gr.HTML("<div class='status-indicator status-yellow pulse'>🟡 STANDBY</div>")

        gr.Markdown("---")
        
        sys_history = gr.State([])
        sys_state = gr.State({})

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
                    # Force this row to have top-level layering to prevent button overlap
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
                        check_gemini = gr.Checkbox(label="Neural Gemini Mode", value=False)
                        btn_ai_single = gr.Button("EXECUTE AI INFERENCE", variant="primary")
                    ai_reasoning = gr.HTML("<div style='color:#8b949e; padding:10px;'>Awaiting neural decision...</div>")
                
                with gr.Row():
                    with gr.Column(elem_classes="glass-card"):
                        gr.Markdown("### 📊 SYSTEM LOG")
                        sys_log = gr.Textbox(label="", value="READY.", interactive=True)
                    
                    with gr.Column(elem_classes="glass-card"):
                        gr.Markdown("### 🧮 MATH AI (BONUS)")
                        math_input = gr.Textbox(label="Math Problem", placeholder="e.g. 52 * 12 + sqrt(144)")
                        math_btn = gr.Button("SOLVE", size="sm")
                        math_out = gr.Textbox(label="Result", interactive=False)

            # --- BOTTOM RIGHT: REWARD DIAGNOSTICS ---
            with gr.Column(scale=4):
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("### 💰 REWARD EXPLAINER")
                    reward_expl = gr.HTML("<div style='color:#8b949e; padding:10px;'>Execute action to analyze rewards...</div>")
                    gr.Markdown("<p style='font-size:0.7rem; color:#8b949e; margin-top:10px;'>Reinforcement Learning Reward Formula: <i>Gain*1.5 - Stress² + Wellbeing + Motivation</i></p>")

    # ==========================
    # EVENT BINDINGS (THEN CHAIN)
    # ==========================
    btns = [btn_apply, btn_step, btn_sim_ai, btn_sim_rnd, btn_reset, btn_ai_single, math_btn]
    
    def ui_busy(): return [gr.update(interactive=False)] * len(btns)
    def ui_ready(): return [gr.update(interactive=True)] * len(btns)
    
    out_all = [sys_history, sys_state, plot_view, telemetry_view, sys_log, reward_expl, ai_status, ai_reasoning]

    btn_reset.click(fn=ui_busy, outputs=btns).then(fn=reset_core, outputs=out_all).then(fn=ui_ready, outputs=btns)
    
    btn_apply.click(fn=ui_busy, outputs=btns).then(fn=apply_custom_scenario, inputs=[sl_e, sl_s, sl_m, sl_p, sys_history, sys_state], outputs=out_all).then(fn=ui_ready, outputs=btns)
    
    act_rev = {v: k for k, v in ACTION_MAP.items()}
    btn_step.click(fn=ui_busy, outputs=btns).then(fn=lambda a, h, s: step_core(act_rev[a], h, s), inputs=[act_drop, sys_history, sys_state], outputs=out_all).then(fn=ui_ready, outputs=btns)
    
    btn_sim_ai.click(fn=ui_busy, outputs=btns).then(fn=simulate_batch, inputs=[gr.State("AI"), gr.State(10), check_gemini, sys_history, sys_state], outputs=out_all).then(fn=ui_ready, outputs=btns)
    btn_sim_rnd.click(fn=ui_busy, outputs=btns).then(fn=simulate_batch, inputs=[gr.State("RND"), gr.State(10), check_gemini, sys_history, sys_state], outputs=out_all).then(fn=ui_ready, outputs=btns)
    
    btn_ai_single.click(fn=ui_busy, outputs=btns).then(fn=handle_ai_intelligence, inputs=[check_gemini, sys_history, sys_state], outputs=out_all).then(fn=ui_ready, outputs=btns)
    
    math_btn.click(fn=handle_math_ai, inputs=math_input, outputs=math_out)

    demo.load(fn=reset_core, outputs=out_all)

# ==========================
# LAUNCH GRADIO 6.0 READY
# ==========================
if __name__ == "__main__":
    demo.queue(default_concurrency_limit=20).launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Base(),
        css=premium_css
    )

