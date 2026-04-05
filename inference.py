"""
inference.py

Gemini-powered decision system for the Burnout Environment.
Uses Gemini API to choose optimal actions based on current state.
Falls back to rule-based logic if Gemini is unavailable or fails.

Actions:
  0 = study      (energy--, stress++, progress++)
  1 = rest       (energy++, stress--, motivation+)
  2 = exercise   (energy+, stress--, motivation++)
  3 = social     (energy+, stress-, motivation++)
  4 = work_hard  (energy---, stress+++, progress+++)
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import random

# =========================================================================
# GEMINI API CLIENT
# =========================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Action name mapping for readable output
ACTION_NAMES = {
    0: "study",
    1: "rest",
    2: "exercise",
    3: "social",
    4: "work_hard",
}


def _build_gemini_prompt(energy: float, stress: float,
                         motivation: float, progress: float) -> str:
    """
    Build a concise prompt for Gemini to choose the next action.
    Returns a structured prompt with state info and action descriptions.
    """
    return f"""You are an AI agent managing a human's productivity and wellbeing.

Current state (all values 0.0 to 1.0):
- Energy: {energy:.3f}
- Stress: {stress:.3f}
- Motivation: {motivation:.3f}
- Progress: {progress:.3f}

Available actions:
0 = study (uses energy, adds stress, gains progress)
1 = rest (restores energy, reduces stress)
2 = exercise (restores energy, reduces stress, boosts motivation)
3 = social (slight energy, reduces stress, boosts motivation)
4 = work_hard (drains energy, high stress, high progress)

Goal: Maximize progress while keeping stress low and energy stable.
Avoid burnout (sustained high stress crashes motivation and energy).

Rules:
- If energy < 0.3, prioritize rest or exercise
- If stress > 0.7, avoid study and work_hard
- If motivation < 0.3, prefer social or exercise
- Balance short-term progress with long-term sustainability

Respond with ONLY a single integer (0, 1, 2, 3, or 4). No explanation."""


def _call_gemini(energy: float, stress: float,
                 motivation: float, progress: float) -> int:
    """
    Call Gemini API to get the next action.

    Attempt order:
      1. google.genai SDK (new, recommended)
      2. google.generativeai SDK (deprecated fallback)
      3. Direct REST API via urllib (no deps)
    Returns action int (0-4) or raises Exception on failure.
    """
    prompt = _build_gemini_prompt(energy, stress, motivation, progress)

    # ---- Attempt 1: google.genai SDK (new) ----
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"temperature": 0.2, "max_output_tokens": 10}
        )
        action = _parse_gemini_response(response.text)
        return action
    except ImportError:
        pass  # New SDK not installed
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in str(e) or "resource_exhausted" in error_msg or "quota" in error_msg:
            print("  [gemini] ⏳ Rate limited (429) — quota exhausted, retrying via REST...")
        elif "401" in str(e) or "403" in str(e) or "invalid" in error_msg or "api_key" in error_msg:
            print("  [gemini] 🔑 Authentication error — check your GEMINI_API_KEY")
        elif "timeout" in error_msg or "network" in error_msg or "connection" in error_msg:
            print("  [gemini] 🌐 Network error — check internet connection")
        else:
            print(f"  [gemini] SDK error: {type(e).__name__}: {e}")

    # ---- Attempt 2: google.generativeai SDK (deprecated) ----
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import google.generativeai as genai_old
        genai_old.configure(api_key=GEMINI_API_KEY)
        model = genai_old.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        action = _parse_gemini_response(response.text)
        return action
    except ImportError:
        pass
    except Exception:
        pass  # Fall through to REST

    # ---- Attempt 3: Direct REST API via urllib (no extra deps) ----
    try:
        import urllib.request
        import urllib.error

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        )
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 10,
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        text = body["candidates"][0]["content"]["parts"][0]["text"]
        action = _parse_gemini_response(text)
        return action

    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RuntimeError("Rate limited (429) — daily quota exhausted")
        elif e.code in (401, 403):
            raise RuntimeError(f"Auth error ({e.code}) — invalid API key")
        else:
            raise RuntimeError(f"HTTP {e.code}: {e.reason}")
    except Exception as e:
        raise RuntimeError(f"All Gemini attempts failed: {e}")


def _parse_gemini_response(text: str) -> int:
    """
    Parse Gemini's response text into a valid action integer (0-4).
    Handles edge cases like extra whitespace, explanations, etc.
    """
    text = text.strip()

    # Try direct integer parse
    if text in ("0", "1", "2", "3", "4"):
        return int(text)

    # Try to find first digit in response
    for char in text:
        if char in "01234":
            return int(char)

    # Try matching action names
    text_lower = text.lower()
    name_to_action = {
        "study": 0,
        "rest": 1,
        "exercise": 2,
        "social": 3,
        "work_hard": 4,
        "work hard": 4,
        "work": 4,
    }
    for name, action in name_to_action.items():
        if name in text_lower:
            return action

    raise ValueError(f"Could not parse Gemini response: '{text}'")


# =========================================================================
# RULE-BASED FALLBACK (with reasoning)
# =========================================================================
def rule_based_action(energy: float, stress: float,
                      motivation: float, progress: float) -> tuple:
    """
    Deterministic rule-based fallback when Gemini is unavailable.
    Returns (action, reason) tuple for observability.

    Priority logic:
      1. Critical energy → rest
      2. Critical stress → exercise or social
      3. Low motivation → social or exercise
      4. Low progress + good state → work_hard or study
      5. Default balanced → study
    """

    # ---- CRITICAL: Prevent burnout ----
    if energy < 0.2:
        return 1, "⚠️  CRITICAL: Energy dangerously low → forced rest"

    if stress > 0.8:
        act = 2 if energy < 0.5 else 3
        return act, "🚨 CRITICAL: Stress dangerously high → emergency de-stress"

    # ---- WARNING: Approaching limits ----
    if energy < 0.35:
        act = 1 if stress > 0.5 else 2
        return act, "⚡ WARNING: Low energy → recovering before collapse"

    if stress > 0.65:
        act = 3 if motivation < 0.4 else 2
        return act, "😰 WARNING: High stress compounding → de-stress now"

    if motivation < 0.25:
        return 3, "💤 WARNING: Motivation depleted → social boost needed"

    # ---- URGENCY: Behind schedule ----
    if progress < 0.5 and energy > 0.4 and stress < 0.6:
        return 0, "📉 URGENCY: Behind schedule → steady study push"

    # ---- OPPORTUNITY: Push for progress ----
    if progress < 0.3 and energy > 0.6 and stress < 0.4:
        return 4, "🚀 OPPORTUNITY: Fresh start + high energy → work hard"

    if progress > 0.7 and energy > 0.5 and stress < 0.5:
        return 4, "🏁 OPPORTUNITY: Near finish line → final push"

    if energy > 0.5 and stress < 0.5:
        return 0, "📚 BALANCED: Healthy state → steady study"

    # ---- BALANCED: Maintain equilibrium ----
    if stress > 0.4:
        return 2, "🏃 BALANCED: Moderate stress → exercise to reset"

    if energy < 0.5:
        return 1, "😴 BALANCED: Low energy → light rest"

    # Default: study
    return 0, "📚 DEFAULT: No urgency detected → study"


# =========================================================================
# MAIN INFERENCE FUNCTION
# =========================================================================
def get_action(energy: float, stress: float,
               motivation: float, progress: float,
               use_gemini: bool = True) -> dict:
    """
    Main entry point: decide the next action given current state.

    Tries Gemini API first, falls back to rule-based logic on any failure.
    Returns a dict with action, source, action_name, and reasoning.
    """
    source = "rule_based"
    action = None
    reason = ""

    # ---- Try Gemini if enabled and API key is set ----
    if use_gemini and GEMINI_API_KEY and GEMINI_API_KEY != "your_key_here":
        try:
            action = _call_gemini(energy, stress, motivation, progress)
            if action not in (0, 1, 2, 3, 4):
                raise ValueError(f"Invalid action from Gemini: {action}")
            source = "gemini"
            reason = "🤖 Gemini API decided this action"
            print(f"  [decision] source=gemini ✅  action={ACTION_NAMES.get(action)}")
        except Exception as e:
            print(f"  [decision] source=fallback ❌  ({type(e).__name__}: {e})")
            action = None

    # ---- Fallback to rule-based ----
    if action is None:
        action, reason = rule_based_action(energy, stress, motivation, progress)
        source = "rule_based"

    return {
        "action": action,
        "action_name": ACTION_NAMES.get(action, "unknown"),
        "source": source,
        "reason": reason,
        "state": {
            "energy": round(energy, 4),
            "stress": round(stress, 4),
            "motivation": round(motivation, 4),
            "progress": round(progress, 4),
        }
    }


# =========================================================================
# GEMINI DIAGNOSTIC TEST
# =========================================================================
def test_gemini() -> dict:
    """
    Simple diagnostic to check if Gemini API is working.
    Sends 'Reply with ONLY the number 1' and checks the response.
    Returns dict with status, response, and error details.
    """
    result = {"status": "unknown", "response": None, "error": None, "sdk": None}

    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_key_here":
        result["status"] = "no_key"
        result["error"] = "GEMINI_API_KEY not set in .env"
        return result

    # Try new SDK
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Reply with ONLY the number 1. No other text.",
            config={"temperature": 0.0, "max_output_tokens": 5}
        )
        result["status"] = "ok"
        result["response"] = response.text.strip()
        result["sdk"] = "google.genai"
        return result
    except ImportError:
        pass
    except Exception as e:
        error_str = str(e).lower()
        if "429" in str(e) or "quota" in error_str or "exhausted" in error_str:
            result["status"] = "rate_limited"
            result["error"] = "429 — Daily free-tier quota exhausted. Resets at midnight PT."
        elif "401" in str(e) or "403" in str(e) or "invalid" in error_str:
            result["status"] = "auth_error"
            result["error"] = "Invalid API key — check your .env file"
        else:
            result["status"] = "error"
            result["error"] = f"{type(e).__name__}: {e}"
        result["sdk"] = "google.genai"
        return result

    # Try deprecated SDK
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import google.generativeai as genai_old
        genai_old.configure(api_key=GEMINI_API_KEY)
        model = genai_old.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Reply with ONLY the number 1.")
        result["status"] = "ok"
        result["response"] = response.text.strip()
        result["sdk"] = "google.generativeai (deprecated)"
        return result
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        result["sdk"] = "google.generativeai"
        return result


# =========================================================================
# VISUAL HELPERS
# =========================================================================
def _bar(value: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """Render a value (0.0-1.0) as a visual bar"""
    filled = int(value * width)
    return fill * filled + empty * (width - filled)


def _color_value(value: float, low_bad: bool = True) -> str:
    """Mark value with emoji severity indicator"""
    if low_bad:
        if value < 0.2: return f"🔴 {value:.3f}"
        if value < 0.4: return f"🟠 {value:.3f}"
        if value < 0.6: return f"🟡 {value:.3f}"
        return f"🟢 {value:.3f}"
    else:  # high is bad (stress)
        if value > 0.8: return f"🔴 {value:.3f}"
        if value > 0.6: return f"🟠 {value:.3f}"
        if value > 0.4: return f"🟡 {value:.3f}"
        return f"🟢 {value:.3f}"


def _detect_alerts(energy, stress, motivation, fatigue, burnout_counter):
    """Detect and return active risk alerts"""
    alerts = []
    if burnout_counter >= 3:
        alerts.append("🔥 BURNOUT ACTIVE — motivation and energy crashing")
    elif burnout_counter >= 2:
        alerts.append("⚠️  Burnout imminent (counter={}/3)".format(burnout_counter))
    if fatigue > 0.6:
        alerts.append("😫 High fatigue — diminished work returns")
    if stress > 0.7:
        alerts.append("📈 Stress compounding — spiral risk")
    if energy < 0.2:
        alerts.append("🪫 Energy critical — collapse risk")
    if motivation < 0.25:
        alerts.append("💔 Motivation depleted")
    return alerts


# =========================================================================
# RUN FULL EPISODE WITH INFERENCE
# =========================================================================
def run_episode(use_gemini: bool = True, max_steps: int = 200,
                verbose: bool = False) -> dict:
    """
    Run a complete episode using the inference system.

    Args:
        use_gemini: whether to use Gemini API
        max_steps: maximum steps per episode
        verbose: if True, print rich step-by-step output

    Returns:
        dict with trajectory and final results
    """
    from environment import ProductivityEnvironment
    from grader import TrajectoryRecorder, grade_trajectory

    env = ProductivityEnvironment()
    recorder = TrajectoryRecorder()
    state = env.reset()

    decisions = []

    if verbose:
        print("\n" + "═" * 64)
        print("  🧠 ADAPTIVE PRODUCTIVITY AGENT — EPISODE START")
        print("  " + "─" * 60)
        src = "Gemini + Fallback" if use_gemini else "Rule-Based Only"
        print(f"  Mode: {src}  |  Max Steps: {max_steps}")
        print("═" * 64)

    for step_num in range(max_steps):
        energy = float(state[0])
        stress = float(state[1])
        motivation = float(state[2])
        progress = float(state[3])

        # Get action from inference
        decision = get_action(energy, stress, motivation, progress, use_gemini)
        action = decision["action"]
        decisions.append(decision)

        # Execute in environment
        next_state, reward, done, info = env.step(action)
        recorder.record(next_state, action, reward, info)

        # ---- Verbose step display ----
        if verbose:
            fatigue = info.get("fatigue_level", 0.0)
            burnout_c = info.get("burnout_counter", 0)
            prog_gain = info.get("progress_gain", 0.0)

            print(f"\n  ┌─ Step {step_num + 1:3d} {'─' * 48}")
            print(f"  │ Energy:     {_bar(energy)} {_color_value(energy)}")
            print(f"  │ Stress:     {_bar(stress)} {_color_value(stress, low_bad=False)}")
            print(f"  │ Motivation: {_bar(motivation)} {_color_value(motivation)}")
            print(f"  │ Progress:   {_bar(progress)} {_color_value(progress)}")
            print(f"  │ Fatigue:    {_bar(fatigue)} {_color_value(fatigue, low_bad=False)}")
            print(f"  │")
            print(f"  │ 🎯 Action:  {decision['action_name'].upper()} (via {decision['source']})")
            print(f"  │ 💡 Reason:  {decision['reason']}")
            print(f"  │ 📊 Reward:  {reward:+.4f}  |  Δ Progress: {prog_gain:+.4f}")

            # Show active alerts
            alerts = _detect_alerts(energy, stress, motivation, fatigue, burnout_c)
            if alerts:
                print(f"  │")
                for alert in alerts:
                    print(f"  │ {alert}")

            print(f"  └{'─' * 58}")

        state = next_state
        if done:
            if verbose:
                # Show why episode ended
                if progress >= 1.0:
                    print("\n  🏆 EPISODE COMPLETE — Progress reached 100%!")
                elif energy <= 0:
                    print("\n  💀 EPISODE ENDED — Energy depleted (collapsed)")
                else:
                    print(f"\n  ⏰ EPISODE ENDED — Max steps ({max_steps}) reached")
            break

    # Grade the trajectory
    results = {
        "total_steps": len(decisions),
        "source_breakdown": {
            "gemini": sum(1 for d in decisions if d["source"] == "gemini"),
            "rule_based": sum(1 for d in decisions if d["source"] == "rule_based"),
        },
        "final_state": {
            "energy": round(float(state[0]), 4),
            "stress": round(float(state[1]), 4),
            "motivation": round(float(state[2]), 4),
            "progress": round(float(state[3]), 4),
        },
        "total_reward": round(recorder.total_reward, 4),
        "grades": {
            diff: grade_trajectory(recorder, diff)
            for diff in ["easy", "medium", "hard"]
        }
    }

    # ---- Verbose summary ----
    if verbose:
        fs = results["final_state"]
        print("\n" + "═" * 64)
        print("  📋 EPISODE SUMMARY")
        print("  " + "─" * 60)
        print(f"  Steps Taken:    {results['total_steps']}")
        print(f"  Total Reward:   {results['total_reward']:+.4f}")
        print(f"  Decision Source: {results['source_breakdown']}")
        print(f"  " + "─" * 60)
        print(f"  Final Energy:     {_bar(fs['energy'])} {fs['energy']:.4f}")
        print(f"  Final Stress:     {_bar(fs['stress'])} {fs['stress']:.4f}")
        print(f"  Final Motivation: {_bar(fs['motivation'])} {fs['motivation']:.4f}")
        print(f"  Final Progress:   {_bar(fs['progress'])} {fs['progress']:.4f}")
        print(f"  " + "─" * 60)
        print("  📝 GRADING RESULTS")
        for diff, grade in results["grades"].items():
            status = "PASS ✅" if grade["passed"] else "FAIL ❌"
            print(f"  [{diff.upper():6s}] {status}  Score: {grade['score']:.4f}")
        print("═" * 64 + "\n")

    return results


# =========================================================================
# MAIN — RICH DEMO
# =========================================================================
if __name__ == "__main__":
    print("\n" + "═" * 64)
    print("  🧠 ADAPTIVE HUMAN PRODUCTIVITY & BURNOUT SYSTEM")
    print("  Meta OpenEnv Hackathon Submission")
    print("═" * 64)

    # ---- 0. Gemini Diagnostic ----
    print("\n  🔬 GEMINI API DIAGNOSTIC")
    print("  " + "─" * 60)
    diag = test_gemini()
    if diag["status"] == "ok":
        print(f"  ✅ source=gemini  |  SDK: {diag['sdk']}")
        print(f"  📡 Response: \"{diag['response']}\"")
    elif diag["status"] == "no_key":
        print("  ⚠️  source=fallback ❌  |  No API key set")
        print("  💡 Add GEMINI_API_KEY to .env file")
    elif diag["status"] == "rate_limited":
        print("  ⏳ source=fallback ❌  |  Rate limited")
        print(f"  💡 {diag['error']}")
    elif diag["status"] == "auth_error":
        print("  🔑 source=fallback ❌  |  Auth error")
        print(f"  💡 {diag['error']}")
    else:
        print(f"  ❌ source=fallback  |  {diag['error']}")

    # ---- 1. Decision Reasoning Demo ----
    print("\n  📌 DECISION REASONING DEMO")
    print("  " + "─" * 60)
    test_cases = [
        (0.1, 0.5, 0.5, 0.3, "Exhausted worker"),
        (0.5, 0.9, 0.5, 0.3, "Burnout crisis"),
        (0.7, 0.2, 0.2, 0.3, "Unmotivated"),
        (0.8, 0.2, 0.8, 0.1, "Fresh start"),
        (0.6, 0.3, 0.6, 0.8, "Almost done"),
    ]
    for energy, stress, motivation, progress, label in test_cases:
        result = get_action(energy, stress, motivation, progress, use_gemini=False)
        print(f"\n  [{label}] E={energy} S={stress} M={motivation} P={progress}")
        print(f"  → {result['action_name'].upper()} | {result['reason']}")

    # ---- 2. Full Episode with Verbose Output ----
    print("\n\n" + "═" * 64)
    print("  🎬 FULL EPISODE — VERBOSE MODE")
    print("═" * 64)
    run_episode(use_gemini=False, verbose=True)

    # ---- 3. Gemini Live Test ----
    print("\n  🔑 GEMINI LIVE ACTION TEST")
    print("  " + "─" * 60)
    if diag["status"] == "ok":
        result = get_action(0.7, 0.3, 0.6, 0.4, use_gemini=True)
        print(f"  🤖 Gemini chose: {result['action_name'].upper()} (source={result['source']})")
        print(f"  💡 {result['reason']}")
    else:
        print(f"  ℹ️  Skipped — Gemini not available ({diag['status']})")
        print("  Rule-based fallback is active and stable")

    print("\n" + "═" * 64)
    print("  ✅ SYSTEM READY FOR DEPLOYMENT")
    print("═" * 64 + "\n")
