"""
visualize.py

Runs a full episode and plots energy, stress, motivation, progress over time.
Highlights burnout zones and low energy zones.
Saves output as burnout_plot.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from environment import ProductivityEnvironment


def run_and_plot():
    """Run one full episode and plot state trajectories with risk zones"""
    env = ProductivityEnvironment()
    state = env.reset()

    # Action pattern: balanced mix of all actions
    actions = [0, 1, 0, 1, 4, 2, 3, 0, 4, 1] * 20

    # Track values at each step
    energy_log = [float(state[0])]
    stress_log = [float(state[1])]
    motivation_log = [float(state[2])]
    progress_log = [float(state[3])]

    for action in actions:
        state, reward, done, info = env.step(action)
        energy_log.append(float(state[0]))
        stress_log.append(float(state[1]))
        motivation_log.append(float(state[2]))
        progress_log.append(float(state[3]))
        if done:
            break

    steps = list(range(len(energy_log)))
    max_stress = max(stress_log)
    final_progress = progress_log[-1]
    burnout_detected = any(s > 0.8 for s in stress_log)
    low_energy_detected = any(e < 0.3 for e in energy_log)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(13, 6.5))

    # Highlight burnout zones (stress > 0.8)
    for i in range(len(steps) - 1):
        if stress_log[i] > 0.8:
            ax.axvspan(steps[i], steps[i + 1], color="#e74c3c", alpha=0.15)

    # Highlight low energy zones (energy < 0.3)
    for i in range(len(steps) - 1):
        if energy_log[i] < 0.3:
            ax.axvspan(steps[i], steps[i + 1], color="#f39c12", alpha=0.12)

    # Plot state trajectories
    ax.plot(steps, energy_log, label="Energy", color="#2ecc71", linewidth=2.2, marker="o", markersize=3)
    ax.plot(steps, stress_log, label="Stress", color="#e74c3c", linewidth=2.2, marker="s", markersize=3)
    ax.plot(steps, motivation_log, label="Motivation", color="#3498db", linewidth=2.2, marker="^", markersize=3)
    ax.plot(steps, progress_log, label="Progress", color="#f39c12", linewidth=2.2, linestyle="--", marker="D", markersize=3)

    # Threshold reference lines
    ax.axhline(y=0.8, color="#e74c3c", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axhline(y=0.3, color="#f39c12", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.text(steps[-1] + 0.3, 0.81, "Burnout", fontsize=8, color="#e74c3c", alpha=0.7)
    ax.text(steps[-1] + 0.3, 0.26, "Low energy", fontsize=8, color="#f39c12", alpha=0.7)

    # Zone legend patches
    zone_handles = []
    if burnout_detected:
        zone_handles.append(mpatches.Patch(color="#e74c3c", alpha=0.2, label="BURNOUT ZONE (stress > 0.8)"))
    if low_energy_detected:
        zone_handles.append(mpatches.Patch(color="#f39c12", alpha=0.2, label="LOW ENERGY ZONE (energy < 0.3)"))

    # Combine legends
    line_legend = ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    if zone_handles:
        ax.add_artist(line_legend)
        ax.legend(handles=zone_handles, loc="lower right", fontsize=9, framealpha=0.9)

    ax.set_title("Adaptive Human Productivity & Burnout Simulation", fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Value (0.0 - 1.0)", fontsize=12)
    ax.set_ylim(-0.05, 1.1)

    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("burnout_plot.png", dpi=150)
    print(f"Plot saved as burnout_plot.png ({len(steps)} steps)")

    # ---- Summary ----
    print("\n--- Episode Summary ---")
    print(f"  Steps completed:  {len(steps) - 1}")
    print(f"  Max stress:       {max_stress:.4f}")
    print(f"  Final progress:   {final_progress:.4f}")
    if burnout_detected:
        print("  Burnout risk:     🔥 YES — stress exceeded 0.8")
    else:
        print("  Burnout risk:     ✅ NO — stress stayed below 0.8")
    if low_energy_detected:
        print("  Low energy event: ⚠️  YES — energy dropped below 0.3")
    else:
        print("  Low energy event: ✅ NO — energy stayed above 0.3")


if __name__ == "__main__":
    run_and_plot()
