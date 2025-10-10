# Debug/heuristics_generator.py
from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    plt = None

# ----- Pathing: allow importing from ../CooldownGenerator -----
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
GEN_DIR = os.path.join(PROJECT_ROOT, "CooldownGenerator")
LOG_DIR = os.path.join(PROJECT_ROOT, "Logs")
os.makedirs(LOG_DIR, exist_ok=True)

if GEN_DIR not in sys.path:
    sys.path.append(GEN_DIR)

from src.CooldownGenerator.cooldown_humanizer import HumanCooldown  # noqa: E402

# ---------------- Configuration ----------------
NUM_ENTRIES = 10_000
HISTOGRAM_BINS = 100
ROUTINE_MAX_CLIP = 6.0

# Log file: Logs/heuristics_generator-logs-<DDMMYYYYHHMMSS>.txt
TIMESTAMP = datetime.now().strftime("%d%m%Y%H%M%S")
LOG_FILENAME = os.path.join(LOG_DIR, f"heuristics_generator-logs-{TIMESTAMP}.txt")

# Create a stateful generator instance (aligned with your HumanCooldown)
human_cooldown = HumanCooldown()
# ------------------------------------------------


def generate_and_analyze_cooldowns():
    """
    Generates cooldown times, logs them, and plots their distribution.
    (Plot types and layout are kept exactly as in your original script.)
    """
    if human_cooldown is None:
        print("\nERROR: HumanCooldown could not be instantiated.")
        return

    print("--- Cooldown Simulation Started ---")
    print(f"Generating {NUM_ENTRIES} human-like cooldown intervals...")

    cooldown_times: list[float] = []

    # 1) Generate the cooldown times
    t0 = time.perf_counter()
    for _ in range(NUM_ENTRIES):
        cooldown_times.append(human_cooldown.next())
    t1 = time.perf_counter()
    print(f"Generation complete in {(t1 - t0):.3f} seconds.")

    cooldown_times_np = np.array(cooldown_times)
    if cooldown_times_np.size == 0:
        print("Error: Cooldown times list is empty.")
        return

    # Stats for logging/plotting
    mean_time = cooldown_times_np.mean()
    min_time = cooldown_times_np.min()
    max_time = cooldown_times_np.max()

    # 2) Log the generated times to file (Logs/<name>-logs-<ts>.txt)
    try:
        with open(LOG_FILENAME, "w", encoding="utf-8") as f:
            f.write(f"--- COOLDOWN SIMULATION ANALYSIS ---\n")
            f.write(f"Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Total Entries: {NUM_ENTRIES}\n")
            f.write(f"Mean Cooldown: {mean_time:.4f}s\n")
            f.write(f"Range: [{min_time:.4f}s, {max_time:.4f}s]\n")
            f.write("-" * 30 + "\n")
            for t in cooldown_times:
                f.write(f"Cooldown: {t:.6f} s\n")
        print(f"Successfully logged {NUM_ENTRIES} cooldowns to {LOG_FILENAME}")
    except OSError as e:
        print(f"Error writing log file {LOG_FILENAME}: {e}")

    # 3) Plot (same figures/layout as before)
    if plt:
        print("Displaying histogram of the distribution...")

        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[3, 1])

        # --- Top-left: Routine Histogram + KDE ---
        ax_hist = fig.add_subplot(gs[0, 0])
        routine_times = cooldown_times_np[cooldown_times_np < ROUTINE_MAX_CLIP]

        ax_hist.hist(
            routine_times,
            bins=HISTOGRAM_BINS,
            color="#1f77b4",
            edgecolor="none",
            alpha=0.9,
            density=True,
            zorder=2,
        )

        # KDE line (unchanged)
        try:
            from scipy.stats import gaussian_kde
            if routine_times.size > 1:
                kde = gaussian_kde(routine_times)
                x_kde = np.linspace(routine_times.min(), routine_times.max(), 500)
                ax_hist.plot(
                    x_kde, kde(x_kde),
                    color="#ff7f0e",
                    linewidth=3,
                    label="Density Estimate",
                    zorder=3,
                )
        except Exception:
            pass  # keep plot types; silently skip KDE if scipy missing

        ax_hist.axvline(
            mean_time, color="r", linestyle="dashed", linewidth=2,
            label=f"Overall Mean: {mean_time:.3f}s"
        )
        ax_hist.set_title(
            f"Detailed Routine Cooldown Distribution (Zoomed to < {ROUTINE_MAX_CLIP:.1f}s)",
            fontsize=16,
        )
        ax_hist.set_xlabel("Cooldown Time (seconds)", fontsize=12)
        ax_hist.set_ylabel("Density / Frequency", fontsize=12)
        ax_hist.grid(axis="y", alpha=0.5, linestyle="--", zorder=1)
        ax_hist.legend()

        # --- Bottom-left: Box Plot (unchanged) ---
        ax_box = fig.add_subplot(gs[1, 0], sharex=ax_hist)
        ax_box.boxplot(
            routine_times,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="#a8c0d8"),
            medianprops=dict(color="red"),
        )
        ax_box.set_yticks([])
        ax_box.set_ylim(0.5, 1.5)
        ax_box.set_xlabel("Time (s)", fontsize=10)

        # --- Top-right: Full Distribution (Log Y) ---
        ax_log = fig.add_subplot(gs[0, 1])
        ax_log.hist(
            cooldown_times_np,
            bins=50,
            color="#9467bd",
            edgecolor="black",
            alpha=0.8,
            log=True,
        )
        ax_log.set_title("Full Distribution (Log Scale)", fontsize=16)
        ax_log.set_xlabel("Time (s)", fontsize=12)
        ax_log.set_ylabel("Log Frequency", fontsize=12)
        ax_log.tick_params(axis="y", which="minor", left=False)

        # bottom-right (unused) removed to keep your layout
        fig.delaxes(fig.add_subplot(gs[1, 1]))

        fig.suptitle(
            f"Human Cooldown Simulation Results (N={NUM_ENTRIES})",
            fontsize=18,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        print("\nNote: Cannot generate plot. Matplotlib is not installed.")


if __name__ == "__main__":
    # Optional pre-flight checks
    if not os.path.exists(os.path.join(GEN_DIR, "cooldown_humanizer.py")):
        print("WARNING: cooldown_humanizer.py not found in CooldownGenerator/")

    try:
        generate_and_analyze_cooldowns()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
