"""
Module:       coilgun_aspect_sweep.py
Purpose:      Step 1 extension — constant-volume aspect ratio sweep.
              Sweeps AR = L_c/R_c from 0.5 to 8.0 (50 points) while holding
              V_ref = π·R_c²·L_c constant at the Step 1 demo coil volume.
              Computes peak |f′| and force scale factor peak(f′)·L_c for each
              geometry, identifies the optimal ratio, and saves a 2-panel
              Neo-Classical figure.
Author:       Rizky Meilandi Saputra
Repository:   github.com/kiki007-lab/hybrid-architect-lab
Project:      Project 4 — Magnetic Linear Accelerator Simulation
Dependencies: coilgun_field_model (Step 1), numpy, matplotlib
Python:       3.10+

Key result
──────────
Both peak |f′| and the force scale factor peak(f′)·L_c are MONOTONICALLY
INCREASING functions of AR = L_c/R_c over all AR > 0, at constant coil
volume. The force scale approaches the analytical asymptote AR/2 from below
and is already within 0.5% of it at AR = 8.

Engineering implication: for a fixed amount of winding material (fixed coil
volume), a longer, narrower coil always produces a larger force scale per unit
volume. The practical optimum is therefore set by external constraints — bore
clearance (R_c must give adequate mechanical clearance around the projectile)
and wire gauge limits on achievable turns density n — not by any geometric
maximum within this sweep range.

Derivation of closed-form expressions
──────────────────────────────────────
At constant volume V, for aspect ratio AR = L_c/R_c:

    R_c = (V / (π·AR))^(1/3)
    L_c = AR·R_c

Peak |f′| is at z = −L_c/2 (coil entry edge; analytically proven as the
global maximum of field_gradient for all valid CoilGeometry). At that z:

    f′_peak = (R_c²/2) · [1/R_c³ − 1/(L_c² + R_c²)^(3/2)]
            = [1 − 1/(AR²+1)^(3/2)] / (2·R_c)

Force scale factor:

    peak(f′)·L_c = (AR/2)·[1 − 1/(AR²+1)^(3/2)]

    → AR/2  asymptotically as AR → ∞
    (correction term decays as 1/(AR²+1)^(3/2) → 0)
"""

import math
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for both scripts and pipelines
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# coilgun_field_model.py is a sibling module in the same directory.
# Adding the script's own directory to sys.path handles both direct execution
# and invocation from a different working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from coilgun_field_model import CoilGeometry, field_gradient


# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCE GEOMETRY AND SWEEP PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Step 1 demo coil — held fixed as the volume reference for the sweep
DEMO_R_C: float = 0.020   # m  (20 mm bore radius)
DEMO_L_C: float = 0.080   # m  (80 mm winding length)
DEMO_AR:  float = DEMO_L_C / DEMO_R_C   # = 4.0

V_REF: float = math.pi * DEMO_R_C**2 * DEMO_L_C
# π·R_c²·L_c = π × 4×10⁻⁴ × 0.080 ≈ 1.005×10⁻⁴ m³ ≈ 100.5 cm³
# Every geometry in the sweep has exactly this coil zone volume.

AR_MIN:   float = 0.5
AR_MAX:   float = 8.0
N_POINTS: int   = 50

FIGURE_WIDTH_IN:     float = 14.0
FIGURE_HEIGHT_IN:    float = 6.5
OUTPUT_DPI:          int   = 150
DEFAULT_OUTPUT_PATH: str   = "coilgun_aspect_sweep.png"


# ═══════════════════════════════════════════════════════════════════════════════
# NEO-CLASSICAL PALETTE  — mirrored exactly from Step 1, no divergence
# ═══════════════════════════════════════════════════════════════════════════════

NC_BACKGROUND: str = "#0a0a0a"
NC_GRID:       str = "#1a1a1a"
NC_SPINE:      str = "#2a2a2a"
NC_TICK:       str = "#888888"
NC_EDGE_MARK:  str = "#404040"
NC_F:          str = "#8b0000"   # deep red   — peak |f′|
NC_FPRIME:     str = "#c9a84c"   # gold       — force scale factor
NC_PRODUCT:    str = "#d8d8d8"   # near-white — asymptote overlay
NC_TITLE:      str = "#c9a84c"
NC_FORMULA:    str = "#484848"


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_aspect_sweep(
    ar_values: np.ndarray,
    V_ref:     float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep AR = L_c/R_c at constant coil volume V_ref = π·R_c²·L_c.

    For each AR the constraint system gives:
        R_c = (V_ref / (π·AR))^(1/3)
        L_c = AR · R_c

    Peak |f′| is evaluated exactly at z = −L_c/2 via a single field_gradient
    call per geometry — no z-grid scan needed. This is valid because the global
    maximum of field_gradient is analytically at the coil entry edge z = −L_c/2:
    at that point z_plus = 0, which maximises the first bracket term in f′(z).
    Any deviation from z = −L_c/2 reduces both terms simultaneously, lowering
    the result. This holds for all AR > 0.

    Parameters
    ──────────
    ar_values : ndarray
        Aspect ratio values to sweep [dimensionless].
    V_ref : float
        Reference coil volume [m³]. Held constant across the sweep.

    Returns
    ───────
    R_c_arr  : ndarray  inner bore radii [m]
    L_c_arr  : ndarray  winding lengths [m]
    fp_arr   : ndarray  peak |f′| values [m⁻¹]
    fs_arr   : ndarray  force scale = peak(f′)·L_c [dimensionless]
    """
    R_c_arr = np.empty_like(ar_values)
    L_c_arr = np.empty_like(ar_values)
    fp_arr  = np.empty_like(ar_values)
    fs_arr  = np.empty_like(ar_values)

    for i, ar in enumerate(ar_values):
        R_c = (V_ref / (math.pi * ar)) ** (1.0 / 3.0)
        L_c = ar * R_c
        coil = CoilGeometry(
            radius_m        = R_c,
            length_m        = L_c,
            turns_per_meter = 2000.0,   # n not used by field_gradient; set for completeness
        )

        # z = −L_c/2 → z_plus = 0 → term_plus = 1/R_c³ (global max of gradient)
        fp_peak = float(field_gradient(-L_c / 2.0, coil))

        R_c_arr[i] = R_c
        L_c_arr[i] = L_c
        fp_arr[i]  = fp_peak
        fs_arr[i]  = fp_peak * L_c

    return R_c_arr, L_c_arr, fp_arr, fs_arr


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

def render_sweep(
    ar_values: np.ndarray,
    R_c_arr:   np.ndarray,
    L_c_arr:   np.ndarray,
    fp_arr:    np.ndarray,
    fs_arr:    np.ndarray,
    output_path: str,
    output_dpi:  int,
) -> None:
    """
    2-panel Neo-Classical figure for the aspect ratio sweep.

    Panel 1 — Peak |f′| [m⁻¹] vs AR.
      Shows how the maximum field gradient grows as the coil narrows at
      constant volume. Grows roughly as AR^(1/3) for large AR (dominated
      by the shrinking R_c in the denominator).

    Panel 2 — Force scale factor peak(f′)·L_c vs AR.
      The dimensionless quantity that directly governs projectile impulse
      for fixed n and I. Approaches the asymptote AR/2 from below; shown
      overlaid as a dashed near-white line. Both the demo coil and the
      sweep maximum are annotated with their exact values.
    """
    # ── reference values at the demo coil geometry ────────────────────────────
    _demo_coil = CoilGeometry(DEMO_R_C, DEMO_L_C, 2000.0)
    demo_fp    = float(field_gradient(-DEMO_L_C / 2.0, _demo_coil))
    demo_fs    = demo_fp * DEMO_L_C

    # ── sweep maximum — both metrics are monotonically increasing ─────────────
    opt_idx = int(np.argmax(fs_arr))   # always the last point (AR_MAX)
    opt_ar  = ar_values[opt_idx]
    opt_fp  = fp_arr[opt_idx]
    opt_fs  = fs_arr[opt_idx]

    # asymptote for force scale: y = AR/2
    asymptote = ar_values / 2.0

    # ── global style ──────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":       "monospace",
        "font.size":         9,
        "axes.facecolor":    NC_BACKGROUND,
        "figure.facecolor":  NC_BACKGROUND,
        "text.color":        NC_TICK,
        "axes.labelcolor":   NC_TICK,
        "xtick.color":       NC_TICK,
        "ytick.color":       NC_TICK,
        "xtick.major.size":  4,
        "ytick.major.size":  4,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        gridspec_kw={"wspace": 0.38},
    )
    fig.patch.set_facecolor(NC_BACKGROUND)

    # ── shared helpers ────────────────────────────────────────────────────────

    def _style(ax, ylabel: str) -> None:
        ax.set_facecolor(NC_BACKGROUND)
        for sp in ax.spines.values():
            sp.set_color(NC_SPINE)
        ax.tick_params(colors=NC_TICK, length=4, width=0.5)
        ax.grid(True, color=NC_GRID, linewidth=0.5, linestyle="--", alpha=0.7)
        ax.set_xlabel("Aspect ratio  L_c / R_c  [dimensionless]",
                      color=NC_TICK, fontsize=9, labelpad=6)
        ax.set_ylabel(ylabel, color=NC_TICK, fontsize=9, labelpad=6)
        ax.set_xlim(AR_MIN - 0.15, AR_MAX + 0.15)

    def _mark_demo(ax, y_demo: float, y_offset_frac: float = 0.06) -> None:
        """Vertical dashed gold line at the demo coil AR with label above."""
        ax.axvline(x=DEMO_AR, color=NC_FPRIME, linewidth=0.9,
                   linestyle="--", alpha=0.50, zorder=2)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(DEMO_AR + 0.10, 1.0 - y_offset_frac,
                f"demo coil\nAR = {DEMO_AR:.1f}",
                transform=trans, fontsize=7.0, color=NC_FPRIME,
                va="top", ha="left", alpha=0.75)

    def _formula_box(ax, lines: str) -> None:
        ax.text(0.04, 0.04, lines,
                transform=ax.transAxes,
                fontsize=6.8, color=NC_FORMULA,
                va="bottom", ha="left", linespacing=1.55,
                bbox=dict(boxstyle="square,pad=0.35",
                          facecolor=NC_BACKGROUND,
                          edgecolor=NC_SPINE,
                          linewidth=0.5))

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 1 — Peak |f′(z)|  [m⁻¹]
    # ═════════════════════════════════════════════════════════════════════════
    _style(ax1, "Peak  |f′(z)|  [m⁻¹]")
    ax1.set_ylim(0, fp_arr.max() * 1.18)

    ax1.plot(ar_values, fp_arr, color=NC_F, linewidth=1.8, zorder=3)
    _mark_demo(ax1, demo_fp)

    # demo coil dot
    ax1.scatter([DEMO_AR], [demo_fp], color=NC_F, s=52, zorder=5,
                edgecolors=NC_TICK, linewidths=0.6)
    ax1.annotate(
        f"AR = {DEMO_AR:.1f}\n{demo_fp:.2f} m⁻¹",
        xy=(DEMO_AR, demo_fp),
        xytext=(DEMO_AR - 2.0, demo_fp * 0.78),
        fontsize=7.5, color=NC_F, ha="center",
        arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7),
    )

    # sweep maximum dot and label
    ax1.scatter([opt_ar], [opt_fp], color=NC_F, s=52, zorder=5,
                edgecolors=NC_TICK, linewidths=0.6)
    ax1.annotate(
        f"Sweep max\nAR = {opt_ar:.1f}\n{opt_fp:.2f} m⁻¹",
        xy=(opt_ar, opt_fp),
        xytext=(opt_ar - 2.2, opt_fp * 0.86),
        fontsize=7.5, color=NC_F, ha="center",
        arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7),
    )

    ax1.set_title("Peak Field Gradient  |f′|  vs Aspect Ratio",
                  color=NC_TITLE, fontsize=10.5, fontweight="bold", pad=10)

    _formula_box(ax1,
        "f′_peak = [1 − 1/(AR²+1)^(3/2)] / (2·R_c)\n"
        "R_c = (V_ref / (π·AR))^(1/3)  [constant volume]\n"
        "Peak grows ∝ AR^(1/3) for large AR  (R_c → 0)")

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Force scale factor  peak(f′)·L_c  [dimensionless]
    # ═════════════════════════════════════════════════════════════════════════
    _style(ax2, "Force scale  peak(f′) · L_c  [dimensionless]")
    ax2.set_ylim(0, fs_arr.max() * 1.18)

    # Asymptote overlay: y = AR/2
    ax2.plot(ar_values, asymptote, color=NC_PRODUCT, linewidth=0.9,
             linestyle="--", alpha=0.38, zorder=2)
    ax2.text(0.96, 0.94, "y = AR/2",
             transform=ax2.transAxes, fontsize=7.0,
             color=NC_PRODUCT, alpha=0.50, ha="right", va="top")

    ax2.plot(ar_values, fs_arr, color=NC_FPRIME, linewidth=1.8, zorder=3)
    _mark_demo(ax2, demo_fs)

    # demo coil dot
    ax2.scatter([DEMO_AR], [demo_fs], color=NC_FPRIME, s=52, zorder=5,
                edgecolors=NC_TICK, linewidths=0.6)
    ax2.annotate(
        f"AR = {DEMO_AR:.1f}\n{demo_fs:.3f}\n({demo_fs / (DEMO_AR/2)*100:.1f}% of AR/2)",
        xy=(DEMO_AR, demo_fs),
        xytext=(DEMO_AR - 2.2, demo_fs * 1.22),
        fontsize=7.5, color=NC_FPRIME, ha="center",
        arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7),
    )

    # sweep maximum
    ax2.scatter([opt_ar], [opt_fs], color=NC_FPRIME, s=52, zorder=5,
                edgecolors=NC_TICK, linewidths=0.6)
    ax2.annotate(
        f"Sweep max\nAR = {opt_ar:.1f}\n{opt_fs:.3f}  (AR/2 = {opt_ar/2:.2f})",
        xy=(opt_ar, opt_fs),
        xytext=(opt_ar - 2.5, opt_fs * 0.83),
        fontsize=7.5, color=NC_FPRIME, ha="center",
        arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7),
    )

    ax2.set_title("Force Scale Factor  peak(f′)·L_c  vs Aspect Ratio",
                  color=NC_TITLE, fontsize=10.5, fontweight="bold", pad=10)

    _formula_box(ax2,
        "Force scale = (AR/2)·[1 − 1/(AR²+1)^(3/2)]\n"
        "Asymptote: AR/2  (correction < 0.5% for AR > 8)\n"
        "Monotonically ↑ — practical limit is bore clearance")

    # ── figure-level title and parameter line ─────────────────────────────────
    fig.suptitle(
        "COILGUN ASPECT RATIO SWEEP — CONSTANT COIL VOLUME",
        color=NC_TITLE, fontsize=12, fontweight="bold", y=1.04,
    )
    fig.text(
        0.5, 0.997,
        (f"V_ref = π·R_c²·L_c = {V_REF * 1e6:.1f} cm³  ·  "
         f"AR swept {AR_MIN} → {AR_MAX}  ·  {N_POINTS} geometries  ·  "
         f"demo coil: R_c = {DEMO_R_C*1e3:.0f} mm, "
         f"L_c = {DEMO_L_C*1e3:.0f} mm, AR = {DEMO_AR:.0f}"),
        ha="center", va="top", color=NC_TICK, fontsize=8.5,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(output_path, dpi=output_dpi, bbox_inches="tight",
                pad_inches=0.15, facecolor=NC_BACKGROUND)
    plt.close(fig)

    print(f"  [render]      → {output_path}")
    print(f"  [render]      {FIGURE_WIDTH_IN:.0f} × {FIGURE_HEIGHT_IN:.1f} in  "
          f"at {output_dpi} DPI")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coilgun Step 1 extension: constant-volume aspect ratio sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python coilgun_aspect_sweep.py --output ./coilgun_aspect_sweep.png',
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help=f"Full path for output PNG. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Orchestrate the constant-volume aspect ratio sweep:
      1. Compute peak |f′| and force scale for 50 AR values
      2. Print a structured results table
      3. Render and save the 2-panel Neo-Classical figure
    """
    args = parse_arguments()

    print()
    print("  ╔════════════════════════════════════════════════════════════╗")
    print("  ║   COILGUN SIMULATION — ASPECT RATIO SWEEP                 ║")
    print("  ║   Step 1 Extension   |  constant-volume geometry study    ║")
    print("  ╚════════════════════════════════════════════════════════════╝")
    print()
    print(f"  [reference]  Demo coil: R_c = {DEMO_R_C*1e3:.1f} mm  "
          f"| L_c = {DEMO_L_C*1e3:.1f} mm  | AR = {DEMO_AR:.1f}")
    print(f"  [reference]  V_ref = π·R_c²·L_c = {V_REF * 1e6:.2f} cm³  "
          f"(held constant across all {N_POINTS} geometries)")
    print(f"  [sweep]      AR from {AR_MIN} to {AR_MAX}  |  {N_POINTS} points")
    print(f"  [output]     {args.output}")
    print()

    # ── run sweep ─────────────────────────────────────────────────────────────
    ar_values = np.linspace(AR_MIN, AR_MAX, N_POINTS)
    R_c_arr, L_c_arr, fp_arr, fs_arr = run_aspect_sweep(ar_values, V_REF)

    # ── print results table ───────────────────────────────────────────────────
    print("  [results]    Selected sweep points:")
    print(f"  {'AR':>6}  {'R_c [mm]':>10}  {'L_c [mm]':>10}  "
          f"{'peak |f′| [m⁻¹]':>17}  {'force scale':>12}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*17}  {'─'*12}")

    sample_indices = [0, 9, 19, 29, 39, 49]   # AR ≈ 0.5, 2.0, 3.5, 5.0, 6.5, 8.0
    for i in sample_indices:
        ar   = ar_values[i]
        R_c  = R_c_arr[i]
        L_c  = L_c_arr[i]
        fp   = fp_arr[i]
        fs   = fs_arr[i]
        marker = " ← demo" if abs(ar - DEMO_AR) < 0.15 else ""
        print(f"  {ar:>6.2f}  {R_c*1e3:>10.2f}  {L_c*1e3:>10.2f}  "
              f"{fp:>17.4f}  {fs:>12.4f}{marker}")

    # ── identify optimal (sweep maximum) ──────────────────────────────────────
    opt_idx = int(np.argmax(fs_arr))
    opt_ar  = ar_values[opt_idx]
    opt_fp  = fp_arr[opt_idx]
    opt_fs  = fs_arr[opt_idx]

    _demo_coil = CoilGeometry(DEMO_R_C, DEMO_L_C, 2000.0)
    demo_fp    = float(field_gradient(-DEMO_L_C / 2.0, _demo_coil))
    demo_fs    = demo_fp * DEMO_L_C

    print()
    print(f"  [analysis]   Force scale is MONOTONICALLY INCREASING over sweep range.")
    print(f"  [analysis]   Optimal in sweep: AR = {opt_ar:.2f}  "
          f"(force scale = {opt_fs:.4f},  asymptote AR/2 = {opt_ar/2:.4f})")
    print(f"  [analysis]   Demo coil (AR = {DEMO_AR:.1f}): "
          f"force scale = {demo_fs:.4f}  "
          f"({demo_fs/(opt_ar/2)*100:.1f}% of asymptote at sweep max)")
    print(f"  [analysis]   Asymptote convergence at AR = {opt_ar:.1f}: "
          f"{opt_fs/(opt_ar/2)*100:.2f}% of AR/2")
    print()
    print("  [insight]    Longer/narrower coil always improves force scale at fixed")
    print("               volume. Real optimum is set by bore clearance and wire gauge,")
    print("               not by any internal geometric maximum in this range.")
    print()

    # ── render ────────────────────────────────────────────────────────────────
    print("  [render]     generating 2-panel Neo-Classical figure...")
    render_sweep(ar_values, R_c_arr, L_c_arr, fp_arr, fs_arr,
                 args.output, OUTPUT_DPI)

    print()
    print("  [done]       Step 1 extension complete.")
    print()


if __name__ == "__main__":
    main()
