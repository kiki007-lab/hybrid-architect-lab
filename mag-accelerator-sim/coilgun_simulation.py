"""
Module:       coilgun_simulation.py
Purpose:      Steps 5-6 of 6 — three-stage magnetic linear accelerator simulation.
              Master integration loop. Corrected trigger timing from Steps 3-4
              diagnosis. Six-panel Neo-Classical output figure.
Author:       Rizky Meilandi Saputra
Repository:   github.com/kiki007-lab/hybrid-architect-lab
Project:      Project 4 — Magnetic Linear Accelerator Simulation
Dependencies: coilgun_field_model, coilgun_rlc_model, coilgun_dynamics
Python:       3.10+

---

Steps 3-4 Timing Diagnosis
-------------------------
Steps 3-4 (z_trigger = z_center − 60mm, v0 = 0) produced v_exit = 34.9 m/s
at η = 9.52%. The velocity profile showed v_max = 246.3 m/s followed by
catastrophic braking — the projectile crossed z_center at t ≈ 2.57ms, which
is 0.6ms AFTER t_peak = 1.975ms. With I still large and the force sign flipped,
most of the kinetic energy was returned to the circuit.

Corrected Timing
----------------
Optimal: projectile reaches z_center near t = π/ω_d ≈ 4.57ms after firing,
when I has returned to zero. At this point:
  - Full positive-current impulse received (accelerating zone complete)
  - I → 0 → F → 0 at the moment of crossing center
  - No residual braking from the first current lobe

For Stage 1 (v0 = 0): z_trigger = z_center − 120mm.
At 120mm before center the field is very weak (f×f′ ≈ 0.003 m⁻¹), so the
force is negligible for the first ~2ms. The projectile accelerates gently,
then rapidly as it enters the coil's strong-field zone (~40mm before center).
This allows the current to build fully before the projectile encounters the
strongest force gradient.

For Stages 2 and 3 (v_entry > 0): trigger offset = min(v_entry × 3ms, 300mm).
The 3ms target represents 65% of π/ω_d, giving the projectile adequate time
in the accelerating zone while keeping trigger positions within barrel geometry.

Multi-Stage Architecture
------------------------
    z_centers = [200, 450, 700] mm   (250mm spacing)

Integration loop per timestep:
    1. Check all unfired stages for trigger condition (z ≥ z_trigger_k)
    2. Fire: record t_fire_k, mark stage active
    3. Sum F_k over all active stages (simultaneous overlap possible)
    4. RK4 step with total force

Trigger positions are computed dynamically (2-pass approach):
    Pass 1: Stage 1 alone → v1_exit
    Pass 2: Stages 1+2   → v2_exit
    Pass 3: Full 3 stages → final records + figure

Known simplifications inherited from Steps 1–4:
    - Constant μ_r (ignores saturation, B_peak ≈ B_sat for this geometry)
    - Point-centroid force (L_proj = L_c/2, ~10% spatial error)
    - Constant-L RLC (real L rises 30–80% as slug enters bore)
    - No eddy current drag, no friction, no gravity
"""

import math
import os
import sys
import argparse
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.transforms import blended_transform_factory

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from coilgun_field_model import CoilGeometry, MU_0
from coilgun_rlc_model   import CapacitorBank, AccelerationStage, rlc_current
from coilgun_dynamics    import Projectile, coupling_force, rk4_step


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

Z_CENTERS_M       = [0.200, 0.450, 0.700]   # stage coil centers [m]
Z_TRIGGER_S1_M    = Z_CENTERS_M[0] - 0.060  # Stage 1: 60mm before center (Steps 3-4 optimum)
TRIGGER_T_TARGET  = 4.0e-3                  # target transit time for Stages 2/3 [s]
TRIGGER_MAX_OFFSET = 0.300                   # cap trigger offset at 300mm [m]
# 300mm chosen so that at v=119m/s (Stage 3 entry), transit = 300/119 = 2.52ms > t_peak = 1.975ms.
# At 200mm cap, transit = 1.68ms < t_peak — Stage 3 always brakes regardless of spacing.


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

DT          = 5e-6    # 5 μs timestep
T_SINGLE    = 15e-3   # Stage-1-only pass [s]
T_DOUBLE    = 22e-3   # Stages-1+2 pass [s]
T_TOTAL     = 30e-3   # Full 3-stage simulation [s]


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

FIGURE_WIDTH_IN  = 20.0
FIGURE_HEIGHT_IN = 13.0
OUTPUT_DPI       = 150
DEFAULT_OUTPUT   = "coilgun_neoclassical.png"
SWEEP_OUTPUT     = "coilgun_spacing_sweep.png"


# ═══════════════════════════════════════════════════════════════════════════════
# NEO-CLASSICAL PALETTE  — locked to Steps 1–4
# ═══════════════════════════════════════════════════════════════════════════════

NC_BG          = "#0a0a0a"
NC_GRID        = "#1a1a1a"
NC_SPINE       = "#2a2a2a"
NC_TICK        = "#888888"
NC_ZERO        = "#383838"
NC_TITLE       = "#c9a84c"
NC_FORMULA     = "#484848"
NC_F           = "#8b0000"   # deep red — alias for Stage 1 color, sweep panels
STAGE_COLORS   = ["#8b0000", "#aa5426", "#c9a84c"]   # deep red → gold


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MultiStageRecord:
    """One timestep snapshot from the multi-stage integration loop."""
    time_s:           float
    position_m:       float
    velocity_ms:      float
    net_force_n:      float
    kinetic_energy_j: float
    stage_currents:   list[float]   # I_k(t) [A] per stage
    stage_forces:     list[float]   # F_k(t) [N] per stage


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def integrate_multi_stage(
    stages:      list[AccelerationStage],
    projectile:  Projectile,
    z0:          float,
    v0:          float,
    dt:          float,
    t_total:     float,
) -> tuple[list[MultiStageRecord], list[float | None]]:
    """
    Multi-stage integration loop. At each timestep:
      1. Check all unfired stages for trigger crossing.
      2. Compute per-stage I(t) and F(z_rel, t).
      3. Sum forces; RK4 step.

    Returns
    -------
    records     : list[MultiStageRecord] — full history
    fired_times : list[float | None]     — wall-clock t when each stage fired
    """
    n_stages    = len(stages)
    n_steps     = int(t_total / dt) + 1
    state       = np.array([z0, v0], dtype=float)
    records:    list[MultiStageRecord] = []
    fired_times: list[float | None]   = [None] * n_stages

    for step in range(n_steps):
        t    = step * dt
        z, v = state

        # ── trigger check ────────────────────────────────────────────────────
        for k, stage in enumerate(stages):
            if fired_times[k] is None and z >= stage.trigger_position_m:
                fired_times[k] = t

        # ── per-stage current and force ──────────────────────────────────────
        stage_currents: list[float] = []
        stage_forces:   list[float] = []
        total_force = 0.0

        for k, stage in enumerate(stages):
            if fired_times[k] is not None:
                t_el = t - fired_times[k]
                I    = float(rlc_current(t_el, stage))
                z_rel = z - stage.center_position_m
                F    = float(coupling_force(z_rel, I, stage.coil, projectile))
            else:
                I = 0.0
                F = 0.0
            stage_currents.append(I)
            stage_forces.append(F)
            total_force += F

        KE = 0.5 * projectile.mass_kg * v ** 2
        records.append(MultiStageRecord(
            time_s           = t,
            position_m       = z,
            velocity_ms      = v,
            net_force_n      = total_force,
            kinetic_energy_j = KE,
            stage_currents   = stage_currents[:],
            stage_forces     = stage_forces[:],
        ))

        state = rk4_step(state, total_force, projectile.mass_kg, dt)

    return records, fired_times


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trigger_offset(v_entry: float) -> float:
    """
    Trigger offset = z_center − z_trigger [m] for a given entry velocity.

    Target: projectile travels from trigger to coil center in T_target = 3ms.
    Treating transit as approximately constant-velocity (conservative):
        offset = v_entry × T_target

    The 3ms target is 65% of π/ω_d = 4.57ms, chosen so the projectile
    arrives while I is declining (not yet zero) — maximising the accelerating
    impulse without overshooting into severe braking.

    Capped at TRIGGER_MAX_OFFSET = 300mm (barrel geometry constraint).
    For v_entry ≈ 0 (Stage 1): returns fixed 120mm.
    """
    if v_entry < 2.0:
        return 0.120
    return min(v_entry * TRIGGER_T_TARGET, TRIGGER_MAX_OFFSET)


def make_stage(
    index:             int,
    coil:              CoilGeometry,
    capacitor:         CapacitorBank,
    wire_resistance_ohm: float,
    z_center_m:        float,
    z_trigger_m:       float,
) -> AccelerationStage:
    """Construct a stage. Raises ValueError if overdamped."""
    return AccelerationStage(
        index                = index,
        coil                 = coil,
        capacitor            = capacitor,
        wire_resistance_ohm  = wire_resistance_ohm,
        center_position_m    = z_center_m,
        trigger_position_m   = z_trigger_m,
    )


def get_exit_velocity(
    records:   list[MultiStageRecord],
    z_center_m: float,
    clearance_m: float = 0.080,
) -> float:
    """
    Velocity when projectile passes z_center + clearance (force ≈ 0 there).
    Falls back to records[-1].velocity_ms if never reached.
    """
    threshold = z_center_m + clearance_m
    for r in records:
        if r.position_m >= threshold:
            return r.velocity_ms
    return records[-1].velocity_ms


def get_entry_velocity(
    records:     list[MultiStageRecord],
    fired_time:  float | None,
    dt:          float,
) -> float:
    """Velocity at the moment a stage fires."""
    if fired_time is None:
        return 0.0
    step = int(round(fired_time / dt))
    step = min(step, len(records) - 1)
    return records[step].velocity_ms


def compute_stage_delta_ke(
    records:  list[MultiStageRecord],
    stages:   list[AccelerationStage],
) -> list[float]:
    """
    Per-stage kinetic energy gain [J].

    Defined as the KE change while the projectile passes through each stage's
    force zone (from entry to z_center + 80mm). The 80mm clearance ensures
    the stage's residual force is negligible at the boundary.
    """
    thresholds = [s.center_position_m + 0.080 for s in stages]

    ke_bounds: list[float] = [records[0].kinetic_energy_j]
    for thresh in thresholds:
        for r in records:
            if r.position_m >= thresh:
                ke_bounds.append(r.kinetic_energy_j)
                break
        else:
            ke_bounds.append(records[-1].kinetic_energy_j)

    delta_ke = [max(0.0, ke_bounds[k + 1] - ke_bounds[k])
                for k in range(len(stages))]
    return delta_ke


# ═══════════════════════════════════════════════════════════════════════════════
# SIX-PANEL RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

def render_six_panel(
    records:     list[MultiStageRecord],
    stages:      list[AccelerationStage],
    projectile:  Projectile,
    fired_times: list[float | None],
    delta_ke:    list[float],
    output_path: str,
    dpi:         int,
) -> None:
    """
    Six-panel Neo-Classical figure: coilgun_neoclassical.png.

    Panel 1 (top-left):   Barrel schematic — coils, projectile, efficiency labels.
    Panel 2 (top-center): Position vs time — coil centers, fire event markers.
    Panel 3 (top-right):  Velocity vs time — PRIMARY RESULT, three acceleration events.
    Panel 4 (bot-left):   Force vs position — accelerating + braking zones.
    Panel 5 (bot-center): Current pulses — overlaid I(t) per stage, relative time.
    Panel 6 (bot-right):  Energy audit — horizontal bar chart, per-stage ΔKE.
    """
    # ── extract arrays ────────────────────────────────────────────────────────
    t_ms   = np.array([r.time_s         * 1e3  for r in records])
    z_mm   = np.array([r.position_m     * 1e3  for r in records])
    v_arr  = np.array([r.velocity_ms           for r in records])
    F_arr  = np.array([r.net_force_n           for r in records])
    KE_arr = np.array([r.kinetic_energy_j      for r in records])

    n_stages = len(stages)
    sc = [[r.stage_currents[k] for r in records] for k in range(n_stages)]
    sf = [[r.stage_forces[k]   for r in records] for k in range(n_stages)]

    v_final    = records[-1].velocity_ms
    KE_final   = records[-1].kinetic_energy_j
    E_total    = sum(s.energy_stored_j for s in stages)
    eta_cumul  = KE_final / E_total * 100.0
    eta_stages = [delta_ke[k] / stages[k].energy_stored_j * 100.0
                  for k in range(n_stages)]

    # peak velocity position (for schematic projectile marker)
    peak_v_idx  = int(np.argmax(v_arr))
    z_peak_v_mm = float(z_mm[peak_v_idx])

    # fired times in ms
    ft_ms = [f * 1e3 if f is not None else None for f in fired_times]

    # ── global style ──────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":       "monospace",
        "font.size":         8.5,
        "axes.facecolor":    NC_BG,
        "figure.facecolor":  NC_BG,
        "text.color":        NC_TICK,
        "axes.labelcolor":   NC_TICK,
        "xtick.color":       NC_TICK,
        "ytick.color":       NC_TICK,
        "xtick.major.size":  3.5,
        "ytick.major.size":  3.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    gs  = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.42,
                            left=0.06, right=0.97, top=0.89, bottom=0.07)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    fig.patch.set_facecolor(NC_BG)

    def _style_ax(ax, xlabel: str, ylabel: str) -> None:
        ax.set_facecolor(NC_BG)
        for sp in ax.spines.values():
            sp.set_color(NC_SPINE)
        ax.tick_params(colors=NC_TICK, length=3.5, width=0.5)
        ax.grid(True, color=NC_GRID, linewidth=0.5, linestyle="--", alpha=0.7)
        ax.axhline(0, color=NC_ZERO, linewidth=0.9, zorder=1)
        ax.set_xlabel(xlabel, color=NC_TICK, fontsize=8.0, labelpad=5)
        ax.set_ylabel(ylabel, color=NC_TICK, fontsize=8.0, labelpad=5)

    def _formula_box(ax, text: str) -> None:
        ax.text(0.04, 0.04, text, transform=ax.transAxes,
                fontsize=6.5, color=NC_FORMULA, va="bottom", ha="left",
                linespacing=1.5,
                bbox=dict(boxstyle="square,pad=0.30", facecolor=NC_BG,
                          edgecolor=NC_SPINE, lw=0.5))

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 1 — Barrel schematic
    # ══════════════════════════════════════════════════════════════════════════
    ax1.set_facecolor(NC_BG)
    ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_color(NC_SPINE)
    ax1.tick_params(colors=NC_TICK)

    Z_TOTAL_MM = 870.0
    ax1.set_xlim(-30, Z_TOTAL_MM)
    ax1.set_ylim(-0.72, 0.90)
    ax1.set_xlabel("z  [mm]  (barrel axis)", color=NC_TICK, fontsize=8.0, labelpad=5)
    ax1.set_title("Barrel Schematic — Three-Stage Coilgun",
                  color=NC_TITLE, fontsize=9.5, fontweight="bold", pad=8)

    # bore interior
    ax1.add_patch(Rectangle((0, -0.25), Z_TOTAL_MM, 0.50,
                             facecolor="#0c0c0c", zorder=2))
    # top barrel wall
    ax1.add_patch(Rectangle((0, 0.25), Z_TOTAL_MM, 0.17,
                             facecolor="#1a1a1a", zorder=3))
    # bottom barrel wall
    ax1.add_patch(Rectangle((0, -0.42), Z_TOTAL_MM, 0.17,
                             facecolor="#1a1a1a", zorder=3))
    # outer outline
    ax1.add_patch(Rectangle((0, -0.42), Z_TOTAL_MM, 0.84,
                             fill=False, edgecolor="#2e2e2e", lw=1.2, zorder=4))

    # coil windings per stage
    L_coil_mm = stages[0].coil.length_m * 1e3   # 80mm
    for k, stage in enumerate(stages):
        z_c = stage.center_position_m * 1000
        col = STAGE_COLORS[k]
        z_l = z_c - L_coil_mm / 2
        # upper winding block
        ax1.add_patch(Rectangle((z_l, 0.25), L_coil_mm, 0.20,
                                 facecolor=col, alpha=0.32, zorder=5))
        # lower winding block
        ax1.add_patch(Rectangle((z_l, -0.45), L_coil_mm, 0.20,
                                 facecolor=col, alpha=0.32, zorder=5))
        # coil outline
        ax1.add_patch(Rectangle((z_l, -0.45), L_coil_mm, 0.90,
                                 fill=False, edgecolor=col, lw=0.9, zorder=6))
        # center dashed line
        ax1.axvline(z_c, ymin=0.08, ymax=0.92, color=col,
                    lw=0.6, ls=":", alpha=0.45, zorder=4)
        # trigger dashed line in bore
        z_tr = stage.trigger_position_m * 1000
        ax1.axvline(z_tr, ymin=0.28, ymax=0.72, color=col,
                    lw=0.7, ls="--", alpha=0.45, zorder=4)
        # efficiency label above coil
        ax1.text(z_c, 0.60,
                 f"S{k+1}\nη={eta_stages[k]:.1f}%",
                 ha="center", va="bottom", fontsize=7.0,
                 color=col, fontfamily="monospace")

    # projectile at peak velocity (gold rectangle in bore)
    proj_half = projectile.length_m / 2 * 1000   # mm
    proj_bore = 0.19   # normalized half-height (7.5mm / 20mm bore × 0.25)
    ax1.add_patch(Rectangle((z_peak_v_mm - proj_half, -proj_bore),
                             2 * proj_half, 2 * proj_bore,
                             facecolor="#c9a84c", edgecolor="#e0c06a",
                             lw=0.7, alpha=0.90, zorder=8))
    ax1.text(z_peak_v_mm, -0.32, f"v_max\n{float(np.max(v_arr)):.0f} m/s",
             ha="center", va="top", fontsize=6.5, color="#c9a84c")

    # entry arrow
    ax1.annotate("", xy=(20, 0), xytext=(-20, 0),
                 arrowprops=dict(arrowstyle="->", color="#c9a84c",
                                 lw=0.9, alpha=0.7),
                 zorder=9)
    ax1.text(0, 0.08, "v₀=0", fontsize=5.8, color=NC_TICK, ha="center")

    # barrel exit label
    ax1.text(Z_TOTAL_MM - 5, 0, "exit →",
             va="center", ha="right", fontsize=6.5, color=NC_TICK, alpha=0.6)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Position vs Time
    # ══════════════════════════════════════════════════════════════════════════
    _style_ax(ax2, "t  [ms]", "z  [mm]")
    ax2.set_title("Position vs Time", color=NC_TITLE,
                  fontsize=9.5, fontweight="bold", pad=8)

    # stage center horizontal reference lines
    for k, stage in enumerate(stages):
        z_c_mm = stage.center_position_m * 1000
        ax2.axhline(z_c_mm, color=STAGE_COLORS[k], lw=0.65,
                    ls=":", alpha=0.55, zorder=2)
        ax2.text(t_ms[-1] * 0.98, z_c_mm + 8,
                 f"z_c{k+1}", fontsize=6.2, color=STAGE_COLORS[k],
                 ha="right", va="bottom", alpha=0.80)

    ax2.plot(t_ms, z_mm, color="#d8d8d8", lw=1.6, zorder=3)

    # fire event markers (upward triangle at z(t_fire))
    for k, ft in enumerate(ft_ms):
        if ft is not None:
            step = int(round(ft / 1e3 / DT))
            step = min(step, len(records) - 1)
            z_f = records[step].position_m * 1e3
            ax2.scatter([ft], [z_f], color=STAGE_COLORS[k], s=40,
                        marker="^", zorder=6)
            ax2.text(ft + 0.25, z_f + 15,
                     f"S{k+1} fires", fontsize=6.2,
                     color=STAGE_COLORS[k], alpha=0.80)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 3 — Velocity vs Time  (PRIMARY RESULT)
    # ══════════════════════════════════════════════════════════════════════════
    _style_ax(ax3, "t  [ms]", "v(t)  [m/s]")
    ax3.set_title("Velocity vs Time  — Primary Result",
                  color=NC_TITLE, fontsize=9.5, fontweight="bold", pad=8)

    # per-stage coloring via fill segments between fire events
    # Build boundaries: [0, t_fire_0, t_fire_1, t_fire_2, t_end]
    fire_ms_vals = [ft for ft in ft_ms if ft is not None]
    boundaries   = [0.0] + sorted(fire_ms_vals) + [t_ms[-1]]

    for k in range(len(boundaries) - 1):
        t_lo, t_hi = boundaries[k], boundaries[k + 1]
        mask = (t_ms >= t_lo) & (t_ms <= t_hi)
        col  = STAGE_COLORS[min(k, n_stages - 1)]
        ax3.fill_between(t_ms[mask], v_arr[mask], 0,
                         color=col, alpha=0.08, zorder=0)

    # main velocity curve
    ax3.plot(t_ms, v_arr, color="#d8d8d8", lw=1.8, zorder=3)

    # fire event vertical lines
    for k, ft in enumerate(ft_ms):
        if ft is not None:
            ax3.axvline(ft, color=STAGE_COLORS[k], lw=0.7, ls="--",
                        alpha=0.50, zorder=2)
            trans = blended_transform_factory(ax3.transData, ax3.transAxes)
            ax3.text(ft + 0.15, 0.97, f"S{k+1}",
                     transform=trans, fontsize=6.5,
                     color=STAGE_COLORS[k], va="top", alpha=0.80)

    # key velocity annotations
    ax3.scatter([0], [0], color="#d8d8d8", s=30, zorder=5)
    ax3.scatter([t_ms[-1]], [v_final], color=NC_TITLE, s=30, zorder=5)
    ax3.annotate(f"v_final = {v_final:.1f} m/s",
                 xy=(t_ms[-1], v_final),
                 xytext=(t_ms[-1] - 6, v_final * 0.82),
                 fontsize=7.5, color=NC_TITLE, ha="left",
                 arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7))

    _formula_box(ax3,
        "dv/dt = ΣF_k(z,t) / m\n"
        f"v_final = {v_final:.1f} m/s\n"
        f"η_total = {eta_cumul:.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 4 — Force vs Position
    # ══════════════════════════════════════════════════════════════════════════
    _style_ax(ax4, "z  [mm]  (barrel position)", "F_net  [N]")
    ax4.set_title("Net Force vs Position",
                  color=NC_TITLE, fontsize=9.5, fontweight="bold", pad=8)

    ax4.fill_between(z_mm, F_arr, 0, where=(F_arr >= 0),
                     color=STAGE_COLORS[0], alpha=0.18, zorder=0)
    ax4.fill_between(z_mm, F_arr, 0, where=(F_arr < 0),
                     color="#1a1a3a", alpha=0.30, zorder=0)
    ax4.plot(z_mm, F_arr, color=STAGE_COLORS[0], lw=1.4, zorder=3)

    # stage center markers
    for k, stage in enumerate(stages):
        ax4.axvline(stage.center_position_m * 1000,
                    color=STAGE_COLORS[k], lw=0.6, ls=":", alpha=0.50, zorder=2)

    peak_F = float(F_arr[np.argmax(np.abs(F_arr))])
    peak_F_z = float(z_mm[np.argmax(np.abs(F_arr))])
    ax4.annotate(f"Peak = {abs(peak_F):.0f} N",
                 xy=(peak_F_z, peak_F),
                 xytext=(peak_F_z + 30, peak_F * 0.80),
                 fontsize=7.0, color=STAGE_COLORS[0],
                 arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7))

    _formula_box(ax4,
        "F_net = Σ_k F_k(z,t)\n"
        "F_k = (μr−1)·V·μ0·n²·I_k²·f·f′\n"
        "sign(F_k) = sign(f′(z_rel_k))")

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 5 — Current Pulses  (relative time, overlaid)
    # ══════════════════════════════════════════════════════════════════════════
    _style_ax(ax5, "t_rel  [ms]  (time since stage fired)", "I(t)  [A]")
    ax5.set_title("Current Pulses  I(t)  per Stage",
                  color=NC_TITLE, fontsize=9.5, fontweight="bold", pad=8)

    t_peak_ms_ref = stages[0].t_peak_s * 1e3  # same for all (same circuit)
    pulse_show_ms = stages[0].pulse_duration_s * 2.5 * 1e3  # x-axis width

    for k, (stage, fired_t) in enumerate(zip(stages, fired_times)):
        if fired_t is None:
            continue
        col     = STAGE_COLORS[k]
        t_arr_s = np.array([r.time_s for r in records])
        mask    = t_arr_s >= fired_t
        t_rel   = (t_arr_s[mask] - fired_t) * 1e3
        I_k     = np.array(sc[k])[mask]

        ax5.plot(t_rel, I_k, color=col, lw=1.5, zorder=3,
                 label=f"S{k+1}  (fires at {fired_t*1e3:.2f} ms)")
        ax5.axvline(t_peak_ms_ref, color=col, lw=0.6, ls=":", alpha=0.55)

    ax5.set_xlim(0, min(pulse_show_ms, t_ms[-1]))
    ax5.legend(loc="upper right", fontsize=6.5, framealpha=0.25,
               edgecolor=NC_SPINE, facecolor=NC_BG)

    # t_peak annotation (same for all stages)
    trans5 = blended_transform_factory(ax5.transData, ax5.transAxes)
    ax5.text(t_peak_ms_ref + 0.1, 0.96,
             f"t_peak\n{t_peak_ms_ref:.2f} ms",
             transform=trans5, fontsize=6.2, color=NC_TICK,
             va="top", ha="left", alpha=0.70)

    _formula_box(ax5,
        f"I(t) = (V₀/ωd·L)·e^(−αt)·sin(ωdt)\n"
        f"t_peak = {t_peak_ms_ref:.3f} ms  (all stages)\n"
        f"π/ωd   = {stages[0].pulse_duration_s*1e3:.3f} ms")

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 6 — Energy Audit
    # ══════════════════════════════════════════════════════════════════════════
    ax6.set_facecolor(NC_BG)
    for sp in ax6.spines.values():
        sp.set_color(NC_SPINE)
    ax6.tick_params(colors=NC_TICK, length=3.5, width=0.5)
    ax6.grid(True, axis="x", color=NC_GRID, lw=0.5, ls="--", alpha=0.7)
    ax6.set_title("Energy Audit — Per Stage",
                  color=NC_TITLE, fontsize=9.5, fontweight="bold", pad=8)
    ax6.set_xlabel("Energy  [J]", color=NC_TICK, fontsize=8.0, labelpad=5)

    E_stored = stages[0].energy_stored_j   # 320 J per stage
    y_positions = [0, 1, 2]
    y_labels    = ["S1", "S2", "S3"]

    for i in range(n_stages):
        col = STAGE_COLORS[i]
        dke = delta_ke[i]
        eta = eta_stages[i]

        # full input bar (background)
        ax6.barh(i, E_stored, height=0.55,
                 color="#1c1c1c", edgecolor="#2e2e2e", lw=0.5)
        # ΔKE portion
        ax6.barh(i, dke, height=0.55, color=col, alpha=0.80)
        # Joule heating label in the gap
        joule = E_stored - dke
        ax6.text(dke + joule / 2, i,
                 f"Q = {joule:.0f} J",
                 va="center", ha="center", fontsize=6.0, color=NC_TICK)
        # efficiency label to the right
        ax6.text(E_stored + 4, i,
                 f"η = {eta:.1f}%",
                 va="center", ha="left", fontsize=7.0, color=col)

    ax6.set_yticks(y_positions)
    ax6.set_yticklabels(y_labels, fontsize=9, color=NC_TICK)
    ax6.set_xlim(0, E_stored * 1.28)
    ax6.set_ylim(-0.6, n_stages - 0.4)
    ax6.axvline(E_stored, color=NC_ZERO, lw=0.7, ls=":", alpha=0.5, zorder=2)
    ax6.text(E_stored + 1, n_stages - 0.55,
             f"E_in = {E_stored:.0f} J",
             fontsize=6.5, color=NC_TICK, va="top", ha="left", alpha=0.70)

    # cumulative summary inside panel
    ax6.text(0.05, 0.04,
             f"Total ΔKE = {KE_final:.1f} J  /  {E_total:.0f} J\n"
             f"η_cumulative = {eta_cumul:.1f}%\n"
             f"v_final = {v_final:.1f} m/s",
             transform=ax6.transAxes, fontsize=7.0, color=NC_TITLE,
             va="bottom", ha="left", linespacing=1.55)

    # ── figure title and subtitle ─────────────────────────────────────────────
    projectile_info = (
        f"m = {projectile.mass_kg*1e3:.0f} g  ·  "
        f"r = {projectile.radius_m*1e3:.1f} mm  ·  "
        f"L = {projectile.length_m*1e3:.0f} mm  ·  "
        f"μ_r = {projectile.relative_permeability:.0f}"
    )
    stage_info = (
        f"3 stages  ·  z_c = [200, 450, 700] mm  ·  "
        f"250mm spacing  ·  "
        f"C = 4 mF  ·  V₀ = 400 V  ·  R = 0.15 Ω  ·  "
        f"E_total = {E_total:.0f} J"
    )

    fig.suptitle(
        "COILGUN THREE-STAGE ACCELERATOR — MULTI-STAGE SIMULATION  |  Steps 5-6 of 6  |  "
        f"v_final = {v_final:.1f} m/s  ·  η = {eta_cumul:.1f}%",
        color=NC_TITLE, fontsize=11.5, fontweight="bold", y=0.95,
    )
    fig.text(0.5, 0.918, projectile_info,
             ha="center", va="top", color=NC_TICK, fontsize=8.0)
    fig.text(0.5, 0.902, stage_info,
             ha="center", va="top", color=NC_TICK, fontsize=7.5)

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                pad_inches=0.15, facecolor=NC_BG)
    plt.close(fig)
    print(f"  [render]      → {output_path}")
    print(f"  [render]      {FIGURE_WIDTH_IN:.0f} × {FIGURE_HEIGHT_IN:.0f} in  at {dpi} DPI")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def console_summary(
    stages:      list[AccelerationStage],
    projectile:  Projectile,
    records:     list[MultiStageRecord],
    fired_times: list[float | None],
    delta_ke:    list[float],
) -> None:
    E_stored  = stages[0].energy_stored_j
    E_total   = E_stored * len(stages)
    KE_final  = records[-1].kinetic_energy_j
    v_final   = records[-1].velocity_ms
    eta_cum   = KE_final / E_total * 100.0

    def _sep(ch="─", w=66):
        print(f"  {ch * w}")

    print()
    _sep("═")
    print("  ║  COILGUN 3-STAGE SIMULATION — FINAL RESULTS")
    _sep("═")
    print()
    print(f"  {'Stage':<8} {'z_trigger':>10} {'z_center':>10} "
          f"{'Entry v':>10} {'ΔKE':>8} {'η':>7}")
    _sep()
    for k, stage in enumerate(stages):
        v_entry = get_entry_velocity(records, fired_times[k], DT)
        dke     = delta_ke[k]
        eta_k   = dke / stage.energy_stored_j * 100.0
        print(f"  Stage {k+1:<3}  "
              f"{stage.trigger_position_m*1e3:>8.0f} mm  "
              f"{stage.center_position_m*1e3:>8.0f} mm  "
              f"{v_entry:>8.1f} m/s  "
              f"{dke:>6.1f} J  "
              f"{eta_k:>5.1f}%")
    _sep()
    print(f"  Total ΔKE   = {KE_final:.1f} J  of {E_total:.0f} J  "
          f"(3 × {E_stored:.0f} J)")
    print(f"  v_final     = {v_final:.2f} m/s")
    print(f"  η_total     = {eta_cum:.2f}%")
    print()
    print("  Known simplifications (all documented in module docstring):")
    print(f"    μ_r = {projectile.relative_permeability:.0f} constant  "
          f"(ignores saturation, B_peak ≈ {4*math.pi*1e-7*2000*839*0.894:.2f} T)")
    print(f"    Centroid force  (L_proj = L_c/2, ~10% spatial error)")
    print(f"    Constant-L RLC  (real L rises 30-80% in bore)")
    print(f"    No eddy drag, no friction, no gravity")
    _sep("═")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stage3_dke_signed(
    records: list[MultiStageRecord],
    stages:  list[AccelerationStage],
) -> float:
    """
    Stage 3 kinetic energy change [J], signed.

    Positive = net accelerator. Negative = net brake.
    Measured from the point the projectile exits Stage 2's force zone
    (z >= z_c2 + 80mm) to when it exits Stage 3's force zone
    (z >= z_c3 + 80mm). Falls back to last record if thresholds not reached.

    Unlike compute_stage_delta_ke(), this returns the raw signed value —
    necessary for crossover detection in the spacing sweep.
    """
    thresh_entry = stages[1].center_position_m + 0.080   # Stage 2 exit
    thresh_exit  = stages[2].center_position_m + 0.080   # Stage 3 exit

    ke_entry: float | None = None
    ke_exit:  float        = records[-1].kinetic_energy_j

    for r in records:
        if ke_entry is None and r.position_m >= thresh_entry:
            ke_entry = r.kinetic_energy_j
        if r.position_m >= thresh_exit:
            ke_exit = r.kinetic_energy_j
            break

    if ke_entry is None:
        ke_entry = records[0].kinetic_energy_j

    return ke_exit - ke_entry


def sweep_stage_spacing(
    coil:          CoilGeometry,
    capacitor:     CapacitorBank,
    wire_r:        float,
    projectile:    Projectile,
    spacing_min_m: float = 0.200,
    spacing_max_m: float = 0.500,
    n_points:      int   = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """
    Sweep stage-to-stage spacing from spacing_min to spacing_max.

    At each spacing: rebuild all three stages (same coil and capacitor),
    run the 2-pass trigger computation, run the full 3-stage simulation,
    and record v_final, η_total, Stage 3 signed ΔKE.

    Returns
    ───────
    spacings_mm   : ndarray  — spacing values [mm]
    v_finals      : ndarray  — final velocity [m/s] per spacing
    eta_totals    : ndarray  — cumulative efficiency [%] per spacing
    stage3_dke    : ndarray  — Stage 3 signed ΔKE [J] per spacing
    crossover_mm  : float | None — spacing where Stage 3 transitions
                    from net-brake to net-accelerator [mm], None if never
    """
    spacings_m = np.linspace(spacing_min_m, spacing_max_m, n_points)
    v_finals   = np.zeros(n_points)
    eta_totals = np.zeros(n_points)
    stage3_dke = np.zeros(n_points)

    Z_C1 = 0.200   # Stage 1 center always at 200mm

    print()
    print(f"  [sweep]  {n_points} spacing points from "
          f"{spacing_min_m*1e3:.0f}mm to {spacing_max_m*1e3:.0f}mm")
    print(f"  [sweep]  {'spacing':>9} {'v_final':>9} {'η':>7} {'S3 ΔKE':>9}")
    print(f"  {'─'*50}")

    for i, sp in enumerate(spacings_m):
        z_centers = [Z_C1, Z_C1 + sp, Z_C1 + 2 * sp]

        # Integration time: sized for projectile to clear Stage 3 at
        # conservative 20 m/s minimum velocity.
        barrel_end  = z_centers[2] + 0.300
        t_total_sw  = barrel_end / 20.0
        t_single_sw = (z_centers[0] + 0.280) / 20.0
        t_double_sw = (z_centers[1] + 0.280) / 20.0

        z_trig1 = z_centers[0] - 0.060   # Stage 1: fixed 60mm offset

        # ── Pass 1: Stage 1 only ──────────────────────────────────────────
        s1  = make_stage(1, coil, capacitor, wire_r, z_centers[0], z_trig1)
        r1, _ = integrate_multi_stage([s1], projectile,
                                       z_trig1, 0.0, DT, t_single_sw)
        v1_exit = get_exit_velocity(r1, z_centers[0])

        # ── Pass 2: Stages 1+2 ────────────────────────────────────────────
        off2   = compute_trigger_offset(v1_exit)
        z_trig2 = max(z_centers[1] - off2, z_centers[0] + 0.020)
        s2  = make_stage(2, coil, capacitor, wire_r, z_centers[1], z_trig2)
        r12, _ = integrate_multi_stage([s1, s2], projectile,
                                        z_trig1, 0.0, DT, t_double_sw)
        v2_exit = get_exit_velocity(r12, z_centers[1])

        # ── Pass 3: Full 3-stage ──────────────────────────────────────────
        off3    = compute_trigger_offset(v2_exit)
        z_trig3 = max(z_centers[2] - off3, z_centers[1] + 0.020)
        s3  = make_stage(3, coil, capacitor, wire_r, z_centers[2], z_trig3)

        records, fired_times = integrate_multi_stage(
            [s1, s2, s3], projectile,
            z_trig1, 0.0, DT, t_total_sw,
        )

        KE_final      = records[-1].kinetic_energy_j
        E_total       = sum(s.energy_stored_j for s in [s1, s2, s3])
        v_finals[i]   = records[-1].velocity_ms
        eta_totals[i] = KE_final / E_total * 100.0
        stage3_dke[i] = compute_stage3_dke_signed(records, [s1, s2, s3])

        marker = " ← current design" if abs(sp - 0.250) < 1e-4 else ""
        print(f"  [{i+1:2d}/{n_points}]  "
              f"{sp*1e3:>6.0f} mm  "
              f"{v_finals[i]:>7.1f} m/s  "
              f"{eta_totals[i]:>5.1f}%  "
              f"{stage3_dke[i]:>+8.1f} J"
              f"{marker}")

    # ── crossover detection ───────────────────────────────────────────────────
    # Find the first sign change of stage3_dke: negative → positive (or zero).
    crossover_mm: float | None = None
    for i in range(n_points - 1):
        if stage3_dke[i] < 0 and stage3_dke[i + 1] >= 0:
            # Linear interpolation between the two bracketing points
            dke_lo, dke_hi = stage3_dke[i], stage3_dke[i + 1]
            sp_lo,  sp_hi  = spacings_m[i] * 1e3, spacings_m[i + 1] * 1e3
            t = -dke_lo / (dke_hi - dke_lo)
            crossover_mm = sp_lo + t * (sp_hi - sp_lo)
            break

    print()
    if crossover_mm is not None:
        print(f"  [sweep]  Stage 3 crossover (brake → accelerate) : "
              f"{crossover_mm:.1f} mm")
    else:
        print("  [sweep]  No crossover detected in swept range. "
              "Stage 3 remains a net brake across all spacings.")

    return spacings_m * 1e3, v_finals, eta_totals, stage3_dke, crossover_mm


def render_spacing_sweep(
    spacings_mm:  np.ndarray,
    v_finals:     np.ndarray,
    eta_totals:   np.ndarray,
    stage3_dke:   np.ndarray,
    crossover_mm: float | None,
    design_mm:    float,
    output_path:  str,
    dpi:          int,
) -> None:
    """
    Two-panel Neo-Classical sweep figure: coilgun_spacing_sweep.png.

    Panel 1 — v_final vs stage spacing (deep red).
      Current design point marked at 250mm. Crossover marked
      with a gold dotted vertical line.

    Panel 2 — Stage 3 ΔKE vs stage spacing (gold).
      Zero line in near-white. Negative region (net brake) filled
      muted blue. Positive region (net accelerator) filled deep red.
      Crossover annotated with spacing value.
    """
    plt.rcParams.update({
        "font.family":       "monospace",
        "font.size":         9,
        "axes.facecolor":    NC_BG,
        "figure.facecolor":  NC_BG,
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
        1, 2, figsize=(13.0, 5.5),
        gridspec_kw={"wspace": 0.42},
    )
    fig.patch.set_facecolor(NC_BG)

    def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
        ax.set_facecolor(NC_BG)
        for sp in ax.spines.values():
            sp.set_color(NC_SPINE)
        ax.tick_params(colors=NC_TICK, length=4, width=0.5)
        ax.grid(True, color=NC_GRID, lw=0.5, ls="--", alpha=0.7)
        ax.set_xlabel(xlabel, color=NC_TICK, fontsize=8.5, labelpad=6)
        ax.set_ylabel(ylabel, color=NC_TICK, fontsize=8.5, labelpad=6)
        ax.set_title(title, color=NC_TITLE, fontsize=10.0,
                     fontweight="bold", pad=10)

    # ── Panel 1: v_final vs spacing ──────────────────────────────────────────
    _style(ax1,
           "Stage spacing  [mm]  (center-to-center)",
           "v_final  [m/s]",
           "Exit Velocity vs Stage Spacing")

    ax1.plot(spacings_mm, v_finals, color=NC_F, lw=1.8, zorder=3)
    ax1.fill_between(spacings_mm, v_finals, v_finals.min() - 5,
                     color=NC_F, alpha=0.08, zorder=0)

    # Current design point at design_mm
    design_idx = int(np.argmin(np.abs(spacings_mm - design_mm)))
    ax1.scatter([spacings_mm[design_idx]], [v_finals[design_idx]],
                color=NC_F, s=60, zorder=5,
                edgecolors="#d8d8d8", linewidths=0.8)
    ax1.annotate(
        f"current design\n{spacings_mm[design_idx]:.0f} mm  "
        f"→  {v_finals[design_idx]:.1f} m/s",
        xy=(spacings_mm[design_idx], v_finals[design_idx]),
        xytext=(spacings_mm[design_idx] + 20,
                v_finals[design_idx] - (v_finals.max() - v_finals.min()) * 0.18),
        fontsize=7.5, color=NC_F, ha="left",
        arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7),
    )

    # Crossover vertical line (where Stage 3 becomes net positive)
    if crossover_mm is not None:
        ax1.axvline(crossover_mm, color=NC_TITLE, lw=1.0, ls=":",
                    alpha=0.75, zorder=4)
        trans = blended_transform_factory(ax1.transData, ax1.transAxes)
        ax1.text(crossover_mm + 4, 0.96,
                 f"crossover\n{crossover_mm:.0f} mm",
                 transform=trans, fontsize=7.2, color=NC_TITLE,
                 va="top", ha="left", alpha=0.85)
        # Also mark v_final at crossover (interpolated)
        v_at_cross = float(np.interp(crossover_mm, spacings_mm, v_finals))
        ax1.scatter([crossover_mm], [v_at_cross],
                    color=NC_TITLE, s=45, zorder=5,
                    edgecolors="#d8d8d8", linewidths=0.7, marker="D")

    ax1.set_xlim(spacings_mm[0] - 10, spacings_mm[-1] + 10)

    # ── Panel 2: Stage 3 ΔKE vs spacing ─────────────────────────────────────
    _style(ax2,
           "Stage spacing  [mm]  (center-to-center)",
           "Stage 3  ΔKE  [J]  (signed)",
           "Stage 3 Energy Balance vs Stage Spacing")

    # Fill regions
    ax2.fill_between(spacings_mm, stage3_dke, 0,
                     where=(stage3_dke < 0),
                     color="#1a1a3a", alpha=0.45, zorder=0,
                     label="net brake  (ΔKE < 0)")
    ax2.fill_between(spacings_mm, stage3_dke, 0,
                     where=(stage3_dke >= 0),
                     color=NC_F, alpha=0.25, zorder=0,
                     label="net accelerator  (ΔKE > 0)")

    # Zero line
    ax2.axhline(0, color="#d8d8d8", lw=1.0, zorder=2)

    # Main ΔKE curve
    ax2.plot(spacings_mm, stage3_dke, color=NC_TITLE, lw=1.8, zorder=3)

    # Region labels
    neg_region = stage3_dke[stage3_dke < 0]
    pos_region = stage3_dke[stage3_dke >= 0]

    mid_neg_x = spacings_mm[len(spacings_mm) // 4]
    ax2.text(mid_neg_x, stage3_dke.min() * 0.55,
             "net brake", ha="center", fontsize=8.0,
             color="#7070b0", alpha=0.80)

    if pos_region.size > 0:
        mid_pos_x = spacings_mm[-(len(spacings_mm) // 5)]
        ax2.text(mid_pos_x, stage3_dke.max() * 0.55,
                 "net\naccelerator", ha="center", fontsize=8.0,
                 color=NC_F, alpha=0.80)

    # Crossover annotation
    if crossover_mm is not None:
        ax2.axvline(crossover_mm, color=NC_TITLE, lw=1.0, ls=":",
                    alpha=0.75, zorder=4)
        trans2 = blended_transform_factory(ax2.transData, ax2.transAxes)
        ax2.text(crossover_mm + 4, 0.96,
                 f"crossover\n{crossover_mm:.0f} mm\n"
                 f"(minimum viable\nstage spacing)",
                 transform=trans2, fontsize=7.2, color=NC_TITLE,
                 va="top", ha="left", alpha=0.85)
        ax2.scatter([crossover_mm], [0.0],
                    color=NC_TITLE, s=55, zorder=5,
                    edgecolors="#d8d8d8", linewidths=0.8, marker="D")

    ax2.legend(loc="lower right", fontsize=6.8, framealpha=0.25,
               edgecolor=NC_SPINE, facecolor=NC_BG)
    ax2.set_xlim(spacings_mm[0] - 10, spacings_mm[-1] + 10)

    # ── figure title ─────────────────────────────────────────────────────────
    cross_str = (f"{crossover_mm:.0f} mm" if crossover_mm is not None
                 else "not found in range")
    fig.suptitle(
        "COILGUN STAGE SPACING SWEEP  |  "
        f"Minimum viable spacing = {cross_str}",
        color=NC_TITLE, fontsize=11.0, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, 0.995,
        "C = 4 mF  ·  V₀ = 400 V  ·  R = 0.15 Ω  ·  "
        "m = 50 g  ·  μ_r = 200  ·  3 stages  ·  fixed Stage 1 offset = 60 mm",
        ha="center", va="top", color=NC_TICK, fontsize=7.8,
    )

    # ── save ─────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                pad_inches=0.15, facecolor=NC_BG)
    plt.close(fig)
    print(f"  [sweep render]  → {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coilgun Steps 5-6: three-stage multi-coil simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output PNG path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--sweep", action="store_true",
                        help="Run stage spacing sweep (200–500mm, 30 points) "
                             "after the main simulation.")
    parser.add_argument("--sweep-output", type=str, default=SWEEP_OUTPUT,
                        help=f"Sweep figure output path. Default: {SWEEP_OUTPUT}")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Steps 5-6 pipeline:
      1. Construct locked geometry (Steps 1–4 parameters).
      2. Pass 1 — Stage 1 only: compute v1_exit.
      3. Compute Stage 2 trigger from v1_exit.
      4. Pass 2 — Stages 1+2: compute v2_exit.
      5. Compute Stage 3 trigger from v2_exit.
      6. Pass 3 — Full 3-stage simulation.
      7. Compute per-stage ΔKE and efficiency.
      8. Console summary and six-panel figure.
    """
    args = parse_arguments()

    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║   COILGUN SIMULATION — STEPS 5-6: THREE-STAGE CHAINING        ║")
    print("  ║   Rizky Meilandi Saputra  |  hybrid-architect-lab           ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")

    # ── locked geometry from Steps 1–4 ──────────────────────────────────────
    coil = CoilGeometry(
        radius_m        = 0.020,
        length_m        = 0.080,
        turns_per_meter = 2000.0,
    )
    capacitor = CapacitorBank(
        capacitance_f     = 4.0e-3,
        initial_voltage_v = 400.0,
    )
    WIRE_R = 0.15   # Ω  — same for all stages

    projectile = Projectile(
        mass_kg               = 0.050,
        radius_m              = 0.0075,
        length_m              = 0.040,
        relative_permeability = 200.0,
    )

    print()
    print(f"  [geometry]  coil: R={coil.radius_m*1e3:.0f}mm  "
          f"L={coil.length_m*1e3:.0f}mm  n={coil.turns_per_meter:.0f}/m  "
          f"→ L_air={coil.inductance_h*1e6:.1f}μH")
    print(f"  [circuit]   C={capacitor.capacitance_f*1e3:.0f}mF  "
          f"V₀={capacitor.initial_voltage_v:.0f}V  R={WIRE_R:.3f}Ω  "
          f"→ E={capacitor.energy_stored_j:.0f}J per stage")
    print(f"  [projectile] m={projectile.mass_kg*1e3:.0f}g  "
          f"r={projectile.radius_m*1e3:.1f}mm  "
          f"L={projectile.length_m*1e3:.0f}mm  "
          f"μ_r={projectile.relative_permeability:.0f}")
    print(f"  [stages]    z_centers = {[f'{z*1e3:.0f}mm' for z in Z_CENTERS_M]}")

    # ── PASS 1: Stage 1 only ─────────────────────────────────────────────────
    stage1 = make_stage(1, coil, capacitor, WIRE_R,
                        Z_CENTERS_M[0], Z_TRIGGER_S1_M)

    print()
    print(f"  [pass 1]  Stage 1  z_trigger = {stage1.trigger_position_m*1e3:.0f} mm  "
          f"(offset = {(stage1.center_position_m - stage1.trigger_position_m)*1e3:.0f} mm)")
    print(f"  [pass 1]  running {T_SINGLE*1e3:.0f} ms simulation...", end="", flush=True)

    r1, ft1 = integrate_multi_stage([stage1], projectile,
                                     stage1.trigger_position_m, 0.0,
                                     DT, T_SINGLE)
    v1_exit = get_exit_velocity(r1, stage1.center_position_m)
    print(f" v1_exit = {v1_exit:.2f} m/s")

    # ── PASS 2: Stages 1+2 ───────────────────────────────────────────────────
    off2   = compute_trigger_offset(v1_exit)
    stage2 = make_stage(2, coil, capacitor, WIRE_R,
                        Z_CENTERS_M[1], Z_CENTERS_M[1] - off2)

    print(f"  [pass 2]  Stage 2  z_trigger = {stage2.trigger_position_m*1e3:.0f} mm  "
          f"(offset = {off2*1e3:.0f} mm from v1={v1_exit:.1f} m/s)")
    print(f"  [pass 2]  running {T_DOUBLE*1e3:.0f} ms simulation...", end="", flush=True)

    r12, ft12 = integrate_multi_stage([stage1, stage2], projectile,
                                       stage1.trigger_position_m, 0.0,
                                       DT, T_DOUBLE)
    v2_exit = get_exit_velocity(r12, stage2.center_position_m)
    print(f" v2_exit = {v2_exit:.2f} m/s")

    # ── PASS 3: Full 3-stage ──────────────────────────────────────────────────
    off3   = compute_trigger_offset(v2_exit)
    stage3 = make_stage(3, coil, capacitor, WIRE_R,
                        Z_CENTERS_M[2], Z_CENTERS_M[2] - off3)

    stages = [stage1, stage2, stage3]

    print(f"  [pass 3]  Stage 3  z_trigger = {stage3.trigger_position_m*1e3:.0f} mm  "
          f"(offset = {off3*1e3:.0f} mm from v2={v2_exit:.1f} m/s)")
    print(f"  [pass 3]  running {T_TOTAL*1e3:.0f} ms full simulation...", end="", flush=True)

    records, fired_times = integrate_multi_stage(
        stages, projectile,
        stage1.trigger_position_m, 0.0,
        DT, T_TOTAL,
    )

    v_final = records[-1].velocity_ms
    print(f" v_final = {v_final:.2f} m/s")

    # ── per-stage ΔKE ─────────────────────────────────────────────────────────
    delta_ke = compute_stage_delta_ke(records, stages)
    E_total  = sum(s.energy_stored_j for s in stages)
    eta_cum  = records[-1].kinetic_energy_j / E_total * 100.0

    # ── console summary ───────────────────────────────────────────────────────
    console_summary(stages, projectile, records, fired_times, delta_ke)

    # ── render six-panel ──────────────────────────────────────────────────────
    print(f"  [render]  generating six-panel Neo-Classical figure...")
    render_six_panel(records, stages, projectile, fired_times,
                     delta_ke, args.output, OUTPUT_DPI)

    print()
    print(f"  [done]  Steps 5-6 complete. Simulation finished.")
    print(f"          v_final = {v_final:.2f} m/s  |  "
          f"ΔKE = {records[-1].kinetic_energy_j:.1f} J  |  "
          f"η = {eta_cum:.1f}%")

    # ── optional spacing sweep ────────────────────────────────────────────────
    if args.sweep:
        print()
        print("  ══════════════════════════════════════════════════════════════")
        print("  STAGE SPACING SWEEP  (200mm → 500mm, 30 points)")
        print("  ══════════════════════════════════════════════════════════════")
        spacings_mm, v_finals, eta_totals, stage3_dke, crossover_mm = \
            sweep_stage_spacing(coil, capacitor, WIRE_R, projectile)

        render_spacing_sweep(
            spacings_mm   = spacings_mm,
            v_finals      = v_finals,
            eta_totals    = eta_totals,
            stage3_dke    = stage3_dke,
            crossover_mm  = crossover_mm,
            design_mm     = 250.0,
            output_path   = args.sweep_output,
            dpi           = OUTPUT_DPI,
        )

        print()
        if crossover_mm is not None:
            print(f"  [result]  Minimum viable stage spacing : {crossover_mm:.1f} mm")
            print(f"            Below this, Stage 3 is a net brake for this")
            print(f"            circuit (C=4mF, V₀=400V, R=0.15Ω) and")
            print(f"            projectile (m=50g, μ_r=200).")
        print()
    print()


if __name__ == "__main__":
    main()
