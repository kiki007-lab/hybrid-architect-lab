"""
Module:       coilgun_dynamics.py
Purpose:      Steps 3-4 of 6 — force coupling and single-stage projectile dynamics.
              Couples the field model (Step 1: f(z), f′(z)) with the circuit
              model (Step 2: I(t)) through the Lorentz force expression, then
              integrates the equations of motion via RK4. Produces the first
              module with a velocity output — the number the simulation exists
              to compute.
Author:       Rizky Meilandi Saputra
Repository:   github.com/kiki007-lab/hybrid-architect-lab
Project:      Project 4 — Magnetic Linear Accelerator Simulation
Dependencies: coilgun_field_model, coilgun_rlc_model, numpy, matplotlib
Python:       3.10+

---

Force Expression
----------------
The axial force on a magnetically soft cylinder in a non-uniform field is
derived from the gradient of the stored magnetic energy:

    F_z = −d/dz [−χ_m · V_proj · B²/(2μ₀)]
        = (χ_m · V_proj / (2μ₀)) · d(B²)/dz

Substituting B_z(z,t) = μ₀ · n · I(t) · f(z) from Step 1:

    d(B²)/dz = 2 · (μ₀·n·I)² · f(z) · f′(z)

    F_z(z, t) = (μ_r − 1) · V_proj · μ₀ · n² · I(t)² · f(z) · f′(z)

Units verification
------------------
K = (μ_r − 1) · V_proj · μ₀ · n²

    [1] × [m³] × [H/m] × [m⁻²]
  = [m³] × [kg·m·s⁻²·A⁻²] × [m⁻²]
  = [kg·m²·s⁻²·A⁻²]
  = [H]   (Henries)

F = K · I² · f · f′

    [H] × [A²] × [1] × [m⁻¹]
  = [kg·m²·s⁻²·A⁻²] × [A²] × [m⁻¹]
  = [kg·m·s⁻²]
  = [N]   ✓

Equations of Motion
-------------------
    dz/dt = v
    dv/dt = F_z(z − z_center, t − t_fired) / m_proj

Integrated via RK4 with constant force per timestep. With dt = 5 μs and
peak force gradient ~500 kN/m, the spatial frequency is ω ≈ 3200 rad/s,
giving ~300 steps per cycle. Well within stability margins.

Known Simplifications (inherited from architecture)
-----------------------------------------------------
1. Constant μ_r — soft iron saturates near B_sat ≈ 1.5–2 T. At peak current,
   B at coil center ≈ 1.88 T, so partial saturation likely occurs.
   Conservative μ_r = 200 underestimates μ_r at low field and overestimates
   at saturation. Net effect: uncertain sign on error, rough 20–40% uncertainty.
2. Point-centroid force — evaluates force at the projectile centre of mass.
   Valid when L_proj << L_coil; here L_proj = L_c/2, so error ≈ 5–15%.
3. Constant-L RLC circuit — real L increases as slug enters bore, reducing
   peak current. Force error: ~30% underestimate of braking, ~15% overestimate
   of accelerating peak.
4. No eddy current drag, no friction, no gravity.
These are all documented in verification output. The simulation gives an
upper-bound efficiency estimate under the stated assumptions.
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
from matplotlib.transforms import blended_transform_factory

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from coilgun_field_model import CoilGeometry, field_profile, field_gradient, MU_0
from coilgun_rlc_model   import CapacitorBank, AccelerationStage, rlc_current


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION AND RENDER PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

DT_DEFAULT:    float = 5e-6      # 5 μs timestep — ~300 steps per force-field cycle
T_TOTAL:       float = 12e-3     # 12 ms total — captures pulse + constant-v coast

FIGURE_WIDTH_IN:     float = 18.0
FIGURE_HEIGHT_IN:    float = 6.5
OUTPUT_DPI:          int   = 150
DEFAULT_OUTPUT_PATH: str   = "coilgun_dynamics.png"


# ═══════════════════════════════════════════════════════════════════════════════
# NEO-CLASSICAL PALETTE — locked to Step 1 and Step 2
# ═══════════════════════════════════════════════════════════════════════════════

NC_BACKGROUND: str = "#0a0a0a"
NC_GRID:       str = "#1a1a1a"
NC_SPINE:      str = "#2a2a2a"
NC_TICK:       str = "#888888"
NC_ZERO_LINE:  str = "#383838"
NC_F:          str = "#8b0000"   # deep red   — force, accelerating zone
NC_FPRIME:     str = "#c9a84c"   # gold       — velocity (primary result)
NC_PRODUCT:    str = "#d8d8d8"   # near-white — kinetic energy
NC_TITLE:      str = "#c9a84c"
NC_FORMULA:    str = "#484848"


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Projectile:
    """
    Physical specification of the accelerated slug.

    Attributes
    ----------
    mass_kg : float
        Slug mass m [kg]. Sets the inertia and thus how strongly the
        force accelerates it. F = ma — lighter slugs reach higher
        velocities for the same impulse.
    radius_m : float
        Slug radius r_proj [m]. Must satisfy r_proj < R_c (coil bore radius)
        with mechanical clearance. Here 7.5 mm < 20 mm (bore). ✓
    length_m : float
        Slug axial length L_proj [m]. The point-centroid approximation
        for force evaluation is valid when L_proj ≪ L_c. Here L_proj = L_c/2,
        so a 5–15% spatial-averaging error is expected and accepted.
    relative_permeability : float
        μ_r [dimensionless] — relative magnetic permeability of the slug
        material. Soft iron: 200–5000. Using μ_r = 200 as a conservative
        lower bound that accounts for partial saturation at high fields.
    """
    mass_kg:                float
    radius_m:               float
    length_m:               float
    relative_permeability:  float

    @property
    def volume_m3(self) -> float:
        """V_proj = π · r² · L [m³]."""
        return math.pi * self.radius_m ** 2 * self.length_m

    @property
    def susceptibility(self) -> float:
        """χ_m = μ_r − 1 [dimensionless]."""
        return self.relative_permeability - 1.0

    @property
    def force_coefficient(self) -> float:
        """
        K = (μ_r − 1) · V_proj · μ₀  [H·m²·m⁻³ = H/m... no]

        Actually K absorbs everything except n², I², f, f′:

            K_full = (μ_r − 1) · V_proj · μ₀ · n²

        This property returns K_partial = (μ_r − 1) · V_proj · μ₀ [H·m]
        so that coupling_force can multiply by n² from the coil geometry.
        Kept here to make the dimensional structure explicit.
        """
        return self.susceptibility * self.volume_m3 * MU_0


@dataclass
class SimulationRecord:
    """One timestep snapshot from the integration loop."""
    time_s:           float    # t [s]
    position_m:       float    # z [m] — absolute barrel position
    velocity_ms:      float    # v [m/s]
    net_force_n:      float    # F_z [N]
    kinetic_energy_j: float    # ½mv² [J]
    z_rel_m:          float    # z − z_center [m] — position relative to coil center
    current_a:        float    # I(t) [A]


# ═══════════════════════════════════════════════════════════════════════════════
# FORCE COUPLING
# ═══════════════════════════════════════════════════════════════════════════════

def coupling_force(
    z_rel:      float,
    I:          float,
    coil:       CoilGeometry,
    projectile: Projectile,
) -> float:
    """
    Axial Lorentz force on the projectile [N].

        F_z = (μ_r − 1) · V_proj · μ₀ · n² · I² · f(z_rel) · f′(z_rel)

    z_rel is the projectile position measured relative to the coil center —
    the same coordinate frame used by field_profile and field_gradient.

    Sign convention (from f′ sign):
        z_rel < 0 → projectile behind center  → f′ > 0 → F_z > 0 (forward)
        z_rel = 0 → at coil center            → f′ = 0 → F_z = 0 (no force)
        z_rel > 0 → projectile past center    → f′ < 0 → F_z < 0 (braking)

    The I² dependence is the key: force scales quadratically with current,
    so doubling I quadruples the force. It also means the force decays at
    rate 2α (not α) — the effective force window is narrower than the current
    pulse. Verified in Step 2 Panel 2.

    Parameters
    ──────────
    z_rel : float
        Position of projectile centroid relative to coil center [m].
    I : float
        Instantaneous coil current [A]. I = 0 → F = 0, always.
    coil : CoilGeometry
        Coil geometry from Step 1.
    projectile : Projectile
        Projectile physical parameters.

    Returns
    ───────
    float — axial force [N]. Positive = forward along barrel axis.
    """
    fz  = float(field_profile( z_rel, coil))
    fpz = float(field_gradient(z_rel, coil))
    return (projectile.susceptibility
            * projectile.volume_m3
            * MU_0
            * coil.turns_per_meter ** 2
            * I ** 2
            * fz * fpz)


# ═══════════════════════════════════════════════════════════════════════════════
# RK4 INTEGRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def rk4_step(
    state:       np.ndarray,
    total_force: float,
    mass_kg:     float,
    dt:          float,
) -> np.ndarray:
    """
    Single RK4 step for the 2-state system y = [z, v].

    Governing ODE:
        dy/dt = [v,  F_z/m]

    With total_force treated as constant over the step (evaluated at step
    entry), the four RK4 slopes for the acceleration are identical:
        k1_v = k2_v = k3_v = k4_v = (F/m) · dt

    This collapses to the exact kinematic equations for constant acceleration:
        new_v = v + (F/m) · dt
        new_z = z + v·dt + ½·(F/m)·dt²

    which are analytically exact for the piecewise-constant-force model.
    The global integration error is first order in the force variation
    over dt, which is negligible at dt = 5 μs.

    Parameters
    ──────────
    state : ndarray [z, v]
        Current position [m] and velocity [m/s].
    total_force : float
        Net axial force on projectile [N], pre-computed at (z, t).
    mass_kg : float
        Projectile mass [kg].
    dt : float
        Timestep [s].

    Returns
    ───────
    ndarray [new_z, new_v]
    """
    z, v    = state
    a       = total_force / mass_kg

    k1_z = v            * dt;  k1_v = a * dt
    k2_z = (v + k1_v/2) * dt;  k2_v = a * dt
    k3_z = (v + k2_v/2) * dt;  k3_v = a * dt
    k4_z = (v + k3_v)   * dt;  k4_v = a * dt

    return np.array([
        z + (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6.0,
        v + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6.0,
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def integrate_single_stage(
    stage:      AccelerationStage,
    projectile: Projectile,
    z0:         float,
    v0:         float,
    dt:         float,
    t_total:    float,
) -> list[SimulationRecord]:
    """
    Integrate equations of motion for one acceleration stage.

    The stage fires the moment the projectile reaches stage.trigger_position_m.
    If z0 >= trigger_position_m (as in the single-stage test), the stage fires
    at t = 0 immediately.

    Force is zero before the stage fires and follows coupling_force after.
    The RLC current I(t_elapsed) is evaluated at elapsed time since firing.
    The field functions evaluate at z_rel = z − stage.center_position_m.

    Parameters
    ──────────
    stage : AccelerationStage   — circuit and geometry of the firing stage
    projectile : Projectile     — slug physical parameters
    z0 : float                  — initial projectile position [m]
    v0 : float                  — initial projectile velocity [m/s]
    dt : float                  — timestep [s]
    t_total : float             — total simulation duration [s]

    Returns
    ───────
    list[SimulationRecord] — one record per timestep, including t=0.
    """
    n_steps     = int(t_total / dt) + 1
    state       = np.array([z0, v0], dtype=float)
    records:    list[SimulationRecord] = []
    fired_time: float | None = None      # wall-clock t when stage fires

    for step in range(n_steps):
        t     = step * dt
        z, v  = state
        z_rel = z - stage.center_position_m

        # ── fire stage when projectile crosses trigger ────────────────────────
        if fired_time is None and z >= stage.trigger_position_m:
            fired_time = t

        # ── compute current and force ─────────────────────────────────────────
        if fired_time is not None:
            t_elapsed = t - fired_time
            I = float(rlc_current(t_elapsed, stage))
            F = float(coupling_force(z_rel, I, stage.coil, projectile))
        else:
            I = 0.0
            F = 0.0

        KE = 0.5 * projectile.mass_kg * v ** 2
        records.append(SimulationRecord(
            time_s           = t,
            position_m       = z,
            velocity_ms      = v,
            net_force_n      = F,
            kinetic_energy_j = KE,
            z_rel_m          = z_rel,
            current_a        = I,
        ))

        state = rk4_step(state, F, projectile.mass_kg, dt)

    return records


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def verify_dynamics(
    stage:      AccelerationStage,
    projectile: Projectile,
) -> list[SimulationRecord]:
    """
    Six analytical checks for coupling_force, rk4_step, and the integrator.
    Raises AssertionError if any check fails. Returns the simulation records
    (already computed for check 6) to avoid running the integration twice.

    Checks
    ──────
    [1] F = 0 when I = 0  (any z)
    [2] F = 0 at z_rel = 0 when I = I_peak  (f′(0) = 0 propagates exactly)
    [3] F > 0 for z_rel < 0  (forward force, approaching center)
    [4] F < 0 for z_rel > 0  (braking force, past center)
    [5] |F(z)| = |F(−z)|  (odd antisymmetry: F(z) = −F(−z))
    [6] ΔKE ≤ E_stored × 1.05  (energy conservation within simplification margin)
    """
    TOL_ZERO: float = 1e-20
    TOL_SYM:  float = 1e-10
    I_test:   float = stage.i_peak_a

    def _sep(ch: str = "─", w: int = 64) -> None:
        print(f"  {ch * w}")

    print()
    _sep("═")
    print("  ║  COILGUN DYNAMICS — ANALYTICAL VERIFICATION")
    _sep("═")
    print()
    print(f"  Projectile: m = {projectile.mass_kg*1e3:.0f} g  "
          f"| r = {projectile.radius_m*1e3:.1f} mm  "
          f"| L = {projectile.length_m*1e3:.0f} mm  "
          f"| μ_r = {projectile.relative_permeability:.0f}")
    print(f"  V_proj = {projectile.volume_m3*1e6:.4f} cm³  "
          f"| χ_m = μ_r − 1 = {projectile.susceptibility:.0f}")
    K_full = (projectile.susceptibility * projectile.volume_m3
              * MU_0 * stage.coil.turns_per_meter**2)
    print(f"  K = (μ_r−1)·V·μ₀·n² = {K_full:.6e} H  "
          f"[units: H × A² × m⁻¹ = N ✓]")
    print()

    # ── [1] F = 0 when I = 0 ──────────────────────────────────────────────────
    F1 = coupling_force(-0.030, 0.0, stage.coil, projectile)
    err1 = abs(F1)
    print("  [1]  coupling_force(z, I=0) = 0  for all z  [I² term kills force]")
    _sep()
    print(f"       Computed at z_rel = −30 mm : {F1:.6e} N")
    print(f"       Error    : {err1:.2e}   {'✓' if err1 < TOL_ZERO else '✗  FAILED'}")
    assert err1 < TOL_ZERO, f"[1] FAILED — F(I=0) = {F1}"

    # ── [2] F = 0 at z_rel = 0 with I = I_peak ────────────────────────────────
    # f′(0) = 0 exactly (Step 1 verification). This propagates exactly.
    F2 = coupling_force(0.0, I_test, stage.coil, projectile)
    err2 = abs(F2)
    print()
    print(f"  [2]  coupling_force(z=0, I=I_peak) = 0  [f′(0) = 0, Step 1 verified]")
    _sep()
    print(f"       I_peak = {I_test:.2f} A  |  z_rel = 0 (coil center)")
    print(f"       Computed : {F2:.6e} N")
    print(f"       Error    : {err2:.2e}   {'✓' if err2 < 1e-6 else '✗  FAILED'}")
    assert err2 < 1e-6, f"[2] FAILED — F at center = {F2:.2e} N"

    # ── [3] F > 0 before coil center (forward accelerating force) ─────────────
    F3 = coupling_force(-0.030, I_test, stage.coil, projectile)
    print()
    print("  [3]  F > 0 for z_rel < 0  (projectile approaching center — forward pull)")
    _sep()
    print(f"       F(z_rel = −30 mm, I_peak) = {F3:.2f} N  "
          f"{'> 0 ✓' if F3 > 0 else '≤ 0 ✗  FAILED'}")
    assert F3 > 0, f"[3] FAILED — F = {F3:.2f} N, expected > 0"

    # ── [4] F < 0 past coil center (braking force) ───────────────────────────
    F4 = coupling_force(+0.030, I_test, stage.coil, projectile)
    print()
    print("  [4]  F < 0 for z_rel > 0  (projectile past center — braking)")
    _sep()
    print(f"       F(z_rel = +30 mm, I_peak) = {F4:.2f} N  "
          f"{'< 0 ✓' if F4 < 0 else '≥ 0 ✗  FAILED'}")
    assert F4 < 0, f"[4] FAILED — F = {F4:.2f} N, expected < 0"

    # ── [5] Force antisymmetry: F(z) = −F(−z) ────────────────────────────────
    z_sym = 0.025   # 25 mm
    Fp = coupling_force(+z_sym, I_test, stage.coil, projectile)
    Fn = coupling_force(-z_sym, I_test, stage.coil, projectile)
    err5 = abs(Fp + Fn)
    print()
    print(f"  [5]  Antisymmetry: F(z) = −F(−z)  at z_rel = ±{z_sym*1e3:.0f} mm")
    _sep()
    print(f"       F(+{z_sym*1e3:.0f} mm) = {Fp:+.4f} N")
    print(f"       F(−{z_sym*1e3:.0f} mm) = {Fn:+.4f} N")
    print(f"       |F(z) + F(−z)| = {err5:.2e}   {'✓' if err5 < TOL_SYM else '✗  FAILED'}")
    assert err5 < TOL_SYM, f"[5] FAILED — symmetry residual {err5:.2e}"

    # ── [6] Energy conservation ───────────────────────────────────────────────
    print()
    print("  [6]  Energy conservation: ΔKE ≤ E_stored  (within simplification margin)")
    _sep()
    print("       Running integration... ", end="", flush=True)
    records = integrate_single_stage(
        stage, projectile,
        z0      = stage.trigger_position_m,
        v0      = 0.0,
        dt      = DT_DEFAULT,
        t_total = T_TOTAL,
    )
    print(f"done ({len(records)} steps)")

    delta_KE   = records[-1].kinetic_energy_j - records[0].kinetic_energy_j
    E_stored   = stage.energy_stored_j
    efficiency = delta_KE / E_stored * 100.0
    margin_ok  = delta_KE <= E_stored * 1.05   # 5% margin for numerical accumulation
    v_final    = records[-1].velocity_ms

    print(f"       E_stored = ½CV₀²    = {E_stored:.2f} J")
    print(f"       ΔKE (simulation)    = {delta_KE:.2f} J")
    print(f"       Stage efficiency    = {efficiency:.2f}%")
    print(f"       v_final             = {v_final:.2f} m/s")
    print(f"       ΔKE ≤ 1.05 × E     : {'✓' if margin_ok else '✗  FAILED  (physics error)'}")
    assert margin_ok, (f"[6] FAILED — ΔKE = {delta_KE:.2f} J > 1.05 × E_stored = "
                       f"{E_stored*1.05:.2f} J")

    print()
    _sep("═")
    print("  ║  All 6 checks passed. Dynamics model verified.")
    _sep("═")
    print()

    return records


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

def render_dynamics(
    records:     list[SimulationRecord],
    stage:       AccelerationStage,
    projectile:  Projectile,
    output_path: str,
    output_dpi:  int,
) -> None:
    """
    3-panel Neo-Classical figure for the single-stage dynamics result.

    Panel 1 — Force vs Position.
      Actual force experienced by the projectile as it traverses the coil,
      plotted as F(z_rel). Positive zone (red fill) = accelerating; negative
      zone (blue-dark fill) = braking. Peak force and coil center annotated.

    Panel 2 — Velocity vs Time.
      The primary simulation output. Shows the full velocity history from
      rest through the pulse. Entry, peak, and final velocities annotated.
      t_peak marked with a vertical reference line.

    Panel 3 — Kinetic Energy vs Time.
      Shows ΔKE gained and the stage efficiency relative to E_stored = ½CV₀².
      Plateaus after the current pulse ends — no further energy input.
    """
    # ── extract arrays from records ───────────────────────────────────────────
    t_ms    = np.array([r.time_s         * 1e3 for r in records])
    z_rel   = np.array([r.z_rel_m        * 1e3 for r in records])   # mm
    F_arr   = np.array([r.net_force_n          for r in records])   # N
    v_arr   = np.array([r.velocity_ms          for r in records])   # m/s
    KE_arr  = np.array([r.kinetic_energy_j     for r in records])   # J

    t_peak_ms    = stage.t_peak_s * 1e3
    v_final      = records[-1].velocity_ms
    KE_final     = records[-1].kinetic_energy_j
    efficiency   = KE_final / stage.energy_stored_j * 100.0

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

    fig, axes = plt.subplots(
        1, 3, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        gridspec_kw={"wspace": 0.42},
    )
    fig.patch.set_facecolor(NC_BACKGROUND)

    def _style(ax, xlabel: str, ylabel: str) -> None:
        ax.set_facecolor(NC_BACKGROUND)
        for sp in ax.spines.values():
            sp.set_color(NC_SPINE)
        ax.tick_params(colors=NC_TICK, length=4, width=0.5)
        ax.grid(True, color=NC_GRID, linewidth=0.5, linestyle="--", alpha=0.7)
        ax.set_xlabel(xlabel, color=NC_TICK, fontsize=8.5, labelpad=6)
        ax.set_ylabel(ylabel, color=NC_TICK, fontsize=8.5, labelpad=6)
        ax.axhline(y=0, color=NC_ZERO_LINE, linewidth=0.9, zorder=1)

    def _formula_box(ax, lines: str) -> None:
        ax.text(0.04, 0.04, lines, transform=ax.transAxes,
                fontsize=6.8, color=NC_FORMULA, va="bottom", ha="left",
                linespacing=1.55,
                bbox=dict(boxstyle="square,pad=0.35", facecolor=NC_BACKGROUND,
                          edgecolor=NC_SPINE, linewidth=0.5))

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 1 — Force vs Position
    # ═════════════════════════════════════════════════════════════════════════
    ax1 = axes[0]
    _style(ax1, "z  [mm]  (relative to coil center)", "F_z(z)  [N]")

    ax1.fill_between(z_rel, F_arr, 0, where=(F_arr >= 0),
                     color=NC_F, alpha=0.18, zorder=0)
    ax1.fill_between(z_rel, F_arr, 0, where=(F_arr < 0),
                     color="#1a1a3a", alpha=0.30, zorder=0)
    ax1.plot(z_rel, F_arr, color=NC_F, linewidth=1.6, zorder=3)

    ax1.axvline(x=0, color=NC_ZERO_LINE, linewidth=0.9, linestyle=":", zorder=2)
    trans1 = blended_transform_factory(ax1.transData, ax1.transAxes)
    ax1.text(1, 0.97, "coil\ncenter", transform=trans1,
             fontsize=6.5, color=NC_ZERO_LINE, va="top", ha="left")

    # peak force annotation
    peak_F_idx  = int(np.argmax(np.abs(F_arr)))
    peak_F      = F_arr[peak_F_idx]
    peak_F_zrel = z_rel[peak_F_idx]
    ax1.scatter([peak_F_zrel], [peak_F], color=NC_F, s=48, zorder=5,
                edgecolors=NC_TICK, linewidths=0.6)
    ax1.annotate(
        f"Peak = {peak_F:.0f} N\nz = {peak_F_zrel:.1f} mm",
        xy=(peak_F_zrel, peak_F),
        xytext=(peak_F_zrel + 8, peak_F * 0.72),
        fontsize=7.5, color=NC_F, ha="left",
        arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7),
    )

    ax1.text(0.62, 0.88, "accelerating\nzone", transform=ax1.transAxes,
             fontsize=7.5, color=NC_F, ha="center", alpha=0.85)
    ax1.text(0.85, 0.12, "braking\nzone", transform=ax1.transAxes,
             fontsize=7.5, color="#7070b0", ha="center", alpha=0.85)

    ax1.set_title("Force vs Position  F_z(z)",
                  color=NC_TITLE, fontsize=10.5, fontweight="bold", pad=10)
    _formula_box(ax1,
        "F_z = (μr−1)·Vproj·μ0·n²·I²·f(z)·f′(z)\n"
        "Force follows projectile as it moves\n"
        "and current builds and decays")

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Velocity vs Time  (primary result)
    # ═════════════════════════════════════════════════════════════════════════
    ax2 = axes[1]
    _style(ax2, "t  [ms]  (time since stage fired)", "v(t)  [m/s]")

    ax2.axvline(x=t_peak_ms, color=NC_FPRIME, linewidth=0.8,
                linestyle="--", alpha=0.50, zorder=2)
    trans2 = blended_transform_factory(ax2.transData, ax2.transAxes)
    ax2.text(t_peak_ms + 0.1, 0.97, f"t_peak\n{t_peak_ms:.2f} ms",
             transform=trans2, fontsize=6.8, color=NC_FPRIME,
             va="top", ha="left", alpha=0.75)

    ax2.plot(t_ms, v_arr, color=NC_FPRIME, linewidth=1.8, zorder=3)
    ax2.fill_between(t_ms, v_arr, 0, color=NC_FPRIME, alpha=0.07, zorder=0)

    # key velocity annotations
    v_entry = records[0].velocity_ms
    v_max   = float(np.max(v_arr))
    t_vmax  = t_ms[int(np.argmax(v_arr))]

    ax2.scatter([t_ms[0]], [v_entry], color=NC_FPRIME, s=40, zorder=5,
                edgecolors=NC_TICK, linewidths=0.5)
    ax2.scatter([t_vmax],  [v_max],   color=NC_FPRIME, s=40, zorder=5,
                edgecolors=NC_TICK, linewidths=0.5)
    ax2.scatter([t_ms[-1]], [v_final], color=NC_FPRIME, s=40, zorder=5,
                edgecolors=NC_TICK, linewidths=0.5)

    ax2.annotate(f"v_entry = {v_entry:.1f} m/s",
                 xy=(t_ms[0], v_entry),
                 xytext=(t_ms[0]+0.5, v_max*0.12),
                 fontsize=7.5, color=NC_FPRIME, ha="left",
                 arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7))
    ax2.annotate(f"v_max = {v_max:.1f} m/s",
                 xy=(t_vmax, v_max),
                 xytext=(t_vmax + 0.5, v_max * 0.85),
                 fontsize=7.5, color=NC_FPRIME, ha="left",
                 arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7))
    ax2.annotate(f"v_exit = {v_final:.1f} m/s",
                 xy=(t_ms[-1], v_final),
                 xytext=(t_ms[-1] - 3.5, v_final * 0.78),
                 fontsize=7.5, color=NC_FPRIME, ha="left",
                 arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7))

    ax2.set_title("Velocity vs Time  v(t)  — Primary Result",
                  color=NC_TITLE, fontsize=10.5, fontweight="bold", pad=10)
    _formula_box(ax2,
        "dv/dt = F_z(z,t) / m_proj\n"
        "dz/dt = v\n"
        "Integrated by RK4  (dt = 5 μs)")

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 3 — Kinetic Energy vs Time
    # ═════════════════════════════════════════════════════════════════════════
    ax3 = axes[2]
    _style(ax3, "t  [ms]  (time since stage fired)", "KE(t)  [J]")

    ax3.axvline(x=t_peak_ms, color=NC_FPRIME, linewidth=0.8,
                linestyle="--", alpha=0.50, zorder=2)

    ax3.plot(t_ms, KE_arr, color=NC_PRODUCT, linewidth=1.8, zorder=3)
    ax3.fill_between(t_ms, KE_arr, 0, color=NC_PRODUCT, alpha=0.07, zorder=0)

    # E_stored reference line
    E_stored = stage.energy_stored_j
    ax3.axhline(y=E_stored, color=NC_ZERO_LINE, linewidth=0.8,
                linestyle=":", alpha=0.60, zorder=2)
    ax3_trans = blended_transform_factory(ax3.transAxes, ax3.transData)
    ax3.text(0.02, E_stored + E_stored*0.02,
             f"E_stored = {E_stored:.0f} J",
             transform=blended_transform_factory(ax3.transAxes, ax3.transData),
             fontsize=7.0, color=NC_ZERO_LINE, alpha=0.70, va="bottom")

    # ΔKE annotation
    ax3.scatter([t_ms[-1]], [KE_final], color=NC_PRODUCT, s=48, zorder=5,
                edgecolors=NC_TICK, linewidths=0.6)
    ax3.annotate(
        f"ΔKE = {KE_final:.1f} J\nη = {efficiency:.1f}%",
        xy=(t_ms[-1], KE_final),
        xytext=(t_ms[-1] - 4.0, KE_final * 0.70),
        fontsize=7.5, color=NC_PRODUCT, ha="left",
        arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7),
    )

    ax3.set_title("Kinetic Energy vs Time  KE(t)",
                  color=NC_TITLE, fontsize=10.5, fontweight="bold", pad=10)
    _formula_box(ax3,
        f"ΔKE = {KE_final:.1f} J  |  E_stored = {E_stored:.0f} J\n"
        f"Efficiency η = ΔKE/E = {efficiency:.1f}%\n"
        "KE plateaus when current pulse ends")

    # ── figure-level title and parameter subtitle ──────────────────────────────
    fig.suptitle(
        "COILGUN SINGLE-STAGE DYNAMICS — FORCE COUPLING & RK4 INTEGRATION  |  Steps 3-4 of 6",
        color=NC_TITLE, fontsize=12, fontweight="bold", y=1.04,
    )
    fig.text(
        0.5, 0.997,
        (f"m = {projectile.mass_kg*1e3:.0f} g  ·  "
         f"r = {projectile.radius_m*1e3:.1f} mm  ·  "
         f"L = {projectile.length_m*1e3:.0f} mm  ·  "
         f"μ_r = {projectile.relative_permeability:.0f}  ·  "
         f"z_trigger = {stage.trigger_position_m*1e3:.0f} mm  ·  "
         f"z_center = {stage.center_position_m*1e3:.0f} mm  ·  "
         f"v_exit = {v_final:.1f} m/s  ·  η = {efficiency:.1f}%"),
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
        description="Coilgun Steps 3-4: force coupling and single-stage dynamics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python coilgun_dynamics.py --output ./coilgun_dynamics.png',
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
    Steps 3-4 pipeline:
      1. Construct locked geometry (Step 1 parameters) + demo circuit (Step 2)
      2. Construct projectile (50 g iron slug)
      3. Run 6-check verification — computes simulation internally for check [6]
      4. Render 3-panel Neo-Classical figure from simulation records

    Trigger geometry: z_trigger = z_center − 60 mm (initial estimate).
    The velocity profile shape in Panel 2 reveals whether timing is correct:
    if v(t) rises monotonically until the pulse ends, timing is adequate;
    if there is a velocity dip (net-negative energy near center), z_trigger
    should be moved farther from center.
    """
    args = parse_arguments()

    print()
    print("  ╔════════════════════════════════════════════════════════════╗")
    print("  ║   COILGUN SIMULATION — STEPS 3-4: FORCE + DYNAMICS          ║")
    print("  ║   Rizky Meilandi Saputra  |  hybrid-architect-lab         ║")
    print("  ╚════════════════════════════════════════════════════════════╝")

    # ── construct locked Step 1/2 geometry ───────────────────────────────────
    coil = CoilGeometry(
        radius_m        = 0.020,
        length_m        = 0.080,
        turns_per_meter = 2000.0,
    )
    capacitor = CapacitorBank(
        capacitance_f     = 4.0e-3,
        initial_voltage_v = 400.0,
    )
    stage = AccelerationStage(
        index                = 1,
        coil                 = coil,
        capacitor            = capacitor,
        wire_resistance_ohm  = 0.15,
        center_position_m    = 0.200,   # 200 mm into barrel
        trigger_position_m   = 0.140,   # z_center − 60 mm  (initial estimate)
    )

    # ── projectile spec ───────────────────────────────────────────────────────
    projectile = Projectile(
        mass_kg               = 0.050,   # 50 g
        radius_m              = 0.0075,  # 7.5 mm  (clearance: 20 mm bore − 7.5 mm = 12.5 mm gap)
        length_m              = 0.040,   # 40 mm  = L_c / 2
        relative_permeability = 200.0,   # conservative soft iron (partial saturation)
    )

    K_full = (projectile.susceptibility * projectile.volume_m3
              * MU_0 * stage.coil.turns_per_meter ** 2)
    print()
    print(f"  [stage]       z_center = {stage.center_position_m*1e3:.0f} mm  "
          f"| z_trigger = {stage.trigger_position_m*1e3:.0f} mm  "
          f"(offset = {(stage.center_position_m-stage.trigger_position_m)*1e3:.0f} mm)")
    print(f"  [projectile]  m = {projectile.mass_kg*1e3:.0f} g  "
          f"| r = {projectile.radius_m*1e3:.1f} mm  "
          f"| L = {projectile.length_m*1e3:.0f} mm  "
          f"| μ_r = {projectile.relative_permeability:.0f}")
    print(f"  [force]       K = (μr-1)·V·μ0·n² = {K_full:.5e} H")
    print(f"  [force]       Peak I²   = {stage.i_peak_a**2:.0f} A²")
    print(f"  [simulation]  dt = {DT_DEFAULT*1e6:.0f} μs  | t_total = {T_TOTAL*1e3:.0f} ms")
    print(f"  [output]      {args.output}")
    print()
    print("  [verify]      running 6 checks (includes simulation for check [6])...")

    # ── verification — returns records from internal simulation run ─────────
    records = verify_dynamics(stage, projectile)

    # ── render ────────────────────────────────────────────────────────────────
    print("  [render]      generating 3-panel Neo-Classical figure...")
    render_dynamics(records, stage, projectile, args.output, OUTPUT_DPI)

    # ── summary ───────────────────────────────────────────────────────────────
    v_final    = records[-1].velocity_ms
    KE_final   = records[-1].kinetic_energy_j
    efficiency = KE_final / stage.energy_stored_j * 100.0
    v_max_idx  = int(np.argmax([r.velocity_ms for r in records]))
    v_max      = records[v_max_idx].velocity_ms
    F_values   = [r.net_force_n for r in records]
    peak_F     = max(F_values, key=abs)

    print()
    print(f"  [result]      v_exit     = {v_final:.2f} m/s")
    print(f"  [result]      v_max      = {v_max:.2f} m/s  (at t = {records[v_max_idx].time_s*1e3:.2f} ms)")
    print(f"  [result]      ΔKE        = {KE_final:.2f} J")
    print(f"  [result]      Efficiency = {efficiency:.2f}%  "
          f"(ΔKE / {stage.energy_stored_j:.0f} J)")
    print(f"  [result]      Peak |F_z| = {abs(peak_F):.0f} N  "
          f"= {abs(peak_F)/9.81:.0f} N  ({abs(peak_F)/projectile.mass_kg/9.81:.0f} g acceleration)")
    print()
    print("  [simplifications]  μ_r assumed constant (ignores saturation above ~1.5 T)")
    print("  [simplifications]  Force at centroid (L_proj = L_c/2: ~10% spatial error)")
    print("  [simplifications]  Constant-L RLC (real L rises 30-80% as slug enters bore)")
    print("  [simplifications]  No eddy current drag, no friction, no gravity")
    print()
    print("  [done]   Steps 3-4 complete. Proceed to Steps 5-6: multi-stage chaining.")
    print()


if __name__ == "__main__":
    main()
