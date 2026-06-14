"""
Module:       coilgun_field_model.py
Purpose:      Phase 1 of 6 — magnetic field model for the coilgun simulation.
              Computes and analytically verifies the on-axis normalized field
              profile f(z) and its gradient f′(z) for a finite solenoid.
              All downstream modules (circuit, force coupling, dynamics, render)
              import CoilGeometry from this file and build on these two functions.
Author:       Rizky Meilandi Saputra
Repository:   github.com/kiki007-lab/hybrid-architect-lab
Project:      Project 4 — Magnetic Linear Accelerator Simulation
Dependencies: numpy, matplotlib
Python:       3.10+

---

Physical Context
----------------
A coilgun accelerates a ferromagnetic projectile through sequenced electromagnetic
coils. Each coil is energized by a capacitor discharge, producing a pulsed axial
field. The projectile — a soft iron cylinder — experiences a force proportional to
the local field gradient. The full force expression (derived in the architecture
document) is:

    F_z(z, t) = (μ_r − 1) · V_proj · μ₀ · n² · I(t)² · f(z) · f′(z)

where f(z) is the normalized field profile and f′(z) is its gradient.
This module computes both functions exactly from the Biot-Savart solution,
verifies them against six closed-form limits, then renders all three curves
as a Neo-Classical reference figure.

Governing Equations
-------------------
On-axis axial field of a finite solenoid with n turns/m, length L_c, radius R_c:

    B_z(z) = (μ₀·n·I / 2) · [ (z + L_c/2) / √((z + L_c/2)² + R_c²)
                              − (z − L_c/2) / √((z − L_c/2)² + R_c²) ]

Factoring out (μ₀ · n · I) isolates geometry from electrics:

    B_z(z, t) = μ₀ · n · I(t) · f(z)

where:
    f(z)  = ½ · [ (z + L_c/2) / √((z + L_c/2)² + R_c²)
                − (z − L_c/2) / √((z − L_c/2)² + R_c²) ]

    f′(z) = (R_c²/2) · [ 1/((z + L_c/2)² + R_c²)^(3/2)
                        − 1/((z − L_c/2)² + R_c²)^(3/2) ]

The factored form is the architectural foundation: f(z) and f′(z) are pure
geometry — stateless, current-free, and testable in complete isolation.

Analytical Limits Verified at Runtime
--------------------------------------
    f(0)      = L_c / √(L_c² + 4·R_c²)   [closed-form center field]
    f′(0)     = 0                          [symmetric maximum; zero gradient]
    f(|z|→∞) → 0                          [far-field extinction]
    f′(|z|→∞)→ 0                          [far-field extinction; ∝ 1/z⁴]
    f(−z)     = f(z)                       [even symmetry about coil center]
    f′(−z)    = −f′(z)                    [odd / antisymmetric gradient]
"""

import os
import argparse
import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MU_0: float = 4.0 * math.pi * 1e-7
# Permeability of free space [H/m].
# Not consumed by f(z) or f′(z) directly — they are dimensionless/[m⁻¹].
# Required in: B_z = μ₀ · n · I · f(z)  and  L = μ₀ · n² · π · R_c² · L_c.


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

FIGURE_WIDTH_IN:  float = 18.0
FIGURE_HEIGHT_IN: float = 6.5
OUTPUT_DPI:       int   = 150
DEFAULT_OUTPUT_PATH: str = r"C:\Users\HP\Downloads\coilgun_field_analysis.png"

Z_SPAN_FACTOR: float = 3.0
# Plot from −(Z_SPAN_FACTOR × L_c) to +(Z_SPAN_FACTOR × L_c).
# A factor of 3 places the coil edges (±L_c/2) well inside the frame
# while showing the far-field decay to near-zero on both sides.

Z_RESOLUTION: int = 4000
# Sample points along z. 4000 resolves the sharp peak of f′(z) near the
# coil edges without aliasing — this matters for force integration later.


# ═══════════════════════════════════════════════════════════════════════════════
# NEO-CLASSICAL PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

NC_BACKGROUND: str = "#0a0a0a"
NC_GRID:       str = "#1a1a1a"
NC_SPINE:      str = "#2a2a2a"
NC_TICK:       str = "#888888"
NC_ZERO_LINE:  str = "#383838"   # reference zero line — visible but subdued
NC_EDGE_MARK:  str = "#404040"   # coil boundary markers (±L_c/2)
NC_COIL_TINT:  str = "#8b0000"   # subtle coil-zone tint behind curves

NC_F:       str = "#8b0000"   # deep red   — f(z) profile
NC_FPRIME:  str = "#c9a84c"   # gold       — f′(z) gradient
NC_PRODUCT: str = "#d8d8d8"   # near-white — f(z)·f′(z) force product
NC_TITLE:   str = "#c9a84c"   # gold       — panel and figure titles
NC_FORMULA: str = "#484848"   # subdued    — formula text annotations
NC_REF:     str = "#3c3c3c"   # darker     — horizontal reference lines


# ═══════════════════════════════════════════════════════════════════════════════
# COIL GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CoilGeometry:
    """
    Geometric specification for one acceleration stage coil.

    Current I(t) is intentionally absent from this class — it belongs to the
    circuit model (Phase 2). This separation means the entire field geometry
    can be constructed, verified, and rendered before any capacitor or resistance
    value is defined. Each module tests its own inputs; nothing assumes the others
    work correctly.

    Attributes
    ----------
    radius_m : float
        Inner bore radius R_c [m]. The Biot-Savart model treats this as the
        radial distance from the barrel axis to the winding turns. In practice,
        this is the barrel bore radius plus the wall thickness.
    length_m : float
        Axial winding length L_c [m]. The solenoid occupies
        z ∈ [−L_c/2, +L_c/2] in the coil-centered frame used by
        field_profile and field_gradient. All z inputs to those functions
        are measured relative to the coil center.
    turns_per_meter : float
        Winding density n [turns/m]. With L_c, gives total turns N = n · L_c.
        Higher n increases peak field linearly but increases resistance as n
        and inductance as n² — a fundamental design tradeoff.
    """
    radius_m:         float
    length_m:         float
    turns_per_meter:  float

    @property
    def total_turns(self) -> float:
        """N = n · L_c [turns]."""
        return self.turns_per_meter * self.length_m

    @property
    def inductance_h(self) -> float:
        """
        Air-core solenoid inductance [H].

            L = μ₀ · n² · π · R_c² · L_c

        KNOWN SIMPLIFICATION — flagged here and documented in Phase 2:
        This is the empty-coil value. When the ferromagnetic slug enters the
        bore, L increases by 30–80% (depending on μ_r), reducing peak current
        and shifting the RLC pulse timing. Treated as a conservative estimate;
        the simulation produces an upper-bound energy output as a result.
        """
        return MU_0 * self.turns_per_meter**2 * math.pi * self.radius_m**2 * self.length_m

    @property
    def f_center_analytical(self) -> float:
        """
        Closed-form value of f(0) derived from the Biot-Savart result.

            f(0) = L_c / √(L_c² + 4·R_c²)

        Derivation: at z = 0 both terms in field_profile are equal in magnitude
        and same sign, giving f(0) = (L_c/2) / √((L_c/2)² + R_c²) which
        simplifies to the expression above. Approaches 1 as L_c >> R_c
        (long solenoid limit). Used as ground truth for verification check [1].
        """
        return self.length_m / math.sqrt(self.length_m**2 + 4.0 * self.radius_m**2)


# ═══════════════════════════════════════════════════════════════════════════════
# FIELD PROFILE — f(z)
# ═══════════════════════════════════════════════════════════════════════════════

def field_profile(
    z:    float | np.ndarray,
    coil: CoilGeometry,
) -> float | np.ndarray:
    """
    Normalized on-axis axial field profile f(z) for a finite solenoid.

    Derived from Biot-Savart for a solenoid with n turns/m, length L_c,
    radius R_c, carrying current I. The full field is:

        B_z(z, t) = μ₀ · n · I(t) · f(z)

    The two bracket terms are the direction cosines of the axial field
    contribution from each coil end-face. In the limit L_c → ∞ both
    terms approach ±1 for any finite z, and f(z) → 1 — recovering the
    uniform interior field of an infinite solenoid.

    Analytical properties
    ─────────────────────
        f(0)      = L_c / √(L_c² + 4·R_c²)   verified by check [1]
        f′(0)     = 0                           verified by check [2]
        f(−z)     = f(z)                        even symmetry; check [5]
        f(|z|→∞) → 0                           check [3]
        0 ≤ f(z) ≤ 1 for all z               bounded by construction

    Parameters
    ──────────
    z : float or ndarray
        Axial position relative to the coil center [m]. NumPy arrays are
        fully supported for vectorized evaluation across a position grid.
    coil : CoilGeometry
        Coil geometric parameters.

    Returns
    ───────
    float or ndarray
        Dimensionless normalized field value(s), bounded [0, 1].
    """
    half_L: float = coil.length_m / 2.0
    R_sq:   float = coil.radius_m ** 2

    # Signed distances from evaluation point z to each coil end-face.
    # z_plus  = z − (−L_c/2) = z + L_c/2  →  rear face (entry side)
    # z_minus = z − (+L_c/2) = z − L_c/2  →  front face (exit side)
    z_plus  = z + half_L
    z_minus = z - half_L

    # Each term is the direction cosine of the axial field from one end-face.
    # The formula integrates the Biot-Savart kernel for a thin ring of current
    # over all rings stacked along [−L_c/2, +L_c/2]. The difference removes
    # the symmetric contributions and isolates the net axial component.
    term_plus  = z_plus  / np.sqrt(z_plus**2  + R_sq)
    term_minus = z_minus / np.sqrt(z_minus**2 + R_sq)

    return 0.5 * (term_plus - term_minus)


# ═══════════════════════════════════════════════════════════════════════════════
# FIELD GRADIENT — f′(z)
# ═══════════════════════════════════════════════════════════════════════════════

def field_gradient(
    z:    float | np.ndarray,
    coil: CoilGeometry,
) -> float | np.ndarray:
    """
    Analytical derivative f′(z) of the normalized field profile.

    Derived by differentiating field_profile term-by-term. The identity:

        d/du [ u / √(u² + R²) ] = R² / (u² + R²)^(3/2)

    gives:

        f′(z) = (R_c²/2) · [ 1/((z + L_c/2)² + R_c²)^(3/2)
                            − 1/((z − L_c/2)² + R_c²)^(3/2) ]

    Physical meaning
    ─────────────────
    Since f(z) ≥ 0 always, the sign of F_z ∝ f(z)·f′(z) is set entirely
    by f′(z):

        z < 0  (approaching center) : f′ > 0  →  forward accelerating force
        z = 0  (at coil center)     : f′ = 0  →  instantaneous zero force
        z > 0  (past center)        : f′ < 0  →  backward braking force

    The sign change at z = 0 is the geometric source of the timing problem.
    The RLC current must decay to near-zero before the projectile crosses
    z = 0, or the braking region dominates and the stage is net-decelerating.
    Panel 3 of the output figure makes this tradeoff visually explicit.

    Analytical properties
    ─────────────────────
        f′(0)     = 0                   verified by check [2]
        f′(−z)    = −f′(z)              odd symmetry; check [6]
        f′(|z|→∞)→ 0                   check [4]; decays ∝ 1/z⁴ far field
        Peak |f′| near z ≈ ±L_c/2      steepest gradient at coil edges

    Parameters
    ──────────
    z : float or ndarray
        Axial position relative to the coil center [m].
    coil : CoilGeometry
        Coil geometric parameters.

    Returns
    ───────
    float or ndarray
        Field gradient [m⁻¹]. Sign is physically meaningful.
    """
    half_L: float = coil.length_m / 2.0
    R_sq:   float = coil.radius_m ** 2

    z_plus  = z + half_L
    z_minus = z - half_L

    # R_c² / (u² + R_c²)^(3/2) is the derivative of the direction-cosine term
    # with respect to u. The net gradient arises from the asymmetric decay of
    # field contributions from the two end-faces: they cancel at z = 0 and
    # reinforce with opposite signs near z = ±L_c/2, producing the peak gradient
    # at the coil edges where the field changes most rapidly.
    term_plus  = 1.0 / (z_plus**2  + R_sq) ** 1.5
    term_minus = 1.0 / (z_minus**2 + R_sq) ** 1.5

    return (R_sq / 2.0) * (term_plus - term_minus)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICAL VERIFICATION — hard gate before any downstream code runs
# ═══════════════════════════════════════════════════════════════════════════════

def verify_analytical_limits(coil: CoilGeometry) -> None:
    """
    Run six analytical checks against known closed-form limits.

    Raises AssertionError (halts execution) if any check fails beyond its
    numerical tolerance. This is a mandatory hard gate: errors in f(z) or
    f′(z) propagate into the force expression as squared quantities, so
    a 1% field error produces a ~2% force error before the integration even
    begins. The simulation must not proceed on a broken field model.

    Tolerances
    ──────────
    TOL_EXACT (1e-10) : for identities that are numerically exact at float64
    TOL_FAR   (1e-5)  : for far-field limits (small but not truly zero)
    TOL_SYM   (1e-12) : for symmetry checks (opposite-sign z pairs)

    Checks
    ──────
    [1] f(0)  matches closed-form center-field formula L_c/√(L_c²+4R_c²)
    [2] f′(0) = 0                 (symmetric maximum, zero gradient)
    [3] f(z_far)  → 0             (far-field profile extinction)
    [4] f′(z_far) → 0             (far-field gradient extinction)
    [5] f(−z) = f(z)              (even symmetry of profile)
    [6] f′(−z) = −f′(z)          (odd / antisymmetric gradient)
    """
    TOL_EXACT: float = 1e-10
    TOL_FAR:   float = 1e-5
    TOL_SYM:   float = 1e-12

    z_far: float = 100.0 * coil.length_m    # 100 × L_c — deep into far field
    z_sym: float =  0.25 * coil.length_m    # L_c/4 — arbitrary symmetry test point

    def _sep(ch: str = "─", w: int = 62) -> None:
        print(f"  {ch * w}")

    print()
    _sep("═")
    print("  ║  COILGUN FIELD MODEL — ANALYTICAL VERIFICATION")
    _sep("═")
    print()
    print(f"  Coil  : R_c = {coil.radius_m * 1e3:.1f} mm  "
          f"| L_c = {coil.length_m * 1e3:.1f} mm  "
          f"| n = {coil.turns_per_meter:.0f} turns/m")
    print(f"         N = {coil.total_turns:.0f} turns  "
          f"| L = {coil.inductance_h * 1e6:.1f} μH  (air-core)  "
          f"| f(0) analytical = {coil.f_center_analytical:.6f}")
    print()

    # ── [1] f(0) vs closed-form center-field ──────────────────────────────────
    f0_num = float(field_profile(0.0, coil))
    f0_ana = coil.f_center_analytical
    err1   = abs(f0_num - f0_ana)

    print("  [1]  f(0) — field at coil center")
    _sep()
    print(f"       Formula  : L_c / √(L_c² + 4·R_c²)")
    print(f"       Expected : {f0_ana:.14f}")
    print(f"       Computed : {f0_num:.14f}")
    print(f"       Error    : {err1:.2e}   {'✓' if err1 < TOL_EXACT else '✗  FAILED'}")
    assert err1 < TOL_EXACT, f"[1] FAILED — f(0) error {err1:.2e} > {TOL_EXACT:.2e}"

    # ── [2] f′(0) = 0 — zero gradient at the field maximum ───────────────────
    fp0  = float(field_gradient(0.0, coil))
    err2 = abs(fp0)

    print()
    print("  [2]  f′(0) — gradient at coil center (must be identically zero)")
    _sep()
    print(f"       Expected : 0.000000000000  (symmetric maximum)")
    print(f"       Computed : {fp0:.6e} m⁻¹")
    print(f"       |Error|  : {err2:.2e}   {'✓' if err2 < TOL_EXACT else '✗  FAILED'}")
    assert err2 < TOL_EXACT, f"[2] FAILED — f′(0) = {fp0:.2e}, expected 0"

    # ── [3] f(z_far) → 0 — far-field profile extinction ─────────────────────
    f_far = float(field_profile(z_far, coil))
    err3  = abs(f_far)

    print()
    print(f"  [3]  f(z_far) — profile at z = {z_far * 1e3:.0f} mm  (100 × L_c)")
    _sep()
    print(f"       Expected : → 0  [far-field limit]")
    print(f"       Computed : {f_far:.6e}")
    print(f"       |Error|  : {err3:.2e}   {'✓' if err3 < TOL_FAR else '✗  FAILED'}")
    assert err3 < TOL_FAR, f"[3] FAILED — f(z_far) = {f_far:.2e}"

    # ── [4] f′(z_far) → 0 — far-field gradient extinction ────────────────────
    fp_far = float(field_gradient(z_far, coil))
    err4   = abs(fp_far)

    print()
    print(f"  [4]  f′(z_far) — gradient at z = {z_far * 1e3:.0f} mm")
    _sep()
    print(f"       Expected : → 0  [decays ∝ 1/z⁴ in far field]")
    print(f"       Computed : {fp_far:.6e} m⁻¹")
    print(f"       |Error|  : {err4:.2e}   {'✓' if err4 < TOL_FAR else '✗  FAILED'}")
    assert err4 < TOL_FAR, f"[4] FAILED — f′(z_far) = {fp_far:.2e}"

    # ── [5] f(−z) = f(z) — even symmetry of profile ──────────────────────────
    f_pos = float(field_profile( z_sym, coil))
    f_neg = float(field_profile(-z_sym, coil))
    err5  = abs(f_pos - f_neg)

    print()
    print(f"  [5]  Even symmetry: f(−z) = f(z)   at z = {z_sym * 1e3:.1f} mm")
    _sep()
    print(f"       f(+{z_sym * 1e3:.1f} mm) = {f_pos:.14f}")
    print(f"       f(−{z_sym * 1e3:.1f} mm) = {f_neg:.14f}")
    print(f"       Asymmetry  : {err5:.2e}   {'✓' if err5 < TOL_SYM else '✗  FAILED'}")
    assert err5 < TOL_SYM, f"[5] FAILED — asymmetry {err5:.2e}"

    # ── [6] f′(−z) = −f′(z) — odd symmetry of gradient ───────────────────────
    fp_pos = float(field_gradient( z_sym, coil))
    fp_neg = float(field_gradient(-z_sym, coil))
    err6   = abs(fp_pos + fp_neg)   # odd: sum of f′(z) + f′(−z) must equal zero

    print()
    print(f"  [6]  Odd symmetry: f′(−z) = −f′(z)   at z = {z_sym * 1e3:.1f} mm")
    _sep()
    print(f"       f′(+{z_sym * 1e3:.1f} mm) = {fp_pos:+.12f} m⁻¹")
    print(f"       f′(−{z_sym * 1e3:.1f} mm) = {fp_neg:+.12f} m⁻¹")
    print(f"       Residual   : {err6:.2e}   {'✓' if err6 < TOL_SYM else '✗  FAILED'}")
    assert err6 < TOL_SYM, f"[6] FAILED — residual {err6:.2e}"

    print()
    _sep("═")
    print("  ║  All 6 checks passed. Field model analytically verified.")
    _sep("═")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERING — 3-panel Neo-Classical figure
# ═══════════════════════════════════════════════════════════════════════════════

def render_field_analysis(
    coil:        CoilGeometry,
    output_path: str,
    output_dpi:  int,
) -> None:
    """
    Produce a 3-panel Neo-Classical figure documenting the field model.

    Panel 1 — f(z): normalized field profile.
      On-axis field strength versus position. Peak at coil center (z = 0),
      decaying to approximately half that value at the coil edges (±L_c/2),
      and toward zero in the far field. The analytical value f(0) is marked.

    Panel 2 — f′(z): field gradient.
      The sign of f′(z) determines the direction of force on the projectile.
      Positive peaks near the coil entrance (accelerating zone); negative
      peaks after the center (braking zone). Both peak values are annotated.
      The zero crossing at z = 0 is marked as the timing boundary.

    Panel 3 — f(z)·f′(z): force product.
      Proportional to d(B_z²)/dz — the quantity that appears directly in
      the force expression F_z ∝ f(z)·f′(z). The positive lobe is the net
      accelerating zone; the negative lobe is the unavoidable braking region
      that grows when the RLC current has not decayed before the projectile
      crosses center. Shaded fills make relative zone areas visually clear.

    Parameters
    ──────────
    coil : CoilGeometry
    output_path : str
    output_dpi : int
    """
    # ── Build z-axis in both meters (for computation) and mm (for display) ────
    z_span_m:  float      = Z_SPAN_FACTOR * coil.length_m
    z_m:       np.ndarray = np.linspace(-z_span_m, z_span_m, Z_RESOLUTION)
    z_mm:      np.ndarray = z_m * 1e3
    half_L_mm: float      = coil.length_m * 500.0   # L_c/2 converted to mm

    # ── Evaluate the three curves ─────────────────────────────────────────────
    f_vals:   np.ndarray = field_profile(z_m, coil)
    fp_vals:  np.ndarray = field_gradient(z_m, coil)
    ffp_vals: np.ndarray = f_vals * fp_vals

    # ── Global Neo-Classical rcParams ─────────────────────────────────────────
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
        1, 3,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        gridspec_kw={"wspace": 0.40},
    )
    fig.patch.set_facecolor(NC_BACKGROUND)

    # ── Inner helpers — closures capture half_L_mm and NC_* constants ─────────

    def _style_ax(ax) -> None:
        """Apply spine, grid, and tick styling to one axis."""
        ax.set_facecolor(NC_BACKGROUND)
        for spine in ax.spines.values():
            spine.set_color(NC_SPINE)
        ax.tick_params(colors=NC_TICK, length=4, width=0.5)
        ax.grid(True, color=NC_GRID, linewidth=0.5, linestyle="--", alpha=0.7)
        ax.set_xlabel("z  [mm]  (relative to coil center)",
                      color=NC_TICK, fontsize=8.5, labelpad=6)

    def _mark_coil_edges(ax) -> None:
        """
        Vertical dashed markers at ±L_c/2 and a subtle coil-zone tint.
        Uses blended coordinates: data-x in mm, axes-fraction y — so the
        edge labels appear at a consistent height regardless of y-scale.
        """
        for x_edge in (-half_L_mm, half_L_mm):
            ax.axvline(x=x_edge, color=NC_EDGE_MARK, linewidth=0.9,
                       linestyle="--", zorder=2)

        # Very faint red wash shows the coil winding region
        ax.axvspan(-half_L_mm, half_L_mm, alpha=0.04,
                   color=NC_COIL_TINT, zorder=0)

        # Coil edge text uses blended transform: data x, axes-fraction y
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(-half_L_mm - 1.5, 0.04, "−L_c/2",
                transform=trans, fontsize=6.5, color=NC_EDGE_MARK,
                ha="right", va="bottom")
        ax.text(+half_L_mm + 1.5, 0.04, "+L_c/2",
                transform=trans, fontsize=6.5, color=NC_EDGE_MARK,
                ha="left", va="bottom")

    def _formula_box(ax, text: str) -> None:
        """Place a formula annotation in the lower-left corner."""
        ax.text(0.04, 0.04, text,
                transform=ax.transAxes,
                fontsize=6.8, color=NC_FORMULA,
                va="bottom", ha="left", linespacing=1.55,
                bbox=dict(boxstyle="square,pad=0.35",
                          facecolor=NC_BACKGROUND,
                          edgecolor=NC_SPINE,
                          linewidth=0.5))

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 1 — f(z)  field profile
    # ═══════════════════════════════════════════════════════════════════════════
    ax1 = axes[0]
    _style_ax(ax1)
    _mark_coil_edges(ax1)

    ax1.plot(z_mm, f_vals, color=NC_F, linewidth=1.8, zorder=3)

    # Horizontal reference at the analytical center-field value
    f0 = coil.f_center_analytical
    ax1.axhline(y=f0, color=NC_REF, linewidth=0.8, linestyle=":", zorder=1)

    # Annotate f(0) — blended transform: axes-fraction x, data y
    trans_ax_data = blended_transform_factory(ax1.transAxes, ax1.transData)
    ax1.text(0.04, f0 + 0.028,
             f"f(0) = {f0:.4f}",
             transform=trans_ax_data,
             fontsize=7.5, color=NC_REF,
             va="bottom", ha="left")

    ax1.set_title("Field Profile  f(z)",
                  color=NC_TITLE, fontsize=10.5, fontweight="bold", pad=10)
    ax1.set_ylabel("f(z)  [dimensionless]",
                   color=NC_TICK, fontsize=8.5, labelpad=6)
    ax1.set_ylim(-0.06, 1.10)

    _formula_box(ax1,
        "f(z) = ½·[ (z + Lc/2) / √((z+Lc/2)²+Rc²)\n"
        "         − (z − Lc/2) / √((z−Lc/2)²+Rc²) ]")

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 2 — f′(z)  field gradient
    # ═══════════════════════════════════════════════════════════════════════════
    ax2 = axes[1]
    _style_ax(ax2)
    _mark_coil_edges(ax2)

    ax2.axhline(y=0, color=NC_ZERO_LINE, linewidth=0.9, zorder=1)
    ax2.plot(z_mm, fp_vals, color=NC_FPRIME, linewidth=1.8, zorder=3)

    # Annotate both peak magnitudes — positive (accelerating) and negative (braking)
    pos_idx   = int(np.argmax(fp_vals))
    neg_idx   = int(np.argmin(fp_vals))
    fp_peak_p = fp_vals[pos_idx]
    fp_peak_n = fp_vals[neg_idx]
    z_peak_p  = z_mm[pos_idx]
    z_peak_n  = z_mm[neg_idx]

    ax2.annotate(
        f"  {fp_peak_p:.2f} m⁻¹",
        xy=(z_peak_p, fp_peak_p),
        xytext=(z_peak_p - 20, fp_peak_p * 0.82),
        fontsize=7.5, color=NC_FPRIME,
        arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7),
    )
    ax2.annotate(
        f"{fp_peak_n:.2f} m⁻¹  ",
        xy=(z_peak_n, fp_peak_n),
        xytext=(z_peak_n + 10, fp_peak_n * 0.82),
        fontsize=7.5, color=NC_FPRIME, ha="left",
        arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7),
    )

    # Mark the timing boundary at z = 0
    trans2 = blended_transform_factory(ax2.transData, ax2.transAxes)
    ax2.text(2.0, 0.52,
             "f′(0) = 0\n← timing boundary",
             transform=trans2,
             fontsize=6.8, color=NC_EDGE_MARK,
             va="bottom", ha="left")

    ax2.set_title("Field Gradient  f′(z)",
                  color=NC_TITLE, fontsize=10.5, fontweight="bold", pad=10)
    ax2.set_ylabel("f′(z)  [m⁻¹]",
                   color=NC_TICK, fontsize=8.5, labelpad=6)

    _formula_box(ax2,
        "f′(z) = (Rc²/2)·[ 1/((z+Lc/2)²+Rc²)^(3/2)\n"
        "                 − 1/((z−Lc/2)²+Rc²)^(3/2) ]")

    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL 3 — f(z)·f′(z)  force product
    # ═══════════════════════════════════════════════════════════════════════════
    ax3 = axes[2]
    _style_ax(ax3)
    _mark_coil_edges(ax3)

    ax3.axhline(y=0, color=NC_ZERO_LINE, linewidth=0.9, zorder=1)

    # Fill accelerating zone (product > 0): projectile has not yet reached center
    ax3.fill_between(z_mm, ffp_vals, 0.0,
                     where=(ffp_vals >= 0),
                     color=NC_F, alpha=0.22, zorder=2)

    # Fill braking zone (product < 0): projectile has passed center
    ax3.fill_between(z_mm, ffp_vals, 0.0,
                     where=(ffp_vals < 0),
                     color="#1a1a3a", alpha=0.40, zorder=2)

    ax3.plot(z_mm, ffp_vals, color=NC_PRODUCT, linewidth=1.8, zorder=3)

    # Zone labels at fixed axes-fraction positions
    ax3.text(0.25, 0.73, "accelerating\n    zone",
             transform=ax3.transAxes,
             fontsize=7.5, color=NC_F, ha="center", alpha=0.85)
    ax3.text(0.75, 0.22, "braking\n zone",
             transform=ax3.transAxes,
             fontsize=7.5, color="#7070b0", ha="center", alpha=0.85)

    ax3.set_title("Force Product  f(z) · f′(z)",
                  color=NC_TITLE, fontsize=10.5, fontweight="bold", pad=10)
    ax3.set_ylabel("f(z)·f′(z)  [m⁻¹]  ∝  d(B_z²)/dz",
                   color=NC_TICK, fontsize=8.5, labelpad=6)

    _formula_box(ax3,
        "F_z ∝ (μr−1)·Vproj·μ0·n²·I(t)²·f(z)·f′(z)\n"
        "Sign of f′(z) sets force direction on slug")

    # ── Figure-level title and parameter subtitle ──────────────────────────────
    fig.suptitle(
        "COILGUN FIELD MODEL — FINITE SOLENOID  |  Phase 1 of 6",
        color=NC_TITLE, fontsize=12, fontweight="bold", y=1.04,
    )
    fig.text(
        0.5, 0.997,
        (f"R_c = {coil.radius_m * 1e3:.1f} mm  ·  "
         f"L_c = {coil.length_m * 1e3:.1f} mm  ·  "
         f"n = {coil.turns_per_meter:.0f} turns/m  ·  "
         f"N = {coil.total_turns:.0f} turns  ·  "
         f"L = {coil.inductance_h * 1e6:.1f} μH  (air-core)  ·  "
         f"f(0) = {coil.f_center_analytical:.4f}"),
        ha="center", va="top",
        color=NC_TICK, fontsize=8.5,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=output_dpi,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor=NC_BACKGROUND,
    )
    plt.close(fig)

    print(f"  [render]      → {output_path}")
    print(f"  [render]      {FIGURE_WIDTH_IN:.0f} × {FIGURE_HEIGHT_IN:.1f} in  "
          f"at {output_dpi} DPI")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_arguments() -> argparse.Namespace:
    """
    Parse CLI arguments. Defined as a function (not module-level) so it never
    fires at import time — same discipline as mandelbrot_visualizer_v2.py.
    """
    parser = argparse.ArgumentParser(
        description="Coilgun Phase 1: compute and verify f(z), f′(z), render field analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python coilgun_field_model.py --output "D:\\renders\\field.png"',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Full path for output PNG. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Orchestrate Phase 1: define coil geometry → verify field functions
    analytically → render the 3-panel Neo-Classical output figure.

    Demo coil parameters (20 mm bore, 80 mm winding, 2000 turns/m)
    represent a realistic single-stage configuration:
    - 20 mm bore fits a 15 mm diameter iron slug with clearance
    - 80 mm winding at 2000 turns/m = 160 turns, achievable in 2–3 layers
      of 0.4 mm magnet wire
    - Inductance ≈ 505 μH paired with 4 mF at 400 V gives a ~10 ms current
      pulse — well within the traversal window at realistic projectile speeds
    - f(0) ≈ 0.894: the coil is wide enough (L_c/R_c = 4) that edge effects
      reduce the center field to ~89% of the infinite-solenoid ideal
    """
    args = parse_arguments()

    print()
    print("  ╔════════════════════════════════════════════════════════╗")
    print("  ║   COILGUN SIMULATION — PHASE 1: FIELD MODEL           ║")
    print("  ║   Rizky Meilandi Saputra  |  hybrid-architect-lab     ║")
    print("  ╚════════════════════════════════════════════════════════╝")

    # ── Define the demo coil ──────────────────────────────────────────────────
    # Phase 2 onward imports CoilGeometry from this module and instantiates
    # its own configurations. This instance is for standalone Phase 1 testing.
    coil = CoilGeometry(
        radius_m        = 0.020,    # 20 mm bore radius
        length_m        = 0.080,    # 80 mm winding length
        turns_per_meter = 2000.0,   # n = 2000 turns/m  →  N = 160 turns
    )

    print()
    print(f"  [parameters]  R_c = {coil.radius_m * 1e3:.1f} mm  "
          f"| L_c = {coil.length_m * 1e3:.1f} mm  "
          f"| n = {coil.turns_per_meter:.0f} turns/m")
    print(f"  [parameters]  N = {coil.total_turns:.0f} turns  "
          f"| L = {coil.inductance_h * 1e6:.1f} μH  "
          f"| f(0) = {coil.f_center_analytical:.6f}")
    print(f"  [output]      {args.output}")
    print()
    print("  [verify]      running 6 analytical checks...")

    # ── Verification — hard gate; nothing proceeds if this raises ─────────────
    verify_analytical_limits(coil)

    # ── Render ────────────────────────────────────────────────────────────────
    print("  [render]      generating 3-panel Neo-Classical figure...")
    render_field_analysis(
        coil        = coil,
        output_path = args.output,
        output_dpi  = OUTPUT_DPI,
    )

    # ── Summary statistics ────────────────────────────────────────────────────
    z_m    = np.linspace(-Z_SPAN_FACTOR * coil.length_m,
                          Z_SPAN_FACTOR * coil.length_m, Z_RESOLUTION)
    fp     = field_gradient(z_m, coil)
    z_peak = z_m[int(np.argmax(fp))] * 1e3    # peak gradient position in mm
    fp_pk  = float(np.max(fp))                # peak gradient value in m⁻¹

    print()
    print(f"  [statistics]  f(0) analytical     = {coil.f_center_analytical:.6f}")
    print(f"  [statistics]  peak |f′| at z      = {z_peak:.1f} mm  "
          f"(value = {fp_pk:.3f} m⁻¹)")
    print(f"  [statistics]  peak |f′| × L_c     = {fp_pk * coil.length_m:.4f}  "
          f"[dimensionless force scale factor]")
    print()
    print("  [done]   Phase 1 complete. "
          "Proceed to Phase 2: RLC circuit model.")
    print()


if __name__ == "__main__":
    main()
