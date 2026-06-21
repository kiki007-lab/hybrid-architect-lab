"""
Module:       coilgun_rlc_model.py
Purpose:      Step 2 of 6 — RLC capacitor discharge circuit model.
              Provides rlc_current(t, stage) returning I(t) in amperes, and
              rlc_envelope(t, stage) returning the exponential decay envelope.
              AccelerationStage enforces underdamped operation on construction.
              All physics verified analytically before any render runs.
Author:       Rizky Meilandi Saputra
Repository:   github.com/kiki007-lab/hybrid-architect-lab
Project:      Project 4 — Magnetic Linear Accelerator Simulation
Dependencies: coilgun_field_model (Step 1), numpy, matplotlib
Python:       3.10+

---

Physical Context
----------------
Each acceleration stage fires by dumping a charged capacitor into its coil.
The resulting current pulse generates the magnetic field B_z = μ₀·n·I·f(z)
(Step 1), which forces the projectile through F_z ∝ f(z)·f′(z)·I²(t).
This module supplies the I(t) half of that expression.

Governing ODE — Series RLC Circuit
------------------------------------
The charge Q(t) on the capacitor obeys:

    L·d²Q/dt² + R·dQ/dt + Q/C = 0

with initial conditions:
    Q(0)  = C·V₀    (capacitor fully charged)
    I(0)  = 0       (no initial current)

Differentiating: L·dI/dt + R·I + Q/C = 0  (circuit voltage law, I = dQ/dt)

Solution for the underdamped regime (α < ω₀, i.e., R < 2√(L/C)):

    ω₀  = 1/√(L·C)             natural frequency     [rad/s]
    α   = R / (2·L)             damping coefficient   [rad/s]
    ω_d = √(ω₀² − α²)          damped frequency      [rad/s]

    I(t) = (V₀ / (ω_d·L)) · e^(−α·t) · sin(ω_d·t)

The underdamped regime is the required operating mode for a coilgun: it
produces a sharp current spike (fast rise, natural decay) that concentrates
force in the first half-cycle. Overdamped or critically damped circuits
cannot produce this pulse — they are rejected at construction.

Key Timing Result
-----------------
Peak current occurs when dI/dt = 0:

    t_peak = arctan(ω_d / α) / ω_d

Substituting sin(arctan(ω_d/α)) = ω_d/ω₀, the peak current simplifies to:

    I_peak = (V₀ / (ω₀·L)) · e^(−α·t_peak)

The exponential envelope ±(V₀/(ω_d·L))·e^(−α·t) bounds |I(t)| from above
and decays at rate α. The force-relevant quantity I²(t) decays at rate 2α —
twice as fast — so the effective force window is substantially narrower than
the current pulse. This is why timing is critical: most of the force is
delivered in the first half of the positive pulse.

Energy Conservation
-------------------
All initial capacitor energy ½·C·V₀² is eventually dissipated in R:

    ∫₀^∞ I²(t)·R dt = ½·C·V₀²    (verified numerically in check [6])

Analytical Limits Verified at Runtime
--------------------------------------
    I(0)               = 0                  [IC1: initial condition]
    dI/dt|_{t=0}       = V₀/L              [IC2: initial slope from ODE]
    I(t_peak)          = I_peak analytical  [peak value]
    Numerical t_peak   ≈ arctan(ω_d/α)/ω_d [peak timing]
    I(π/ω_d)           = 0                  [first zero crossing, exact]
    ∫I²·R dt           ≈ ½CV₀²             [energy conservation]
"""

import math
import os
import sys
import argparse
from dataclasses import dataclass, field

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from coilgun_field_model import CoilGeometry


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

FIGURE_WIDTH_IN:     float = 14.0
FIGURE_HEIGHT_IN:    float = 6.5
OUTPUT_DPI:          int   = 150
DEFAULT_OUTPUT_PATH: str   = "coilgun_rlc_analysis.png"

T_SPAN_FACTOR: float = 2.5
# Plot from t = 0 to T_SPAN_FACTOR × pulse_duration (first zero of I).
# Factor of 2.5 shows the full first positive lobe, the negative return,
# and the beginning of the second positive lobe — complete pulse picture.

T_RESOLUTION: int = 3000


# ═══════════════════════════════════════════════════════════════════════════════
# NEO-CLASSICAL PALETTE  — locked to Step 1 and Step 1 extension
# ═══════════════════════════════════════════════════════════════════════════════

NC_BACKGROUND: str = "#0a0a0a"
NC_GRID:       str = "#1a1a1a"
NC_SPINE:      str = "#2a2a2a"
NC_TICK:       str = "#888888"
NC_ZERO_LINE:  str = "#383838"
NC_F:          str = "#8b0000"   # deep red   — I(t) primary curve
NC_FPRIME:     str = "#c9a84c"   # gold       — envelope overlay
NC_PRODUCT:    str = "#d8d8d8"   # near-white — I²(t)
NC_TITLE:      str = "#c9a84c"
NC_FORMULA:    str = "#484848"
NC_SPINE_ALT:  str = "#2a2a2a"


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CapacitorBank:
    """
    Electrical specification of one capacitor discharge bank.

    Attributes
    ----------
    capacitance_f : float
        Capacitance C [Farads]. Together with coil inductance L, sets the
        natural frequency ω₀ = 1/√(LC) and the critical resistance threshold
        2√(L/C). Larger C slows the pulse (lower ω₀) and tightens the damping
        margin — a design tradeoff between pulse speed and electrical efficiency.
    initial_voltage_v : float
        Pre-charge voltage V₀ [Volts]. Stored energy = ½·C·V₀². Force scales
        as I² ∝ V₀² — doubling voltage quadruples peak force, which is why
        voltage is the primary performance lever in coilgun design.
    """
    capacitance_f:     float
    initial_voltage_v: float

    @property
    def energy_stored_j(self) -> float:
        """½·C·V₀²  — energy stored before discharge [J]."""
        return 0.5 * self.capacitance_f * self.initial_voltage_v ** 2


@dataclass
class AccelerationStage:
    """
    Complete specification of one coilgun acceleration stage.

    Combines the coil geometry (Step 1), capacitor bank, and resistive
    losses into a single object from which all RLC circuit parameters can
    be derived. All electrical properties are computed from stored fields —
    nothing is pre-baked as a magic number.

    Construction raises ValueError if the circuit is overdamped or critically
    damped (α ≥ ω₀). This is a hard gate: an overdamped stage cannot produce
    the sharp current spike needed for effective projectile acceleration.

    Attributes
    ----------
    index : int
        Stage number along the barrel, 1-indexed. Used in console output
        and multi-stage simulation bookkeeping.
    coil : CoilGeometry
        Geometric specification of the winding. Supplies inductance L.
    capacitor : CapacitorBank
        Capacitor bank specification. Supplies C and V₀.
    wire_resistance_ohm : float
        Total coil winding resistance R [Ω]. For a solenoid wound with
        magnet wire, this is ρ·l_wire/A_wire where l_wire = N·2π·R_mean.
        Drives both α (damping) and Joule heating losses.
    center_position_m : float
        Axial position of the coil center z_c [m] along the barrel. This
        is the origin of the coil-centered frame used by field_profile and
        field_gradient in Step 1.
    trigger_position_m : float
        Projectile position z_T [m] at which this stage fires. Must satisfy
        z_T < z_c (projectile has not yet reached the coil center). Optimally
        chosen so I(t) peaks as the projectile arrives at z_c.
    fired_time_s : float | None
        Wall-clock time [s] when this stage fired during simulation. Set by
        the integration loop; None means the stage has not yet fired.
    """
    index:               int
    coil:                CoilGeometry
    capacitor:           CapacitorBank
    wire_resistance_ohm: float
    center_position_m:   float
    trigger_position_m:  float
    fired_time_s:        float | None = None

    def __post_init__(self) -> None:
        """
        Validate underdamped condition on construction.

        Checks α < ω₀, equivalently R < 2√(L/C).
        Raises ValueError with diagnostic information if violated.
        This is called automatically by the dataclass machinery after all
        fields are set — every AccelerationStage is guaranteed underdamped.
        """
        omega_0 = self.omega_0
        alpha   = self.alpha
        r_crit  = self.r_critical_ohm

        if alpha >= omega_0:
            raise ValueError(
                f"\n  Stage {self.index}: overdamped or critically damped — cannot fire.\n"
                f"  α = {alpha:.4f} rad/s  ≥  ω₀ = {omega_0:.4f} rad/s\n"
                f"  Require R < 2√(L/C) = {r_crit:.6f} Ω\n"
                f"  Current R = {self.wire_resistance_ohm:.6f} Ω\n"
                f"  Fix: reduce wire resistance, increase L (more turns or wider bore),\n"
                f"       or reduce C (smaller capacitor, raises impedance threshold)."
            )

    # ── Derived circuit parameters ─────────────────────────────────────────────

    @property
    def omega_0(self) -> float:
        """ω₀ = 1/√(L·C)  — natural (undamped) resonant frequency [rad/s]."""
        return 1.0 / math.sqrt(self.coil.inductance_h * self.capacitor.capacitance_f)

    @property
    def alpha(self) -> float:
        """α = R/(2·L)  — damping coefficient [rad/s]."""
        return self.wire_resistance_ohm / (2.0 * self.coil.inductance_h)

    @property
    def omega_d(self) -> float:
        """ω_d = √(ω₀² − α²)  — damped resonant frequency [rad/s]."""
        return math.sqrt(self.omega_0 ** 2 - self.alpha ** 2)

    @property
    def damping_ratio(self) -> float:
        """ζ = α/ω₀  — damping ratio. Guaranteed < 1 by __post_init__."""
        return self.alpha / self.omega_0

    @property
    def r_critical_ohm(self) -> float:
        """2√(L/C)  — critical resistance threshold [Ω]. R must be below this."""
        return 2.0 * math.sqrt(self.coil.inductance_h / self.capacitor.capacitance_f)

    @property
    def i_amplitude(self) -> float:
        """Pre-exponential amplitude V₀/(ω_d·L) [A].

        This is the peak value the envelope reaches at t = 0 (before the
        sine has risen to 1 and the exponential has not yet decayed).
        The actual peak current I_peak is lower by e^(-α·t_peak)·sin(ω_d·t_peak).
        """
        return self.capacitor.initial_voltage_v / (self.omega_d * self.coil.inductance_h)

    @property
    def t_peak_s(self) -> float:
        """
        Time of peak current [s].

            t_peak = arctan(ω_d / α) / ω_d

        Derived by setting dI/dt = 0 and solving for the first positive root.
        Uses atan2(ω_d, α) to handle edge cases cleanly (α → 0 gives t_peak → π/(2ω_d)).
        """
        return math.atan2(self.omega_d, self.alpha) / self.omega_d

    @property
    def i_peak_a(self) -> float:
        """
        Peak current magnitude [A].

            I_peak = (V₀ / (ω₀·L)) · e^(−α·t_peak)

        Simplified form: substitutes sin(arctan(ω_d/α)) = ω_d/ω₀ into I(t_peak).
        """
        return (self.capacitor.initial_voltage_v / (self.omega_0 * self.coil.inductance_h)) \
               * math.exp(-self.alpha * self.t_peak_s)

    @property
    def pulse_duration_s(self) -> float:
        """
        Time of the first zero crossing of I(t) [s].

            t_zero = π / ω_d

        I(t) = 0 when sin(ω_d·t) = 0 → t = π/ω_d is the end of the
        first positive pulse. The projectile must pass the coil center
        before this time for net-positive energy transfer.
        """
        return math.pi / self.omega_d

    @property
    def energy_stored_j(self) -> float:
        """½·C·V₀²  — initial stored energy [J]. Delegates to capacitor."""
        return self.capacitor.energy_stored_j


# ═══════════════════════════════════════════════════════════════════════════════
# RLC CURRENT AND ENVELOPE
# ═══════════════════════════════════════════════════════════════════════════════

def rlc_current(
    t:     float | np.ndarray,
    stage: AccelerationStage,
) -> float | np.ndarray:
    """
    Exact RLC discharge current I(t) [A] for an underdamped series circuit.

        I(t) = (V₀ / (ω_d·L)) · e^(−α·t) · sin(ω_d·t)

    This is the closed-form solution to:
        L·dI/dt + R·I + Q/C = 0,   Q(0) = C·V₀,   I(0) = 0

    AccelerationStage.__post_init__ guarantees α < ω₀, so this formula is
    always valid when called on a constructed stage. No branching needed.

    Parameters
    ──────────
    t : float or ndarray
        Elapsed time since stage firing [s]. t = 0 is the moment of discharge.
        Accepts NumPy arrays for vectorized evaluation.
    stage : AccelerationStage
        Fully constructed stage (underdamped condition already validated).

    Returns
    ───────
    float or ndarray  — current [A]. Positive = forward, negative = reverse.
    """
    return (stage.i_amplitude
            * np.exp(-stage.alpha * t)
            * np.sin(stage.omega_d * t))


def rlc_envelope(
    t:     float | np.ndarray,
    stage: AccelerationStage,
) -> float | np.ndarray:
    """
    Positive exponential decay envelope of I(t): (V₀/(ω_d·L)) · e^(−α·t)

    The envelope bounds |I(t)| ≤ envelope(t) for all t. It decays at rate α.

    Critical distinction from I²(t):
    - I(t) envelope decays as  e^(−α·t)
    - I²(t) envelope decays as e^(−2α·t)  — twice as fast

    Since F_z ∝ I²(t), the force window is significantly narrower than the
    current pulse window. This is the quantitative reason why precise timing
    matters: waiting for the projectile to fully enter the coil before firing
    sacrifices most of the available force.

    Returns the positive (upper) envelope only. Negate for the lower bound.
    """
    return stage.i_amplitude * np.exp(-stage.alpha * t)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICAL VERIFICATION — hard gate before any render runs
# ═══════════════════════════════════════════════════════════════════════════════

def verify_rlc_model(stage: AccelerationStage) -> None:
    """
    Run six analytical checks against known closed-form limits.

    Raises AssertionError and halts if any check exceeds its tolerance.
    These checks together confirm:
        - Both initial conditions (I(0) = 0 and correct initial slope)
        - Correct peak timing and peak value
        - Exact zero crossing at t = π/ω_d
        - Energy conservation over the full pulse lifetime

    Tolerances
    ──────────
    TOL_EXACT (1e-10) : for identically zero or analytically exact quantities
    TOL_SLOPE (5e-4)  : for the finite-difference initial slope (O(α·ε) error)
    TOL_ENERGY (2e-3) : for numerical energy integration (trapezoidal quadrature)

    Checks
    ──────
    [1] I(0) = 0                     IC1: initial condition
    [2] dI/dt|_{t→0} = V₀/L         IC2: from circuit ODE at t=0
    [3] I(t_peak) = I_peak_analytical peak value, simplified formula
    [4] Numerical t_peak ≈ analytical arctan(ω_d/α)/ω_d
    [5] I(π/ω_d) = 0                 exact first zero crossing
    [6] ∫I²·R dt ≈ ½CV₀²            energy conservation
    """
    TOL_EXACT:  float = 1e-10
    TOL_SLOPE:  float = 5e-4    # finite-difference truncation O(α·ε) ≈ 1.5e-6 relative
    TOL_ENERGY: float = 2e-3    # trapezoidal integration over N periods

    def _sep(ch: str = "─", w: int = 64) -> None:
        print(f"  {ch * w}")

    print()
    _sep("═")
    print("  ║  COILGUN RLC MODEL — ANALYTICAL VERIFICATION")
    _sep("═")
    print()
    print(f"  Stage {stage.index} circuit:")
    print(f"    C = {stage.capacitor.capacitance_f * 1e3:.1f} mF  "
          f"| V₀ = {stage.capacitor.initial_voltage_v:.0f} V  "
          f"| R = {stage.wire_resistance_ohm:.4f} Ω  "
          f"| L = {stage.coil.inductance_h * 1e6:.1f} μH")
    print(f"    ω₀ = {stage.omega_0:.2f} rad/s  "
          f"| α = {stage.alpha:.2f} rad/s  "
          f"| ω_d = {stage.omega_d:.2f} rad/s  "
          f"| ζ = {stage.damping_ratio:.4f}")
    print(f"    Underdamped: R = {stage.wire_resistance_ohm:.4f} Ω  "
          f"< R_crit = {stage.r_critical_ohm:.4f} Ω  ✓")
    print(f"    E_stored = {stage.energy_stored_j:.1f} J  "
          f"| t_peak = {stage.t_peak_s * 1e3:.3f} ms  "
          f"| I_peak = {stage.i_peak_a:.2f} A")
    print()

    # ── [1] I(0) = 0 ──────────────────────────────────────────────────────────
    i0    = float(rlc_current(0.0, stage))
    err1  = abs(i0)
    print("  [1]  I(0) = 0  — initial condition (no current at t = 0)")
    _sep()
    print(f"       Expected : 0.000000000000 A")
    print(f"       Computed : {i0:.6e} A")
    print(f"       Error    : {err1:.2e}   {'✓' if err1 < TOL_EXACT else '✗  FAILED'}")
    assert err1 < TOL_EXACT, f"[1] FAILED — I(0) = {i0:.2e}"

    # ── [2] dI/dt|_{t=0} = V₀/L — initial slope from ODE ────────────────────
    # At t=0: L·dI/dt = V₀ (all capacitor voltage appears across inductor since I=0).
    # Numerically: forward difference with ε = 1e-9 s.
    eps     = 1e-9
    slope_n = float((rlc_current(eps, stage) - 0.0) / eps)
    slope_a = stage.capacitor.initial_voltage_v / stage.coil.inductance_h
    err2    = abs(slope_n - slope_a) / slope_a

    print()
    print("  [2]  dI/dt|_{t=0} = V₀/L  — initial slope (circuit ODE at t=0)")
    _sep()
    print(f"       Analytical : V₀/L = {slope_a:.2f} A/s")
    print(f"       Numerical  : (I(ε) − I(0)) / ε = {slope_n:.2f} A/s  [ε = {eps:.0e} s]")
    print(f"       Rel. error : {err2:.2e}   {'✓' if err2 < TOL_SLOPE else '✗  FAILED'}")
    assert err2 < TOL_SLOPE, f"[2] FAILED — slope error {err2:.2e}"

    # ── [3] I(t_peak) matches simplified analytical formula ───────────────────
    t_pk    = stage.t_peak_s
    i_pk_n  = float(rlc_current(t_pk, stage))                       # from rlc_current
    i_pk_a  = stage.i_peak_a                                         # (V₀/(ω₀·L))·e^(-α·t_peak)
    err3    = abs(i_pk_n - i_pk_a) / i_pk_a

    print()
    print(f"  [3]  I(t_peak) = (V₀/(ω₀·L))·e^(−α·t_peak)  — peak value")
    _sep()
    print(f"       Formula    : (V₀/(ω₀·L))·e^(−α·t_peak)")
    print(f"       Analytical : {i_pk_a:.8f} A")
    print(f"       Computed   : {i_pk_n:.8f} A")
    print(f"       Rel. error : {err3:.2e}   {'✓' if err3 < TOL_EXACT else '✗  FAILED'}")
    assert err3 < TOL_EXACT, f"[3] FAILED — peak value error {err3:.2e}"

    # ── [4] Numerical t_peak ≈ arctan(ω_d/α)/ω_d ─────────────────────────────
    # Find argmax over a dense grid. With 500,000 points, resolution ≈ 1 ns.
    t_dense   = np.linspace(0.0, stage.pulse_duration_s, 500_000)
    i_dense   = rlc_current(t_dense, stage)
    t_pk_num  = float(t_dense[int(np.argmax(i_dense))])
    t_pk_ana  = stage.t_peak_s
    err4      = abs(t_pk_num - t_pk_ana)

    print()
    print(f"  [4]  Numerical t_peak ≈ arctan(ω_d/α)/ω_d")
    _sep()
    print(f"       Analytical : arctan(ω_d/α)/ω_d = {t_pk_ana * 1e3:.6f} ms")
    print(f"       Numerical  : argmax(I) = {t_pk_num * 1e3:.6f} ms  "
          f"[grid ≈ {stage.pulse_duration_s/500_000*1e9:.1f} ns resolution]")
    print(f"       |Error|    : {err4 * 1e9:.1f} ns   {'✓' if err4 < 1e-6 else '✗  FAILED'}")
    assert err4 < 1e-6, f"[4] FAILED — t_peak discrepancy {err4*1e9:.1f} ns"

    # ── [5] I(π/ω_d) = 0 — first zero crossing is exact ─────────────────────
    t_zero = stage.pulse_duration_s       # = π/ω_d
    i_zero = float(rlc_current(t_zero, stage))
    err5   = abs(i_zero)

    print()
    print(f"  [5]  I(π/ω_d) = 0  — first zero crossing  "
          f"[t = {t_zero * 1e3:.3f} ms]")
    _sep()
    print(f"       Expected : 0.000000000000 A  (sin(π) = 0, exact)")
    print(f"       Computed : {i_zero:.6e} A")
    print(f"       Error    : {err5:.2e}   {'✓' if err5 < TOL_EXACT else '✗  FAILED'}")
    assert err5 < TOL_EXACT, f"[5] FAILED — I(π/ω_d) = {i_zero:.2e}"

    # ── [6] Energy conservation: ∫I²·R dt ≈ ½CV₀² ───────────────────────────
    # Integrate over 20 pulse durations. After 20 × π/ω_d, the envelope has
    # decayed by e^(-2α×20π/ω_d): for α=148 and ω_d=688, this is e^(-27) ≈ 2×10⁻¹².
    # Essentially all energy is captured. 200,000 points gives fine quadrature.
    N_periods  = 20
    t_integ    = np.linspace(0.0, N_periods * stage.pulse_duration_s, 200_000)
    i_sq       = rlc_current(t_integ, stage) ** 2
    e_numerical = float(np.trapezoid(i_sq * stage.wire_resistance_ohm, t_integ))
    e_stored    = stage.energy_stored_j
    err6        = abs(e_numerical - e_stored) / e_stored

    print()
    print(f"  [6]  ∫I²·R dt = ½CV₀²  — energy conservation  "
          f"[integrated over {N_periods} × π/ω_d]")
    _sep()
    print(f"       E_stored    = ½·C·V₀²    = {e_stored:.4f} J")
    print(f"       E_numerical = ∫I²·R dt   = {e_numerical:.4f} J")
    print(f"       Rel. error  : {err6:.2e}   {'✓' if err6 < TOL_ENERGY else '✗  FAILED'}")
    assert err6 < TOL_ENERGY, f"[6] FAILED — energy error {err6:.2e} > {TOL_ENERGY:.2e}"

    print()
    _sep("═")
    print("  ║  All 6 checks passed. RLC model verified.")
    _sep("═")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERING — 2-panel Neo-Classical figure
# ═══════════════════════════════════════════════════════════════════════════════

def render_rlc_analysis(
    stage:       AccelerationStage,
    output_path: str,
    output_dpi:  int,
) -> None:
    """
    2-panel Neo-Classical figure documenting the RLC discharge model.

    Panel 1 — I(t) with exponential envelope.
      Shows the full current pulse from t = 0 to T_SPAN_FACTOR × pulse_duration.
      The golden dashed envelope bounds |I(t)| and decays at rate α.
      t_peak and I_peak are annotated. The timing boundary (t = pulse_duration,
      the first zero crossing) is marked — the projectile must pass the coil
      center before this point for net-positive energy transfer.

    Panel 2 — I²(t): the force-relevant current profile.
      Since F_z ∝ I²(t), this panel shows the actual shape of the force pulse
      over time. The envelope² (gold dashed) decays at rate 2α, demonstrating
      that the effective force window is substantially narrower than the current
      pulse window. The ratio of areas under I²(t) in successive half-cycles
      shows how quickly the accelerating force dominates the subsequent braking.
    """
    # ── time axis ─────────────────────────────────────────────────────────────
    T_MAX: float = T_SPAN_FACTOR * stage.pulse_duration_s
    t_s:  np.ndarray = np.linspace(0.0, T_MAX, T_RESOLUTION)
    t_ms: np.ndarray = t_s * 1e3    # milliseconds for display

    # ── evaluate curves ───────────────────────────────────────────────────────
    i_t:   np.ndarray = rlc_current(t_s, stage)
    env_t: np.ndarray = rlc_envelope(t_s, stage)
    isq_t: np.ndarray = i_t ** 2
    esq_t: np.ndarray = env_t ** 2    # envelope of I²(t)

    t_peak_ms:  float = stage.t_peak_s * 1e3
    t_zero_ms:  float = stage.pulse_duration_s * 1e3
    i_peak:     float = stage.i_peak_a
    isq_peak:   float = i_peak ** 2

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
        gridspec_kw={"wspace": 0.40},
    )
    fig.patch.set_facecolor(NC_BACKGROUND)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _style(ax, ylabel: str) -> None:
        ax.set_facecolor(NC_BACKGROUND)
        for sp in ax.spines.values():
            sp.set_color(NC_SPINE)
        ax.tick_params(colors=NC_TICK, length=4, width=0.5)
        ax.grid(True, color=NC_GRID, linewidth=0.5, linestyle="--", alpha=0.7)
        ax.set_xlabel("t  [ms]  (time since stage fired)",
                      color=NC_TICK, fontsize=9, labelpad=6)
        ax.set_ylabel(ylabel, color=NC_TICK, fontsize=9, labelpad=6)
        ax.axhline(y=0, color=NC_ZERO_LINE, linewidth=0.9, zorder=1)

    def _mark_timing(ax, is_squared: bool = False) -> None:
        """Vertical markers at t_peak and first zero crossing."""
        # t_peak — gold dashed
        ax.axvline(x=t_peak_ms, color=NC_FPRIME, linewidth=0.9,
                   linestyle="--", alpha=0.55, zorder=2)
        # First zero crossing — deeper separator
        ax.axvline(x=t_zero_ms, color=NC_ZERO_LINE, linewidth=1.0,
                   linestyle=":", alpha=0.7, zorder=2)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(t_peak_ms + 0.08, 0.96,
                f"t_peak\n{t_peak_ms:.2f} ms",
                transform=trans, fontsize=6.8, color=NC_FPRIME,
                va="top", ha="left", alpha=0.80)
        ax.text(t_zero_ms + 0.08, 0.72,
                f"t_zero\n{t_zero_ms:.2f} ms",
                transform=trans, fontsize=6.8, color=NC_ZERO_LINE,
                va="top", ha="left", alpha=0.65)

    def _formula_box(ax, lines: str) -> None:
        ax.text(0.04, 0.04, lines,
                transform=ax.transAxes,
                fontsize=6.8, color=NC_FORMULA,
                va="bottom", ha="left", linespacing=1.55,
                bbox=dict(boxstyle="square,pad=0.35",
                          facecolor=NC_BACKGROUND,
                          edgecolor=NC_SPINE_ALT,
                          linewidth=0.5))

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 1 — I(t) with exponential envelope
    # ═════════════════════════════════════════════════════════════════════════
    _style(ax1, "I(t)  [A]")
    _mark_timing(ax1)

    # Subtle fill — positive half (accelerating) and negative half (decelerating)
    ax1.fill_between(t_ms, i_t, 0, where=(i_t >= 0),
                     color=NC_F, alpha=0.10, zorder=0)
    ax1.fill_between(t_ms, i_t, 0, where=(i_t < 0),
                     color="#1a1a3a", alpha=0.18, zorder=0)

    # Envelope — positive and negative bounds
    ax1.plot(t_ms,  env_t, color=NC_FPRIME, linewidth=0.9,
             linestyle="--", alpha=0.60, zorder=2)
    ax1.plot(t_ms, -env_t, color=NC_FPRIME, linewidth=0.9,
             linestyle="--", alpha=0.60, zorder=2)

    # Primary I(t) curve
    ax1.plot(t_ms, i_t, color=NC_F, linewidth=1.8, zorder=3)

    # Peak annotation
    ax1.scatter([t_peak_ms], [i_peak], color=NC_F, s=50, zorder=5,
                edgecolors=NC_TICK, linewidths=0.6)
    ax1.annotate(
        f"I_peak = {i_peak:.0f} A",
        xy=(t_peak_ms, i_peak),
        xytext=(t_peak_ms + 1.2, i_peak * 0.80),
        fontsize=7.5, color=NC_F, ha="left",
        arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7),
    )

    # Envelope label
    ax1.text(0.97, 0.76, "±envelope\n(V₀/(ω_d·L))·e^(−αt)",
             transform=ax1.transAxes, fontsize=6.8, color=NC_FPRIME,
             va="top", ha="right", alpha=0.70)

    # Circuit parameter box — upper right
    param_str = (
        f"ω₀ = {stage.omega_0:.1f} rad/s\n"
        f"α  = {stage.alpha:.1f} rad/s\n"
        f"ω_d = {stage.omega_d:.1f} rad/s\n"
        f"ζ  = {stage.damping_ratio:.3f}\n"
        f"R_crit = {stage.r_critical_ohm:.4f} Ω"
    )
    ax1.text(0.97, 0.97, param_str,
             transform=ax1.transAxes, fontsize=7.0, color=NC_TICK,
             va="top", ha="right",
             bbox=dict(boxstyle="square,pad=0.35",
                       facecolor=NC_BACKGROUND,
                       edgecolor=NC_SPINE_ALT, linewidth=0.5))

    ax1.set_title("Current Pulse  I(t)  with Exponential Envelope",
                  color=NC_TITLE, fontsize=10.5, fontweight="bold", pad=10)

    _formula_box(ax1,
        "I(t) = (V₀/(ω_d·L))·e^(−α·t)·sin(ω_d·t)\n"
        "t_peak = arctan(ω_d/α) / ω_d\n"
        "I_peak = (V₀/(ω₀·L))·e^(−α·t_peak)")

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 2 — I²(t): force-relevant current profile
    # ═════════════════════════════════════════════════════════════════════════
    _style(ax2, "I²(t)  [A²]")
    _mark_timing(ax2, is_squared=True)

    # Fill under I²(t) — positive only (I² ≥ 0 always)
    ax2.fill_between(t_ms, isq_t, 0,
                     color=NC_PRODUCT, alpha=0.07, zorder=0)

    # Envelope of I²(t): decays at 2α
    ax2.plot(t_ms, esq_t, color=NC_FPRIME, linewidth=0.9,
             linestyle="--", alpha=0.60, zorder=2)

    # Primary I²(t) curve
    ax2.plot(t_ms, isq_t, color=NC_PRODUCT, linewidth=1.8, zorder=3)

    # Peak annotation
    ax2.scatter([t_peak_ms], [isq_peak], color=NC_PRODUCT, s=50, zorder=5,
                edgecolors=NC_TICK, linewidths=0.6)
    ax2.annotate(
        f"I²_peak = {isq_peak:.0f} A²",
        xy=(t_peak_ms, isq_peak),
        xytext=(t_peak_ms + 1.2, isq_peak * 0.78),
        fontsize=7.5, color=NC_PRODUCT, ha="left",
        arrowprops=dict(arrowstyle="-", color=NC_FORMULA, lw=0.7),
    )

    # 2α decay rate annotation
    ax2.text(0.97, 0.76,
             f"envelope decays at 2α\n= {2*stage.alpha:.1f} rad/s\n"
             f"(current decays at α\n= {stage.alpha:.1f} rad/s)",
             transform=ax2.transAxes, fontsize=6.8, color=NC_FPRIME,
             va="top", ha="right", alpha=0.70)

    ax2.set_title("Force Profile  I²(t)  — Force Scales as I²",
                  color=NC_TITLE, fontsize=10.5, fontweight="bold", pad=10)

    _formula_box(ax2,
        "I²(t) = (V₀/(ω_d·L))²·e^(−2α·t)·sin²(ω_d·t)\n"
        "F_z ∝ I²(t)  →  decays at 2α, not α\n"
        "Force window narrower than current pulse")

    # ── figure-level title and parameter subtitle ──────────────────────────────
    fig.suptitle(
        "COILGUN RLC CIRCUIT MODEL — CAPACITOR DISCHARGE  |  Step 2 of 6",
        color=NC_TITLE, fontsize=12, fontweight="bold", y=1.04,
    )
    fig.text(
        0.5, 0.997,
        (f"C = {stage.capacitor.capacitance_f * 1e3:.1f} mF  ·  "
         f"V₀ = {stage.capacitor.initial_voltage_v:.0f} V  ·  "
         f"R = {stage.wire_resistance_ohm:.3f} Ω  ·  "
         f"L = {stage.coil.inductance_h * 1e6:.1f} μH  ·  "
         f"E = {stage.energy_stored_j:.0f} J  ·  "
         f"t_peak = {stage.t_peak_s * 1e3:.2f} ms  ·  "
         f"I_peak = {stage.i_peak_a:.0f} A"),
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
        description="Coilgun Step 2: RLC circuit model — capacitor discharge current.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python coilgun_rlc_model.py --output ./coilgun_rlc_analysis.png',
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
    Step 2 pipeline: construct demo stage → verify → render.

    Demo stage parameters are chosen to represent a realistic single-stage
    coilgun with the Step 1 locked geometry (R_c=20mm, L_c=80mm, n=2000):

    Coil: L = 505.3 μH (air-core inductance from Step 1 geometry).
    Capacitor: C = 4 mF, V₀ = 400 V → E = 320 J per discharge.
    Resistance: R = 0.15 Ω — achievable with AWG 14 magnet wire or
                parallel-wound sections. Sets ζ = 0.211 (well underdamped).

    Critical damping threshold: 2√(L/C) = 0.711 Ω.
    With R = 0.15 Ω, the margin to criticality is 0.561 Ω — generous.

    This produces:
        ω₀ ≈ 703 rad/s,  α ≈ 148 rad/s,  ω_d ≈ 688 rad/s
        t_peak ≈ 1.97 ms,  I_peak ≈ 840 A,  pulse_duration ≈ 4.57 ms
    """
    args = parse_arguments()

    print()
    print("  ╔════════════════════════════════════════════════════════╗")
    print("  ║   COILGUN SIMULATION — STEP 2: RLC CIRCUIT MODEL     ║")
    print("  ║   Rizky Meilandi Saputra  |  hybrid-architect-lab     ║")
    print("  ╚════════════════════════════════════════════════════════╝")

    # ── construct demo stage ──────────────────────────────────────────────────
    demo_coil = CoilGeometry(
        radius_m        = 0.020,    # R_c = 20 mm  — locked from Step 1
        length_m        = 0.080,    # L_c = 80 mm  — locked from Step 1
        turns_per_meter = 2000.0,   # n = 2000 turns/m  — locked from Step 1
    )
    demo_capacitor = CapacitorBank(
        capacitance_f     = 4.0e-3,   # 4 mF
        initial_voltage_v = 400.0,    # 400 V  →  E = 320 J
    )
    # AccelerationStage.__post_init__ validates underdamped condition here.
    # ValueError is raised and halts execution if overdamped.
    demo_stage = AccelerationStage(
        index                = 1,
        coil                 = demo_coil,
        capacitor            = demo_capacitor,
        wire_resistance_ohm  = 0.15,   # 0.15 Ω  →  ζ = 0.211 (well underdamped)
        center_position_m    = 0.20,   # stage 1 center at 200 mm into barrel
        trigger_position_m   = 0.155,  # fire when projectile is 45 mm before center
    )

    print()
    print(f"  [stage]       Constructed: Stage {demo_stage.index}")
    print(f"  [stage]       L = {demo_stage.coil.inductance_h * 1e6:.1f} μH  "
          f"| R_crit = {demo_stage.r_critical_ohm:.4f} Ω  "
          f"| R = {demo_stage.wire_resistance_ohm:.4f} Ω  (ζ = {demo_stage.damping_ratio:.3f})")
    print(f"  [stage]       ω₀ = {demo_stage.omega_0:.1f} rad/s  "
          f"| α = {demo_stage.alpha:.1f} rad/s  "
          f"| ω_d = {demo_stage.omega_d:.1f} rad/s")
    print(f"  [stage]       t_peak = {demo_stage.t_peak_s * 1e3:.3f} ms  "
          f"| I_peak = {demo_stage.i_peak_a:.1f} A  "
          f"| E_stored = {demo_stage.energy_stored_j:.0f} J")
    print(f"  [output]      {args.output}")
    print()
    print("  [verify]      running 6 analytical checks...")

    # ── verification — hard gate ───────────────────────────────────────────────
    verify_rlc_model(demo_stage)

    # ── render ────────────────────────────────────────────────────────────────
    print("  [render]      generating 2-panel Neo-Classical figure...")
    render_rlc_analysis(
        stage       = demo_stage,
        output_path = args.output,
        output_dpi  = OUTPUT_DPI,
    )

    print()
    print(f"  [statistics]  pulse_duration = {demo_stage.pulse_duration_s * 1e3:.3f} ms  "
          f"(π/ω_d)")
    print(f"  [statistics]  I² peak        = {demo_stage.i_peak_a**2:.0f} A²")
    print(f"  [statistics]  I² decay rate  = 2α = {2*demo_stage.alpha:.1f} rad/s  "
          f"(vs α = {demo_stage.alpha:.1f} for I)")
    print(f"  [statistics]  t_peak/t_zero  = {demo_stage.t_peak_s/demo_stage.pulse_duration_s:.3f}  "
          f"(peak at {demo_stage.t_peak_s/demo_stage.pulse_duration_s*100:.1f}% of pulse)")
    print()
    print("  [done]   Step 2 complete. Proceed to Steps 3-4: force coupling and dynamics.")
    print()


if __name__ == "__main__":
    main()
