"""
Module:      vtol_calculator.py
Purpose:     VTOL Firefighting Drone — Parameter & Feasibility Calculator
Author:      Rizky Meilandi Saputra
Repository:  github.com/kiki007-lab
Version:     1.0.0
Dependencies: None — pure Python 3.14+ standard library only

Context:
    This calculator is a direct tool from the VTOL firefighting drone
    development project. It takes physical configuration parameters and
    outputs a full feasibility analysis covering thrust, current draw,
    flight endurance, and operational safety margins.

    Engineering basis:
        - Thrust-to-weight ratio (TWR) is the single most critical metric
          in multirotor design. A hover TWR of 2:1 is the industry baseline,
          meaning each motor collectively generates twice the drone's weight
          in thrust. This headroom covers wind resistance, payload variation,
          and emergency climb authority.
        - Current draw is estimated using empirical motor efficiency constants
          derived from brushless motor datasheets across typical multirotor
          configurations.
        - Flight time is derived from Peukert-simplified energy consumption:
          capacity_Ah / current_A * 60, adjusted for a 80% usable battery
          threshold — discharging lithium-polymer cells below 20% causes
          accelerated degradation and increases thermal runaway risk.
"""

import math


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL AND OPERATIONAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

GRAVITY_MS2: float = 9.81
# Standard gravitational acceleration, m/s². Used to convert mass → force.

NEWTON_TO_KGF: float = 1.0 / GRAVITY_MS2
# Conversion factor: 1 N = 1/9.81 kgf. Useful for sanity-checking against
# motor thrust specifications, which are almost always listed in grams-force.

BATTERY_USABLE_FRACTION: float = 0.80
# LiPo cells should not be discharged below 20% state-of-charge. This
# constant represents the usable energy window. Dropping below it causes
# voltage sag that destabilizes ESC output and accelerates cell aging.

MOTOR_EFFICIENCY_CONSTANT: float = 7.5
# Empirical thrust-to-current ratio: approximately 7.5 grams of thrust
# per watt of input power, derived from mid-range brushless motor datasheets
# in the 2216–3508 size class. This is a conservative estimate — higher-
# quality motors may achieve 9–12 g/W. A conservative constant is intentional:
# undershooting current draw is far more dangerous than overshooting it.

PROPELLER_PITCH_ESTIMATE: float = 4.5
# Default assumed pitch in inches for a standard multirotor propeller.
# Pitch and diameter together define the theoretical thrust envelope via
# disk actuator theory. Used only when thrust verification by prop formula
# is cross-checked against TWR calculation.

# Safe operating boundaries — used to trigger warnings
MIN_VOLTAGE_V: float = 11.1    # Below 3S LiPo minimum; system likely unstable
MAX_VOLTAGE_V: float = 51.8    # Above 12S LiPo maximum; risks ESC and motor ratings
MIN_MOTORS: int = 4            # Quads are the minimum stable multirotor configuration
MAX_MOTORS: int = 12           # Beyond 12 motors, control complexity grows nonlinearly
MIN_PROP_DIAMETER_IN: float = 8.0   # Below this, firefighting-class payload is unrealistic
MAX_PROP_DIAMETER_IN: float = 30.0  # Above this, structural resonance becomes a concern
MIN_WEIGHT_KG: float = 0.5    # Below this, a VTOL firefighting role is not credible
MAX_WEIGHT_KG: float = 50.0   # Above this, requires industrial-grade motor specs
MIN_TWR: float = 1.5           # Below 1.5 the drone cannot maintain altitude under load
MAX_TWR: float = 4.0           # Above 4.0 is aggressively powerful; unusual for firefighting


# ─────────────────────────────────────────────────────────────────────────────
# INPUT ACQUISITION
# ─────────────────────────────────────────────────────────────────────────────

def prompt_positive_float(label: str, unit: str) -> float:
    """
    Prompt the user for a positive float value. Loops until valid input
    is received — a drone calculator that accepts negative weight or zero
    motors would produce meaningless output silently, which is worse than
    crashing loudly.
    """
    while True:
        raw: str = input(f"  {label} [{unit}]: ").strip()
        try:
            value: float = float(raw)
            if value <= 0:
                print(f"    [!] Value must be greater than zero. Got: {value}")
                continue
            return value
        except ValueError:
            print(f"    [!] Expected a number. Got: '{raw}'")


def prompt_positive_int(label: str, unit: str) -> int:
    """
    Prompt the user for a positive integer. Motor count and similar
    discrete quantities must be whole numbers — 3.5 motors is not a
    physical configuration.
    """
    while True:
        raw: str = input(f"  {label} [{unit}]: ").strip()
        try:
            value: float = float(raw)
            if not value.is_integer():
                print(f"    [!] Must be a whole number. Got: {value}")
                continue
            value_int: int = int(value)
            if value_int <= 0:
                print(f"    [!] Value must be greater than zero. Got: {value_int}")
                continue
            return value_int
        except ValueError:
            print(f"    [!] Expected a whole number. Got: '{raw}'")


def collect_user_inputs() -> dict:
    """
    Collect all configuration parameters from the user via CLI prompts.
    Returns a typed dictionary of raw inputs — no validation logic here.
    Validation is handled separately so it can be reused or unit-tested
    independently of I/O.
    """
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │         CONFIGURATION INPUT — VTOL SYSTEM        │")
    print("  └─────────────────────────────────────────────────┘")
    print()

    return {
        "total_weight_kg":       prompt_positive_float("Total drone weight (with payload)", "kg"),
        "motor_count":           prompt_positive_int("Number of motors", "count"),
        "battery_voltage_v":     prompt_positive_float("Battery voltage", "V"),
        "battery_capacity_mah":  prompt_positive_float("Battery capacity", "mAh"),
        "propeller_diameter_in": prompt_positive_float("Propeller diameter", "inches"),
        "target_twr":            prompt_positive_float("Target hover thrust-to-weight ratio", "e.g. 2.0"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAFETY BOUNDARY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_inputs(inputs: dict) -> list[str]:
    """
    Check each input against safe operating boundaries.
    Returns a list of warning strings — empty list means all clear.

    This is kept as a pure function (no side effects) so the validation
    logic can be tested independently of print statements.
    """
    warnings: list[str] = []

    weight = inputs["total_weight_kg"]
    motors = inputs["motor_count"]
    voltage = inputs["battery_voltage_v"]
    prop = inputs["propeller_diameter_in"]
    twr = inputs["target_twr"]

    if weight < MIN_WEIGHT_KG:
        warnings.append(
            f"Weight {weight:.2f} kg is below {MIN_WEIGHT_KG} kg — "
            f"too light for a firefighting-class VTOL role."
        )
    if weight > MAX_WEIGHT_KG:
        warnings.append(
            f"Weight {weight:.2f} kg exceeds {MAX_WEIGHT_KG} kg — "
            f"industrial-grade motor specs required. Verify ESC and frame ratings."
        )
    if motors < MIN_MOTORS:
        warnings.append(
            f"{motors}-motor configuration is below the minimum stable multirotor count ({MIN_MOTORS})."
        )
    if motors > MAX_MOTORS:
        warnings.append(
            f"{motors} motors exceed recommended maximum ({MAX_MOTORS}). "
            f"Control complexity and failure modes increase nonlinearly above this."
        )
    if voltage < MIN_VOLTAGE_V:
        warnings.append(
            f"Battery voltage {voltage:.1f} V is below minimum safe threshold ({MIN_VOLTAGE_V} V / 3S LiPo). "
            f"System likely cannot power motors at this weight class."
        )
    if voltage > MAX_VOLTAGE_V:
        warnings.append(
            f"Battery voltage {voltage:.1f} V exceeds {MAX_VOLTAGE_V} V (12S LiPo max). "
            f"Verify ESC voltage rating before proceeding."
        )
    if prop < MIN_PROP_DIAMETER_IN:
        warnings.append(
            f"Propeller diameter {prop:.1f}\" is below {MIN_PROP_DIAMETER_IN}\" — "
            f"thrust density insufficient for a firefighting payload."
        )
    if prop > MAX_PROP_DIAMETER_IN:
        warnings.append(
            f"Propeller diameter {prop:.1f}\" exceeds {MAX_PROP_DIAMETER_IN}\". "
            f"Structural resonance and tip speed may exceed safe limits."
        )
    if twr < MIN_TWR:
        warnings.append(
            f"Target TWR {twr:.2f} is below {MIN_TWR}:1 — "
            f"drone cannot maintain altitude under wind or payload variation."
        )
    if twr > MAX_TWR:
        warnings.append(
            f"Target TWR {twr:.2f} exceeds {MAX_TWR}:1 — "
            f"overpowered for most firefighting scenarios. "
            f"Check if target TWR is intentional (aggressive climb, rapid payload drop)."
        )

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# CORE CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_required_thrust_per_motor(
    total_weight_kg: float,
    motor_count: int,
    target_twr: float
) -> dict:
    """
    Calculate the required thrust output per motor to achieve the target
    thrust-to-weight ratio at hover.

    Formula:
        Total thrust required = drone_weight_kg × gravity × target_TWR
        Thrust per motor      = total_thrust_N / motor_count

    Engineering note:
        TWR at hover is not 1:1. At 1:1 the drone is just barely neutrally
        buoyant with zero authority margin. The target TWR accounts for
        maneuverability, wind loading, and payload variation headroom.
        A TWR of 2.0 means each motor collectively produces 2× the drone's
        weight-force, leaving a 50% throttle at hover (theoretical) — the
        upper half of the throttle range is reserved for control authority.
    """
    total_weight_force_n: float = total_weight_kg * GRAVITY_MS2
    total_thrust_required_n: float = total_weight_force_n * target_twr
    thrust_per_motor_n: float = total_thrust_required_n / motor_count
    thrust_per_motor_gf: float = thrust_per_motor_n / GRAVITY_MS2 * 1000  # grams-force

    return {
        "total_weight_force_n":    total_weight_force_n,
        "total_thrust_required_n": total_thrust_required_n,
        "thrust_per_motor_n":      thrust_per_motor_n,
        "thrust_per_motor_gf":     thrust_per_motor_gf,
    }


def calculate_hover_current_draw(
    thrust_per_motor_n: float,
    motor_count: int,
    battery_voltage_v: float
) -> dict:
    """
    Estimate total current draw at hover using an empirical efficiency constant.

    Formula:
        Thrust per motor in gf  = thrust_per_motor_N × 1000 / g
        Power per motor (W)     = thrust_gf / MOTOR_EFFICIENCY_CONSTANT
        Total system power (W)  = power_per_motor_W × motor_count
        Current draw (A)        = total_power_W / battery_voltage_V

    Engineering note:
        The motor efficiency constant (g/W) is the core approximation here.
        Real motors have efficiency curves — they are less efficient at very
        low and very high throttle. This calculation assumes the motor operates
        near its peak efficiency band, which aligns with hover throttle on a
        well-sized configuration (typically 40–60% throttle at hover).
        If the motor is significantly oversized or undersized for the prop,
        real current draw may deviate by 15–30% from this estimate.
    """
    thrust_per_motor_gf: float = (thrust_per_motor_n / GRAVITY_MS2) * 1000
    power_per_motor_w: float = thrust_per_motor_gf / MOTOR_EFFICIENCY_CONSTANT
    total_power_w: float = power_per_motor_w * motor_count
    hover_current_a: float = total_power_w / battery_voltage_v

    return {
        "thrust_per_motor_gf": thrust_per_motor_gf,
        "power_per_motor_w":   power_per_motor_w,
        "total_system_power_w": total_power_w,
        "hover_current_a":      hover_current_a,
    }


def calculate_flight_time(
    battery_capacity_mah: float,
    hover_current_a: float
) -> dict:
    """
    Estimate hover flight time in minutes using the usable battery capacity.

    Formula:
        Usable capacity (Ah) = battery_capacity_mAh × BATTERY_USABLE_FRACTION / 1000
        Flight time (h)      = usable_capacity_Ah / hover_current_A
        Flight time (min)    = flight_time_h × 60

    Engineering note:
        The 80% usable threshold is a conservative but correct engineering choice
        for LiPo-powered aircraft. The remaining 20% acts as a mandatory reserve.
        In a firefighting scenario, this reserve is not optional — a cell that
        hits undervoltage cutoff mid-flight is a crash, not a soft landing.
        Real-world flight time will also be lower than this estimate because:
            1. Takeoff and climb draw more current than hover
            2. Wind resistance increases current draw dynamically
            3. Battery internal resistance rises as the cell discharges
        Apply a further 15–20% real-world correction factor on top of this output.
    """
    usable_capacity_ah: float = (battery_capacity_mah * BATTERY_USABLE_FRACTION) / 1000
    flight_time_hours: float = usable_capacity_ah / hover_current_a
    flight_time_minutes: float = flight_time_hours * 60

    return {
        "usable_capacity_ah":   usable_capacity_ah,
        "flight_time_hours":    flight_time_hours,
        "flight_time_minutes":  flight_time_minutes,
    }


def evaluate_twr_compliance(
    total_thrust_required_n: float,
    total_weight_force_n: float,
    target_twr: float
) -> dict:
    """
    Determine whether the configuration achieves the target thrust-to-weight
    ratio and by how much it exceeds or falls short.

    The actual TWR is derived from total thrust / total weight force.
    This should equal target_TWR by construction (since thrust was calculated
    from it), but this function is structured to support future extensions
    where thrust_per_motor is taken from a motor datasheet rather than
    derived from the target — making the compliance check non-trivial.
    """
    actual_twr: float = total_thrust_required_n / total_weight_force_n
    twr_margin: float = actual_twr - target_twr
    twr_met: bool = actual_twr >= target_twr

    return {
        "actual_twr":  actual_twr,
        "twr_margin":  twr_margin,
        "twr_met":     twr_met,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def render_separator(char: str = "─", width: int = 55) -> None:
    print(f"  {char * width}")


def render_summary(
    inputs: dict,
    thrust_data: dict,
    current_data: dict,
    flight_data: dict,
    twr_data: dict,
    warnings: list[str]
) -> None:
    """
    Print the full structured summary block.
    Each section is separated visually. Formulas are printed inline with
    their result so the output functions as both a result and a reference sheet.
    """

    print()
    render_separator("═")
    print("  ║  VTOL FIREFIGHTING DRONE — PARAMETER ANALYSIS")
    render_separator("═")

    # ── CONFIGURATION ECHO ───────────────────────────────────────────────────
    print()
    print("  CONFIGURATION INPUT")
    render_separator()
    print(f"  Total drone weight        : {inputs['total_weight_kg']:.2f} kg")
    print(f"  Number of motors          : {inputs['motor_count']}")
    print(f"  Battery voltage           : {inputs['battery_voltage_v']:.1f} V")
    print(f"  Battery capacity          : {inputs['battery_capacity_mah']:.0f} mAh")
    print(f"  Propeller diameter        : {inputs['propeller_diameter_in']:.1f} inches")
    print(f"  Target hover TWR          : {inputs['target_twr']:.2f} : 1")

    # ── THRUST ANALYSIS ───────────────────────────────────────────────────────
    print()
    print("  THRUST ANALYSIS")
    render_separator()
    print(f"  Formula : F_weight = m × g")
    print(f"            {inputs['total_weight_kg']:.2f} kg × {GRAVITY_MS2} m/s²"
          f"  =  {thrust_data['total_weight_force_n']:.2f} N")
    print()
    print(f"  Formula : F_total = F_weight × TWR")
    print(f"            {thrust_data['total_weight_force_n']:.2f} N × {inputs['target_twr']:.2f}"
          f"  =  {thrust_data['total_thrust_required_n']:.2f} N")
    print()
    print(f"  Formula : F_motor = F_total / n_motors")
    print(f"            {thrust_data['total_thrust_required_n']:.2f} N ÷ {inputs['motor_count']}"
          f"  =  {thrust_data['thrust_per_motor_n']:.2f} N  "
          f"({thrust_data['thrust_per_motor_gf']:.0f} gf)")

    # ── POWER & CURRENT ───────────────────────────────────────────────────────
    print()
    print("  POWER & CURRENT DRAW")
    render_separator()
    print(f"  Efficiency constant used  : {MOTOR_EFFICIENCY_CONSTANT} g / W")
    print(f"  Formula : P_motor = thrust_gf / η")
    print(f"            {current_data['thrust_per_motor_gf']:.0f} gf ÷ {MOTOR_EFFICIENCY_CONSTANT}"
          f"  =  {current_data['power_per_motor_w']:.2f} W per motor")
    print()
    print(f"  Formula : P_total = P_motor × n_motors")
    print(f"            {current_data['power_per_motor_w']:.2f} W × {inputs['motor_count']}"
          f"  =  {current_data['total_system_power_w']:.2f} W total")
    print()
    print(f"  Formula : I = P / V")
    print(f"            {current_data['total_system_power_w']:.2f} W ÷ {inputs['battery_voltage_v']:.1f} V"
          f"  =  {current_data['hover_current_a']:.2f} A")

    # ── FLIGHT TIME ───────────────────────────────────────────────────────────
    print()
    print("  FLIGHT ENDURANCE")
    render_separator()
    print(f"  Usable battery fraction   : {int(BATTERY_USABLE_FRACTION * 100)}%  "
          f"(LiPo 20% reserve, avoid undervoltage cutoff)")
    print(f"  Formula : C_usable = C_total × 0.80 / 1000")
    print(f"            {inputs['battery_capacity_mah']:.0f} mAh × 0.80 ÷ 1000"
          f"  =  {flight_data['usable_capacity_ah']:.3f} Ah")
    print()
    print(f"  Formula : t = C_usable / I × 60")
    print(f"            {flight_data['usable_capacity_ah']:.3f} Ah ÷ {current_data['hover_current_a']:.2f} A × 60"
          f"  =  {flight_data['flight_time_minutes']:.1f} minutes")
    print()
    print(f"  Real-world estimate       : ~{flight_data['flight_time_minutes'] * 0.82:.1f} min")
    print(f"  (applies -18% correction for climb, wind load, battery resistance rise)")

    # ── TWR COMPLIANCE ────────────────────────────────────────────────────────
    print()
    print("  THRUST-TO-WEIGHT RATIO")
    render_separator()
    print(f"  Target TWR                : {inputs['target_twr']:.2f} : 1")
    print(f"  Actual TWR (derived)      : {twr_data['actual_twr']:.2f} : 1")
    twr_status = "PASS" if twr_data["twr_met"] else "FAIL"
    twr_note = (
        f"  Configuration MEETS target TWR. Margin: +{twr_data['twr_margin']:.2f}"
        if twr_data["twr_met"]
        else f"  Configuration FAILS to meet target TWR. Deficit: {abs(twr_data['twr_margin']):.2f}"
    )
    print(f"  TWR compliance            : [{twr_status}]")
    print(twr_note)

    # ── WARNINGS ──────────────────────────────────────────────────────────────
    print()
    if warnings:
        print("  SAFETY WARNINGS")
        render_separator()
        for i, warning in enumerate(warnings, start=1):
            print(f"  [{i}] {warning}")
    else:
        print("  SAFETY CHECK")
        render_separator()
        print("  All parameters within defined safe operating ranges.")

    # ── CLOSING ───────────────────────────────────────────────────────────────
    print()
    render_separator("═")
    print("  END OF ANALYSIS")
    render_separator("═")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Top-level orchestrator. Keeps I/O, calculation, and rendering as
    three distinct phases — none of them should know about each other's
    internals. This separation makes any individual phase replaceable:
    the input layer could become a file parser, the calc layer stays
    identical, and the output layer could become a PDF report generator.
    """

    print()
    print("  ╔═══════════════════════════════════════════════════════╗")
    print("  ║     VTOL FIREFIGHTING DRONE — PARAMETER CALCULATOR    ║")
    print("  ║     Rizky Meilandi Saputra  |  v1.0.0                 ║")
    print("  ╚═══════════════════════════════════════════════════════╝")
    print()
    print("  Enter your drone configuration below.")
    print("  All inputs must be positive numbers.")

    # Phase 1: Collect inputs
    inputs: dict = collect_user_inputs()

    # Phase 2: Validate against safe operating boundaries
    warnings: list[str] = validate_inputs(inputs)

    # Phase 3: Run calculations — each function is independent and testable
    thrust_data: dict = calculate_required_thrust_per_motor(
        total_weight_kg=inputs["total_weight_kg"],
        motor_count=inputs["motor_count"],
        target_twr=inputs["target_twr"],
    )

    current_data: dict = calculate_hover_current_draw(
        thrust_per_motor_n=thrust_data["thrust_per_motor_n"],
        motor_count=inputs["motor_count"],
        battery_voltage_v=inputs["battery_voltage_v"],
    )

    flight_data: dict = calculate_flight_time(
        battery_capacity_mah=inputs["battery_capacity_mah"],
        hover_current_a=current_data["hover_current_a"],
    )

    twr_data: dict = evaluate_twr_compliance(
        total_thrust_required_n=thrust_data["total_thrust_required_n"],
        total_weight_force_n=thrust_data["total_weight_force_n"],
        target_twr=inputs["target_twr"],
    )

    # Phase 4: Render results
    render_summary(inputs, thrust_data, current_data, flight_data, twr_data, warnings)


if __name__ == "__main__":
    main()
