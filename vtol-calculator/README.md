# VTOL Drone Parameter Calculator

> A feasibility analysis tool for VTOL firefighting multirotors.
> Single-file, zero dependencies, output is both a result and an engineering reference sheet.

---

## What it is

An interactive command-line calculator that takes six physical configuration
parameters of a VTOL drone (weight, motor count, battery voltage, battery
capacity, propeller diameter, target thrust-to-weight ratio) and returns a
full feasibility analysis: required thrust per motor, hover current draw,
estimated flight time with real-world correction, TWR compliance check, and
a safety audit against industry operating boundaries.

Built as a direct engineering tool for an active VTOL firefighting drone
development project.

## Engineering context

The calculator implements three load-bearing concepts from multirotor design:

- **Thrust-to-weight ratio (TWR)** — the single most critical metric in
  multirotor design. A hover TWR of 2:1 is the industry baseline: each
  motor contributes thrust equal to twice the drone's weight collectively,
  leaving 50% throttle headroom for wind resistance, payload variation, and
  emergency climb authority. Anything below 1.5:1 cannot maintain altitude
  under load.

- **Current estimation via empirical motor efficiency** — real brushless
  motors have nonlinear efficiency curves, but they cluster around 7.5 g
  of thrust per watt of input power at hover-throttle band (40–60%).
  Using this as a constant is intentionally conservative — under-predicting
  current draw is far less dangerous than over-predicting it.

- **Peukert-simplified flight time with 80% usable threshold** —
  lithium-polymer cells discharged below 20% state-of-charge suffer
  accelerated degradation and increase thermal runaway risk. In a
  firefighting scenario, the bottom 20% is not a reserve, it is a
  mandatory non-negotiable margin. A real-world correction factor of
  -18% on top of the theoretical estimate accounts for climb, wind
  loading, and battery internal resistance rise during discharge.

## What it produces

Run with a realistic firefighting hexacopter config and you get:

```
  ═══════════════════════════════════════════════════════
  ║  VTOL FIREFIGHTING DRONE — PARAMETER ANALYSIS
  ═══════════════════════════════════════════════════════

  CONFIGURATION INPUT
  ───────────────────────────────────────────────────────
  Total drone weight        : 12.50 kg
  Number of motors          : 6
  Battery voltage           : 44.4 V
  Battery capacity          : 22000 mAh
  Propeller diameter        : 15.0 inches
  Target hover TWR          : 2.20 : 1

  THRUST ANALYSIS
  ───────────────────────────────────────────────────────
  Formula : F_weight = m × g
            12.50 kg × 9.81 m/s²  =  122.62 N

  Formula : F_total = F_weight × TWR
            122.62 N × 2.20  =  269.77 N

  Formula : F_motor = F_total / n_motors
            269.77 N ÷ 6  =  44.96 N  (4583 gf)

  POWER & CURRENT DRAW
  ───────────────────────────────────────────────────────
  Efficiency constant used  : 7.5 g / W
  System power at hover     : 3666.67 W
  Hover current draw        : 82.58 A

  FLIGHT ENDURANCE
  ───────────────────────────────────────────────────────
  Usable battery fraction   : 80%  (LiPo 20% reserve)
  Theoretical flight time   : 12.8 minutes
  Real-world estimate       : ~10.5 minutes

  THRUST-TO-WEIGHT RATIO
  ───────────────────────────────────────────────────────
  Target TWR                : 2.20 : 1
  Actual TWR (derived)      : 2.20 : 1
  TWR compliance            : [PASS]

  SAFETY CHECK
  ───────────────────────────────────────────────────────
  All parameters within defined safe operating ranges.

  ═══════════════════════════════════════════════════════
  END OF ANALYSIS
  ═══════════════════════════════════════════════════════
```

Every formula is printed inline with its result, so the output doubles
as both a calculation summary and an engineering reference sheet for
junior team members.

## How to run it

```bash
python vtol_calculator.py
```

Then enter the six configuration values when prompted. The tool will
loop on invalid input rather than crash — non-numeric entries, negative
numbers, and non-integer motor counts are all rejected with a specific
error message.

**No dependencies.** Pure Python standard library. No `pip install`
ceremony, no virtual environment required.

## The interesting bits

### Phase separation — input, validation, calculation, rendering

`main()` is structured as four distinct phases that don't share internals:

```
collect_user_inputs()  →  validate_inputs()  →  calculate_*  →  render_summary()
```

Each layer could be swapped without touching the others. The input layer
could become a file parser. The validation layer is pure and testable.
The calculation layer has no I/O. The rendering layer could become a PDF
report generator. This separation is what makes engineering software
maintainable rather than disposable.

### Validation as a pure function

`validate_inputs()` returns a `list[str]` of warning messages — it does
no printing, no logging, no side effects. This means the warning logic
can be unit-tested independently of the I/O layer, and the same validator
could power a web frontend, a CI check, or a batch analysis script
without modification.

### Constants block as the entire knob panel

Every adjustable parameter — gravity, motor efficiency constant, usable
battery fraction, safe operating boundaries for weight/motors/voltage/
prop diameter/TWR — lives in a single named-constants block at the top of
the file. No magic numbers buried inside functions. Tuning the calculator
for a different drone class (FPV racer, surveying, agricultural) is a
matter of editing 10 named constants rather than hunting through code.

### Output as reference sheet

The summary print block doesn't just show results — it prints the formula
that produced each number alongside the substituted values. A junior
engineer reading the output learns the underlying equation, not just the
answer:

```
  Formula : F_total = F_weight × TWR
            122.62 N × 2.20  =  269.77 N
```

The calculator is also a teaching tool.

### Forward-compatible TWR compliance

The `evaluate_twr_compliance()` function is structured so that future
versions can accept actual motor thrust from a datasheet rather than
deriving it from the target — making the compliance check non-trivial.
The current implementation returns a tautology by construction, but the
function shape is ready for the upgrade without API changes.

## Stack

- Python 3.10+ minimum (developed on 3.14)
- Standard library only — no external packages, no `pip install` required

## Files

```
vtol-calculator/
├── vtol_calculator.py    # The calculator
└── README.md             # This file
```

---

> *In aviation, every number on the page becomes a force in the air.
> The 18% margin is not pessimism — it is the difference between a soft
> landing and a crash.*
