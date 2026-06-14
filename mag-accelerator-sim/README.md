# Magnetic Linear Accelerator Simulation — Project 4

Multi-stage coilgun physics simulation built in Python.
Models electromagnetic coil staging, RLC capacitor discharge 
dynamics, and projectile motion through a three-stage barrel.
All governing equations explicit in code. No black-box physics 
libraries.

## Physics Model

Three decoupled equation families, each in a dedicated module:

**Field model** — finite solenoid Biot-Savart solution

`B_z(z) = (u0*n*I/2) * [cosine terms from each coil face]`

Verified analytically: f(0) = Lc/sqrt(Lc^2 + 4Rc^2),
f'(0) = 0, odd symmetry f'(-z) = -f'(z)

**Circuit model** — underdamped RLC capacitor discharge

`I(t) = [V0/(wd*L)] * exp(-a*t) * sin(wd*t)`

Energy verified: integral(I^2 * R dt) = 0.5*C*V0^2
to floating-point precision

**Dynamics** — force coupling and RK4 integration

`F_z = (ur-1) * Vproj * u0 * n^2 * I(t)^2 * f(z) * f'(z)`

Integrated at dt = 5 us. Force scales as I^2, quadratically
with voltage.

## Parameters

| Parameter | Value |
|---|---|
| Coil radius R_c | 20 mm |
| Coil length L_c | 80 mm (AR = 4.0) |
| Turns density n | 2000 turns/m |
| Capacitance C | 4 mF |
| Charge voltage V0 | 400 V |
| Energy per stage E | 320 J |
| Resistance R | 0.150 ohm |
| t_peak | 1.975 ms |
| Projectile mass | 50 g |
| Projectile radius | 7.5 mm |
| Relative permeability ur | 200 (soft iron) |
| Stage spacing | 250 mm center-to-center |
| Minimum viable spacing | 284 mm (sweep result) |

## Results — Three-Stage Simulation

| Stage | Entry v | ΔKE | η |
|---|---|---|---|
| Stage 1 | 0.0 m/s | 31.2 J | 9.8% |
| Stage 2 | 34.9 m/s | 324.5 J | 101.4%* |
| Stage 3 | 89.3 m/s | 24.4 J | 7.6% |

*Boundary attribution artifact. Full-system energy check holds.

**v_final = 122.57 m/s | ΔKE = 375.6 J | η_total = 39.1%**

Stage spacing sweep found minimum viable spacing at 284 mm.
Below this threshold, Stage 3 is a net decelerator for this
circuit and projectile combination. Above 360 mm, v_final
plateaus near 146 m/s as trigger geometry saturates.

## Known Simplifications

All flagged explicitly in module docstrings:
- ur constant (ignores magnetic saturation above ~1.5 T)
- Force evaluated at projectile centroid (approx 10% spatial error)
- Constant-L approximation (real L rises 30-80% as slug enters bore)
- No eddy current drag, no friction, no gravity

## Files

| File | Phase | Description |
|---|---|---|
| coilgun_field_model.py | 1 | Finite solenoid field model |
| coilgun_aspect_sweep.py | 1 ext | Coil geometry optimisation |
| coilgun_rlc_model.py | 2 | RLC capacitor discharge |
| coilgun_dynamics.py | 3 | Single-stage force and motion |
| coilgun_simulation.py | 4 | Three-stage chaining + spacing sweep |
| coilgun_field_analysis.png | 1 | Field profile render |
| coilgun_aspect_sweep.png | 1 ext | Aspect ratio sweep render |
| coilgun_rlc_analysis.png | 2 | Current pulse render |
| coilgun_dynamics.png | 3 | Single-stage dynamics render |
| coilgun_neoclassical.png | 4 | Six-panel final render |
| coilgun_spacing_sweep.png | 4 ext | Stage spacing sweep render |

## Stack

Python 3.14, NumPy, Matplotlib. No external physics libraries.
