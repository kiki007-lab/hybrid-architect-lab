# VTOL Motor Arm — Parametric CAD

Parametric motor arm designed for VTOL drone applications.
Built in Onshape using a fully variable-driven workflow.

## Parameter Table

| Variable               | Value  | Purpose                        |
|------------------------|--------|--------------------------------|
| `#arm_length`          | 180mm  | Total arm span                 |
| `#arm_width`           | 20mm   | Cross-section width            |
| `#arm_height`          | 14mm   | Cross-section height           |
| `#wall_thickness`      | 2.5mm  | Shell wall thickness           |
| `#wire_bore_diameter`  | 6mm    | Wire routing channel           |
| `#motor_boss_diameter` | 38mm   | Motor mount boss diameter      |
| `#motor_boss_height`   | 8mm    | Boss extrusion depth           |
| `#bolt_circle_diameter`| 16mm   | Motor bolt circle BCD          |
| `#bolt_diameter`       | 3mm    | M3 bolt clearance              |
| `#rib_height`          | 8mm    | Internal rib height            |
| `#rib_thickness`       | 1.8mm  | Internal rib thickness         |
| `#fillet_radius`       | 1.5mm  | Internal and boss fillets      |
| `#chamfer_size`        | 0.8mm  | Edge and bolt hole chamfers    |

## Build Specification

- **Features:** 21
- **Export format:** STEP AP242 Edition 2
- **Configurations:** Two validated — 45mm/19mm BCD (primary) and 38mm/16mm BCD
- **Render:** Blender Cycles, 3-point lighting, Neo-Classical material setup

## Files

- `vtol_arm_neoclassical.png` — final render
- `vtol_arm_render_v3.py` — Blender render script
- `VTOL_Motor_Arm.step` — geometry export (AP242)

## Onshape Document

[VTOL Motor Arm with Integrated Motor Mount](https://cad.onshape.com/documents/83b89dd03b2c505ae399a8bb/w/f876e47ff6f40dede03fed47/e/afa810a5c7523899e6760c96)
