"""
Module: mandelbrot_visualizer.py
Purpose: Render a high-resolution Mandelbrot set fractal with Neo-Classical aesthetic
Author: Rizky Meilandi Saputra
Dependencies: numpy, matplotlib
Python: 3.10+

---

Chaos Theory Context
--------------------
The Mandelbrot set is one of the canonical demonstrations of deterministic chaos:
a system governed by a simple rule that produces infinitely complex, unpredictable
behavior at its boundary. It lives at the border between order and chaos — the
set interior is bounded (stable), the exterior diverges to infinity (chaotic),
and the fractal boundary between them is infinite in length yet contained in a
finite region of the complex plane.

This is not randomness. It is sensitivity to initial conditions rendered visible.

Governing Equation
------------------
    z_(n+1) = z_n^2 + c

Where:
    - z is a complex number, initialized to 0 at each new point
    - c is the complex coordinate of the pixel being tested
    - We iterate until either |z| > escape_radius (diverges), or we hit max_iterations

A point c belongs to the Mandelbrot set if the sequence {z_n} remains bounded
forever. In practice, we use max_iterations as a proxy for "bounded" — if a
point hasn't escaped after N iterations, we assume it belongs to the set.

The escape count per pixel is the signal we use for coloring. Points that escape
quickly sit far outside the set. Points that take longer are near the boundary.
Points that never escape are inside the set (colored black in this render).

Color Mapping Philosophy (Neo-Classical)
-----------------------------------------
Standard matplotlib colormaps (viridis, plasma) are built for scientific
neutrality. This render uses a custom LinearSegmentedColormap that moves:

    Black (#0a0a0a) → Deep Red (#8b0000) → Gold (#c9a84c) → White (#ffffff)

The choice is intentional:
    - Black: the interior of the set. Total stability. The void from which structure emerges.
    - Deep Red: points just outside the boundary. High iteration count, near-escape.
      These are the most structurally complex regions of the fractal.
    - Gold: mid-range escape. The ornamental layer — detail without chaos.
    - White: fast-escape regions. Far from the set. Pure divergence.

The progression from dark to light mirrors the mathematical progression from
bounded to unbounded. Structure collapses into pure escape at the bright edge.

Smooth Coloring (Renormalization)
----------------------------------
Raw iteration count produces harsh color banding — visible rings around the set.
The smooth coloring formula corrects this by computing a fractional escape value:

    smooth_count = iteration - log2(log2(|z|))

This uses the fact that |z| grows roughly exponentially after escape, so the
logarithm of log2(|z|) measures "how far into" the current iteration the point
escaped. Subtracting it from the integer iteration count gives a continuous value
rather than a stepped one. The result is smooth color gradients across the
fractal boundary — no visible rings.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Constants — all magic numbers live here, never inline
# ---------------------------------------------------------------------------

# Render resolution. Higher = more detail, more compute time.
# 1200x900 is a good balance for portfolio display.
RENDER_WIDTH_PX: int = 1200
RENDER_HEIGHT_PX: int = 900

# Maximum iterations per pixel. Determines how deep we probe the boundary.
# Higher values reveal more detail at the fractal edge but increase render time.
# 256 is sufficient for a clean portfolio render at this resolution.
MAX_ITERATIONS: int = 256

# Escape radius. Once |z| exceeds this, the point is confirmed diverging.
# Must be > 2.0 (proven lower bound for divergence). Using 256.0 because the
# smooth coloring formula requires log2(log2(|z|)) to be valid, which needs
# |z| to be sufficiently large before we stop iterating.
ESCAPE_RADIUS: float = 256.0

# Complex plane bounds — the region of the plane we render.
# These values frame the full Mandelbrot set with a small margin.
REAL_MIN: float = -2.5
REAL_MAX: float = 1.0
IMAG_MIN: float = -1.25
IMAG_MAX: float = 1.25

# Default output path. Overridable at runtime via: --output "D:\path\file.png"
DEFAULT_OUTPUT_PATH: str = r"C:\Users\HP\Downloads\mandelbrot_neoclassical.png"

# DPI for saved file. 150 is good for web; 300 for print.
OUTPUT_DPI: int = 150


# ---------------------------------------------------------------------------
# FIX 1: argparse isolated in its own function, never at module level.
#
# If parse_args() runs at module level, it fires at import time — meaning any
# script, test runner, or notebook that imports this module will immediately
# try to parse sys.argv from whatever called it. That either silently swallows
# unrelated arguments or throws. Function-scoped parsing runs only when called.
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments. Called explicitly from main(), not at import time.

    Returns
    -------
    argparse.Namespace
        Parsed arguments. Exposes:
            output (str): full path for the saved PNG.
    """
    parser = argparse.ArgumentParser(
        description="Render a Mandelbrot set fractal with Neo-Classical aesthetic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python mandelbrot_visualizer_v2.py --output "D:\\renders\\fractal.png"',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Full path for the output PNG. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Color system — Neo-Classical palette built as a custom colormap
# ---------------------------------------------------------------------------

def build_neoclassical_colormap() -> LinearSegmentedColormap:
    """
    Construct a 4-stop LinearSegmentedColormap anchored in the Neo-Classical
    engineering palette: void black → deep red → gold → white.

    The color stops are not evenly distributed. Deep red is placed early
    (position 0.05) to concentrate visual weight at the fractal boundary,
    where iteration counts are highest and detail is most dense.
    Gold at 0.35 gives the mid-range a warm ornamental quality before
    collapsing into white in the outer escape field.

    Returns
    -------
    LinearSegmentedColormap
        Ready to pass directly to imshow or any matplotlib call.
    """
    palette = [
        (0.00, "#0a0a0a"),   # Void — set interior and near-zero escape
        (0.05, "#8b0000"),   # Deep red — boundary proximity, high iteration density
        (0.35, "#c9a84c"),   # Gold — ornamental mid-range escape field
        (1.00, "#ffffff"),   # White — full divergence, maximum escape speed
    ]

    positions = [stop[0] for stop in palette]
    hex_colors = [stop[1] for stop in palette]

    return LinearSegmentedColormap.from_list(
        name="neoclassical",
        colors=list(zip(positions, hex_colors)),
        N=2048,  # Number of discrete color levels — 2048 gives sub-banding resolution
    )


# ---------------------------------------------------------------------------
# Core computation — vectorized Mandelbrot iteration using NumPy
# ---------------------------------------------------------------------------

def compute_smooth_escape_field(
    real_axis: np.ndarray,
    imag_axis: np.ndarray,
    max_iterations: int,
    escape_radius: float,
) -> np.ndarray:
    """
    Compute the smooth (continuous) escape count for every pixel in the render.

    This is the mathematical core of the visualizer. For each point c in the
    complex plane (one per pixel), we iterate z = z^2 + c until either:
      - |z| exceeds escape_radius (the point diverges — outside the Mandelbrot set)
      - we reach max_iterations (the point is presumed inside the set)

    Vectorization strategy
    ----------------------
    Instead of a Python loop over every pixel (which would be extremely slow),
    we operate on the entire grid simultaneously using NumPy arrays. A boolean
    mask tracks which pixels have not yet escaped. Each iteration updates only
    the unescaped pixels. This yields roughly 10-50x speed improvement over
    pure Python loops at this resolution.

    Smooth coloring
    ---------------
    Raw integer escape counts produce banding artifacts. The smooth value:

        smooth = iteration_count - log2(log2(|z_final|))

    uses the final magnitude of z at escape to interpolate between integer
    iteration levels. This requires escape_radius to be large enough that
    log2(log2(|z|)) is positive and well-defined — hence our choice of 256.0.

    Parameters
    ----------
    real_axis : np.ndarray
        1D array of real (x) coordinate values across the image width.
    imag_axis : np.ndarray
        1D array of imaginary (y) coordinate values across the image height.
    max_iterations : int
        Maximum number of iterations before a point is declared bounded.
    escape_radius : float
        Threshold |z| value at which a point is declared divergent.

    Returns
    -------
    np.ndarray
        2D array of shape (height, width) containing smooth escape values.
        Interior points (never escaped) have value 0.0.
        Exterior points have values in [0, max_iterations].
    """

    # Build the 2D complex plane grid from the two 1D axis arrays.
    # real_grid[i,j] = real component of pixel (i,j)
    # imag_grid[i,j] = imaginary component of pixel (i,j)
    real_grid, imag_grid = np.meshgrid(real_axis, imag_axis)

    # c is the complex constant for each pixel — fixed for the entire iteration sequence
    c_plane = real_grid + 1j * imag_grid

    # z starts at 0 for every pixel (standard Mandelbrot definition)
    z_current = np.zeros_like(c_plane, dtype=complex)

    # Track how many iterations each pixel survived before escaping.
    # dtype=float because we'll write fractional smooth values here.
    escape_field = np.zeros(c_plane.shape, dtype=float)

    # Mask: True = pixel has not yet escaped. Starts fully True (nothing has escaped).
    # We only iterate on unescaped pixels — once a pixel escapes, it's done.
    still_iterating = np.ones(c_plane.shape, dtype=bool)

    for iteration in range(max_iterations):

        # Core Mandelbrot recurrence: z = z^2 + c
        # Only applied to pixels that haven't escaped yet.
        z_current[still_iterating] = (
            z_current[still_iterating] ** 2 + c_plane[still_iterating]
        )

        # Test escape condition: |z| > escape_radius
        escaped_this_step = (
            still_iterating & (np.abs(z_current) > escape_radius)
        )

        if escaped_this_step.any():
            # Smooth coloring — fractional correction to remove banding.
            # For pixels escaping on this iteration:
            #   smooth = iteration - log2(log2(|z|))
            # This compresses the escape count into a continuous float.
            z_mag_at_escape = np.abs(z_current[escaped_this_step])

            # Guard: log2(log2(x)) requires x > 1. Our escape_radius=256 ensures
            # |z| >> 1 at escape, so this is safe.
            smooth_correction = np.log2(np.log2(z_mag_at_escape))
            escape_field[escaped_this_step] = iteration - smooth_correction

        # Remove newly escaped pixels from the active iteration mask.
        still_iterating &= ~escaped_this_step

        # Early termination: if every pixel has escaped, no point continuing.
        if not still_iterating.any():
            break

    # Interior pixels (still_iterating=True after all iterations) remain at 0.0.
    # These are the black regions — confirmed bounded under our iteration budget.
    return escape_field


# ---------------------------------------------------------------------------
# Rendering — turn escape field into a visual output
# ---------------------------------------------------------------------------

def render_mandelbrot(
    escape_field: np.ndarray,
    colormap: LinearSegmentedColormap,
    output_path: str,
    output_dpi: int,
) -> None:
    """
    Apply the colormap to the escape field and save the rendered image.

    The escape field is normalized to [0, 1] before mapping — this is required
    because LinearSegmentedColormap expects values in that range. The normalization
    uses a square-root scale rather than linear, which compresses the high end
    of the escape range. This visually amplifies the detail in the boundary
    region (low escape counts) where the interesting mathematics lives.

    Parameters
    ----------
    escape_field : np.ndarray
        2D array of smooth escape values from compute_smooth_escape_field().
    colormap : LinearSegmentedColormap
        The Neo-Classical colormap to apply.
    output_path : str
        Full path to write the output PNG.
    output_dpi : int
        Dots per inch for the saved image.
    """

    # Normalize escape field to [0, 1].
    # Square-root normalization: n_normalized = sqrt(n / max_n)
    # Deliberate aesthetic choice: redistributes color range so that the fractal
    # boundary (low escape count, small values) receives more color resolution.
    # Linear normalization would compress all boundary detail into a thin dark band.
    field_max = escape_field.max()
    if field_max > 0:
        normalized_field = np.sqrt(escape_field / field_max)
    else:
        # Degenerate case: everything inside the set. Shouldn't happen with a
        # correct parameter range, but handled cleanly rather than crashing.
        normalized_field = escape_field

    # FIX 2: os.makedirs with exist_ok=True — atomic, no race condition.
    #
    # The manual pattern (if not exists: makedirs) has a TOCTOU race: another
    # process can create the directory between the check and the call, causing
    # an OSError. exist_ok=True collapses both operations into one atomic call:
    # creates the directory if absent, does nothing if already present.
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Build the figure. No axes, no margins — pure image output.
    fig, ax = plt.subplots(
        figsize=(RENDER_WIDTH_PX / 100, RENDER_HEIGHT_PX / 100),
        dpi=100,
    )

    # Render the escape field using the colormap.
    # origin="lower" ensures the imaginary axis increases upward (mathematically correct).
    ax.imshow(
        normalized_field,
        cmap=colormap,
        origin="lower",
        interpolation="bilinear",
        aspect="auto",
    )

    # Strip all axes, labels, ticks — the math speaks through the image alone.
    ax.axis("off")
    fig.patch.set_facecolor("#0a0a0a")
    plt.tight_layout(pad=0)

    plt.savefig(
        output_path,
        dpi=output_dpi,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="#0a0a0a",
    )
    plt.close(fig)

    print(f"[render complete] → {output_path}")
    print(f"[resolution]      → {RENDER_WIDTH_PX}x{RENDER_HEIGHT_PX}px at {output_dpi} DPI")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full render pipeline:
      1. Parse arguments — output path configurable via --output flag
      2. Build the coordinate axes for the complex plane region
      3. Compute the vectorized escape field
      4. Apply the Neo-Classical colormap
      5. Save the output image
    """

    # FIX 3: parse_arguments() called here inside main(), not at module level.
    # See parse_arguments() docstring for why this placement is required.
    args = parse_arguments()

    print("[mandelbrot] initializing render...")
    print(f"[parameters] {RENDER_WIDTH_PX}x{RENDER_HEIGHT_PX} | {MAX_ITERATIONS} iterations | escape radius {ESCAPE_RADIUS}")
    print(f"[output]     {args.output}")

    # Build 1D coordinate arrays for the real and imaginary axes.
    # These define exactly which complex numbers correspond to which pixels.
    real_axis = np.linspace(REAL_MIN, REAL_MAX, RENDER_WIDTH_PX)
    imag_axis = np.linspace(IMAG_MIN, IMAG_MAX, RENDER_HEIGHT_PX)

    print("[mandelbrot] computing escape field (this may take 10-30 seconds)...")

    escape_field = compute_smooth_escape_field(
        real_axis=real_axis,
        imag_axis=imag_axis,
        max_iterations=MAX_ITERATIONS,
        escape_radius=ESCAPE_RADIUS,
    )

    print("[mandelbrot] building colormap and rendering...")

    neoclassical_cmap = build_neoclassical_colormap()

    render_mandelbrot(
        escape_field=escape_field,
        colormap=neoclassical_cmap,
        output_path=args.output,
        output_dpi=OUTPUT_DPI,
    )

    # Summary statistics — useful for understanding the render
    interior_pixel_count = np.sum(escape_field == 0.0)
    total_pixels = RENDER_WIDTH_PX * RENDER_HEIGHT_PX
    interior_fraction = interior_pixel_count / total_pixels * 100

    print(f"[statistics]  interior pixels (bounded): {interior_pixel_count:,} ({interior_fraction:.1f}% of frame)")
    print(f"[statistics]  exterior pixels (escaped): {total_pixels - interior_pixel_count:,}")
    print(f"[statistics]  max smooth escape value:   {escape_field.max():.4f}")
    print("[mandelbrot] done.")


if __name__ == "__main__":
    main()
