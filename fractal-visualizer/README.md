# Fractal Visualizer

> Mandelbrot set rendered in the Neo-Classical engineering palette.
> Vectorized NumPy computation, smooth-coloring boundary, custom 4-stop colormap.

![Mandelbrot set rendered in deep red, gold, and white against black — the Neo-Classical palette](mandelbrot_neoclassical.png)

---

## What it is

A high-resolution Mandelbrot set renderer that produces a single 1200×900 PNG
of the canonical fractal, colored in a custom 4-stop palette (black → deep red
→ gold → white). The boundary detail is enhanced with smooth-coloring
renormalization to eliminate the iteration-count banding that ruins most
naive Mandelbrot renders.

## What it produces

A PNG file like the one above. Default render:

- **Resolution:** 1200 × 900 px at 150 DPI
- **Iteration depth:** 256
- **Escape radius:** 256.0 (high enough for smooth coloring to remain numerically stable)
- **Render time:** ~10–30 seconds on a typical laptop
- **Interior fraction:** roughly 18% of the frame is bounded (the black set itself)

Output filename and path are configurable via the `--output` flag.

## How to run it

```bash
# install dependencies
pip install -r requirements.txt

# default render — saves to ./mandelbrot_neoclassical.png
python mandelbrot_visualizer.py

# custom output path
python mandelbrot_visualizer.py --output ./renders/fractal.png
```

## The interesting bits

### Vectorized iteration with active-pixel masking

The naive approach loops over every pixel in Python, which is unusably slow at
this resolution. This implementation operates on the entire 1200×900 grid as
NumPy arrays simultaneously, with a boolean mask tracking which pixels haven't
yet escaped. Each iteration step only updates the unescaped subset.

Result: roughly **10–50× faster** than a Python loop at this resolution.

### Smooth coloring (renormalization)

Raw iteration counts produce visible ring-shaped color bands around the set.
The standard fix is to compute a fractional escape value:

    smooth_count = iteration - log2(log2(|z|))

This uses the fact that `|z|` grows roughly exponentially after escape, so
log-of-log measures how far *into* the current iteration the point escaped.
Subtracting that from the integer iteration gives a continuous float across
the boundary — no rings.

### Why the colormap stops are not evenly spaced

The 4-stop Neo-Classical palette concentrates color weight at the boundary,
not at the outer escape field. The stops are:

| Position | Color    | What it represents                          |
|----------|----------|---------------------------------------------|
| 0.00     | `#0a0a0a` | Set interior — confirmed bounded            |
| 0.05     | `#8b0000` | Boundary proximity — densest detail         |
| 0.35     | `#c9a84c` | Mid-range escape — ornamental layer         |
| 1.00     | `#ffffff` | Fast escape — divergence field              |

Red at 0.05 (not 0.25) compresses the most visually complex region into a
narrow color band, which makes the fractal boundary feel *dense* rather than
*washed out*. This is an aesthetic choice, not a mathematical requirement.

### Square-root normalization

After computing the escape field, values are normalized using `sqrt(n/max_n)`
rather than `n/max_n`. Square-root compresses the upper end of the escape
range, giving boundary detail (low escape counts) more color resolution.
Linear normalization would push all the interesting structure into a thin
dark band.

## Stack

- Python 3.10+
- NumPy ≥ 1.24
- Matplotlib ≥ 3.7

No other dependencies. Two libraries, one file, ~300 lines.

## Files

```
fractal-visualizer/
├── mandelbrot_visualizer.py     # The renderer
├── mandelbrot_neoclassical.png  # Sample output
├── requirements.txt              # Pinned minimum versions
└── README.md                     # This file
```

## Theory note

The Mandelbrot set lives at the boundary between order and chaos — interior
points are bounded forever under `z → z² + c`, exterior points diverge to
infinity, and the boundary between them is infinite in length yet contained
in a finite region of the complex plane. This is deterministic chaos rendered
visible: simple rule, infinite complexity. Not randomness.
