# Bad Apple Ising Simulation

Recreates the **Bad Apple** video using a two-dimensional Ising model. Each video frame is converted into spatially varying magnetic and temperature fields, which drive a lattice of spins through Metropolis Monte Carlo dynamics.

The output shows:

- the evolving Ising spin lattice
- the local temperature field, concentrated around image boundaries
- the magnetic field derived from the signed distance to the target silhouette
- the lattice magnetisation and spin-flip acceptance rate.

## How it works

For each input video frame, the simulation:

1. Converts the frame to grayscale, centre-crops it, and resizes it to the lattice dimensions.
2. Smooths and thresholds the image into a binary silhouette.
3. Computes a signed distance field from the silhouette boundary.
4. Maps the signed distance to an external magnetic field:

   ```text
   H(x, y) = H_strength tanh(d(x, y) / distance_scale)
   ```

5. Raises the local temperature near edges using the distance field and image gradient.
6. Evolves the spins using checkerboard Metropolis sweeps with periodic boundary conditions.
7. Renders the spin state and the corresponding temperature and magnetic-field surfaces.

The external field causes spins to form domains corresponding to the frames black/white regions. Temperature is used to control fluctuations inside of domains of one colour.

## Requirements

- Python 3.10 or later
- FFmpeg available on the system path
- OpenCV
- Matplotlib
- NumPy
- SciPy
- tqdm

Command to install:

```bash
pip install opencv-python matplotlib numpy scipy tqdm
```

## Input video

Put a copy of the Bad Apple MV in the project directory and name it:

```text
bad-apple.mp4
```

Or, change `video_path` in the config block at the bottom of the script.

> The input video is not included in this repository.

## Usage

Run the script directly:

```bash
python bad_apple_ising.py
```

The default configuration writes the animation to:

```text
ising_bad_apple_full.mp4
```

A minimal example:

```python
simulation = BadAppleIsing(
    video_path="bad-apple.mp4",
    N=150,
    H_strength=1.75,
    T_bulk=1.35,
    T_edge=2.55,
    sweeps_per_frame=8,
    seed=None,
)

simulation.run(
    output_path="ising_bad_apple_full.mp4",
    output_fps=24,
    dpi=120,
)
```

Pass `output_path=None` to display the Matplotlib animation without encoding it to a video file.

## Main parameters

| Parameter | Description | Default |
|---|---|---:|
| `N` | Width and height of the square spin lattice | `150` |
| `J` | Nearest-neighbour coupling strength | `1.0` |
| `H_strength` | Maximum external magnetic-field magnitude | `1.75` |
| `distance_scale` | Controls how quickly the field saturates away from an edge | `5.0` |
| `T_bulk` | Temperature away from image boundaries | `1.35` |
| `T_edge` | Maximum temperature near boundaries | `2.55` |
| `edge_width` | Width of the locally heated edge region | `5.0` |
| `spatial_blur` | Gaussian smoothing applied before thresholding | `1.5` |
| `temporal_smoothing` | Inter-frame field update fraction | `0.18` |
| `field_spatial_smoothing` | Additional smoothing of the generated fields | `0.75` |
| `sweeps_per_frame` | Monte Carlo sweeps performed per video frame | `8` |
| `initial_sweeps` | Equilibration sweeps for the first frame | `10 × sweeps_per_frame` |
| `invert_video` | Reverses the target spin polarity | `False` |
| `seed` | Random-number seed for reproducible evolution | `None` |

## Physical model

The Hamiltonian is

```text
E = -J Σ_<ij> s_i s_j - Σ_i H_i s_i,
```

where each spin is `s_i = ±1`, `J` is the ferromagnetic nearest-neighbour coupling, and `H_i` is the position-dependent field derived from the current video frame.

A proposed spin flip has energy change

```text
ΔE = 2 s_i (J Σ_j s_j + H_i).
```

The flip is accepted when `ΔE <= 0`; otherwise, it is accepted with probability

```text
P = exp(-ΔE / T_i).
```

Updates are split across the two colours of a checkerboard lattice, allowing each sublattice to be updated with vectorised NumPy operations.

The simulation uses reduced units with `k_B = 1`. For comparison, the zero-field critical temperature of the infinite square-lattice Ising model is approximately `T_c = 2.269 J`. The bulk lies below this value, while the edge temperature lies above it, helping preserve ordered regions while keeping boundaries mobile.

## Performance notes

Runtime scales approximately with

```text
number of frames × sweeps per frame × N².
```

The two animated 3D surfaces are also relatively expensive to redraw. For faster testing, reduce `N`, `sweeps_per_frame`, `dpi`, or the number of input frames.

For example:

```python
simulation = BadAppleIsing(
    video_path="bad-apple.mp4",
    N=80,
    sweeps_per_frame=3,
    seed=42,
)
```

## Notes

- Periodic boundary conditions are used for both the spin interactions and image-derived fields.
- Fixing `seed` makes the stochastic spin evolution reproducible.
- Increasing `H_strength` improves tracking but can suppress visible thermal dynamics (less isingy).
- Increasing `sweeps_per_frame` improves equilibration but increases rendering time (also less isingy).
- Strong temporal smoothing produces steadier fields but may cause the spin pattern to lag behind rapid motion.

## License

idk
