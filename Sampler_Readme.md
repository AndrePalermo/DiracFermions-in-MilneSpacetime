# Milne Sampler

Parallel sampler to compute thermodynamic and polarization observables on a 3D momentum grid and store results in HDF5 files.

**Producers:** A. Palermo, D. Roselli

---

## What this code does

The sampler evaluates physical observables on a grid in momentum space:

- transverse momenta: `p_x`, `p_y`
- longitudinal variable: `μ`

For each value of a scan parameter (`omega` or `SP`), it:

1. builds a 3D grid
2. evaluates observables using `libMilne_FINAL.py`
3. saves one HDF5 file per scan point
4. optionally merges all slices into a single dataset

---

## Units and conventions

The code uses natural units:

- Temperature `T`, mass `m`, angular velocity `Ω` → **GeV**
- Proper time `τ` → **GeV⁻¹**

### Momentum variables

The longitudinal variable `μ` is related to the physical momentum `p_z` by:

    μ = τ · p_z

So the grid in `μ` corresponds to a grid in `p_z`.

---

## Momentum grid

You provide a single parameter:

    xmax

The sampler builds a **physically symmetric momentum grid**:

- `p_x ∈ [-xmax, xmax]`
- `p_y ∈ [-xmax, xmax]`
- `p_z ∈ [-xmax, xmax]`

Internally this is implemented as:

    μ_max = τ · xmax

So:

- you only set `xmax`
- the longitudinal direction is automatically consistent
- no extra tuning is needed

---

## Sampling modes

- `standard` → canonical + Belinfante quantities
- `polarization` → polarization only
- `full` → everything

---

## Usage

### Compute data

```bash
python sampler.py compute \
  --xmax 2.0 \
  --npt 41 \
  --T 0.15 \
  --mass 0.139 \
  --tau 1.0 \
  --scan omega \
  --scan-min -0.02 \
  --scan-max 0.02 \
  --nscan 9 \
  --sample-mode full
```

### Merge slices

```bash
python sampler.py merge \
  --outdir output \
  --merged-file merged.h5
```

---

## Output

Each scan value produces a file:

    slice_*.h5

These contain:

- grid: `px`, `py`, `mu`
- observables (depending on mode)
- metadata (T, mass, τ, etc.)

Merged files stack all slices along a new axis.

---

## Dependencies

```bash
pip install numpy h5py mpmath scipy
```

---

## Project structure

```
project/
├── sampler.py
├── libMilne_FINAL.py
└── README.md
```
