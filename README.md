# DiracFermions-in-MilneSpacetime

This repository contains all the code produced during the writing of the paper xxxx.xxxx. If you use this code, please cite the relevant publication:

@article{...}

This project is written in Python. It uses standard python libraries plus mpmath in order to calculate with high precision Hankel functions with complex order. The calculation of integrals is done by constructing a grid and using scipy.simpson. Temperature, mass and proper spin potential (denoted as `Ω` in our paper) are intended in GeV, whereas `τ` shall be given in units GeV⁻¹. 

**!! Warning:** When we started writing the python code, we were using a convention for the sign of the spin potential which is opposite to the one adopted in the final version of our manuscript. Therefore, the diagonalization procedure in our library produces the results associated to `-Ω` instead of `Ω`. In most functions this is taken into account, but not everywhere. This notation problem may be addressed in the future, if necessary. For the time being, users should keep this sign in mind. 

## Main library

The main library of our code is libMilne.py. This library contains:

- the special functions appearing in spinor products in Milne space.
- the exact, analytical eigenvectors of the effective Hamiltonian with a finite spin density
- diagonalization matrices, which are used to obtain the expectation values of Milne modes
- sampler functions which evaluate the integrands of the partition function, energy density, pressures (both for the Belinfante and canonical pseudogauges), spin density, torque, and spin polarization
- functions returning the integrals of interest, apt to be executed in parallel for a list of values of spin potentials
- utils functions, such as for plotting integrands, merge output files, etc...

The slowest part in our code is the evaluation of the Hankel functions, for which we are limited by mpmath efficiency. Nonetheless, thermal expectation values for 24 values of spin potential are evaluated on a grid of 31³ points in about 1h on our laptops, with our default working precision of 80 bits. Working precision is affected by values of mass and `τ`, as if these values become large, the integration domain in `μ` must be cut at higher and higer values, which require a higher precision: in the case of the Lambda polarization plots, we have cut the integration domain at |`μ`|=60, which requires a working precision of ~300 bits. The higher the precision used, the slower the integration.

## Tutorial notebook

The notebook milneNumerics.ipynb is a self-contained commented tutorial, with relevant functions that can be used to make tests and reproduce our results. The plotting script plots.ipynb can be used to reproduce the plots presented in the paper.



## Computing expectation values and integrand grid with Milne Sampler

Parallel sampler to compute thermodynamic and polarization observables on a 3D momentum grid and store results in HDF5 files is done with SAMPLER_libMilne.py
The sampler evaluates physical observables on a grid in momentum space:

- transverse momenta: `p_x`, `p_y`
- longitudinal variable: `μ`

For each value of a scan parameter (`omega` or `SP`), it:

1. builds a 3D grid
2. evaluates observables using `libMilne.py`
3. saves integrands of expectation values (the outputs of tabulating functions) in one HDF5 file per scan point
4. optionally merges all slices into a single dataset

The grid is constructed from a single parameter: xmax-

The sampler builds a physically symmetric momentum grid:

- `p_x ∈ [-xmax, xmax]`
- `p_y ∈ [-xmax, xmax]`
- `p_z ∈ [-xmax, xmax]`

Internally this is implemented as:

    μ_max = τ · xmax.

There are three sampling modes:

- `standard` → canonical + Belinfante quantities (Energy momentum tensor)
- `polarization` → polarization only
- `full` → everything

To use the sampler, for example, do:

```bash
python SAMPLER_libMilne.py compute\
  --xmax 2.0 \
  --npt 41 \
  --T 0.1 \
  --mass 1.0 \
  --tau 1.0 \
  --scan omega \
  --scan-min 0.0001 \
  --scan-max 3 \
  --nscan 12 \
  --sample-mode standard
```

Each scan value produces a file:

    slice_*.h5

These contain:

- grid: `px`, `py`, `mu`
- observables (depending on mode)
- metadata (T, mass, τ, etc.)

To merge the files do

```bash
python SAMPLER_libMilne.py merge \
  --outdir output \
  --merged-file merged.h5
```

Merged files stack all slices along a new axis.
The merged file is saved in the same directory of SAMPLER_libMilne

Once you have the merged file, use 
```bash
python Integrals_Obs.py merged.h5 
```
to generate .txt files corresponding to the integrated quantities. The txt files are saved in the Observables_txt directory. 
The plots can be produced using the notebook plots.ipynb.
