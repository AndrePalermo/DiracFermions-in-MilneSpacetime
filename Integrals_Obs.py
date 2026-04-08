#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    from scipy.integrate import simpson
except ImportError:
    simpson = None


def load(h5, path):
    if path not in h5:
        raise KeyError(f"Dataset '{path}' not found.")
    return h5[path][:]


def integrate_3d(values, px, py, mu):
    """
    Integrate values(px, py, mu) over mu, py, px.
    Expected shape: (len(px), len(py), len(mu)).
    """
    if simpson is not None:
        tmp = simpson(values, x=mu, axis=2)
        tmp = simpson(tmp, x=py, axis=1)
        return float(simpson(tmp, x=px, axis=0))

    tmp = np.trapz(values, x=mu, axis=2)
    tmp = np.trapz(tmp, x=py, axis=1)
    return float(np.trapz(tmp, x=px, axis=0))


def safe_ratio(num, den):
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.abs(den) > 1e-30
    out[mask] = num[mask] / den[mask]
    return out


def save_xy(filename, x, y, header):
    np.savetxt(filename, np.column_stack([x, y]), header=header)


def main():
    parser = argparse.ArgumentParser(
        description="Integrate observables from a Milne HDF5 file and save txt files."
    )
    parser.add_argument("input_file", help="Input HDF5 file")
    parser.add_argument("--outdir", default="Observables_txt", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input_file, "r") as h5:
        px = load(h5, "grid/px")
        py = load(h5, "grid/py")
        mu = load(h5, "grid/mu")

        if "grid/omega_values" in h5:
            x = load(h5, "grid/omega_values")
            xname = "Omega"
        elif "grid/scan_values" in h5:
            x = load(h5, "grid/scan_values")
            xname = "x"
        else:
            raise KeyError("Could not find 'grid/omega_values' or 'grid/scan_values'.")

        can_energy = load(h5, "canonical/energy_density")
        bel_energy = load(h5, "belinfante/energy_density")

        can_pt = load(h5, "canonical/transv_pressure")
        bel_pt = load(h5, "belinfante/transv_pressure")

        can_pl = load(h5, "canonical/long_pressure")
        bel_pl = load(h5, "belinfante/long_pressure")

        can_spin = load(h5, "canonical/spin_density")
        can_torque = load(h5, "canonical/torque")

    n = len(x)

    E_can = np.empty(n)
    E_bel = np.empty(n)
    PT_can = np.empty(n)
    PT_bel = np.empty(n)
    PL_can = np.empty(n)
    PL_bel = np.empty(n)
    S_can = np.empty(n)
    TQ_can = np.empty(n)

    for i in range(n):
        E_can[i] = integrate_3d(can_energy[i], px, py, mu)
        E_bel[i] = integrate_3d(bel_energy[i], px, py, mu)

        PT_can[i] = integrate_3d(can_pt[i], px, py, mu)
        PT_bel[i] = integrate_3d(bel_pt[i], px, py, mu)

        PL_can[i] = integrate_3d(can_pl[i], px, py, mu)
        PL_bel[i] = integrate_3d(bel_pl[i], px, py, mu)

        S_can[i] = integrate_3d(can_spin[i], px, py, mu)
        TQ_can[i] = integrate_3d(can_torque[i], px, py, mu)

    E_ratio = safe_ratio(E_can, E_bel)
    PT_ratio = safe_ratio(PT_can, PT_bel)
    PL_ratio = safe_ratio(PL_can, PL_bel)
    anis_can = safe_ratio(PL_can, PT_can)

    save_xy(outdir / "energy_ratio.txt", x, E_ratio, f"{xname}  E_can_over_E_bel")
    save_xy(outdir / "transverse_pressure_ratio.txt", x, PT_ratio, f"{xname}  PT_can_over_PT_bel")
    save_xy(outdir / "longitudinal_pressure_ratio.txt", x, PL_ratio, f"{xname}  PL_can_over_PL_bel")
    save_xy(outdir / "anisotropy_factor.txt", x, anis_can, f"{xname}  PL_can_over_PT_can")
    save_xy(outdir / "spin_density.txt", x, S_can, f"{xname}  spin_density")
    save_xy(outdir / "spin_torque.txt", x, TQ_can, f"{xname}  spin_torque")

    print("Saved:")
    print(outdir / "energy_ratio.txt")
    print(outdir / "transverse_pressure_ratio.txt")
    print(outdir / "longitudinal_pressure_ratio.txt")
    print(outdir / "anisotropy_factor.txt")
    print(outdir / "spin_density.txt")
    print(outdir / "spin_torque.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)