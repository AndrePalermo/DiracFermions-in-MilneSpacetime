#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import mpmath as mp

from libMilne_FINAL import (
    tabulating_canonical,
    tabulating_belinfante,
    tabulating_Polarization,
)


def safe_real_float(x, imag_tol: float = 1e-12) -> float:
    xr = mp.re(x)
    xi = mp.im(x)
    if abs(float(xi)) > imag_tol:
        print(f"[warning] discarding non-negligible imaginary part: {x}", flush=True)
    return float(xr)


def build_grid(xmax: float, npt: int) -> np.ndarray:
    return np.linspace(-xmax, xmax, npt, dtype=np.float64)


def make_slice_filename(
    outdir: str,
    scan_name: str,
    scan_value: float,
    mass: float,
    T: float,
    tau: float,
    precision: int,
    sample_mode: str,
) -> str:
    sval = f"{scan_value:+.12e}".replace("+", "p").replace("-", "m")
    return os.path.join(
        outdir,
        f"slice_{sample_mode}_{scan_name}_{sval}_m{mass:.6g}_T{T:.6g}_tau{tau:.6g}_prec{precision}.h5",
    )


def compute_one_slice(
    scan_name: str,
    scan_value: float,
    xmax: float,
    npt: int,
    T: float,
    mass: float,
    tau: float,
    precision: int,
    outdir: str,
    progress_stride: int = 100,
    sample_mode: str = "full",
) -> str:
    os.makedirs(outdir, exist_ok=True)

    beta = 1.0 / T
    if scan_name.lower() == "omega":
        omega = scan_value
        sp = omega / T
    elif scan_name.lower() == "sp":
        sp = scan_value
        omega = sp * T
    else:
        raise ValueError("scan_name must be 'omega' or 'sp'.")

    do_standard = sample_mode in ("standard", "full")
    do_polarization = sample_mode in ("polarization", "full")

    px_vals = build_grid(xmax, npt)
    py_vals = build_grid(xmax, npt)
    mu_vals = build_grid(xmax, npt)

    shape = (npt, npt, npt)

    # standard quantities
    if do_standard:
        can_energy = np.empty(shape, dtype=np.float64)
        can_pt = np.empty(shape, dtype=np.float64)
        can_pl = np.empty(shape, dtype=np.float64)
        can_spin = np.empty(shape, dtype=np.float64)
        can_torque = np.empty(shape, dtype=np.float64)

        bel_energy = np.empty(shape, dtype=np.float64)
        bel_pt = np.empty(shape, dtype=np.float64)
        bel_pl = np.empty(shape, dtype=np.float64)

    # polarization
    if do_polarization:
        pol0 = np.empty(shape, dtype=np.float64)
        pol1 = np.empty(shape, dtype=np.float64)
        pol2 = np.empty(shape, dtype=np.float64)
        pol3 = np.empty(shape, dtype=np.float64)

    total_points = npt * npt * npt
    counter = 0
    t0 = time.time()

    pid = os.getpid()
    print(
        f"[slice start][pid={pid}] mode={sample_mode} {scan_name}={scan_value:.8g} "
        f"(Omega={omega:.8g}, SP={sp:.8g}) | grid={npt}^3={total_points}",
        flush=True,
    )

    for i, px in enumerate(px_vals):
        for j, py in enumerate(py_vals):
            for k, mu in enumerate(mu_vals):
                if do_standard:
                    e_c, pt_c, pl_c, s_c, tq_c = tabulating_canonical(
                        mass=mass,
                        px=px,
                        py=py,
                        μ=mu,
                        τ=tau,
                        β=beta,
                        SP=sp,
                        precision=precision,
                    )

                    can_energy[i, j, k] = safe_real_float(e_c)
                    can_pt[i, j, k] = safe_real_float(pt_c)
                    can_pl[i, j, k] = safe_real_float(pl_c)
                    can_spin[i, j, k] = safe_real_float(s_c)
                    can_torque[i, j, k] = safe_real_float(tq_c)

                    e_b, pt_b, pl_b = tabulating_belinfante(
                        mass=mass,
                        px=px,
                        py=py,
                        μ=mu,
                        τ=tau,
                        β=beta,
                        precision=precision,
                    )

                    bel_energy[i, j, k] = safe_real_float(e_b)
                    bel_pt[i, j, k] = safe_real_float(pt_b)
                    bel_pl[i, j, k] = safe_real_float(pl_b)

                if do_polarization:
                    p0, p1, p2, p3 = tabulating_Polarization(
                        mass=mass,
                        px=px,
                        py=py,
                        μ=mu,
                        τ=tau,
                        β=beta,
                        SP=sp,
                        precision=precision,
                    )

                    pol0[i, j, k] = safe_real_float(p0)
                    pol1[i, j, k] = safe_real_float(p1)
                    pol2[i, j, k] = safe_real_float(p2)
                    pol3[i, j, k] = safe_real_float(p3)

                counter += 1
                if counter % progress_stride == 0 or counter == total_points:
                    elapsed = time.time() - t0
                    frac = counter / total_points
                    rate = counter / elapsed if elapsed > 0 else 0.0
                    eta = (total_points - counter) / rate if rate > 0 else math.inf
                    print(
                        f"[slice progress][pid={pid}] mode={sample_mode} {scan_name}={scan_value:.8g} "
                        f"{counter}/{total_points} ({100*frac:.1f}%) "
                        f"| elapsed={elapsed:.1f}s | eta={eta:.1f}s",
                        flush=True,
                    )

    outfile = make_slice_filename(
        outdir=outdir,
        scan_name=scan_name,
        scan_value=scan_value,
        mass=mass,
        T=T,
        tau=tau,
        precision=precision,
        sample_mode=sample_mode,
    )

    with h5py.File(outfile, "w") as h5:
        h5.attrs["scan_name"] = scan_name
        h5.attrs["scan_value"] = scan_value
        h5.attrs["omega"] = omega
        h5.attrs["SP"] = sp
        h5.attrs["mass"] = mass
        h5.attrs["T"] = T
        h5.attrs["beta"] = beta
        h5.attrs["tau"] = tau
        h5.attrs["precision"] = precision
        h5.attrs["xmax"] = xmax
        h5.attrs["npt"] = npt
        h5.attrs["sample_mode"] = sample_mode

        ggrid = h5.create_group("grid")
        ggrid.create_dataset("px", data=px_vals)
        ggrid.create_dataset("py", data=py_vals)
        ggrid.create_dataset("mu", data=mu_vals)

        if do_standard:
            gcan = h5.create_group("canonical")
            gcan.create_dataset("energy_density", data=can_energy, compression="gzip")
            gcan.create_dataset("transv_pressure", data=can_pt, compression="gzip")
            gcan.create_dataset("long_pressure", data=can_pl, compression="gzip")
            gcan.create_dataset("spin_density", data=can_spin, compression="gzip")
            gcan.create_dataset("torque", data=can_torque, compression="gzip")

            gbel = h5.create_group("belinfante")
            gbel.create_dataset("energy_density", data=bel_energy, compression="gzip")
            gbel.create_dataset("transv_pressure", data=bel_pt, compression="gzip")
            gbel.create_dataset("long_pressure", data=bel_pl, compression="gzip")

        if do_polarization:
            gpol = h5.create_group("polarization")
            gpol.create_dataset("P0", data=pol0, compression="gzip")
            gpol.create_dataset("P1", data=pol1, compression="gzip")
            gpol.create_dataset("P2", data=pol2, compression="gzip")
            gpol.create_dataset("P3", data=pol3, compression="gzip")

    elapsed = time.time() - t0
    print(
        f"[slice done][pid={pid}] mode={sample_mode} {scan_name}={scan_value:.8g} "
        f"saved to {outfile} | elapsed={elapsed:.1f}s",
        flush=True,
    )
    return outfile


def merge_slice_files(outdir: str, merged_file: str) -> str:
    slice_files = sorted(glob.glob(os.path.join(outdir, "slice_*.h5")))
    if not slice_files:
        raise FileNotFoundError(f"No slice_*.h5 files found in {outdir}")

    with h5py.File(slice_files[0], "r") as h5:
        px = h5["grid/px"][:]
        py = h5["grid/py"][:]
        mu = h5["grid/mu"][:]

        mass = h5.attrs["mass"]
        T = h5.attrs["T"]
        beta = h5.attrs["beta"]
        tau = h5.attrs["tau"]
        precision = h5.attrs["precision"]
        scan_name = h5.attrs["scan_name"]
        sample_mode = h5.attrs.get("sample_mode", "standard")
        if isinstance(sample_mode, bytes):
            sample_mode = sample_mode.decode()

        has_standard = ("canonical" in h5) and ("belinfante" in h5)
        has_polarization = ("polarization" in h5)

        if has_standard:
            c_shape = h5["canonical/energy_density"].shape
            b_shape = h5["belinfante/energy_density"].shape

        if has_polarization:
            p_shape = h5["polarization/P0"].shape

    nslices = len(slice_files)

    scan_values = np.empty(nslices, dtype=np.float64)
    omega_values = np.empty(nslices, dtype=np.float64)
    sp_values = np.empty(nslices, dtype=np.float64)

    if has_standard:
        can_energy = np.empty((nslices,) + c_shape, dtype=np.float64)
        can_pt = np.empty((nslices,) + c_shape, dtype=np.float64)
        can_pl = np.empty((nslices,) + c_shape, dtype=np.float64)
        can_spin = np.empty((nslices,) + c_shape, dtype=np.float64)
        can_torque = np.empty((nslices,) + c_shape, dtype=np.float64)

        bel_energy = np.empty((nslices,) + b_shape, dtype=np.float64)
        bel_pt = np.empty((nslices,) + b_shape, dtype=np.float64)
        bel_pl = np.empty((nslices,) + b_shape, dtype=np.float64)

    if has_polarization:
        pol0 = np.empty((nslices,) + p_shape, dtype=np.float64)
        pol1 = np.empty((nslices,) + p_shape, dtype=np.float64)
        pol2 = np.empty((nslices,) + p_shape, dtype=np.float64)
        pol3 = np.empty((nslices,) + p_shape, dtype=np.float64)

    print(f"[merge] merging {nslices} slice files...", flush=True)

    for idx, fpath in enumerate(slice_files, start=1):
        print(f"[merge] reading {idx}/{nslices}: {fpath}", flush=True)
        with h5py.File(fpath, "r") as h5:
            scan_values[idx - 1] = h5.attrs["scan_value"]
            omega_values[idx - 1] = h5.attrs["omega"]
            sp_values[idx - 1] = h5.attrs["SP"]

            if has_standard:
                can_energy[idx - 1] = h5["canonical/energy_density"][:]
                can_pt[idx - 1] = h5["canonical/transv_pressure"][:]
                can_pl[idx - 1] = h5["canonical/long_pressure"][:]
                can_spin[idx - 1] = h5["canonical/spin_density"][:]
                can_torque[idx - 1] = h5["canonical/torque"][:]

                bel_energy[idx - 1] = h5["belinfante/energy_density"][:]
                bel_pt[idx - 1] = h5["belinfante/transv_pressure"][:]
                bel_pl[idx - 1] = h5["belinfante/long_pressure"][:]

            if has_polarization:
                pol0[idx - 1] = h5["polarization/P0"][:]
                pol1[idx - 1] = h5["polarization/P1"][:]
                pol2[idx - 1] = h5["polarization/P2"][:]
                pol3[idx - 1] = h5["polarization/P3"][:]

    order = np.argsort(scan_values)

    with h5py.File(merged_file, "w") as h5:
        h5.attrs["scan_name"] = scan_name
        h5.attrs["mass"] = mass
        h5.attrs["T"] = T
        h5.attrs["beta"] = beta
        h5.attrs["tau"] = tau
        h5.attrs["precision"] = precision
        h5.attrs["nslices"] = nslices
        h5.attrs["sample_mode"] = sample_mode

        ggrid = h5.create_group("grid")
        ggrid.create_dataset("scan_values", data=scan_values[order])
        ggrid.create_dataset("omega_values", data=omega_values[order])
        ggrid.create_dataset("SP_values", data=sp_values[order])
        ggrid.create_dataset("px", data=px)
        ggrid.create_dataset("py", data=py)
        ggrid.create_dataset("mu", data=mu)

        if has_standard:
            gcan = h5.create_group("canonical")
            gcan.create_dataset("energy_density", data=can_energy[order], compression="gzip")
            gcan.create_dataset("transv_pressure", data=can_pt[order], compression="gzip")
            gcan.create_dataset("long_pressure", data=can_pl[order], compression="gzip")
            gcan.create_dataset("spin_density", data=can_spin[order], compression="gzip")
            gcan.create_dataset("torque", data=can_torque[order], compression="gzip")

            gbel = h5.create_group("belinfante")
            gbel.create_dataset("energy_density", data=bel_energy[order], compression="gzip")
            gbel.create_dataset("transv_pressure", data=bel_pt[order], compression="gzip")
            gbel.create_dataset("long_pressure", data=bel_pl[order], compression="gzip")

        if has_polarization:
            gpol = h5.create_group("polarization")
            gpol.create_dataset("P0", data=pol0[order], compression="gzip")
            gpol.create_dataset("P1", data=pol1[order], compression="gzip")
            gpol.create_dataset("P2", data=pol2[order], compression="gzip")
            gpol.create_dataset("P3", data=pol3[order], compression="gzip")

    print(f"[merge done] saved merged file to {merged_file}", flush=True)
    return merged_file


def run_compute(args: argparse.Namespace) -> None:
    scan_values = np.linspace(args.scan_min, args.scan_max, args.nscan, dtype=np.float64)
    workers = args.workers if args.workers is not None else args.nscan
    workers = max(1, min(workers, len(scan_values)))

    print("[main] starting computation", flush=True)
    print(f"[main] scan variable = {args.scan}", flush=True)
    print(f"[main] sample mode = {args.sample_mode}", flush=True)
    print(f"[main] number of slices = {len(scan_values)}", flush=True)
    print(f"[main] workers = {workers}", flush=True)
    print(f"[main] grid = {args.npt} x {args.npt} x {args.npt}", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                compute_one_slice,
                args.scan,
                float(sval),
                args.xmax,
                args.npt,
                args.T,
                args.mass,
                args.tau,
                args.precision,
                args.outdir,
                args.progress_stride,
                args.sample_mode,
            )
            for sval in scan_values
        ]

        total = len(futures)
        done = 0
        for fut in as_completed(futures):
            _ = fut.result()
            done += 1
            print(f"[main] completed slices: {done}/{total}", flush=True)

    print("[main] all slices completed", flush=True)

    if args.auto_merge:
        merged_file = args.merged_file or os.path.join(args.outdir, "merged_all_slices.h5")
        merge_slice_files(args.outdir, merged_file)


def run_merge(args: argparse.Namespace) -> None:
    merge_slice_files(args.outdir, args.merged_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_compute = subparsers.add_parser("compute")
    p_compute.add_argument("--xmax", type=float, required=True)
    p_compute.add_argument("--npt", type=int, required=True)
    p_compute.add_argument("--T", type=float, required=True)
    p_compute.add_argument("--mass", type=float, required=True)
    p_compute.add_argument("--tau", type=float, required=True)
    p_compute.add_argument("--scan", choices=["omega", "sp"], required=True)
    p_compute.add_argument("--scan-min", type=float, required=True)
    p_compute.add_argument("--scan-max", type=float, required=True)
    p_compute.add_argument("--nscan", type=int, required=True)
    p_compute.add_argument("--precision", type=int, default=50)
    p_compute.add_argument("--outdir", type=str, default="output")
    p_compute.add_argument("--workers", type=int, default=None)
    p_compute.add_argument("--progress-stride", type=int, default=100)
    p_compute.add_argument(
        "--sample-mode",
        choices=["standard", "polarization", "full"],
        default="standard",
        help="standard = canonical+belinfante, polarization = only polarization, full = both",
    )
    p_compute.add_argument("--auto-merge", action="store_true")
    p_compute.add_argument("--merged-file", type=str, default=None)
    p_compute.set_defaults(func=run_compute)

    p_merge = subparsers.add_parser("merge")
    p_merge.add_argument("--outdir", type=str, required=True)
    p_merge.add_argument("--merged-file", type=str, required=True)
    p_merge.set_defaults(func=run_merge)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()