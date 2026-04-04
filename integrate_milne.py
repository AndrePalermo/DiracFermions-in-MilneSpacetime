#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import plot_params as pl

try:
    from scipy.integrate import simpson
except ImportError:
    simpson = None


def integrate_3d_simpson(values: np.ndarray, px: np.ndarray, py: np.ndarray, mu: np.ndarray) -> float:
    """
    Integrate values(px, py, mu) over mu, py, px using Simpson's rule.
    Expected shape: (len(px), len(py), len(mu)).
    """
    if simpson is None:
        raise ImportError(
            "scipy is required for Simpson integration.\n"
            "Install it with: pip install scipy"
        )

    tmp_mu = simpson(values, x=mu, axis=2)
    tmp_py = simpson(tmp_mu, x=py, axis=1)
    result = simpson(tmp_py, x=px, axis=0)
    return float(result)


def load_dataset(h5: h5py.File, path: str) -> np.ndarray:
    if path not in h5:
        raise KeyError(f"Dataset '{path}' not found in file.")
    return h5[path][:]


def load_scalar_or_attr(h5: h5py.File, keys: list[str], dataset_paths: list[str]) -> float | None:
    for key in keys:
        if key in h5.attrs:
            value = h5.attrs[key]
            if isinstance(value, bytes):
                value = value.decode()
            return float(value)

    for path in dataset_paths:
        if path in h5:
            data = h5[path][()]
            return float(np.asarray(data))

    return None


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    mask = np.abs(den) > 1e-30
    out[mask] = num[mask] / den[mask]
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    pl.set_mpl()
    mpl.rc("axes", grid=False)
    plt.rcParams.update({
        "savefig.bbox": "tight",
    })


def make_figure(scale_x: float = 2.4, scale_y: float = 2.2):
    return plt.figure(figsize=(pl.pWidth * scale_x, pl.pHeight * scale_y))


def save_line_plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    *,
    color: str = "tab:blue",
    add_reference_one: bool = False,
    reference_value: float = 1.0,
    scale_x: float = 2.4,
    scale_y: float = 2.2,
    linewidth: float = 2.2,
) -> None:
    fig = make_figure(scale_x=scale_x, scale_y=scale_y)
    ax = fig.add_subplot(111)

    ax.plot(x, y, "-", color=color, linewidth=linewidth, zorder=3)

    if add_reference_one:
        ax.axhline(reference_value, color="grey", linewidth=3.0, zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)


def save_two_line_plot(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    *,
    label1: str,
    label2: str,
    color: str = "tab:blue",
    linestyle1: str = "-",
    linestyle2: str = "--",
    add_reference_one: bool = False,
    reference_value: float = 1.0,
    scale_x: float = 2.4,
    scale_y: float = 2.2,
    linewidth: float = 2.2,
) -> None:
    fig = make_figure(scale_x=scale_x, scale_y=scale_y)
    ax = fig.add_subplot(111)

    ax.plot(x, y1, linestyle1, color=color, linewidth=linewidth, label=label1, zorder=3)
    ax.plot(x, y2, linestyle2, color=color, linewidth=linewidth, label=label2, zorder=3)

    if add_reference_one:
        ax.axhline(reference_value, color="grey", linewidth=3.0, zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrate merged Milne HDF5 data and plot observables vs SP or Omega."
    )
    parser.add_argument("input_file", type=str, help="Merged HDF5 file from the sampler.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="integrated_output",
        help="Directory for plots and integrated results.",
    )
    parser.add_argument(
        "--xvar",
        choices=["auto", "sp", "omega"],
        default="auto",
        help="Choose x-axis variable for the original ratio/spin/torque plots. Default: infer from file.",
    )
    parser.add_argument(
        "--scale-x",
        type=float,
        default=2.4,
        help="Horizontal scaling factor applied to plot_params.pWidth.",
    )
    parser.add_argument(
        "--scale-y",
        type=float,
        default=2.2,
        help="Vertical scaling factor applied to plot_params.pHeight.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    setup_plot_style()

    with h5py.File(args.input_file, "r") as h5:
        scan_name = h5.attrs.get("scan_name", "scan")
        if isinstance(scan_name, bytes):
            scan_name = scan_name.decode()

        px = load_dataset(h5, "grid/px")
        py = load_dataset(h5, "grid/py")
        mu = load_dataset(h5, "grid/mu")

        scan_values = load_dataset(h5, "grid/scan_values")
        omega_values = load_dataset(h5, "grid/omega_values")
        sp_values = load_dataset(h5, "grid/SP_values")

        can_energy = load_dataset(h5, "canonical/energy_density")
        can_pt = load_dataset(h5, "canonical/transv_pressure")
        can_pl = load_dataset(h5, "canonical/long_pressure")
        can_spin = load_dataset(h5, "canonical/spin_density")
        can_torque = load_dataset(h5, "canonical/torque")

        bel_energy = load_dataset(h5, "belinfante/energy_density")
        bel_pt = load_dataset(h5, "belinfante/transv_pressure")
        bel_pl = load_dataset(h5, "belinfante/long_pressure")

        tau = load_scalar_or_attr(
            h5,
            keys=["tau", "Tau", "proper_time"],
            dataset_paths=["tau", "grid/tau", "params/tau", "milne/tau"],
        )

    if args.xvar == "auto":
        if str(scan_name).lower() == "sp":
            x = sp_values
            x_label = r"$\mathrm{SP}$"
        elif str(scan_name).lower() == "omega":
            x = omega_values
            x_label = r"$\Omega\,[\mathrm{GeV}]$"
        else:
            x = scan_values
            x_label = rf"${scan_name}$"
    elif args.xvar == "sp":
        x = sp_values
        x_label = r"$\mathrm{SP}$"
    else:
        x = omega_values
        x_label = r"$\Omega\,[\mathrm{GeV}]$"

    x_omega = omega_values
    x_omega_label = r"$\Omega\,[\mathrm{GeV}]$"

    nscan = len(x)

    E_can = np.empty(nscan, dtype=np.float64)
    E_bel = np.empty(nscan, dtype=np.float64)
    PT_can = np.empty(nscan, dtype=np.float64)
    PT_bel = np.empty(nscan, dtype=np.float64)
    PL_can = np.empty(nscan, dtype=np.float64)
    PL_bel = np.empty(nscan, dtype=np.float64)
    S_can = np.empty(nscan, dtype=np.float64)
    TQ_can = np.empty(nscan, dtype=np.float64)

    print("\nIntegrating slices")
    print("-" * 120)

    for i in range(nscan):
        E_can[i] = integrate_3d_simpson(can_energy[i], px, py, mu)
        E_bel[i] = integrate_3d_simpson(bel_energy[i], px, py, mu)

        PT_can[i] = integrate_3d_simpson(can_pt[i], px, py, mu)
        PT_bel[i] = integrate_3d_simpson(bel_pt[i], px, py, mu)

        PL_can[i] = integrate_3d_simpson(can_pl[i], px, py, mu)
        PL_bel[i] = integrate_3d_simpson(bel_pl[i], px, py, mu)

        S_can[i] = integrate_3d_simpson(can_spin[i], px, py, mu)
        TQ_can[i] = integrate_3d_simpson(can_torque[i], px, py, mu)

        print(
            f"{x_label}={x[i]:.8g} | "
            f"E_can={E_can[i]:.10e} E_bel={E_bel[i]:.10e} | "
            f"PT_can={PT_can[i]:.10e} PT_bel={PT_bel[i]:.10e} | "
            f"PL_can={PL_can[i]:.10e} PL_bel={PL_bel[i]:.10e} | "
            f"S_can={S_can[i]:.10e} TQ_can={TQ_can[i]:.10e}"
        )

    E_ratio = safe_ratio(E_can, E_bel)
    PT_ratio = safe_ratio(PT_can, PT_bel)
    PL_ratio = safe_ratio(PL_can, PL_bel)

    anis_can = safe_ratio(PL_can, PT_can)
    anis_bel = safe_ratio(PL_bel, PT_bel)

    W_can = safe_ratio((2.0 * PT_can + PL_can) / 3.0, E_can)
    W_bel = safe_ratio((2.0 * PT_bel + PL_bel) / 3.0, E_bel)
    W_ratio = safe_ratio(W_can, W_bel)

    trace_can = E_can - 2.0 * PT_can - PL_can
    trace_bel = E_bel - 2.0 * PT_bel - PL_bel
    trace_ratio = safe_ratio(trace_can, trace_bel)

    if tau is None:
        raise KeyError(
            "Could not determine tau from file. Needed for "
            "dot{S} = Torque - (1/tau) Spin density.\n"
            "Expected an attribute like 'tau' or a dataset like 'grid/tau'."
        )

    Sdot_can = TQ_can - S_can / tau

    out_h5 = outdir / "integrated_results.h5"
    with h5py.File(out_h5, "w") as h5:
        h5.attrs["x_label"] = x_label
        h5.attrs["omega_x_label"] = x_omega_label
        h5.attrs["tau"] = tau

        h5.create_dataset("x", data=x)
        h5.create_dataset("omega_x", data=x_omega)

        h5.create_dataset("E_can", data=E_can)
        h5.create_dataset("E_bel", data=E_bel)

        h5.create_dataset("PT_can", data=PT_can)
        h5.create_dataset("PT_bel", data=PT_bel)

        h5.create_dataset("PL_can", data=PL_can)
        h5.create_dataset("PL_bel", data=PL_bel)

        h5.create_dataset("S_can", data=S_can)
        h5.create_dataset("TQ_can", data=TQ_can)
        h5.create_dataset("Sdot_can", data=Sdot_can)

        h5.create_dataset("E_ratio", data=E_ratio)
        h5.create_dataset("PT_ratio", data=PT_ratio)
        h5.create_dataset("PL_ratio", data=PL_ratio)

        h5.create_dataset("anis_can", data=anis_can)
        h5.create_dataset("anis_bel", data=anis_bel)

        h5.create_dataset("W_can", data=W_can)
        h5.create_dataset("W_bel", data=W_bel)
        h5.create_dataset("W_ratio", data=W_ratio)

        h5.create_dataset("trace_can", data=trace_can)
        h5.create_dataset("trace_bel", data=trace_bel)
        h5.create_dataset("trace_ratio", data=trace_ratio)

    save_line_plot(
        x_omega,
        E_ratio,
        x_omega_label,
        r"$\mathcal{E}_{\mathrm{can}}/\mathcal{E}_{\mathrm{Bel}}$",
        outdir / "energy_ratio.pdf",
        add_reference_one=True,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )

    save_line_plot(
        x_omega,
        PT_ratio,
        x_omega_label,
        r"$\mathcal{P}_{\mathrm{T},\mathrm{can}}/\mathcal{P}_{\mathrm{T},\mathrm{Bel}}$",
        outdir / "transverse_pressure_ratio.pdf",
        add_reference_one=True,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )

    save_line_plot(
        x_omega,
        PL_ratio,
        x_omega_label,
        r"$\mathcal{P}_{\mathrm{L},\mathrm{can}}/\mathcal{P}_{\mathrm{L},\mathrm{Bel}}$",
        outdir / "longitudinal_pressure_ratio.pdf",
        add_reference_one=True,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )

    # New combined normalized pressure plot
    save_two_line_plot(
        x_omega,
        PT_ratio,
        PL_ratio,
        x_omega_label,
        r"$\mathcal{P}_{\mathrm{can}}/\mathcal{P}_{\mathrm{Bel}}$",
        outdir / "normalized_pressures_combined.pdf",
        label1=r"$\mathcal{P}_{\mathrm{T}}$",
        label2=r"$\mathcal{P}_{\mathrm{L}}$",
        linestyle1="-",
        linestyle2="--",
        add_reference_one=True,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )

    save_line_plot(
        x_omega,
        S_can,
        x_omega_label,
        r"$\mathcal{S}\,[\mathrm{GeV}^3]$",
        outdir / "spin_density.pdf",
        add_reference_one=False,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )

    save_line_plot(
        x_omega,
        TQ_can,
        x_omega_label,
        r"$\mathcal{T}\,[\mathrm{GeV}^4]$",
        outdir / "spin_torque.pdf",
        add_reference_one=False,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )

    save_line_plot(
        x_omega,
        Sdot_can,
        x_omega_label,
        r"$\dot{\mathcal{S}}\,[\mathrm{GeV}^4]$",
        outdir / "spin_density_time_derivative_vs_omega.pdf",
        add_reference_one=False,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )

    save_line_plot(
        x_omega,
        anis_can,
        x_omega_label,
        r"$\mathcal{P}_{\mathrm{L}}/\mathcal{P}_{\mathrm{T}}$",
        outdir / "anisotropy_factor_vs_omega.pdf",
        add_reference_one=True,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )

    save_line_plot(
        x_omega,
        W_can,
        x_omega_label,
        r"$\mathcal{P}_{\mathrm{eff}}/\mathcal{E}$",
        outdir / "equation_of_state_vs_omega.pdf",
        add_reference_one=False,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )

    save_line_plot(
        x_omega,
        W_ratio,
        x_omega_label,
        r"$W/W_{\mathrm{Bel}}$",
        outdir / "normalized_equation_of_state_vs_omega.pdf",
        add_reference_one=True,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )

    save_line_plot(
        x_omega,
        trace_can,
        x_omega_label,
        r"$T^\mu_{\ \mu}=\mathcal{E}-2\mathcal{P}_{\mathrm{T}}-\mathcal{P}_{\mathrm{L}}\,[\mathrm{GeV}^4]$",
        outdir / "trace_vs_omega.pdf",
        add_reference_one=False,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )

    print("\nSaved files")
    print("-" * 120)
    print(out_h5)
    print(outdir / "energy_ratio.pdf")
    print(outdir / "transverse_pressure_ratio.pdf")
    print(outdir / "longitudinal_pressure_ratio.pdf")
    print(outdir / "normalized_pressures_combined.pdf")
    print(outdir / "spin_density.pdf")
    print(outdir / "spin_torque.pdf")
    print(outdir / "spin_density_time_derivative_vs_omega.pdf")
    print(outdir / "anisotropy_factor_vs_omega.pdf")
    print(outdir / "equation_of_state_vs_omega.pdf")
    print(outdir / "normalized_equation_of_state_vs_omega.pdf")
    print("-" * 120)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)