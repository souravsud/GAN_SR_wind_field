"""
tools/test_plotting.py
Apache License

Matplotlib-based plotting utilities for test/inference output.
Generates comparison figures (LR vs HR vs SR vs TL) and error figures
for each velocity component at a mid-height slice, then saves them to disk.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch


# ── component labels used when none are supplied ─────────────────────────────
_DEFAULT_COMP_NAMES = ["Ux", "Uy", "Uz"]


# ── figure builders ───────────────────────────────────────────────────────────

def create_comparison_figure(
    wind_height_index,
    wind_comp_LR,
    wind_comp_HR,
    wind_comp_SR,
    wind_comp_TL,
):
    """2×2 grid: LR / HR / SR / TL for one velocity component at one height."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    vmin, vmax = float(wind_comp_HR.min()), float(wind_comp_HR.max())

    axes[0, 0].pcolor(wind_comp_LR, vmin=vmin, vmax=vmax, cmap="viridis", edgecolor="none")
    axes[0, 0].set_title("LR")
    axes[0, 1].pcolor(wind_comp_HR, vmin=vmin, vmax=vmax, cmap="viridis", edgecolor="none")
    axes[0, 1].set_title("HR")
    axes[1, 1].pcolor(wind_comp_SR, vmin=vmin, vmax=vmax, cmap="viridis", edgecolor="none")
    axes[1, 1].set_title("SR")
    axes[1, 0].pcolor(wind_comp_TL, vmin=vmin, vmax=vmax, cmap="viridis", edgecolor="none")
    axes[1, 0].set_title("TL")

    fig.suptitle(f"z index = {wind_height_index}", fontsize=10)
    fig.subplots_adjust(hspace=0.3)

    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_clim(vmin=vmin, vmax=vmax)
    fig.colorbar(sm, ax=axes)
    return fig


def create_error_figure(
    wind_height_index,
    wind_comp_HR,
    wind_comp_SR,
    wind_comp_TL,
    average_SR_error,
    average_TL_error,
):
    """2×3 grid: signed error / field / absolute error for both SR and TL."""
    hr = wind_comp_HR
    sr = wind_comp_SR
    tl = wind_comp_TL

    vmin_wf = float(np.minimum(sr, tl).min())
    vmax_wf = float(np.maximum(sr, tl).max())

    err_sr = sr - hr
    err_tl = tl - hr
    vmin_err = float(np.concatenate([err_sr.ravel(), err_tl.ravel()]).min())
    vmax_err = float(np.concatenate([err_sr.ravel(), err_tl.ravel()]).max())

    abs_max = max(abs(vmax_err), abs(vmin_err))
    vmin_abs, vmax_abs = 0.0, abs_max

    sm_wf = plt.cm.ScalarMappable(cmap="viridis")
    sm_wf.set_clim(vmin=vmin_wf, vmax=vmax_wf)
    sm_err = plt.cm.ScalarMappable(cmap="coolwarm")
    sm_err.set_clim(vmin=vmin_err, vmax=vmax_err)
    sm_abs = plt.cm.ScalarMappable(cmap="jet")
    sm_abs.set_clim(vmin=vmin_abs, vmax=vmax_abs)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True, sharex=True)

    # Row 0 – SR
    axes[0, 0].pcolor(err_sr, vmin=vmin_err, vmax=vmax_err, cmap="coolwarm")
    axes[0, 0].set_title("Error SR−HR (m/s)")
    axes[0, 1].pcolor(sr, vmin=vmin_wf, vmax=vmax_wf, cmap="viridis", edgecolor="none")
    axes[0, 1].set_title(f"SR  avg|err|={average_SR_error:.3f} m/s")
    axes[0, 2].pcolor(np.abs(err_sr), vmin=vmin_abs, vmax=vmax_abs, cmap="jet", edgecolor="none")
    axes[0, 2].set_title("SR Absolute Error (m/s)")

    # Row 1 – TL
    axes[1, 0].pcolor(err_tl, vmin=vmin_err, vmax=vmax_err, cmap="coolwarm")
    axes[1, 0].set_title("Error TL−HR (m/s)")
    axes[1, 1].pcolor(tl, vmin=vmin_wf, vmax=vmax_wf, cmap="viridis", edgecolor="none")
    axes[1, 1].set_title(f"TL  avg|err|={average_TL_error:.3f} m/s")
    axes[1, 2].pcolor(np.abs(err_tl), vmin=vmin_abs, vmax=vmax_abs, cmap="jet", edgecolor="none")
    axes[1, 2].set_title("TL Absolute Error (m/s)")

    for row in range(2):
        fig.colorbar(sm_err, ax=axes[row, 0])
        fig.colorbar(sm_wf,  ax=axes[row, 1])
        fig.colorbar(sm_abs, ax=axes[row, 2])

    fig.suptitle(f"z index = {wind_height_index}", fontsize=10)
    fig.subplots_adjust(hspace=0.3)
    return fig


# ── main entry-point called from test.py ─────────────────────────────────────

def save_test_plots(
    LR: torch.Tensor,
    HR: torch.Tensor,
    SR: torch.Tensor,
    TL: torch.Tensor,
    folder_path: str,
    field_name: str,
    component_names=None,
    height_indices=None,
):
    """Generate and save comparison + error PNG figures for a test sample.

    Plots are written to  ``<folder_path>/plots/``.

    Args:
        LR: (C, Nx_lr, Ny_lr, Nk) – low-resolution input.
        HR: (C, Nx_hr, Ny_hr, Nk) – high-resolution ground truth.
        SR: (C, Nx_hr, Ny_hr, Nk) – super-resolved output.
        TL: (C_vel, Nx_hr, Ny_hr, Nk) – trilinear-upsampled LR (first 3 ch).
        folder_path: directory that *contains* the ``plots/`` sub-folder.
        field_name: unique label used in saved filenames.
        component_names: list of channel-name strings (default: Ux, Uy, Uz).
        height_indices: list of z-indices to plot; defaults to the mid slice.
    """
    if component_names is None:
        component_names = _DEFAULT_COMP_NAMES

    plots_dir = os.path.join(folder_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Convert to numpy arrays for plotting
    def _to_np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().float().numpy()
        return np.asarray(t, dtype=np.float32)

    lr_np = _to_np(LR)
    hr_np = _to_np(HR)
    sr_np = _to_np(SR)
    tl_np = _to_np(TL)

    n_k = hr_np.shape[-1]
    if height_indices is None:
        height_indices = [n_k // 2]

    # Only iterate over velocity components (first 3 channels)
    n_comp = min(3, hr_np.shape[0], tl_np.shape[0])

    for height_idx in height_indices:
        for c in range(n_comp):
            comp_name = component_names[c] if c < len(component_names) else f"ch{c}"

            lr_sl = lr_np[c, :, :, height_idx]
            hr_sl = hr_np[c, :, :, height_idx]
            sr_sl = sr_np[c, :, :, height_idx]
            tl_sl = tl_np[c, :, :, height_idx]

            avg_sr_err = float(np.mean(np.abs(hr_sl - sr_sl)))
            avg_tl_err = float(np.mean(np.abs(hr_sl - tl_sl)))

            base = f"{field_name}_{comp_name}_z{height_idx}"

            fig1 = create_comparison_figure(height_idx, lr_sl, hr_sl, sr_sl, tl_sl)
            fig1.savefig(
                os.path.join(plots_dir, f"{base}_comparison.png"),
                dpi=100,
                bbox_inches="tight",
            )
            plt.close(fig1)

            fig2 = create_error_figure(
                height_idx, hr_sl, sr_sl, tl_sl, avg_sr_err, avg_tl_err
            )
            fig2.savefig(
                os.path.join(plots_dir, f"{base}_error.png"),
                dpi=100,
                bbox_inches="tight",
            )
            plt.close(fig2)
