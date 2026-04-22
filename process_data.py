"""
process_data.py
Zarr-based Wind-Terrain CFD dataset loading.
Replaces the previous NetCDF/pickle-based HARMONIE-SIMRA pipeline.
"""

import os
import re
import pickle

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import xarray as xr

# Reference velocity used by all simulations (m/s)
REFERENCE_VELOCITY = 10.0


class WindTerrainDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for the Zarr Wind-Terrain CFD dataset.

    Each case is a single Zarr store containing steady-state RANS wind fields
    over a terrain patch at a fixed reference velocity of 10 m/s.

    __getitem__ returns (LR, HR, Z) tensors with the same semantics as the
    former CustomizedDataset so that train.py / test.py are unchanged:
      - HR  : normalised target wind field, shape (out_ch, nx, ny, nk)
      - LR  : spatially coarsened input + optional auxiliary channels,
              shape (in_ch, nx//coarseness_factor, ny//coarseness_factor, nk)
      - Z   : absolute elevation used for vertical gradient computation,
              shape (1, nx, ny, nk)
    """

    def __init__(
        self,
        zarr_paths,
        case_ids,
        Z_MIN,
        Z_MAX,
        UVW_MAX,
        P_MIN,
        P_MAX,
        Z_ABOVE_GROUND_MAX,
        x,
        y,
        include_pressure=False,
        include_z_channel=True,
        include_above_ground_channel=False,
        COARSENESS_FACTOR=4,
        crop_nx=300,
        crop_ny=300,
        crop_nk=64,
        data_aug_rot=True,
        data_aug_flip=True,
        for_plotting=False,
        is_test=False,
        # Legacy parameters kept for interface compatibility
        enable_slicing=False,
        slice_size=64,
        terrain=None,
    ):
        self.zarr_paths = zarr_paths
        self.case_ids = case_ids
        self.Z_MIN = Z_MIN
        self.Z_MAX = Z_MAX
        self.Z_ABOVE_GROUND_MAX = Z_ABOVE_GROUND_MAX
        self.UVW_MAX = UVW_MAX
        self.P_MIN = P_MIN
        self.P_MAX = P_MAX
        self.x = x
        self.y = y
        self.include_pressure = include_pressure
        self.include_z_channel = include_z_channel
        self.include_above_ground_channel = include_above_ground_channel
        self.coarseness_factor = COARSENESS_FACTOR
        self.crop_nx = crop_nx
        self.crop_ny = crop_ny
        self.crop_nk = crop_nk
        self.data_aug_rot = data_aug_rot
        self.data_aug_flip = data_aug_flip
        self.for_plotting = for_plotting
        self.is_test = is_test

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.zarr_paths)

    def __getitem__(self, index):
        # ---- crop indices --------------------------------------------------
        # Full domain: ni=300, nj=300, nk=64.  Centre crop in i/j; keep
        # the lowest crop_nk layers from the ground (k=0 is the ground layer).
        ni, nj = 300, 300
        i_start = (ni - self.crop_nx) // 2
        i_end = i_start + self.crop_nx
        j_start = (nj - self.crop_ny) // 2
        j_end = j_start + self.crop_ny

        # ---- load from Zarr ------------------------------------------------
        ds = xr.open_zarr(self.zarr_paths[index])
        Ux = ds["Ux"].values[i_start:i_end, j_start:j_end, : self.crop_nk].astype(
            np.float32
        )
        Uy = ds["Uy"].values[i_start:i_end, j_start:j_end, : self.crop_nk].astype(
            np.float32
        )
        Uz = ds["Uz"].values[i_start:i_end, j_start:j_end, : self.crop_nk].astype(
            np.float32
        )
        Z_abs = ds["Z"].values[i_start:i_end, j_start:j_end, : self.crop_nk].astype(
            np.float32
        )
        h_agl = ds["h_agl"].values[
            i_start:i_end, j_start:j_end, : self.crop_nk
        ].astype(np.float32)
        p = None
        if self.include_pressure:
            p = ds["p"].values[i_start:i_end, j_start:j_end, : self.crop_nk].astype(
                np.float32
            )
        ds.close()

        # ---- build LR/HR/Z tensors -----------------------------------------
        LR, HR, Z_tensor = reformat_to_torch(
            Ux,
            Uy,
            Uz,
            p,
            Z_abs,
            h_agl,
            self.Z_MIN,
            self.Z_MAX,
            self.Z_ABOVE_GROUND_MAX,
            self.UVW_MAX,
            self.P_MIN,
            self.P_MAX,
            coarseness_factor=self.coarseness_factor,
            include_pressure=self.include_pressure,
            include_z_channel=self.include_z_channel,
            include_above_ground_channel=self.include_above_ground_channel,
            for_plotting=self.for_plotting,
        )

        # ---- data augmentation ---------------------------------------------
        if self.data_aug_rot:
            amount_of_rotations = np.random.randint(0, 4)
            LR = torch.rot90(LR, amount_of_rotations, [1, 2])
            HR = torch.rot90(HR, amount_of_rotations, [1, 2])
            Z_tensor = torch.rot90(Z_tensor, amount_of_rotations, [1, 2])

            if amount_of_rotations == 1:
                HR[:2] = torch.concatenate(
                    (
                        -torch.index_select(HR, 0, torch.tensor(1)),
                        torch.index_select(HR, 0, torch.tensor(0)),
                    ),
                    0,
                )
                LR[:2] = torch.concatenate(
                    (
                        -torch.index_select(LR, 0, torch.tensor(1)),
                        torch.index_select(LR, 0, torch.tensor(0)),
                    ),
                    0,
                )
            if amount_of_rotations == 2:
                HR[:2] = torch.concatenate(
                    (
                        -torch.index_select(HR, 0, torch.tensor(0)),
                        -torch.index_select(HR, 0, torch.tensor(1)),
                    ),
                    0,
                )
                LR[:2] = torch.concatenate(
                    (
                        -torch.index_select(LR, 0, torch.tensor(0)),
                        -torch.index_select(LR, 0, torch.tensor(1)),
                    ),
                    0,
                )
            if amount_of_rotations == 3:
                HR[:2] = torch.concatenate(
                    (
                        torch.index_select(HR, 0, torch.tensor(1)),
                        -torch.index_select(HR, 0, torch.tensor(0)),
                    ),
                    0,
                )
                LR[:2] = torch.concatenate(
                    (
                        torch.index_select(LR, 0, torch.tensor(1)),
                        -torch.index_select(LR, 0, torch.tensor(0)),
                    ),
                    0,
                )

        if self.data_aug_flip:
            if np.random.rand() > 0.5:
                LR = torch.flip(LR, [1])
                HR = torch.flip(HR, [1])
                Z_tensor = torch.flip(Z_tensor, [1])
                LR[0] = -LR[0]
                HR[0] = -HR[0]
            if np.random.rand() > 0.5:
                LR = torch.flip(LR, [2])
                HR = torch.flip(HR, [2])
                Z_tensor = torch.flip(Z_tensor, [2])
                LR[1] = -LR[1]
                HR[1] = -HR[1]

        if self.is_test:
            return LR, HR, Z_tensor, self.case_ids[index], 0, 0

        return LR, HR, Z_tensor


# ---------------------------------------------------------------------------
# Tensor construction helpers (kept for backward compat with wind_field_GAN_3D)
# ---------------------------------------------------------------------------


def calculate_div_z(HR_data: torch.Tensor, Z: torch.Tensor):
    dZ = torch.tile(
        Z[:, :, :, :, 1:] - Z[:, :, :, :, :-1], [1, HR_data.shape[1], 1, 1, 1]
    )

    derivatives = torch.zeros_like(HR_data)

    derivatives[:, :, :, :, 1:-1] = (
        dZ[:, :, :, :, :-1] ** 2 * HR_data[:, :, :, :, 2:]
        + (dZ[:, :, :, :, 1:] ** 2 - dZ[:, :, :, :, :-1] ** 2)
        * HR_data[:, :, :, :, 1:-1]
        - dZ[:, :, :, :, 1:] ** 2 * HR_data[:, :, :, :, :-2]
    ) / (
        dZ[:, :, :, :, :-1]
        * dZ[:, :, :, :, 1:]
        * (dZ[:, :, :, :, :-1] + dZ[:, :, :, :, 1:])
    )

    derivatives[:, :, :, :, -1] = (
        HR_data[:, :, :, :, -1] - HR_data[:, :, :, :, -2]
    ) / dZ[:, :, :, :, -1]
    derivatives[:, :, :, :, 0] = (
        HR_data[:, :, :, :, 1] - HR_data[:, :, :, :, 0]
    ) / dZ[:, :, :, :, 0]

    return derivatives


@torch.jit.script
def calculate_gradient_of_wind_field(HR_data, x, y, Z):
    grad_x, grad_y = torch.gradient(HR_data, dim=(2, 3), spacing=(x, y))
    grad_z = calculate_div_z(HR_data, Z)

    return torch.cat(
        (
            grad_x,
            grad_y,
            grad_z,
        ),
        dim=1,
    )


def reformat_to_torch(
    u,
    v,
    w,
    p,
    z,
    z_above_ground,
    Z_MIN,
    Z_MAX,
    Z_ABOVE_GROUND_MAX,
    UVW_MAX,
    P_MIN,
    P_MAX,
    coarseness_factor=4,
    include_pressure=False,
    include_z_channel=False,
    include_above_ground_channel=False,
    for_plotting=False,
):
    HR_arr = (
        np.concatenate(
            (u[np.newaxis, :, :, :], v[np.newaxis, :, :, :], w[np.newaxis, :, :, :]),
            axis=0,
        )
        / UVW_MAX
    )
    del u, v, w

    if include_pressure:
        arr_norm_LR = np.concatenate(
            (HR_arr, (p[np.newaxis, :, :, :] - P_MIN) / (P_MAX - P_MIN)), axis=0
        )[:, ::coarseness_factor, ::coarseness_factor, :]
        if for_plotting:
            HR_arr = np.concatenate(
                (HR_arr, (p[np.newaxis, :, :, :] - P_MIN) / (P_MAX - P_MIN)), axis=0
            )
    else:
        arr_norm_LR = HR_arr[:, ::coarseness_factor, ::coarseness_factor, :]

    if include_z_channel:
        if include_above_ground_channel:
            arr_norm_LR = np.concatenate(
                (
                    arr_norm_LR,
                    z_above_ground[
                        np.newaxis, ::coarseness_factor, ::coarseness_factor, :
                    ]
                    / Z_ABOVE_GROUND_MAX,
                    (z - z_above_ground - Z_MIN)[
                        np.newaxis, ::coarseness_factor, ::coarseness_factor, :
                    ]
                    / (Z_MAX - Z_MIN - Z_ABOVE_GROUND_MAX),
                ),
                axis=0,
            )
            del z_above_ground
        else:
            arr_norm_LR = np.concatenate(
                (
                    arr_norm_LR,
                    (z[np.newaxis, ::coarseness_factor, ::coarseness_factor, :] - Z_MIN)
                    / (Z_MAX - Z_MIN),
                ),
                axis=0,
            )

    HR_data = torch.from_numpy(HR_arr).float()
    LR_data = torch.from_numpy(arr_norm_LR).float()
    z = torch.from_numpy(z[np.newaxis, :, :, :]).float()

    return (
        LR_data,
        HR_data,
        z,
    )


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def compute_norm_factors(zarr_paths, crop_nx, crop_ny, crop_nk, include_pressure):
    """Scan training Zarr files to derive global normalisation constants.

    UVW_MAX is fixed at REFERENCE_VELOCITY (10 m/s) because all simulations
    use the same inlet boundary condition and linear scaling applies.
    """
    ni, nj = 300, 300
    i_start = (ni - crop_nx) // 2
    i_end = i_start + crop_nx
    j_start = (nj - crop_ny) // 2
    j_end = j_start + crop_ny

    Z_MIN = np.inf
    Z_MAX = -np.inf
    Z_ABOVE_GROUND_MAX = -np.inf
    UVW_MAX = REFERENCE_VELOCITY
    P_MIN = np.inf
    P_MAX = -np.inf

    for path in zarr_paths:
        try:
            ds = xr.open_zarr(path)
            Z_arr = ds["Z"].values[i_start:i_end, j_start:j_end, :crop_nk]
            h_agl_arr = ds["h_agl"].values[i_start:i_end, j_start:j_end, :crop_nk]
            Z_MIN = min(Z_MIN, float(np.nanmin(Z_arr)))
            Z_MAX = max(Z_MAX, float(np.nanmax(Z_arr)))
            Z_ABOVE_GROUND_MAX = max(
                Z_ABOVE_GROUND_MAX, float(np.nanmax(h_agl_arr))
            )
            if include_pressure:
                p_arr = ds["p"].values[i_start:i_end, j_start:j_end, :crop_nk]
                P_MIN = min(P_MIN, float(np.nanmin(p_arr)))
                P_MAX = max(P_MAX, float(np.nanmax(p_arr)))
            ds.close()
        except Exception as exc:
            print(f"Warning: could not read {path} during norm scan: {exc}")

    if not include_pressure:
        # Unused defaults; prevent div-by-zero inside reformat_to_torch
        P_MIN, P_MAX = 0.0, 1.0

    return Z_MIN, Z_MAX, Z_ABOVE_GROUND_MAX, UVW_MAX, P_MIN, P_MAX


# ---------------------------------------------------------------------------
# Main entry point called by run.py
# ---------------------------------------------------------------------------


def preprosess(
    destination_folder,
    processed_data_folder,
    train_eval_test_ratio=0.8,
    sample_start=1,
    sample_end=100,
    crop_nx=300,
    crop_ny=300,
    crop_nk=64,
    include_pressure=False,
    include_z_channel=True,
    include_above_ground_channel=False,
    COARSENESS_FACTOR=4,
    train_aug_rot=True,
    train_aug_flip=True,
    val_aug_rot=False,
    val_aug_flip=False,
    for_plotting=False,
    isDownload=False,
    dataset="wind_terrain",
    data_source=None,
    # Legacy parameters kept for interface compatibility
    Z_DICT=None,
    start_date=None,
    end_date=None,
    interpolate_z=False,
    enable_slicing=False,
    slice_size=64,
):
    """Prepare train / validation / test datasets from the Zarr CFD dataset.

    Parameters
    ----------
    data_source : str
        Path to the dataset root directory (must contain ``data/`` and
        ``metadata/case_index.csv``).
    sample_start : int
        First sample to use (1-based row index in case_index.csv, inclusive).
    sample_end : int
        Last sample to use (1-based row index, inclusive).
    crop_nx, crop_ny : int
        Number of horizontal cells to keep after centre-cropping (must be
        divisible by ``COARSENESS_FACTOR``).
    crop_nk : int
        Number of vertical layers to keep from the ground up (k=0 is ground).
    """
    # Resolve dataset root
    dataset_root = data_source if data_source else destination_folder
    zarr_data_dir = os.path.join(dataset_root, "data")
    case_index_path = os.path.join(dataset_root, "metadata", "case_index.csv")

    # ---- load and filter case index ----------------------------------------
    idx = pd.read_csv(case_index_path)

    # 1-based sample range selection
    idx = idx.iloc[sample_start - 1 : sample_end].reset_index(drop=True)

    # Quality filters
    idx = idx[idx["converged"] & idx["mesh_mesh_ok"]].reset_index(drop=True)

    if len(idx) == 0:
        raise ValueError(
            f"No valid cases found in sample range {sample_start}–{sample_end} "
            "after quality filtering (converged=True, mesh_mesh_ok=True)."
        )

    # ---- build zarr paths --------------------------------------------------
    def case_id_to_zarr_path(case_id):
        return os.path.join(zarr_data_dir, case_id + ".zarr")

    # Extract terrain key: capture the 'terrain_XXXX_...' portion so that
    # both wind directions of the same terrain share the same key.
    def get_terrain_key(case_id):
        m = re.search(r"(terrain_.*?)_\d+deg$", case_id)
        return m.group(1) if m else case_id

    idx = idx.copy()
    idx["zarr_path"] = idx["case_id"].apply(case_id_to_zarr_path)
    idx["terrain_key"] = idx["case_id"].apply(get_terrain_key)

    # Warn about missing files but don't abort — allows partial datasets
    missing = [p for p in idx["zarr_path"] if not os.path.exists(p)]
    if missing:
        print(
            f"Warning: {len(missing)} Zarr store(s) not found on disk. "
            f"First missing: {missing[0]}"
        )
        idx = idx[idx["zarr_path"].apply(os.path.exists)].reset_index(drop=True)

    if len(idx) == 0:
        raise ValueError(
            "No Zarr files found on disk. Check dataset_root and case IDs."
        )

    # ---- terrain-grouped split to avoid leakage ----------------------------
    unique_terrains = sorted(idx["terrain_key"].unique())
    rng = np.random.default_rng(seed=2001)
    shuffled_terrains = rng.permutation(unique_terrains).tolist()

    n_terrains = len(shuffled_terrains)
    n_train = max(1, int(n_terrains * train_eval_test_ratio))
    n_remaining = n_terrains - n_train
    n_val = max(0, n_remaining // 2)

    train_terrains = set(shuffled_terrains[:n_train])
    val_terrains = set(shuffled_terrains[n_train : n_train + n_val])
    test_terrains = set(shuffled_terrains[n_train + n_val :])

    train_idx = idx[idx["terrain_key"].isin(train_terrains)]
    val_idx = idx[idx["terrain_key"].isin(val_terrains)]
    test_idx = idx[idx["terrain_key"].isin(test_terrains)]

    train_paths = train_idx["zarr_path"].tolist()
    train_ids = train_idx["case_id"].tolist()
    val_paths = val_idx["zarr_path"].tolist()
    val_ids = val_idx["case_id"].tolist()
    test_paths = test_idx["zarr_path"].tolist()
    test_ids = test_idx["case_id"].tolist()

    print(
        f"Dataset: {len(idx)} cases total  |  "
        f"train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}"
    )

    # ---- validate crop compatibility ---------------------------------------
    assert crop_nx % COARSENESS_FACTOR == 0, (
        f"crop_nx={crop_nx} must be divisible by scale factor "
        f"COARSENESS_FACTOR={COARSENESS_FACTOR}"
    )
    assert crop_ny % COARSENESS_FACTOR == 0, (
        f"crop_ny={crop_ny} must be divisible by scale factor "
        f"COARSENESS_FACTOR={COARSENESS_FACTOR}"
    )

    # ---- normalisation factors ---------------------------------------------
    os.makedirs(processed_data_folder, exist_ok=True)
    norm_cache_path = os.path.join(processed_data_folder, "norm_factors_zarr.pkl")

    if os.path.exists(norm_cache_path):
        with open(norm_cache_path, "rb") as f:
            Z_MIN, Z_MAX, Z_ABOVE_GROUND_MAX, UVW_MAX, P_MIN, P_MAX = pickle.load(f)
        print(
            f"Loaded norm factors from cache: "
            f"Z=[{Z_MIN:.1f}, {Z_MAX:.1f}], "
            f"Z_AGL_MAX={Z_ABOVE_GROUND_MAX:.1f}, "
            f"UVW_MAX={UVW_MAX:.1f}"
        )
    else:
        print("Computing normalisation factors from training cases (first run only)…")
        Z_MIN, Z_MAX, Z_ABOVE_GROUND_MAX, UVW_MAX, P_MIN, P_MAX = compute_norm_factors(
            train_paths, crop_nx, crop_ny, crop_nk, include_pressure
        )
        with open(norm_cache_path, "wb") as f:
            pickle.dump(
                [Z_MIN, Z_MAX, Z_ABOVE_GROUND_MAX, UVW_MAX, P_MIN, P_MAX], f
            )
        print(
            f"Computed norm factors: "
            f"Z=[{Z_MIN:.1f}, {Z_MAX:.1f}], "
            f"Z_AGL_MAX={Z_ABOVE_GROUND_MAX:.1f}, "
            f"UVW_MAX={UVW_MAX:.1f}"
        )

    # ---- coordinate arrays (regular ~30 m resolution grid) ----------------
    x = np.arange(crop_nx, dtype=np.float32) * 30.0
    y = np.arange(crop_ny, dtype=np.float32) * 30.0

    # ---- construct datasets ------------------------------------------------
    common_kwargs = dict(
        Z_MIN=Z_MIN,
        Z_MAX=Z_MAX,
        UVW_MAX=UVW_MAX,
        P_MIN=P_MIN,
        P_MAX=P_MAX,
        Z_ABOVE_GROUND_MAX=Z_ABOVE_GROUND_MAX,
        x=x,
        y=y,
        include_pressure=include_pressure,
        include_z_channel=include_z_channel,
        include_above_ground_channel=include_above_ground_channel,
        COARSENESS_FACTOR=COARSENESS_FACTOR,
        crop_nx=crop_nx,
        crop_ny=crop_ny,
        crop_nk=crop_nk,
    )

    dataset_train = WindTerrainDataset(
        train_paths,
        train_ids,
        data_aug_rot=train_aug_rot,
        data_aug_flip=train_aug_flip,
        for_plotting=for_plotting,
        **common_kwargs,
    )

    dataset_test = WindTerrainDataset(
        test_paths,
        test_ids,
        data_aug_rot=False,
        data_aug_flip=False,
        is_test=True,
        **common_kwargs,
    )

    dataset_validation = WindTerrainDataset(
        val_paths,
        val_ids,
        data_aug_rot=val_aug_rot,
        data_aug_flip=val_aug_flip,
        **common_kwargs,
    )

    return (
        dataset_train,
        dataset_test,
        dataset_validation,
        torch.from_numpy(x).float(),
        torch.from_numpy(y).float(),
        sample_start,
        sample_end,
    )


if __name__ == "__main__":
    preprosess()
