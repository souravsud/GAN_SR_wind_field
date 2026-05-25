"""
process_data.py
NPZ-based Wind-Terrain CFD dataset loading.
Each simulation case is a directory containing maps.npz and fields.npz.
Replaces the previous Zarr/NetCDF pipeline.
"""

import json
import os
import re
import pickle

import numpy as np
import pandas as pd
import torch
import torch.utils.data

# Reference velocity used by all simulations (m/s)
REFERENCE_VELOCITY = 10.0


class WindTerrainDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for the NPZ Wind-Terrain CFD dataset.

    Each case is a directory containing maps.npz (terrain geometry) and
    fields.npz (CFD flow variables) for a steady-state RANS simulation at
    a fixed reference velocity of 10 m/s.

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
        case_dirs,
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
        enable_slicing=False,
        slice_size=64,
        terrain=None,
        z_skip=0,
    ):
        self.case_dirs = case_dirs
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
        self.z_skip = z_skip
        self.data_aug_rot = data_aug_rot
        self.data_aug_flip = data_aug_flip
        self.for_plotting = for_plotting
        self.is_test = is_test
        self.enable_slicing = enable_slicing
        self.slice_size = slice_size

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, index):
        # ---- crop / slice indices ------------------------------------------
        # Full domain: ni=300, nj=300, nk=64.  Keep the lowest crop_nk layers
        # (k=0 is ground).
        # Without slicing: fixed centre crop to (crop_nx, crop_ny).
        # With slicing: random (slice_size, slice_size) window drawn from
        #   Beta(0.25, 0.25) so corners and centre are sampled more often.
        ni, nj = 300, 300
        if self.enable_slicing:
            i_start = round(np.random.beta(0.25, 0.25) * (ni - self.slice_size))
            j_start = round(np.random.beta(0.25, 0.25) * (nj - self.slice_size))
            i_end = i_start + self.slice_size
            j_end = j_start + self.slice_size
        else:
            i_start = (ni - self.crop_nx) // 2
            i_end = i_start + self.crop_nx
            j_start = (nj - self.crop_ny) // 2
            j_end = j_start + self.crop_ny

        # ---- load from NPZ -------------------------------------------------
        case_dir = self.case_dirs[index]
        maps = np.load(os.path.join(case_dir, "maps.npz"), allow_pickle=True)
        fields = np.load(os.path.join(case_dir, "fields.npz"))

        # z_slice: take the lowest crop_nk layers then keep every (z_skip+1)-th.
        z_slice = slice(None, self.crop_nk, self.z_skip + 1)

        dem = maps["dem"].astype(np.float32)[i_start:i_end, j_start:j_end]
        h_agl = maps["h_agl"].astype(np.float32)[i_start:i_end, j_start:j_end, z_slice]
        Ux = fields["Ux"].astype(np.float32)[i_start:i_end, j_start:j_end, z_slice]
        Uy = fields["Uy"].astype(np.float32)[i_start:i_end, j_start:j_end, z_slice]
        Uz = fields["Uz"].astype(np.float32)[i_start:i_end, j_start:j_end, z_slice]
        # Reconstruct absolute elevation from ground DEM + height-above-ground
        Z_abs = dem[:, :, np.newaxis] + h_agl
        # Pressure is not stored in the NPZ format
        p = None

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
            z_agl_filled = np.where(np.isnan(z_above_ground), 0.0, z_above_ground)
            z_agl_norm = (
                z_agl_filled[np.newaxis, ::coarseness_factor, ::coarseness_factor, :]
                / Z_ABOVE_GROUND_MAX
            )
            z_abs_filled = np.where(np.isnan(z - z_above_ground), Z_MIN, z - z_above_ground)
            z_abs_norm = (
                (z_abs_filled - Z_MIN)[
                    np.newaxis, ::coarseness_factor, ::coarseness_factor, :
                ]
                / (Z_MAX - Z_MIN - Z_ABOVE_GROUND_MAX)
            )
            arr_norm_LR = np.concatenate(
                (arr_norm_LR, z_agl_norm, z_abs_norm),
                axis=0,
            )
            del z_above_ground, z_agl_filled, z_agl_norm, z_abs_filled, z_abs_norm
        else:
            z_filled = np.where(np.isnan(z), Z_MIN, z)
            z_norm = (
                (z_filled[np.newaxis, ::coarseness_factor, ::coarseness_factor, :] - Z_MIN)
                / (Z_MAX - Z_MIN)
            )
            arr_norm_LR = np.concatenate(
                (arr_norm_LR, z_norm),
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


def compute_norm_factors(case_dirs, crop_nx, crop_ny, crop_nk, include_pressure):
    """Scan training NPZ case directories to derive global normalisation constants.

    UVW_MAX is fixed at REFERENCE_VELOCITY (10 m/s) because all simulations
    use the same inlet boundary condition and linear scaling applies.
    Pressure is not stored in the NPZ format; P_MIN/P_MAX are unused stubs.
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
    P_MIN = 0.0
    P_MAX = 1.0

    for case_dir in case_dirs:
        try:
            maps = np.load(os.path.join(case_dir, "maps.npz"), allow_pickle=True)
            dem = maps["dem"].astype(np.float32)[i_start:i_end, j_start:j_end]
            h_agl_arr = maps["h_agl"].astype(np.float32)[
                i_start:i_end, j_start:j_end, :crop_nk
            ]
            Z_arr = dem[:, :, np.newaxis] + h_agl_arr
            Z_MIN = min(Z_MIN, float(np.nanmin(Z_arr)))
            Z_MAX = max(Z_MAX, float(np.nanmax(Z_arr)))
            Z_ABOVE_GROUND_MAX = max(
                Z_ABOVE_GROUND_MAX, float(np.nanmax(h_agl_arr))
            )
        except Exception as exc:
            print(f"Warning: could not read {case_dir} during norm scan: {exc}")

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
    z_skip=0,
    enable_slicing=False,
    slice_size=64,
    # Legacy parameters kept for interface compatibility with the old NetCDF pipeline
    Z_DICT=None,
    start_date=None,
    end_date=None,
    interpolate_z=False,
):
    """Prepare train / validation / test datasets from the NPZ CFD dataset.

    Parameters
    ----------
    data_source : str
        Path to the dataset root directory (must contain ``data/`` and
        ``case_index.csv``).
    sample_start : int
        First sample to use (1-based row index in case_index.csv, inclusive).
    sample_end : int
        Last sample to use (1-based row index, inclusive).
    crop_nx, crop_ny : int
        Number of horizontal cells to keep after centre-cropping (must be
        divisible by ``COARSENESS_FACTOR``). Ignored when ``enable_slicing``
        is True (the test dataset still centre-crops to this size).
    crop_nk : int
        Number of vertical layers to keep from the ground up (k=0 is ground).
    enable_slicing : bool
        If True, train and validation datasets return a randomly positioned
        ``slice_size × slice_size`` horizontal window instead of a fixed
        centre crop. The window is drawn from Beta(0.25, 0.25) so corners
        and centre are sampled more often. The test dataset always uses a
        fixed centre crop regardless of this flag.
    slice_size : int
        Side length of the spatial window when ``enable_slicing`` is True.
        Must be divisible by ``COARSENESS_FACTOR``.
    """
    # Resolve dataset root
    dataset_root = data_source if data_source else destination_folder
    data_dir = os.path.join(dataset_root, "data")

    # case_index.csv lives at the dataset root
    case_index_path = os.path.join(dataset_root, "case_index.csv")
    if not os.path.exists(case_index_path):
        # Fallback for datasets that still have metadata/ sub-directory
        fallback = os.path.join(dataset_root, "metadata", "case_index.csv")
        if os.path.exists(fallback):
            case_index_path = fallback
        else:
            raise FileNotFoundError(
                f"case_index.csv not found at {case_index_path} or {fallback}"
            )

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

    # ---- build case directory paths ----------------------------------------
    def case_id_to_dir(case_id):
        return os.path.join(data_dir, case_id)

    # Extract terrain key: capture the 'terrain_XXXX_...' portion so that
    # both wind directions of the same terrain share the same key.
    def get_terrain_key(case_id):
        m = re.search(r"(terrain_.*?)_\d+deg$", case_id)
        return m.group(1) if m else case_id

    idx = idx.copy()
    idx["case_dir"] = idx["case_id"].apply(case_id_to_dir)
    idx["terrain_key"] = idx["case_id"].apply(get_terrain_key)

    # Warn about missing files but don't abort — allows partial datasets
    missing = [
        d for d in idx["case_dir"]
        if not os.path.exists(os.path.join(d, "fields.npz"))
    ]
    if missing:
        print(
            f"Warning: {len(missing)} case(s) missing fields.npz on disk. "
            f"First missing: {missing[0]}"
        )
        idx = idx[
            idx["case_dir"].apply(
                lambda d: os.path.exists(os.path.join(d, "fields.npz"))
            )
        ].reset_index(drop=True)

    if len(idx) == 0:
        raise ValueError(
            "No NPZ case directories found on disk. Check dataset_root and case IDs."
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

    train_dirs = train_idx["case_dir"].tolist()
    train_ids = train_idx["case_id"].tolist()
    val_dirs = val_idx["case_dir"].tolist()
    val_ids = val_idx["case_id"].tolist()
    test_dirs = test_idx["case_dir"].tolist()
    test_ids = test_idx["case_id"].tolist()

    print(
        f"Dataset: {len(idx)} cases total  |  "
        f"train={len(train_dirs)}, val={len(val_dirs)}, test={len(test_dirs)}"
    )

    # ---- validate crop / slice compatibility --------------------------------
    assert crop_nx % COARSENESS_FACTOR == 0, (
        f"crop_nx={crop_nx} must be divisible by scale factor "
        f"COARSENESS_FACTOR={COARSENESS_FACTOR}"
    )
    assert crop_ny % COARSENESS_FACTOR == 0, (
        f"crop_ny={crop_ny} must be divisible by scale factor "
        f"COARSENESS_FACTOR={COARSENESS_FACTOR}"
    )
    if enable_slicing:
        assert slice_size % COARSENESS_FACTOR == 0, (
            f"slice_size={slice_size} must be divisible by scale factor "
            f"COARSENESS_FACTOR={COARSENESS_FACTOR}"
        )

    # ---- normalisation factors ---------------------------------------------
    os.makedirs(processed_data_folder, exist_ok=True)
    norm_cache_path = os.path.join(processed_data_folder, "norm_factors_npz.pkl")

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
            train_dirs, crop_nx, crop_ny, crop_nk, include_pressure
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
    # When slicing is enabled, train/val output slice_size × slice_size patches,
    # so the returned x/y used for gradient computation must have that length.
    # The test dataset always centre-crops to crop_nx × crop_ny, but x/y are
    # only used during training, so this is consistent.
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
        z_skip=z_skip,
    )

    dataset_train = WindTerrainDataset(
        train_dirs,
        train_ids,
        data_aug_rot=train_aug_rot,
        data_aug_flip=train_aug_flip,
        for_plotting=for_plotting,
        enable_slicing=enable_slicing,
        slice_size=slice_size,
        **common_kwargs,
    )

    dataset_test = WindTerrainDataset(
        test_dirs,
        test_ids,
        data_aug_rot=False,
        data_aug_flip=False,
        is_test=True,
        **common_kwargs,
    )

    dataset_validation = WindTerrainDataset(
        val_dirs,
        val_ids,
        data_aug_rot=val_aug_rot,
        data_aug_flip=val_aug_flip,
        enable_slicing=enable_slicing,
        slice_size=slice_size,
        **common_kwargs,
    )

    x_out = x[:slice_size] if enable_slicing else x
    y_out = y[:slice_size] if enable_slicing else y

    return (
        dataset_train,
        dataset_test,
        dataset_validation,
        torch.from_numpy(x_out).float(),
        torch.from_numpy(y_out).float(),
        sample_start,
        sample_end,
    )


if __name__ == "__main__":
    preprosess()
