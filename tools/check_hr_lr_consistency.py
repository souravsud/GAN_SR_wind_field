"""
Read-only consistency checks between HR and LR Wind-Terrain NPZ datasets.

Part 1 (terrain orientation), for N random cases present in the LR dataset and
each HR dataset:
  a) corr(LR dem, block-averaged HR dem), HR dem as stored vs transposed.
     Tells whether HR terrain is in the same orientation as LR terrain.
  b) corr(Uz, U . grad(dem)) near the ground, dem as stored vs transposed.
     Flow over terrain follows the slope, so this tells whether a dataset's
     terrain is aligned with its own wind field. Both axis conventions
     (x along array axis 0 or axis 1) are reported; the best one is used.
  c) corr(wind speed near the ground, dem), as stored vs transposed.
     Wind speeds up over hills.
  d) std(dem + h_agl[..., top]) as stored vs transposed. A smaller value means
     h_agl has the same orientation as dem.

Part 2 (near-ground LR vs HR mismatch), for the given cases and each HR dataset:
  LR wind is interpolated in height onto the HR layer heights (block-averaged
  over each LR cell, like the training code), then compared with the
  block-averaged HR wind. Columns are split by the LR column's
  z0 > first-cell centre height (h_agl[..., 0]). If check (b)/(c) says HR
  terrain is transposed, HR maps are transposed before use.

Nothing is written unless --out_dir is given.

Usage:
  python tools/check_hr_lr_consistency.py \
      --hr v2=/path/dataset_v2 --hr old=/path/dataset --lr /path/LR/dataset \
      [--n 20] [--seed 0] [--near_ground_cases ID ...] [--out_dir DIR]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

SMOKE_CASES = [
    "case_00104_terrain_0052_N50_018_E011_141_285deg",
    "case_01270_terrain_0634_N37_116_E097_350_326deg",
    "case_01350_terrain_0678_N42_428_E140_971_251deg",
]
K_LOW = 2  # near-ground layer used by checks b and c (about 10 m)


def corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def block(a, f):
    n0, n1 = a.shape[0] // f, a.shape[1] // f
    return a[: n0 * f, : n1 * f].reshape(n0, f, n1, f, *a.shape[2:]).mean(axis=(1, 3))


def case_dirs(root):
    return set(os.listdir(os.path.join(root, "data")))


def read_index(root):
    for p in (os.path.join(root, "case_index.csv"), os.path.join(root, "metadata", "case_index.csv")):
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def load(root, cid, fields=("Ux", "Uy", "Uz")):
    d = os.path.join(root, "data", cid)
    m = np.load(os.path.join(d, "maps.npz"), allow_pickle=False)
    f = np.load(os.path.join(d, "fields.npz"))
    out = {k: m[k].astype(np.float32) for k in ("dem", "h_agl", "roughness") if k in m.files}
    out["meta"] = json.loads(str(m["metadata"])) if "metadata" in m.files else {}
    out.update({k: f[k].astype(np.float32) for k in fields})
    return out


def slope_corr(c, dem):
    """corr(Uz, U.grad(dem)) at K_LOW for x along axis 0 and x along axis 1."""
    g0, g1 = np.gradient(dem)
    ux, uy, uz = (c[k][..., K_LOW] for k in ("Ux", "Uy", "Uz"))
    return corr(uz, ux * g0 + uy * g1), corr(uz, ux * g1 + uy * g0)


def orientation_checks(c):
    dem = c["dem"]
    st0, st1 = slope_corr(c, dem)
    tr0, tr1 = slope_corr(c, dem.T)
    speed = np.hypot(c["Ux"][..., K_LOW + 1], c["Uy"][..., K_LOW + 1])
    top = c["h_agl"][..., -1]
    return {
        "slope_stored_x0": st0, "slope_stored_x1": st1,
        "slope_T_x0": tr0, "slope_T_x1": tr1,
        "speed_dem_stored": corr(speed, dem), "speed_dem_T": corr(speed, dem.T),
        "ztop_std_stored": float(np.nanstd(dem + top)), "ztop_std_T": float(np.nanstd(dem.T + top)),
    }


def verdict(r):
    s = max(r["slope_stored_x0"], r["slope_stored_x1"]) - max(r["slope_T_x0"], r["slope_T_x1"])
    v = r["speed_dem_stored"] - r["speed_dem_T"]
    if s > 0 and v > 0:
        return "OK"
    if s < 0 and v < 0:
        return "TRANSPOSED"
    return "unclear"


def interp_columns(h_src, v_src, h_tgt):
    """Linear interpolation in height of each column; clamps outside the source column."""
    hi = np.clip((h_src[..., None, :] < h_tgt[..., None]).sum(-1), 1, h_src.shape[-1] - 1)
    lo = hi - 1
    h_lo, h_hi = np.take_along_axis(h_src, lo, -1), np.take_along_axis(h_src, hi, -1)
    w = np.clip((h_tgt - h_lo) / (h_hi - h_lo), 0, 1)
    return (1 - w) * np.take_along_axis(v_src, lo, -1) + w * np.take_along_axis(v_src, hi, -1)


def near_ground(hr, lr, transpose_hr_maps):
    f = hr["dem"].shape[0] // lr["dem"].shape[0]
    h_hr = hr["h_agl"].transpose(1, 0, 2) if transpose_hr_maps else hr["h_agl"]
    h_tgt = block(h_hr, f)
    spd_hr = block(np.hypot(hr["Ux"], hr["Uy"]), f)
    spd_lr = interp_columns(lr["h_agl"], np.hypot(lr["Ux"], lr["Uy"]), h_tgt)
    rough = lr["roughness"] > lr["h_agl"][..., 0]
    groups = {"z0 > first-cell centre": rough, "z0 <= first-cell centre": ~rough, "all columns": np.ones_like(rough)}
    med_h = np.nanmedian(h_tgt, axis=(0, 1))
    layers = sorted({int(np.argmin(np.abs(med_h - z))) for z in (0, 10, 50, 100)})
    rows = []
    for g, mask in groups.items():
        for k in layers:
            d = spd_lr[..., k][mask] - spd_hr[..., k][mask]
            rows.append({
                "group": g, "n_columns": int(mask.sum()), "hr_layer": k,
                "height_m": round(float(med_h[k]), 1),
                "HR_speed": float(np.nanmean(spd_hr[..., k][mask])) if mask.any() else np.nan,
                "LR_speed": float(np.nanmean(spd_lr[..., k][mask])) if mask.any() else np.nan,
                "bias_LR_minus_HR": float(np.nanmean(d)) if mask.any() else np.nan,
                "MAE": float(np.nanmean(np.abs(d))) if mask.any() else np.nan,
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hr", action="append", required=True, help="NAME=PATH, repeatable")
    ap.add_argument("--lr", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--near_ground_cases", nargs="*", default=SMOKE_CASES)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    hrs = dict(s.split("=", 1) for s in args.hr)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.precision", 3)

    print("== Dataset overview")
    lr_ids = case_dirs(args.lr)
    print(f"LR  {args.lr}: {len(lr_ids)} case dirs")
    hr_ids = {}
    for name, root in hrs.items():
        hr_ids[name] = case_dirs(root)
        idx = read_index(root)
        cols = list(idx.columns) if idx is not None else "no case_index.csv"
        print(f"HR {name} {root}: {len(hr_ids[name])} case dirs, {len(hr_ids[name] & lr_ids)} also in LR")
        print(f"   case_index columns: {cols}")
        if idx is not None and "converged_residualControl" in idx.columns:
            print(f"   converged_residualControl counts: {idx['converged_residualControl'].value_counts(dropna=False).to_dict()}")

    common = lr_ids.intersection(*hr_ids.values())
    pool = common if len(common) >= args.n else lr_ids & next(iter(hr_ids.values()))
    print(f"Cases in LR and all HR datasets: {len(common)}. Sampling {args.n} from "
          f"{'that set' if pool is common else 'LR ∩ first HR dataset (too few in all)'}.")
    picked = sorted(np.random.default_rng(args.seed).choice(sorted(pool), min(args.n, len(pool)), replace=False))

    print("\n== Part 1: terrain orientation")
    rows = []
    for cid in picked:
        lr = load(args.lr, cid)
        r = {"case_id": cid, "dataset": "LR", **orientation_checks(lr)}
        r["verdict"] = verdict(r)
        rows.append(r)
        for name, root in hrs.items():
            if cid not in hr_ids[name]:
                continue
            try:
                hr = load(root, cid)
            except Exception as exc:
                print(f"  {name} {cid}: could not load ({exc})")
                continue
            r = {"case_id": cid, "dataset": f"HR {name}", **orientation_checks(hr)}
            r["verdict"] = verdict(r)
            f = hr["dem"].shape[0] // lr["dem"].shape[0]
            if f * lr["dem"].shape[0] == hr["dem"].shape[0] and hr["dem"].shape[0] == hr["dem"].shape[1]:
                r["demLR_stored"] = corr(lr["dem"], block(hr["dem"], f))
                r["demLR_T"] = corr(lr["dem"], block(hr["dem"].T, f))
            r["hr_shape"] = str(hr["Ux"].shape)
            r["converged_residualControl"] = hr["meta"].get("converged_residualControl")
            rows.append(r)
    df = pd.DataFrame(rows)
    show = ["case_id", "dataset", "verdict", "demLR_stored", "demLR_T", "slope_stored_x0", "slope_stored_x1",
            "slope_T_x0", "slope_T_x1", "speed_dem_stored", "speed_dem_T", "ztop_std_stored", "ztop_std_T"]
    print(df[[c for c in show if c in df]].to_string(index=False))
    print("\nVerdict counts (OK = terrain aligned with its own wind as stored):")
    print(df.groupby("dataset")["verdict"].value_counts().unstack(fill_value=0).to_string())
    print("\nMeans per dataset:")
    print(df.groupby("dataset")[[c for c in show[3:] if c in df]].mean().to_string())
    hr_verdicts = {d: g["verdict"].mode().iat[0] for d, g in df.groupby("dataset")}

    print("\n== Part 2: near-ground LR vs HR wind speed (m/s)")
    ng = []
    for cid in args.near_ground_cases:
        if cid not in lr_ids:
            print(f"  {cid}: not in LR dataset, skipped")
            continue
        lr = load(args.lr, cid)
        for name, root in hrs.items():
            if cid not in hr_ids[name]:
                print(f"  {cid}: not in HR {name}, skipped")
                continue
            hr = load(root, cid)
            v = verdict({**orientation_checks(hr)})
            for r in near_ground(hr, lr, transpose_hr_maps=(v == "TRANSPOSED")):
                ng.append({"case_id": cid, "dataset": f"HR {name}", "hr_maps_transposed": v == "TRANSPOSED", **r})
    ng = pd.DataFrame(ng)
    if len(ng):
        print(ng.to_string(index=False))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        df.to_csv(os.path.join(args.out_dir, "terrain_orientation.csv"), index=False)
        ng.to_csv(os.path.join(args.out_dir, "near_ground.csv"), index=False)
        print(f"\nWrote CSVs to {args.out_dir}")
    print("\nMajority verdict per dataset:", hr_verdicts)


if __name__ == "__main__":
    main()
