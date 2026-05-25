# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 3D GAN for super-resolving wind fields from HARMONIE-SIMRA / Wind-Terrain CFD data. The model takes low-resolution (LR) wind fields and upscales them by a configurable factor (default 4×) in the horizontal plane, producing high-resolution (HR) outputs. Inspired by ESRGAN, adapted to 3D volumetric wind data.

## Commands

```bash
# Full pipeline: download/process data then train
python run.py --download --train --cfg config/wind_field_GAN_3D_config_local.ini

# Train only (data already processed)
python run.py --train --cfg config/wind_field_GAN_3D_config_local.ini

# Test a trained model
python run.py --test --cfg config/wind_field_GAN_3D_config_local.ini

# CPU smoke-test (no GPU needed, tiny model, 20 iterations)
python run.py --train --test --cfg config/wind_field_GAN_3D_config_local_cpu_test.ini

# Hyperparameter search (uses Ray Tune + Optuna)
python run.py --param_search --cfg config/wind_field_GAN_3D_config_local.ini
```

Flags can be combined: `--train --test` trains then immediately tests.

## Configuration System

All hyperparameters are set in `.ini` files under `config/`. The default config is `config/wind_field_GAN_3D_config_local.ini`. Pass a different config with `--cfg`.

Key config sections and their roles:
- `[DEFAULT]` — name, model, scale factor, GPU id, `load_model_from_save`
- `[ENV]` — file paths (`data_path`, `data_source`, `runs_subpath`), checkpoint load paths
- `[GAN]` — dataset shape: `crop_nx`, `crop_ny`, `number_of_z_layers`, `sample_start`/`sample_end`, `include_z_channel`, `include_pressure`
- `[GENERATOR]` / `[DISCRIMINATOR]` — architecture: `num_features`, `num_RRDB`, `dropout_probability`, `use_mixed_precision`
- `[TRAINING]` — loss weights, learning rates, `niter`, `val_period`, `save_model_period`

**Critical**: `crop_nx`, `crop_ny`, and `number_of_z_layers` must be consistent between `[GAN]`, `[GENERATOR]`, and `[DISCRIMINATOR]` sections — they determine tensor shapes and the discriminator's `flat_size`. If resuming training, set `load_model_from_save = True` and `resume_training_from_save = True`; checkpoint paths are auto-discovered from the runs folder.

## Dataset Format

The pipeline uses NPZ-formatted Wind-Terrain CFD data:
- Dataset root must contain `case_index.csv` and a `data/` subdirectory
- Each case is a directory with `maps.npz` (terrain: `dem`, `h_agl`) and `fields.npz` (CFD: `Ux`, `Uy`, `Uz`)
- `case_index.csv` must have columns: `case_id`, `converged`, `mesh_mesh_ok`; only cases where both are `True` are used
- `data_source` in `[ENV]` must point to this root directory
- `sample_start`/`sample_end` are 1-based row indices into `case_index.csv`

Domain: 300×300 horizontal cells, 64 vertical layers. `crop_nx`/`crop_ny` centre-crop the horizontal domain; `number_of_z_layers` keeps the lowest k layers from the ground (k=0).

## Architecture

### Data flow
`run.py` → `process_data.preprosess()` → `WindTerrainDataset` → `train()` / `test()`

Each dataset `__getitem__` returns `(LR, HR, Z)`:
- `HR`: normalised wind field `(out_ch, crop_nx, crop_ny, nk)` — channels are `[Ux, Uy, Uz, ...]`
- `LR`: coarsened `(in_ch, crop_nx//scale, crop_ny//scale, nk)` — optionally includes normalised elevation (`include_z_channel`) or pressure (`include_pressure`)
- `Z`: absolute elevation `(1, crop_nx, crop_ny, nk)` for gradient loss computation

Normalisation: `UVW_MAX` is fixed at 10.0 m/s (REFERENCE_VELOCITY); `Z_MIN`/`Z_MAX` are scanned from training cases.

### GAN model (`GAN_models/wind_field_GAN_3D.py`)
- Inherits `BaseGAN` (`GAN_models/baseGAN.py`) which holds `G`, `D`, and checkpoint save/load logic
- Generator: `Generator_3D` — RRDB-based ESRGAN-style network, upsamples horizontally by `scale` factor, leaves vertical dimension unchanged
- Discriminator: `Discriminator_3D` — VGG-style 3D discriminator; `flat_size` is computed at construction from `crop_nx`, `crop_ny`, and `number_of_z_layers`

### Loss terms (weighted in `[TRAINING]`)
- Pixel loss (`l1`)
- Adversarial loss (relativistic avg GAN)
- XY gradient loss
- Z gradient loss
- Divergence loss
- XY divergence loss
- Feature discriminator loss (disabled by default, `use_D_feature_extractor_cost = False`)

### Output locations
- Checkpoints: `runs/<name>/G_<iter>.pth`, `D_<iter>.pth`, `state_<iter>.pth`
- Validation images (pickled): `runs/<name>/images/val_imgs__it_<iter>.pkl`
- Test plots: `runs/<name>/plots/`
- Test metrics CSV: `test_output/<name>____metrics.csv`, `test_output/averages.csv`
- TensorBoard logs: `tensorboard_log/<name>/`
- Status logs: `log/<name>.log`

## Coding Guidelines

### Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### Simplicity First
Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

Every changed line should trace directly to the user's request.

### Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Fix the bug" → "Reproduce it, then make it not occur"
- "Refactor X" → "Ensure the pipeline runs identically before and after"

For multi-step tasks, state a brief plan with a concrete check for each step.

## Key Files

| File | Purpose |
|------|---------|
| `run.py` | Entry point; parses args, sets up env, calls train/test/param_search |
| `config/config.py` | `Config` dataclass hierarchy; reads `.ini` files via `ConfigParser` |
| `process_data.py` | `WindTerrainDataset`, `preprosess()`, `reformat_to_torch()`, normalisation |
| `train.py` | Training loop with TensorBoard logging and periodic validation |
| `test.py` | Inference loop; saves metrics CSVs and plots |
| `GAN_models/wind_field_GAN_3D.py` | Loss computation, optimizer steps, metric tracking |
| `CNN_models/Generator_3D_Resnet_ESRGAN.py` | Generator architecture |
| `CNN_models/Discriminator_3D.py` | Discriminator architecture |
| `CNN_models/torch_blocks.py` | Shared building blocks (RRDB, conv layers, upsampling) |
| `tools/test_plotting.py` | `save_test_plots()` — PNG comparison/error figures |
| `param_search.py` | Ray Tune + Optuna hyperparameter search |
