#!/bin/bash
# Terrain-architecture runs, single stage. Same data, split v3 and recipe as
# run1197_LR_v2_s1 (the baseline); only the generator changes:
#   run1197_LR_merge4     branch hr_merge_convs            4 post-merge convs (5,3,3,3)
#   run1197_LR_merge4_z0  branch hr_merge_convs_roughness  same + log10(z0) roughness input
# Each branch runs from its own git worktree pinned to the branch's current commit
# on origin, so switching branches in the main checkout can't affect queued jobs.
# Checkpoints/test fields go to the big project ($BIG) via runs/<name> symlinks;
# logs, tensorboard and test_output stay in the main checkout, and data/ is shared
# so the cached norm factors (data/full_dataset_files) match the baseline's.
#
#   DRY=1 scripts/slurm/submit_terrain_arch.sh     set up + check, submit nothing
#   scripts/slurm/submit_terrain_arch.sh           train (+ test after it)
#   RESUME=1 scripts/slurm/submit_terrain_arch.sh  continue from the latest checkpoint
#                                                  (e.g. after hitting the time limit)
set -euo pipefail
REPO=/projects/F202509582CPCAA2/superRes/GAN_SR_wind_field_steady
WTROOT=/projects/F202509582CPCAA2/superRes
BIG=/projects/F202511947CPCAA2/sr_runs
CD=$REPO/config/experiments/2026-11_terrain_arch
VARIANTS="run1197_LR_merge4:hr_merge_convs run1197_LR_merge4_z0:hr_merge_convs_roughness"
ACC=f202509582cpcaa2g; PART=normal-a100-80
TRAIN_TIME=1-00:00:00; TEST_TIME=02:00:00
MODULES="ml purge; unset PYTHONPATH; ml load Python/3.11.3-GCCcore-12.3.0 CUDA/11.8.0; source /projects/F202511947CPCAA2/0_venv/ml_env/bin/activate"

cd $REPO
set +u; eval "$MODULES"; set -u   # module/venv scripts use unset variables
git fetch -q origin

# ---- 1. data: every split case has HR PT files (with roughness) and LR NPZ ----
cfg_get() { grep "^$1 *=" $CD/run1197_LR_merge4.ini | sed "s/^$1 *= *//"; }
PT=$(cfg_get pt_cache_path); LR=$(cfg_get lr_data_source); SPLIT=$(cfg_get split_file)
echo "pt_cache: $PT"; echo "lr:       $LR"; echo "split:    $SPLIT"
python - "$PT" "$LR" "$SPLIT" <<'EOF'
import sys, os, torch
pt, lr, split = sys.argv[1:]
ids = [l.split(",")[0] for l in open(split) if l.startswith("case_")]
bad = []
for c in ids:
    m, f = (os.path.join(pt, "data", c, n) for n in ("maps.pt", "fields.pt"))
    if not (os.path.exists(m) and os.path.exists(f) and os.path.exists(os.path.join(lr, "data", c, "fields.npz"))):
        bad.append(c + " (missing HR PT or LR NPZ)")
    elif "roughness" not in torch.load(m, mmap=True, weights_only=True):
        bad.append(c + " (maps.pt has no roughness)")
print(f"split cases checked: {len(ids)}, problems: {len(bad)}")
for b in bad[:20]: print("  ", b)
sys.exit(1 if bad else 0)
EOF

mkdir -p log tensorboard_log runs test_output
[ -f data/full_dataset_files/norm_factors_npz.pkl ] || { echo "no cached norm factors in $REPO/data"; exit 1; }
for v in $VARIANTS; do
  NAME=${v%%:*}; BRANCH=${v#*:}
  C=$CD/$NAME.ini; T=$CD/${NAME}_test.ini
  [ -f $C ] && [ -f $T ] || { echo "missing config $C or $T"; exit 1; }

  # ---- 2. worktree pinned to the branch commit ----
  COMMIT=$(git rev-parse --short origin/$BRANCH)
  WT=$WTROOT/gan_sr_${BRANCH}_$COMMIT
  [ -d $WT ] || git worktree add -q --detach $WT $COMMIT
  for d in data log tensorboard_log runs test_output; do
    [ -e $WT/$d ] && [ ! -L $WT/$d ] && { echo "$WT/$d exists and is not a link"; exit 1; }
    ln -sfn $REPO/$d $WT/$d
  done

  # ---- 3. output folders in the big project ----
  for n in $NAME ${NAME}_test; do
    mkdir -p $BIG/$n; [ -e runs/$n ] || ln -s $BIG/$n runs/$n
  done
  if [ "${RESUME:-0}" = 1 ]; then
    C=$CD/${NAME}_resume.ini
    sed -e "s|^load_model_from_save = .*|load_model_from_save = True|" \
        -e "s|^resume_training_from_save = .*|resume_training_from_save = True|" $CD/$NAME.ini > $C
    ls $BIG/$NAME/G_*.pth >/dev/null 2>&1 || { echo "RESUME=1 but no checkpoint in $BIG/$NAME"; exit 1; }
  elif ls $BIG/$NAME/G_*.pth >/dev/null 2>&1; then
    echo "$BIG/$NAME already has checkpoints - use RESUME=1 or remove them"; exit 1
  fi

  echo "=== $NAME  ($BRANCH @ $COMMIT)"
  echo "code:   $WT"; echo "config: $C"; echo "runs:   $(readlink -f $WT/runs/$NAME)"
  grep -E "^(name|split_file|pt_cache_path|include_roughness|num_hr_merge_convs|hr_merge_kern_size|save_model_period|niter|load_model_from_save|resume_training_from_save) " $C
  echo "test G: $(grep '^generator_load_path' $T)"
  [ "${DRY:-0}" = 1 ] && continue

  # ---- 4. submit: train, then test on G_100000 if training finished ----
  ENV="$MODULES; cd $WT; git rev-parse HEAD"
  J1=$(sbatch --parsable -J $NAME -A $ACC -p $PART -t $TRAIN_TIME -N1 -n1 -c32 --gres=gpu:1 --mem=80G \
       -o $REPO/log/${NAME}_%j.out -e $REPO/log/${NAME}_%j.err --wrap="$ENV; python run.py --train --cfg $C")
  J2=$(sbatch --parsable --dependency=afterok:$J1 -J ${NAME}_test -A $ACC -p $PART -t $TEST_TIME -N1 -n1 -c16 --gres=gpu:1 --mem=80G \
       -o $REPO/log/${NAME}_test_%j.out -e $REPO/log/${NAME}_test_%j.err --wrap="$ENV; test -f $BIG/$NAME/G_100000.pth && python run.py --test --cfg $T")
  echo "submitted: train=$J1  test=$J2 (afterok)"
done
[ "${DRY:-0}" = 1 ] && echo "DRY run - nothing submitted"
squeue -u $USER
