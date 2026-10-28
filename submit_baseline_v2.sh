#!/bin/bash
# Baseline v2: corrected dataset (280 HR cases rerun), same 2-stage recipe, same split.
set -euo pipefail
REPO=/projects/F202509582CPCAA2/superRes/GAN_SR_wind_field_steady
D=$(readlink -f /home/souravsudhakaran/dataset/DATASET)
HR=$D/HR_30m; LR=$D/LR_120m; PT=$HR/pt_cache
BIG=$(readlink -f /home/souravsudhakaran/dataset)/sr_runs
BASE=$REPO/config/best_config_wold_et_al.ini
SPLIT0=$REPO/splits/split_fixed_v3.csv
TAG=run1197_LR_v2; S1=${TAG}_s1; S2=${TAG}_s2
CD=$REPO/config/experiments/2026-10_v2; mkdir -p $CD
ACC=f202509582cpcaa2g; PART=normal-a100-80
ENV="ml purge; unset PYTHONPATH; ml load Python/3.11.3-GCCcore-12.3.0 CUDA/11.8.0; source /projects/F202511947CPCAA2/0_venv/ml_env/bin/activate; cd $REPO; git rev-parse HEAD"
cd $REPO

# ---- 1. every split case must exist in HR and LR ----
missing=$(awk -F, '/^case_[0-9]/{print $1}' $SPLIT0 | while read c; do
  [ -f $HR/data/$c/fields.npz ] && [ -f $LR/data/$c/fields.npz ] || echo $c; done)
if [ -n "$missing" ]; then
  echo "MISSING in HR or LR (excluded from split):"; echo "$missing" | while read c; do grep "^$c," $SPLIT0; done
  SPLIT=$REPO/splits/split_v2_1-1200_v2avail.csv
  grep -v -F -f <(echo "$missing") $SPLIT0 > $SPLIT
else SPLIT=$SPLIT0; fi
echo "split: $SPLIT"; awk -F, '/^case_[0-9]/{print $NF}' $SPLIT | sort | uniq -c

# ---- 2. configs ----
C1=$CD/$S1.ini; C2=$CD/$S2.ini; T1=$CD/${S1}_test.ini; T2=$CD/${S2}_test.ini
sed -e "0,/^name = .*/s||name = $S1|" \
    -e "s|^data_source.*|data_source = $HR/|" -e "s|^lr_data_source.*|lr_data_source = $LR|" \
    -e "s|^pt_cache_path.*|pt_cache_path = $PT|" -e "s|^split_file.*|split_file = $SPLIT|" \
    -e "s|^pixel_loss_weight = .*|pixel_loss_weight = 0.034|" -e "s|^niter = .*|niter = 100000|" \
    -e "s|^save_model_period = .*|save_model_period = 10000|" \
    -e "s|^load_model_from_save = .*|load_model_from_save = False|" \
    -e "s|^resume_training_from_save = .*|resume_training_from_save = False|" \
    -e "s|^generator_load_path.*|generator_load_path|" -e "s|^discriminator_load_path.*|discriminator_load_path|" \
    -e "s|^state_load_path.*|state_load_path|" \
    -e "0,/^num_workers = .*/s||num_workers = 16|" $BASE > $C1
sed -e "0,/^name = .*/s||name = $S2|" -e "s|^pixel_loss_weight = .*|pixel_loss_weight = 0.136|" \
    -e "s|^load_model_from_save = .*|load_model_from_save = True|" \
    -e "s|^generator_load_path.*|generator_load_path = $REPO/runs/$S1/G_100000.pth|" \
    -e "s|^discriminator_load_path.*|discriminator_load_path = none|" $C1 > $C2
sed -e "0,/^name = .*/s||name = ${S1}_test|" -e "s|^load_model_from_save = .*|load_model_from_save = True|" \
    -e "s|^generator_load_path.*|generator_load_path = $REPO/runs/$S1/G_100000.pth|" \
    -e "s|^discriminator_load_path.*|discriminator_load_path = none|" $C1 > $T1
sed -e "0,/^name = .*/s||name = ${S2}_test|" \
    -e "s|^generator_load_path.*|generator_load_path = $REPO/runs/$S2/G_100000.pth|" $C2 > $T2
for c in $C1 $C2 $T1 $T2; do echo "-- $(basename $c)"; grep -E "^name = run|^data_source|^lr_data_source|^pt_cache_path|^split_file|^pixel_loss_weight|^niter|^save_model_period|^load_model|^resume|^generator_load_path|^discriminator_load_path|^state_load_path|^use_lr_dataset|^num_workers = 16" $c; done

# ---- 3. output folders in the big project (links in runs/) ----
for n in $S1 $S2 ${S1}_test ${S2}_test; do
  mkdir -p $BIG/$n; [ -e runs/$n ] || ln -s $BIG/$n runs/$n; done
ls -l runs/ | grep $TAG

[ "${DRY:-0}" = 1 ] && { echo "DRY run - nothing submitted"; exit 0; }

# ---- 4. chained jobs ----
J0=$(sbatch --parsable -J ${TAG}_pt  -A $ACC -p $PART -t 03:00:00  -N1 -n1 -c16 --gres=gpu:1 --mem=80G -o ${TAG}_pt_%j.out  -e ${TAG}_pt_%j.err  --wrap="$ENV; python run.py --convert_to_pt --cfg $C1")
J1=$(sbatch --parsable --dependency=afterok:$J0 -J ${S1} -A $ACC -p $PART -t 1-00:00:00 -N1 -n1 -c32 --gres=gpu:1 --mem=80G -o ${S1}_%j.out -e ${S1}_%j.err --wrap="$ENV; python run.py --train --cfg $C1")
J1t=$(sbatch --parsable --dependency=afterok:$J1 -J ${S1}_test -A $ACC -p $PART -t 02:00:00 -N1 -n1 -c16 --gres=gpu:1 --mem=80G -o ${S1}_test_%j.out -e ${S1}_test_%j.err --wrap="$ENV; python run.py --test --cfg $T1")
J2=$(sbatch --parsable --dependency=afterok:$J1 -J ${S2} -A $ACC -p $PART -t 1-00:00:00 -N1 -n1 -c32 --gres=gpu:1 --mem=80G -o ${S2}_%j.out -e ${S2}_%j.err --wrap="$ENV; test -f $REPO/runs/$S1/G_100000.pth && python run.py --train --cfg $C2")
J2t=$(sbatch --parsable --dependency=afterok:$J2 -J ${S2}_test -A $ACC -p $PART -t 02:00:00 -N1 -n1 -c16 --gres=gpu:1 --mem=80G -o ${S2}_test_%j.out -e ${S2}_test_%j.err --wrap="$ENV; python run.py --test --cfg $T2")
echo "PT=$J0  S1=$J1  S1test=$J1t  S2=$J2  S2test=$J2t"; squeue -u $USER
