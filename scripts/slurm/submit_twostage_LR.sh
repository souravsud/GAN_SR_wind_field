#!/bin/bash
set -euo pipefail
REPO=/projects/F202509582CPCAA2/superRes/GAN_SR_wind_field_steady
BASE=$REPO/config/default_cluster.ini
TAG=run1000_LR_2stage_v1
S1=${TAG}_s1; S2=${TAG}_s2
SPLIT=$(grep "^split_file" $REPO/config/default_cluster.ini | sed "s/^split_file *= *//")
C1=$REPO/config/experiments/2026-10_v1_run1000/${S1}.ini; C2=$REPO/config/experiments/2026-10_v1_run1000/${S2}.ini; CT=$REPO/config/experiments/2026-10_v1_run1000/${S2}_test.ini
RUNS=$REPO/runs
ACC=f202509582cpcaa2g; PART=normal-a100-80
ENV="ml purge; unset PYTHONPATH; ml load Python/3.11.3-GCCcore-12.3.0 CUDA/11.8.0; source /projects/F202511947CPCAA2/0_venv/ml_env/bin/activate; cd $REPO; git rev-parse HEAD; git status --short"

cd $REPO
[ -f "$BASE" ] || { echo "missing $BASE"; exit 1; }

# ---- shared split (once) ----
sed "s|^split_file =.*|split_file = $SPLIT|" $BASE > $C1
echo "using split: $SPLIT"
[ -f "$SPLIT" ] || { echo "split not created"; exit 1; }
echo "split lines: $(wc -l < $SPLIT)"

# ---- stage 1: pixel 0.034, scratch, 100k ----
sed -i -e "0,/^name = .*/s||name = $S1|" \
       -e "s|^pixel_loss_weight = .*|pixel_loss_weight = 0.034|" \
       -e "s|^niter = .*|niter = 100000|" \
       -e "s|^save_model_period = .*|save_model_period = 5000|" \
       -e "s|^load_model_from_save = .*|load_model_from_save = False|" \
       -e "s|^resume_training_from_save = .*|resume_training_from_save = False|" $C1

# ---- stage 2: warm restart from S1/G_100000, pixel 0.136, fresh schedule, 100k ----
sed -e "0,/^name = .*/s||name = $S2|" \
    -e "s|^pixel_loss_weight = .*|pixel_loss_weight = 0.136|" \
    -e "s|^load_model_from_save = .*|load_model_from_save = True|" \
    -e "s|^generator_load_path.*|generator_load_path = $RUNS/$S1/G_100000.pth|" \
    -e "s|^discriminator_load_path.*|discriminator_load_path = none|" \
    -e "s|^state_load_path.*|state_load_path|" \
    -e "s|^resume_training_from_save = .*|resume_training_from_save = False|" $C1 > $C2

# ---- test: FINAL stage-2 checkpoint (not the loaded one) ----
sed -e "s|^generator_load_path.*|generator_load_path = $RUNS/$S2/G_100000.pth|" $C2 > $CT

echo "=== key settings ==="
for c in $C1 $C2 $CT; do echo "-- $(basename $c)"; grep -E "^name = run|^pixel_loss_weight|^niter|^save_model_period|^load_model_from_save|^resume_training|^generator_load_path|^discriminator_load_path|^state_load_path|^split_file|^use_lr_dataset" $c; done

J1=$(sbatch --parsable -J ${S1} -A $ACC -p $PART -t 1-06:00:00 -N1 -n1 -c8 --gres=gpu:1 --mem=80G \
     -o ${S1}_%j.out -e ${S1}_%j.err --wrap="$ENV; python run.py --train --cfg $C1")
J2=$(sbatch --parsable --dependency=afterok:$J1 -J ${S2} -A $ACC -p $PART -t 1-06:00:00 -N1 -n1 -c8 --gres=gpu:1 --mem=80G \
     -o ${S2}_%j.out -e ${S2}_%j.err --wrap="$ENV; test -f $RUNS/$S1/G_100000.pth && python run.py --train --cfg $C2")
J3=$(sbatch --parsable --dependency=afterok:$J2 -J ${S2}_test -A $ACC -p $PART -t 02:00:00 -N1 -n1 -c8 --gres=gpu:1 --mem=80G \
     -o ${S2}_test_%j.out -e ${S2}_test_%j.err --wrap="$ENV; python run.py --test --cfg $CT")
echo "submitted: S1=$J1  S2=$J2 (afterok)  TEST=$J3 (afterok)"
