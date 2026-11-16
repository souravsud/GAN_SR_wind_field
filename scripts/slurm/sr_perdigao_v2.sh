#!/bin/bash
#SBATCH --job-name=sr_perdigao_v2
#SBATCH --account=f202509582cpcaa2g
#SBATCH --partition=dev-a100-40
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=log/perdigao_v2_%j.out
#SBATCH --error=log/perdigao_v2_%j.err
ml load Python/3.11.3-GCCcore-12.3.0 CUDA/11.8.0
source /projects/F202511947CPCAA2/0_venv/ml_env/bin/activate
cd /projects/F202509582CPCAA2/superRes/GAN_SR_wind_field_steady
for v in B2sig P_log; do echo "=== $v ==="; python run.py --test --cfg /projects/F202509582CPCAA2/superRes/GAN_SR_wind_field_steady/config/experiments/2026-11_perdigao/perdigao_v2s2_100k_${v}_cluster.ini; done
