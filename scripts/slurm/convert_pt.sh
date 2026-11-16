#!/bin/bash
#SBATCH --job-name=pt_cache_v2
#SBATCH --account=f202509582cpcaa2g
#SBATCH --partition=normal-a100-80
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=pt_cache_%j.out
#SBATCH --error=pt_cache_%j.err

# NPZ -> PT conversion only: stops after conversion (no --train/--test).
ml purge
unset PYTHONPATH
ml load Python/3.11.3-GCCcore-12.3.0 CUDA/11.8.0
source /projects/F202511947CPCAA2/0_venv/ml_env/bin/activate
cd /projects/F202509582CPCAA2/superRes/GAN_SR_wind_field_steady
python run.py --convert_to_pt --cfg /projects/F202509582CPCAA2/superRes/GAN_SR_wind_field_steady/config/default_cluster.ini
