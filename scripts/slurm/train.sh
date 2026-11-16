#!/bin/bash
#SBATCH --job-name=sr_gan_steady
#SBATCH --account=f202509582cpcaa2g
#SBATCH --partition=normal-a100-80
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

ml load Python/3.11.3-GCCcore-12.3.0 CUDA/11.8.0
source /projects/F202511947CPCAA2/0_venv/ml_env/bin/activate
cd /projects/F202509582CPCAA2/superRes/GAN_SR_wind_field_steady
python run.py --convert_to_pt --train --test --cfg /projects/F202509582CPCAA2/superRes/GAN_SR_wind_field_steady/config/default_cluster.ini
