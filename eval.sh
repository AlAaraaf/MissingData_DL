#!/bin/bash

#SBATCH --time=1-0:00:00  # max job runtime
#SBATCH --cpus-per-task=1  # number of processor cores
#SBATCH --nodes=1  # number of nodes
#SBATCH --partition=gpu  # partition(s)
#SBATCH --gres=gpu:1
#SBATCH --mem=5G  # max memory
#SBATCH -J "mdi-gq-t"  # job name
#SBATCH --mail-user=sjx@iastate.edu  # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#module load python/3.10.8-5qsesua
#source /work/LAS/zhanruic-lab/jiaxin/vaeac/jiaxin/bin/activate

# bash for evaluating other models than GAN
# the prefix is better to set corresponding to model's hyperparameters choice
# for example, try mrun.sh for the GAN output result

model_name="gain"
dataset="boston"
mr=0.3
sample_size=500
sample_id=0
prefix="prefix_for_model_metric"

python ./eval_main.py -id $sample_id -dataset $dataset -model $model_name -mr $mr -size $sample_size \
    -onlylog 1 \
    -prefix ${prefix}

echo 'finish'
