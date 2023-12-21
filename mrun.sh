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

model_name="gain"
dataset_list=("sim1" "sim2" "sim3" "sim4" "boston" "credit" "nhanes" "house")
mr=0.3
sample_size=(10000 10000 10000 10000 500 10000 10000 10000)
sample_id=0
dlr_list=(0.0005 0.0007 0.001 0.005 0.01 0.02)
glr_list=(0.0003 0.0005 0.0007 0.003 0.007 0.01)
d_step=(4 4 4 3 2 2)
g_step=(3 3 2 2 1 1)
alpha=(20 15 10)
batch_size=512
iteration=50

for dataset_i in `seq 0 7`
do
    for alpha_i in `seq 0 2`
    do
        for lr_i in `seq 0 5`
        do
            python ./main.py -id $sample_id \
            -dataset ${dataset_list[$dataset_i]} \
            -model $model_name \
            -mr $mr \
            -size ${sample_size[dataset_i]} \
            -batch_size $batch_size \
            -alpha ${alpha[$alpha_i]} \
            -iterations $iteration \
            -dlr ${dlr_list[$lr_i]} \
            -glr ${glr_list[$lr_i]} \
            -d_gradstep ${d_step[$lr_i]} \
            -g_gradstep ${g_step[$lr_i]} \
            -log_name ${model_name}_$dataset/tuning/

            python ./eval_main.py -id $sample_id \
            -dataset ${dataset_list[$dataset_i]} \
            -model $model_name \
            -mr $mr \
            -size ${sample_size[dataset_i]} \
            -batch_size $batch_size \
            -alpha ${alpha[$alpha_i]} \
            -iterations $iteration \
            -dlr ${dlr_list[$lr_i]} \
            -glr ${glr_list[$lr_i]} \
            -d_gradstep ${d_step[$lr_i]} \
            -g_gradstep ${g_step[$lr_i]}
        done
    done
done
echo 'finish'
