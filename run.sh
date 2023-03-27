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

degree=1
model_name="gain_qreg"
dataset="income"
mr=0.3
sample_size=10000
sample_id=$1
dlr_list=(0.0005 0.0007 0.001 0.002 0.005)
glr_list=(0.0025 0.0035 0.0005 0.001 0.002)
d_step=(2 3 4 5)
g_step=(1 2 3)

for lr_i in `seq 0 4`
do
    for ds_i in `seq 0 3`
    do
        for gs_i in `seq 0 2`
        do
            python ./main.py -id $sample_id -dataset $dataset -model $model_name -mr $mr -size $sample_size \
            -batch_size 256 \
            -alpha 20 \
            -iterations 100 \
            -dlr ${dlr_list[$lr_i]} \
            -glr ${glr_list[$lr_i]} \
            -d_gradstep ${d_step[$ds_i]} \
            -g_gradstep ${g_step[$gs_i]} \
            -log_name ${model_name}_$dataset/tuning/

            python ./calculate_estimands.py -dataset $dataset -model $model_name -num 1 -mr $mr -size $sample_size -completedir ../training_data/samples/${dataset}/complete_${mr}_${sample_size}/ -missingdir ../training_data/samples/${dataset}/MCAR_${mr}_${sample_size}/ -imputedir ../training_data/results/$dataset/MCAR_${mr}_${sample_size}/$model_name/
            python ./evaluate_estimands.py -dataset $dataset -model $model_name
            python ./show_tables.py -dataset $dataset -output ../metrics/$dataset/${model_name}_${mr}_${sample_size}_${sample_id}_${dlr_list[$lr_i]}_${glr_list[$lr_i]}_${d_step[$ds_i]}_${g_step[$gs_i]}
        done 
    done
done
echo 'finish'
