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

dataset_list=('sim1' 'sim2' 'sim3' 'sim4' 'boston' 'credit' 'nhanes' 'house')
sample_number=1
mr=0.3
sample_size=(10000 10000 10000 10000 500 10000 10000 10000)
missing_column=(5 5 5 5 1 3 4 14)

for i in `seq 0 7`
do
    python ./sampler.py -data ${dataset_list[$i]} \
    -num_samp ${sample_number} \
    -mr $mr \
    -size ${sample_size[$i]} \
    -missc ${missing_column[$i]}

echo 'finish'