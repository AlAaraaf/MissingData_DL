#!/bin/bash
model='cart'
dataset='income'
sample_id=0
mr=0.3
sample_size=20000

python ./calculate_estimands.py -dataset $dataset -model $model -num 1 -mr $mr -size $sample_size -completedir ../training_data/samples/$dataset/complete_${mr}_${sample_size}/ -missingdir ../training_data/samples/$dataset/MCAR_${mr}_${sample_size}/ -imputedir ../training_data/results/$dataset/MCAR_${mr}_${sample_size}/$model/
python ./evaluate_estimands.py -dataset $dataset -model $model
python ./show_tables.py -dataset $dataset -output ../metrics/$dataset/${model}_${mr}_${sample_size}

echo 'finish'
