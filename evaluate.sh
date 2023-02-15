#!/bin/bash
python ./calculate_estimands.py -dataset sim_1_tiny -model gain -num 10 -completedir ./samples/sim_1_tiny/complete_0.3_5000/ -missingdir ./samples/sim_1_tiny/MCAR_0.3_5000/ -imputedir ./results/sim_1_tiny/MCAR_0.3_5000/gain/
python ./evaluate_estimands.py -dataset sim_1_tiny -model gain
python ./show_tables.py -dataset sim_1_tiny -output gain_sim_1_tiny_0.3_5000
echo 'finish'
