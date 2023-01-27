#!/bin/bash
python ./calculate_estimands.py -dataset boston -completedir ../vaeac/samples/boston/complete/ -missingdir ../vaeac/samples/boston/MCAR/ -imputedir ../vaeac/results/boston/
python ./evaluate_estimands.py -dataset boston
python ./show_tables.py -dataset boston -output vaeac_boston_mr_size
echo 'finish'
