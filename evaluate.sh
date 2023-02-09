#!/bin/bash
python ./calculate_estimands.py -dataset house -num 10 -completedir ./samples/house/complete_0.3_10000/ -missingdir ./samples/house/MCAR_0.3_10000/ -imputedir ./results/house/MCAR_0.3_10000/cart/
python ./evaluate_estimands.py -dataset hosue
python ./show_tables.py -dataset house -output cart_house_0.3_10000
echo 'finish'
