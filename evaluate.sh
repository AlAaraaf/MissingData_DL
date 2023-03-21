#!/bin/bash
# python ./calculate_estimands.py -dataset house -model gain -num 1 -mr 0.3 -size 10000 -completedir ../training_data/samples/house/complete_0.3_10000/ -missingdir ../training_data/samples/house/MCAR_0.3_10000/ -imputedir ../training_data/results/house/MCAR_0.3_10000/gain/
# python ./evaluate_estimands.py -dataset house -model gain
# python ./show_tables.py -dataset house -output ../metrics/gain_0.3_10000

# python ./calculate_estimands.py -dataset house -model cart -num 1 -mr 0.3 -size 10000 -completedir ../training_data/samples/house/complete_0.3_10000/ -missingdir ../training_data/samples/house/MCAR_0.3_10000/ -imputedir ../training_data/results/house/MCAR_0.3_10000/cart/
# python ./evaluate_estimands.py -dataset house -model cart
# python ./show_tables.py -dataset house -output ../metrics/cart_0.3_10000

python ./calculate_estimands.py -dataset house -model vaeac -num 1 -mr 0.3 -size 10000 -completedir ../training_data/samples/house/complete_0.3_10000/ -missingdir ../training_data/samples/house/MCAR_0.3_10000/ -imputedir ../training_data/results/house/MCAR_0.3_10000/vaeac/
python ./evaluate_estimands.py -dataset house -model vaeac
python ./show_tables.py -dataset house -output ../metrics/vaeac_0.3_10000

# python ./calculate_estimands.py -dataset house -model gain_qreg -num 1 -mr 0.3 -size 10000 -completedir ../training_data/samples/house/complete_0.3_10000/ -missingdir ../training_data/samples/house/MCAR_0.3_10000/ -imputedir ../training_data/results/house/MCAR_0.3_10000/gain_qreg/
# python ./evaluate_estimands.py -dataset house -model gain_qreg
# python ./show_tables.py -dataset house -output ../metrics/gain_qreg_0.3_10000
echo 'finish'
