#!/bin/bash
degree=1
for i in `seq 0 0`
do
    python ./main.py -id $i -dataset house -model gain_qreg -mr 0.3 -size 10000 \
    -batch_size 256 \
    -alpha 20 \
    -iterations 10 \
    -dlr 0.0005 \
    -glr 0.0025 \
    -d_gradstep 1 \
    -g_gradstep 1 \
    -log_name gain_house/tuning/
    echo $i
    #[ `expr $i % $degree` -eq 0 ] && wait
done
echo 'finish'
