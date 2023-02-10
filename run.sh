#!/bin/bash
degree=1
for i in `seq 0 9`
do
    python ./main.py -id $i -dataset sim_1_tiny -mr 0.3 -size 5000
    echo $i
    #[ `expr $i % $degree` -eq 0 ] && wait
done
echo 'finish'
