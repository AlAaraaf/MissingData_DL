#!/bin/bash
secs=$SECONDS
bgtime=$(date)
echo $bgtime
Rscript mice-cart.R
edtime=$(date)
echo $edtime
hrs=$(( secs/3600 )); mins=$(( (secs-hrs*3600)/60 )); secs=$(( secs-hrs*3600-mins*60 ))
echo 'Time spent %02d:%02d:%02d\n' $hrs $mins $secs
echo 'finish'