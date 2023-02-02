#!/bin/bash

conda activate DMCG

# for number in 100 300 600 1000; do
for number in 300 600 1000; do

    python generatec_confs.py \
        --remove-hs \
        --use-ff \
        --number $number \
        --sdffile "/data/git-repo/DMCG/compounds/SDTI_1194/SDTI_1194.sdf" \
        --outpkl "SDTI_1194_gen$number.pkl" \
        --folder "dataset/SDTI_1194" \
        --eval-from "/data/git-repo/DMCG/DMCG/Large_Drugs/checkpoint_94.pt"
        &> log &

done