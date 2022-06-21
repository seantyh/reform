#! /bin/bash

N_JOB=10

for i in $(seq 0 $(($N_JOB-1))); do
    echo "Starting JOB: ${i}/${N_JOB}"
    bash run_mediapipe.sh $i $N_JOB &> run_mediapipe_$i.log &
done
