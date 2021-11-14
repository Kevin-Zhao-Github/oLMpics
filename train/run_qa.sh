#!/bin/bash

TRAIN_DATA=('conjunction_filt4_train.jsonl' 'composition_v2_train.jsonl')
EVAL_DATA=('conjunction_filt4_dev.jsonl' 'composition_v2_dev.jsonl')
NUM_CHOICES=(3 3)

MODELS=('t5-large' 't5-base')
#MODELS=('t5-base')

for seed in 123; do
    for model in "${MODELS[@]}"; do
        for i in "${!TRAIN_DATA[@]}"; do
            # if [[ "$model" == *"t5"* || "$model" == *"albert"* ]]; then
            #     if [[ "$i" -eq 3 ]]; then
            #         continue
            #     fi
            # fi
            python eval_qa.py $model "data/${TRAIN_DATA[i]}" "data/${EVAL_DATA[i]}" "${NUM_CHOICES[i]}" \
            --learning_rate 5e-5 \
            --num_train_epochs 0
        break 3
        done
    done
done
