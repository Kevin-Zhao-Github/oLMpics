#!/bin/bash

TRAIN_DATA=('antonym_synonym_negation_train.jsonl' 'coffee_cats_quantifiers_train.jsonl' 'compositional_comparison_train.jsonl' 'hypernym_conjunction_train.jsonl' 'number_comparison_age_compare_masked_train.jsonl' 'size_comparison_train.jsonl')
EVAL_DATA=('antonym_synonym_negation_dev.jsonl' 'coffee_cats_quantifiers_dev.jsonl' 'compositional_comparison_dev.jsonl' 'hypernym_conjunction_dev.jsonl' 'number_comparison_age_compare_masked_dev.jsonl' 'size_comparison_dev.jsonl')
NUM_CHOICES=(2 5 3 3 2 2)

MODELS=('bert-base-uncased' 'distilbert-base-uncased' 'bert-large-uncased' 'bert-large-uncased-whole-word-masking' 'roberta-base' 'roberta-large' 'facebook/bart-large' 't5-base' 'albert-large-v1')

for seed in 123; do
    for i in "${!TRAIN_DATA[@]}"; do
        for model in "${MODELS[@]}"; do
            python train_mlm.py $model "data/${TRAIN_DATA[i]}" "data/${EVAL_DATA[i]}" "${NUM_CHOICES[i]}" \
            --gradient_accumulation_steps 2 \
            --learning_rate 0.5e-6 \
            --num_train_epochs 5
        done
    done
done
