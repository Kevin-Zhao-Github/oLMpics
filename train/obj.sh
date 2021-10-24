python train_mlm.py \
bert-base-uncased \
data/antonym_synonym_negation_train.jsonl \
data/antonym_synonym_negation_dev.jsonl \
2 \
--gradient_accumulation_steps 2 \
--learning_rate 0.5e-6 \
--num_train_epochs 5 \
