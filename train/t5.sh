python train_mlm.py \
t5-large \
data/antonym_synonym_negation_train.jsonl \
data/antonym_synonym_negation_dev.jsonl \
2 \
--gradient_accumulation_steps 64 \
--learning_rate 0.5e-6 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
