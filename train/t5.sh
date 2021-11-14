python train_16_mlm.py \
t5-11b \
data/antonym_synonym_negation_train.jsonl \
data/antonym_synonym_negation_dev.jsonl \
2 \
--learning_rate 0 \
--num_train_epochs 0 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
