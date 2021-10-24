python train_mlm.py \
bert-large-uncased \
data/number_comparison_age_compare_masked_train.jsonl \
data/number_comparison_age_compare_masked_dev.jsonl \
2 \
--learning_rate 0.3e-5 \
--num_train_epochs 8 \
