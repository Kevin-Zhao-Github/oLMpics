MODEL="bart-large"

rm -ri "results-$MODEL"
mkdir "results-$MODEL"

python bart_responses.py output/ $MODEL
python prediction_accuracy_tests.py \
  --preddir output \
  --resultsdir "results-$MODEL" \
  --models $MODEL \
  --k_values 5 \
  --role_stim datasets/ROLE-88/ROLE-88.tsv \
  --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
  --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
  --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv

python sensitivity_tests.py \
  --probdir output \
  --resultsdir "results-$MODEL" \
  --models $MODEL \
  --role_stim datasets/ROLE-88/ROLE-88.tsv \
  --negnat_stim datasets/NEG-88/NEG-88-NAT.tsv \
  --negsimp_stim datasets/NEG-88/NEG-88-SIMP.tsv \
  --cprag_stim datasets/CPRAG-34/CPRAG-34.tsv

