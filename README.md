Please add test.stm, train.stm and dev-libri_time.stm or adjust the paths in the .sh files

1. Run run-model.sh to train the WhisperForCTC model
2. Run run-lexicon.sh to generate a own lexicon for the LM
3. Run run-lm.sh to train the LM
4. Run run-decode.sh to decode with greedy decoder and beam search decoder (+ LM)
