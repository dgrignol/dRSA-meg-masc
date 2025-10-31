TODO:

- choose sensors (temporal, speech, ATL)
- different models 
- lock to the word onset:
    -


RUN NEXT:
DONE    qsub -t 1-27 s2_submit_A1_preproc.sh -- --overwrite
DONE    qsub -t 1-27 s2_submit_A2_concat.sh -- --overwrite
DONE    qsub s2_submit_C1_subject.sh -- --analysis-name lock_to_onset_overlap --lock-subsample-to-word-onset --allow-overlap
DONE    qsub -t 1-27 s2_submit_C2_plot.sh -- --analysis-name lock_to_onset_overlap
DONE    qsub s2_submit_D1_group.sh -- --analysis-name lock_to_onset_overlap

try new dRSA matrix permutation
DONE python D1_group_cluster_analysis.py --analysis-name lock_to_onset_overlap --subjects $(seq -w 1 27) --models "Envelope" "Phoneme Voicing" "Word Frequency" "GloVe" "GloVe Norm" --n-permutations 1000 --log-level DEBUG --matrix-downsample-factor 4
