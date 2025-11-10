TODO:
- run PCR/Lasso/ridge/elasticNet
- choose sensors (temporal, speech, ATL)
- Add a flag to decide how much in advance to start subsample in respect to word onset (try centering the window on the word onset first e.g. 2.5 seconds in advance with a 5 seconds subsample window)
- Other models:
Voice activation (0 where silence 1 where speech)
spectrogram
wav2vec 2.0
syntactic
word2vec
BERT 
- check sentence mask definition and make plot with words printed
- do simulations (autocorrelations) with different number of subsamples (e.g. 10, 100, 1000)
- simulations will plot of autocorrelation one for subsample strategy
- take out first N word onset from possibilities (start with N = 1)









RUN NEXT:
DONE    qsub -t 1-27 s2_submit_A1_preproc.sh -- --overwrite
DONE    qsub -t 1-27 s2_submit_A2_concat.sh -- --overwrite
DONE    qsub s2_submit_C1_subject.sh -- --analysis-name lock_to_onset_overlap --lock-subsample-to-word-onset --allow-overlap
DONE    qsub -t 1-27 s2_submit_C2_plot.sh -- --analysis-name lock_to_onset_overlap
DONE    qsub s2_submit_D1_group.sh -- --analysis-name lock_to_onset_overlap

try new dRSA matrix permutation
DONE python D1_group_cluster_analysis.py --analysis-name lock_to_onset_overlap --subjects $(seq -w 1 27) --models "Envelope" "Phoneme Voicing" "Word Frequency" "GloVe" "GloVe Norm" --n-permutations 1000 --log-level DEBUG --matrix-downsample-factor 4