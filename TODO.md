TODO:
- 27 noises for autocorrelation - same number of features as in the MEG
- Let D1 plot of the single subjects overlapping under the avg plot
- Run analysis on wordlist time to compare results 
- run PCR/Lasso/ridge/elasticNet
- Check lag plots generation
- D1 should create a second lag plot per model where the diagonal average is taken just in the second after 0 (in model time, and all timings in neural time), and do also separate stats on that reduced lagged plot too. C1 has to be changed as well in that case because I think it is C1 that creates the *_lag_curves.npy files needed for D1. So there will be a *_lag_curves.npy like before and another *_lag_curves_1secondlocked.npy. maybe we can base the start of the window to keep in model time on the type of analysis (if the subsample start on the onset of the word or centered on the onset of the word) 
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
- run `python C1_dRSA_run.py --simulation ...` to generate autocorrelations with different numbers of subsamples (e.g. 10, 100, 1000)
- use `python C2_plot_dRSA.py ...` to automatically emit plots for every simulation run alongside the standard analysis
- take out first N word onset from possibilities (start with N = 1)









RUN NEXT:
python D1_group_cluster_analysis.py --analysis-name center_on_onset_overlap --subjects $(seq -w 1 27) --simulation-noise --simulation-neural-label "Random Noise 208"




DONE    qsub -t 1-27 s2_submit_A1_preproc.sh -- --overwrite
DONE    qsub -t 1-27 s2_submit_A2_concat.sh -- --overwrite
DONE    qsub s2_submit_C1_subject.sh -- --analysis-name lock_to_onset_overlap --lock-subsample-to-word-onset --allow-overlap
DONE    qsub -t 1-27 s2_submit_C2_plot.sh -- --analysis-name lock_to_onset_overlap
DONE    qsub s2_submit_D1_group.sh -- --analysis-name lock_to_onset_overlap

try new dRSA matrix permutation
DONE python D1_group_cluster_analysis.py --analysis-name lock_to_onset_overlap --subjects $(seq -w 1 27) --models "Envelope" "Phoneme Voicing" "Word Frequency" "GloVe" "GloVe Norm" --n-permutations 1000 --log-level DEBUG --matrix-downsample-factor 4





