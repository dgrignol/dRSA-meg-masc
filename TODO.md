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

