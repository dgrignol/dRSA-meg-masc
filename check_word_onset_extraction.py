# HOW TO RUN THIS SCRIPT:
# quick: 
# 'python check_word_onset_extraction.py'  # generates a plot 'word_onsets.png'
# Detailed:
# 1. Ensure the necessary preprocessed artefacts exist:
#    - `*_concatenated_envelope_100Hz.npy`
#    - `*_concatenated_word_onsets_sec.npy`
#    - `*_concatenated_sentence_mask_100Hz.npy`
# 2. Change the 'subject' variable to the desired subject ID (e.g., 'sub-01').
# 3. Run this script with 'python check_word_onset_extraction.py' in an environment where numpy, pandas, and matplotlib are installed.
# 4. The script plots the mean audio envelope, shades regions where the concatenated sentence mask is True, and overlays vertical lines at the extracted word onsets.
# 5. The resulting PNG (`word_onsets.png`) is saved in the current directory; adjust `time_window` to explore different sections.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


subject = "sub-01"
time_window = 800  # starting point of 30s window, adjust if want to see another window
repo_root = Path(".").resolve()
deriv = repo_root / "derivatives" / "preprocessed" / subject / "concatenated"

# Load the seconds array
onset_path = deriv / f"{subject}_concatenated_word_onsets_sec.npy"
onsets = np.load(onset_path)
print(f"{subject}: {onsets.size} word onsets, "
      f"{onsets.min():.2f}s–{onsets.max():.2f}s (first 10: {onsets[:10]})")

# Optional: load the concatenated envelope to use as a backdrop
env_path = (
    repo_root
    / "derivatives"
    / "Models"
    / "envelope"
    / subject
    / "concatenated"
    / f"{subject}_concatenated_envelope_100Hz.npy"
)
envelope = np.load(env_path)
envelope = np.squeeze(envelope)
if envelope.ndim == 1:
    env_trace = envelope
else:
    env_trace = envelope.mean(axis=0)
time = np.arange(env_trace.shape[-1]) / 100.0  # 100 Hz

sentence_mask_path = deriv / f"{subject}_concatenated_sentence_mask_100Hz.npy"
sentence_mask = np.load(sentence_mask_path).astype(bool).ravel()
if sentence_mask.shape[0] != env_trace.shape[-1]:
    raise ValueError(
        f"Mask length ({sentence_mask.shape[0]}) does not match envelope length ({env_trace.shape[-1]})."
    )

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(time, env_trace, lw=0.6, color="slateblue", alpha=0.7, label="Envelope (mean)", zorder=2)
ylim = ax.get_ylim()
ax.fill_between(
    time,
    ylim[0],
    ylim[1],
    where=sentence_mask,
    color="gold",
    alpha=0.15,
    step="post",
    zorder=0,
    label="Sentence mask",
)
ax.set_ylim(ylim)
ax.vlines(
    onsets,
    ymin=ylim[0],
    ymax=ylim[1],
    colors="crimson",
    linewidth=0.6,
    alpha=0.6,
    zorder=1,
    label="Word onsets",
)
ax.set_xlim(onsets.min() - 1 + time_window, onsets.min() + 30+ time_window)  # zoom into the first 30 s; tweak as needed
ax.set_xlabel("Time (s)")
ax.set_ylabel("Envelope (avg across channels)")
ax.set_title(f"{subject} word onsets vs. envelope")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("word_onsets.png", dpi=300)
