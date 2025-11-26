

import numpy as np
a1 = np.squeeze(np.load("derivatives/Models/wordfreq/sub-01/concatenated/sub-01_concatenated_wordfreq_100Hz.npy"))
a2 = np.squeeze(np.load("derivatives/Models/wordfreq/sub-02/concatenated/sub-02_concatenated_wordfreq_100Hz.npy"))
a1.shape, a2.shape           # lengths
np.count_nonzero(a1), np.count_nonzero(a2)
i1 = np.argmax(a1 != 0); i2 = np.argmax(a2 != 0)
i1, i2                        # first nonzero indices
a1[a1 != 0][:20]              # first 20 nonzero values
a2[a2 != 0][:20]
a1[i1-2:i1+5], a2[i2-2:i2+5] 

import matplotlib.pyplot as plt
t1 = np.arange(len(a1)) / 100.0
t2 = np.arange(len(a2)) / 100.0
plt.plot(t1, a1, alpha=0.6)
plt.plot(t2, a2, alpha=0.6)
plt.xlim(0, 60)   # zoom as needed
plt.show()

b1 = a1[a1 != 0]
b2 = a2[a2 != 0]
t1 = np.arange(len(b1)) / 100.0
t2 = np.arange(len(b2)) / 100.0
plt.plot(t1, b1, alpha=0.6)
plt.plot(t2, b2, alpha=0.6)
plt.xlim(0, 60)   # zoom as needed
plt.show()