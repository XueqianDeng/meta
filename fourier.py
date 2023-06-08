import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/down.csv")

n = data.shape[0]
x = data - np.mean(data, axis=0)

up_fft = np.fft.rfft(x, axis=0)
amp = np.abs(up_fft/n)
amp[2:-1] = 2 * amp[2:-1]

f = 144 * np.arange(0, n/2) / n

fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
fig.text(0.5, 0.04, 'common X', ha='center')
fig.text(0.06, 0.5, 'common Y', va='center', rotation='vertical')
for i, axis in enumerate(ax.flat):
    axis.plot(f, amp[:, i])
    axis.set_xlim([0, 10])

plt.show()
