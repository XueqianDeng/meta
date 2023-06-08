import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df2 = pd.read_csv("data/MAV_sample.csv")

df = pd.read_csv('data/emg_sample.csv')

df.loc[~(df==0).all(axis=1)]
pos = df2.iloc[:, 16:18]

Nchannels = 16

x1 = np.linspace(0, 8, 250)
y1 = np.linspace(0, 0, 250)

x2 = np.linspace(8, -8, 500)
y2 = np.linspace(0, 0, 500)

x3 = np.linspace(-8, 8, 500)
y3 = np.linspace(0, 0, 500)

x4 = np.linspace(8, 0, 250)
y4 = np.linspace(0, 0, 250)

x = np.concatenate([x1, x2, x3, x4, y1, y2, y3, y4])
y = np.concatenate([y1, y2, y3, y4, x1, x2, x3, x4])

offset = 0
# MAV_all = MAV_all.iloc[offset:, :].reset_index(drop=True)
pos = pos.iloc[offset:, :].reset_index(drop=True)

step = 1/2000
time = np.arange(0, df.shape[0] * step, step)

plt.figure(1)
for i in range(Nchannels):
    plt.plot(time, df.iloc[:, i] + 0.0006 * i - np.mean(df.iloc[:, i]), linewidth=0.3)

step = 1/144
time = np.arange(0, pos.shape[0] * step, step)

plt.plot(time, pos.iloc[:, 0] * 0.0001 - 0.001, color='black')
plt.plot(time, pos.iloc[:, 1] * 0.0001 - 0.002, color='grey')

plt.xticks(np.arange(0, 30, 5))
plt.xlim([0, 21])
plt.yticks([])
plt.xlabel('Time (s)')
plt.show()


# plt.figure(2)
# plt.subplot(2, 1, 1)
# plt.plot(time, pos.iloc[:, 0])
# plt.plot(time, x[offset:])
# plt.xlim([0, 21])
# plt.xticks([])
# plt.ylabel('X position (cm)')
# plt.legend(['Cursor', 'Target'])
#
# plt.subplot(2, 1, 2)
# plt.plot(time, pos.iloc[:, 1])
# plt.plot(time, y[offset:])
# plt.xticks(np.arange(0, 30, 5))
# plt.xlim([0, 21])
# plt.xlabel('Time (s)')
# plt.ylabel('Y position (cm)')
# plt.show()
