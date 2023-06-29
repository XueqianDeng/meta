##############################################
# This file collects MAV data for an experiment where one points their hand at a sum-of-sines stimulus
#
# The code outputs a .csv file which can be used for training a model in pointing_position_training.csv and
# pointing_velocity_training.csv
##############################################

from psychopy import visual, core, event, monitors  # import some libraries from PsychoPy
from psychopy.visual import circle, shape

import numpy as np
import pandas as pd
import math
import json
import websockets
import time
import asyncio
import random
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


# function to read data from wristband
async def listen():
    url = 'ws://127.0.0.1:9999'

    async with websockets.connect(url) as ws:

        # begin data stream from wristband
        await ws.send(json.dumps({
            "api_version": "0.12",
            "api_request": {
                "request_id": 1,
                "start_stream_request": {
                    "stream_id": "test_stream_id",
                    "app_id": "my-application",
                    "raw_emg": {}
                }
            }
        }))

        global MAV, run
        result = await ws.recv() # get rid of junk from first call to ws.recv()
        timestamp = 0  # initialize timestamp so it can be used outside while loop
        Nseconds = 10  # number of seconds to run while loop

        initTime = time.time()  # initial time of data collection
        t = initTime
        while run:
            result = await ws.recv()  # read data from wristband
            temp = json.loads(result)  # convert into readable format

            # samples is a nested list which is indexed as samples[data batch][data type]
            #   data batch: data from all channels collected at a single timepoint (timestamp_s); batch is indexed from
            #               0 to Nsamples-1
            #   data type: either raw emg data or two different timestamps, indexed as one of the below fields
            #              'data': the raw emg data; this is further indexed by channel from 0 to 15
            #              'timestamp_s': the time at which the data batch was collected
            #              'produced_timestamp_s': I think this is the time that ws.recv() is called, but not sure
            samples = temp['stream_batch']['raw_emg']['samples']

            Nsamples = len(samples)  # number of batches received from call to ws.recv()
            timestamp = samples[Nsamples-1]['timestamp_s'] - initTime  # time of last data batch

            # remove old samples
            channel[:, :-Nsamples] = channel[:, Nsamples:]
            timeRecord[:-Nsamples] = timeRecord[Nsamples:]

            # store new data
            for j in range(Nsamples):
                channel[:, -(Nsamples - j)] = samples[j]['data']
                timeRecord[-(Nsamples - j)] = samples[j]['timestamp_s']

        # end data stream from wristband
        await ws.send(json.dumps({
            "api_version": "0.12",
            "api_request": {
                "request_id": 1,
                "end_stream_request": {
                    "stream_id": "test_stream_id",
                }
            }
        }))


# function to run experiment
async def experiment():

    global x, y, idx, MAV, run, missing
    initTime = time.time()  # initial time the experiment is run

    while run:

        # if ESC key is pressed, then exit experiment
        keys = event.getKeys(keyList=['escape'])
        if keys:
            myWin.close()
            core.quit()

        # get current time
        currentTime = time.time()
        time_all.append(currentTime)

        # the wristband takes some time to start collecting data, so this will record data only after data collection
        # has started
        if np.sum(np.sum(channel)) != 0:
            if idx >= missing - 1:
                idx2 = timeRecord > currentTime - 0.25
                MAV = np.mean(np.abs(channel[:, idx2]), axis=1)
            else:
                missing += 1

        # store data from each channel in MAV_all
        for i in range(Nchannels):
            MAV_all[i].append(MAV[i])

        x_pos = x_all[idx]
        y_pos = y_all[idx]

        # set target position
        newPos = [x_pos, y_pos]
        target.setPos(newPos=newPos)

        # record target position history
        x.append(x_pos)
        y.append(y_pos)
        idx += 1

        if idx == len(x_all):
            run = False

        # draw stimuli on screen
        myWin.flip()

        # included for asynchronous function to work
        await asyncio.sleep(0.001)


# main asynchronous function that runs the wristband and experiment
async def main():
    await asyncio.gather(listen(), experiment())

#######################################################
# CODE FOR RUNNING WRISTBAND
#######################################################

np.random.seed(1)
idx = 0
missing = 0  # records number of samples not recorded by wristband
Nchannels = 16  # number of wristband channels
Npoints = 2000  # number of data points to record in history

# initialize variables to record data
channel = np.zeros([Nchannels, Npoints])  # record raw EMG over a time window set by Npoints
timeRecord = np.zeros(Npoints)  # time record for calculating MAV from raw EMG
MAV = np.zeros(Nchannels)  # MAV at the current time
MAV_all = [[] for i in range(Nchannels)]  # record history of MAV
time_all = []  # record times at which MAV was calculated
x = []  # x target position
y = []  # y target position


#######################################################
# CODE FOR RUNNING PSYCHOPY EXPERIMENT
#######################################################

# set target positions


x1 = np.linspace(0, 0, 20)
y1 = np.linspace(0, 0, 20)
x2 = []
y2 = []

# target vector1 -22.5; vector2 22.5;vector3 157.5;vector4 202.5
for a in range(5000):
    x2.append(a * 0.01 * math.sin(math.radians(202.5)))
    y2.append(a * 0.01 * math.cos(math.radians(202.5)))

x_all = np.concatenate([x1, x2])
y_all = np.concatenate([y1, y2])

# create a window
mon = monitors.Monitor(name='testMonitor', width=38.1)
mon.setSizePix([1920, 1080])
myWin = visual.Window(size=[1920, 1080], fullscr=True, allowGUI=True, monitor=mon, color=[-.6, -.6, -.6],
                      units='cm')

target = circle.Circle(myWin, radius=0.3, lineColor=(1, 1, 1), fillColor=(0.75, 0, 0.75), autoDraw=True, pos=(0, 0))  # target circle

run = True
gameTimer = core.Clock()

asyncio.get_event_loop().run_until_complete(main())  # run wristband and experiment code

# reshape data into a form that can be concatenated into a pandas dataframe
x = np.reshape(np.array(x), [-1, 1])
y = np.reshape(np.array(y), [-1, 1])
time_all = np.reshape(np.array(time_all), [-1, 1])
MAV_all = np.array(MAV_all).T
allData = np.concatenate([MAV_all, x, y, time_all], axis=1)

myWin.close()

data = allData
data_abs = np.abs(data)
Nsamples = data.shape[0]
MAV = np.zeros([data.shape[0]-500, data.shape[1]])

idx = 0
for i in range(500, Nsamples):
    MAV[idx, :] = np.mean(data_abs[i-500:i, :], axis=0)
    idx += 1

pca = PCA()
result = pca.fit(MAV)
var = pca.explained_variance_ratio_

pc = pca.components_[0:2, :]
np.savetxt("data/pc.csv", pc, delimiter=",")

plt.subplot(1, 2, 1)
plt.plot(np.arange(1, 20), var)
plt.ylabel('Proportion of explained variance')
plt.xlabel('Principal component number')

plt.subplot(1, 2, 2)
plt.plot(np.arange(1, 20), np.cumsum(var))
plt.ylabel('Proportion of explained variance')
plt.xlabel('Principal component number')

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(2, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=4).fit_transform(data_abs)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=y,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)

print(X_reduced)
print(len(X_reduced))

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.zaxis.set_ticklabels([])

plt.show()

