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

        # # set variables for drawing future trajectory of the target
        # timeFromStart = currentTime - initTime  # time at which to start computing target position
        # future_times = np.linspace(timeFromStart, timeFromStart + 0.5, num=Ntimes)
        # x_pos = 0
        # y_pos = 0
        # vertices = np.zeros([Ntimes, 2])
        #
        # # calculate vertices for the ShapeStim line
        # for i in range(Nfreqs):
        #     x_pos += 2 * np.cos(2 * math.pi * freqX[i] * timeFromStart + phaseX[i])
        #     y_pos += 2 * np.cos(2 * math.pi * freqY[i] * timeFromStart + phaseY[i])
        #     vertices[:, 0] += 2 * np.cos(2 * math.pi * freqX[i] * future_times + phaseX[i])
        #     vertices[:, 1] += 2 * np.cos(2 * math.pi * freqY[i] * future_times + phaseY[i])

        x_pos = x_all[idx]
        y_pos = y_all[idx]

        # set target position
        newPos = [x_pos, y_pos]
        target.setPos(newPos=newPos)

        # set vertices
        line.setVertices(vertices)

        # record target position history
        x.append(x_pos)
        y.append(y_pos)
        idx += 1

        # end the experiment after a set number of seconds
        # if timeFromStart > 20:
        #     run = False

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

# x1 = np.linspace(0, 8, 250)
# y1 = np.linspace(0, 0, 1500)
# x2 = np.linspace(8, -8, 500)
# y2 = np.linspace(0, 0, 500)
# x3 = np.linspace(-8, 8, 500)
# y3 = np.linspace(0, 0, 500)
# x4 = np.linspace(8, 0, 250)
# y4 = np.linspace(0, 0, 250)
x1 = np.linspace(0, 0, 10)
y1 = np.linspace(0, 0, 10)
x2 = []
y2 = []

freqs1 = [1.5/250, 6/250, 1/25]
amps1 = [1/2, 1/1.7, 1/5]
phase1 = []
freqs2 = [3/250, 4.5/250, 7.5/250]
amps2 = [1/2, 1/4, 1/6]
phase2 = []
tempx = 0
tempy = 0
for r in range(len(freqs1)):
    phase1.append(random.random() * 2 * math.pi)
    phase2.append(random.random() * 2 * math.pi)
for a in range(8000):
    for h in range(len(freqs1)):
        tempx = tempx + amps1[h] * math.sin(float(freqs1[h]) * float(a) + float(phase1[h]))
        tempy = tempy + amps2[h] * math.sin(float(freqs2[h]) * float(a) + float(phase2[h]))
    tempx = tempx*8*max((min(.001*(a-100),1)),0)
    tempy = tempy*8*max((min(.001*(a-100),1)),0)
    x2.append(tempx)
    y2.append(tempy)
    tempx = 0
    tempy = 0
#for a in range(8000):
#    temp = ((math.sin(1.5*a/250)/2+math.sin(2*a*3/250+math.pi/5)/1.7+math.sin(2*a*5/250+5*math.pi/9)/5)*8)*max((min(.001*(a-100),1)),0)
#   temp = sum(amps1.*sin(freqs1*a + phases))
#    x2.append(temp)

#for b in range(8000):
#    temp = ((math.sin(1.5*b*2/250)/2+math.sin(1.5*b*3/250)/4+math.sin(1.5*b*5/250)/6)*8)*max((min(.001*(b-100),1)),0)
    # temp = (math.sin(1.5*b*2/250)/2+math.sin(1.5*b*3/250)/4+math.sin(1.5*b*5/250)/6)*8
#    y2.append(temp)

x_all = np.concatenate([x1, x2])
y_all = np.concatenate([y1, y2])

# # set frequencies for sum of sines
# base_freq = 0.05
# primes = np.array([2, 3, 5, 7, 11, 13, 17, 19])
# freqX = base_freq * primes[::2]
# freqY = base_freq * primes[1::2]
# Nfreqs = len(freqX)
#
# # set phases for sum of sines
# phaseX = 2 * np.pi * (np.random.rand(Nfreqs) - 0.5)
# phaseY = 2 * np.pi * (np.random.rand(Nfreqs) - 0.5)

Ntimes = 40  # number of vertices to draw look ahead trajectory for sum of sines
vertices = np.zeros([Ntimes, 2])

# create a window
mon = monitors.Monitor(name='testMonitor', width=38.1)
mon.setSizePix([1920, 1080])
myWin = visual.Window(size=[1920, 1080], fullscr=True, allowGUI=True, monitor=mon, color=[-.6, -.6, -.6],
                      units='cm')

line = shape.ShapeStim(myWin, vertices=vertices, lineColor=(1, 1, 1), autoDraw=True, closeShape=False)  # line for look ahead trajectory
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

# create data frame of MAV from each channel, target positions, and time
df = pd.DataFrame(data=allData,
                  columns=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
                           'C13', 'C14', 'C15', 'C16', 'x_pos', 'y_pos', 'time'])

subject = "Hokin"
# save data to .csv file
df.to_csv('data/' + subject + '/point_training_AMH.csv', index=False)

myWin.close()
core.quit()
