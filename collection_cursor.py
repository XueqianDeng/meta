#####


from psychopy import visual, core, event, monitors  # import some libraries from PsychoPy
from psychopy.visual import circle

import numpy as np
import pandas as pd
import json
import websockets
import time
import asyncio
import pickle
import math
import os
import random
from scipy import signal


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
        }))  # start data stream

        global run
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

            channel[:, :-Nsamples] = channel[:, Nsamples:]  # remove old samples
            timeRecord[:-Nsamples] = timeRecord[Nsamples:]

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


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(ae, ex, x_prev):
    return ae * ex + (1 - ae) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, ex):
        t_e = t - self.t_prev
        a_d = smoothing_factor(t_e, self.d_cutoff)
        if t_e == 0:
            dx = 0

        else:
            dx = (ex - self.x_prev) / t_e

        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        ae = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(ae, ex, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


# function for running experiment
async def experiment():

    global x, y, idx, run, missing, MAV, x_pos, y_pos, sensitivity, ofile, subjectnum, subjectdate, subjectblock
    counter = 0
    trialcount = 0
    drawim = False

    while run:

        # quit experiment if ESC is pressed
        keys = event.getKeys(keyList=['escape'])
        if keys:
            myWin.close()
            core.quit()

        currentTime = time.time()

        # check to see if wristband has started collecting data
        if np.sum(np.sum(channel)) != 0:

            # runs if data has started collecting
            if idx >= step + missing - 1:

                # calculate MAV within 250 ms time window
                idx2 = timeRecord > currentTime - 0.25
                MAV = np.mean(np.abs(channel[:, idx2]), axis=1)
                drawim = True
                #MAV_all[idx, :] = MAV

                # filter MAV
                # for j in range(Nchannels):
                #    MAV_window = MAV_all[idx - step + 1: idx + 1, j]
                #    window, _ = signal.lfilter(b, a, MAV_window, zi=zi * MAV_window[0])
                #    MAV_filtered[j] = window[-1]

                df = pd.DataFrame(data=MAV.reshape([1, 16]),
                                      columns=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
                                               'C13', 'C14', 'C15', 'C16'])
                df.fillna(0)

                vel_predict = model.predict(df)[0]
                #vel_predict, _ = signal.lfilter(b, a, vel_predict, zi=zi * vel_predict[0])
                #one_euro_filter = OneEuroFilter(0, vel_predict[0], min_cutoff=.004, beta=.7)
                vel_predict = one_euro_filter_V(currentTime, vel_predict)


        # runs if wristband has not started collecting data
        else:
            #MAV_all[idx, :] = MAV
            vel_predict = np.array([0, 0])
            missing += 1

        x_pos = x_pos + vel_predict[0] * sensitivity
        y_pos = y_pos + vel_predict[1] * sensitivity
        #x_pos = one_euro_filter_X(currentTime, x_pos)
        #y_pos = one_euro_filter_Y(currentTime, y_pos)
        #y_pos = 0

        # set maximum bounds for the cursor position
        if x_pos > 19:
            x_pos = 19
        elif x_pos < -19:
            x_pos = -19

        if y_pos > 10.75:
            y_pos = 10.75
        elif y_pos < -10.75:
            y_pos = -10.75

        fixSpot.setPos((x_pos, y_pos))
        stimCirc.setPos((x[idx], y[idx]))

        if drawim:
            stimCirc.draw()
            fixSpot.draw()

        tnumber.setText(trialcount)
        tnumber.draw()
        MAV_data = np.append(MAV, [stimCirc.pos[0], stimCirc.pos[1], x_pos, y_pos, currentTime])
        mdata = np.array_str(MAV_data)
        formatdata = ' '.join(map(str, MAV_data))
        if drawim:
            ofile.write(formatdata + "\n")


        # cPos = [x_pos, y_pos]
        # cursor.setPos(newPos=cPos)

        # tPos = [x[idx], y[idx]]
        # target.setPos(newPos=tPos)

        #t_all[idx] = currentTime
        idx += 1

        # if idx == len(x):
        #    run = False

        myWin.flip()
        await asyncio.sleep(0.001)

        if trialcount == 61:
            run = False


async def main():
    await asyncio.gather(listen(), experiment())


# set target positions

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

x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])

#######################################################
# CODE FOR RUNNING WRISTBAND
#######################################################

# initialize variables for experiment
idx = 0
missing = 0
# Npos = len(x)+10000000000
Nchannels = 16
Npoints = 4000  # number of data points to record in history
x_pos = 0
y_pos = 0
sensitivity = 0.0025
subjectnum = "Test"
subjectdate = "day1"
subjectblock = "Block1"

channel = np.zeros([Nchannels, Npoints])
timeRecord = np.zeros(Npoints)

MAV = np.zeros(Nchannels)
MAV_filtered = np.empty([Nchannels])
#MAV_all = np.zeros([Npos, Nchannels])
#t_all = np.zeros([Npos])

#ofile = open("data/" + subjectnum + "/" + subjectdate + "/" + subjectblock + "/fluid" + str(time.time()) + ".txt", "w")
ofile = open("data\pilot_data.txt", "w")
ofile.write("Channel Values\n")
ofile.write("Subject: " + subjectnum + " " + subjectdate + " " + subjectblock + "\n")
ofile.write("C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C12, C13, C14, C15, C16, targ_x, targ_y, curs_x, curs_Y, time\n")

# load model for predicting cursor velocity from MAV
with open('data/' + subjectnum + '/pointing_linear_pos_YD.pkl', 'rb') as f:
    model = pickle.load(f)

# set filter parameters
Fs = 144
b, a = signal.butter(N=5, Wn=3, btype='lowpass', fs=Fs)
zi = signal.lfilter_zi(b, a)
step = 100
one_euro_filter_X = OneEuroFilter(0, 0, min_cutoff=.004, beta=.1)
one_euro_filter_Y = OneEuroFilter(0, 0, min_cutoff=.004, beta=.1)
one_euro_filter_V = OneEuroFilter(0, 0, min_cutoff=.004, beta=.1)
#######################################################
# CODE FOR RUNNING PSYCHOPY EXPERIMENT
#######################################################

# create a window
mon = monitors.Monitor(name='testMonitor', width=38.1)
mon.setSizePix([1920, 1080])
myWin = visual.Window(size=[1920, 1080], fullscr=True, allowGUI=True, monitor=mon, color=[-.6, -.6, -.6],
                      units='cm')

target = circle.Circle(myWin, radius=0.3, lineColor=(1, 1, 1), fillColor=(0.75, 0, 0.75), autoDraw=False, pos=(0, 0))
cursor = circle.Circle(myWin, radius=0.1, lineColor=(1, 1, 1), fillColor=(1, 1, 1), autoDraw=False)
fixSpot = visual.Circle(myWin,
                        pos=(0, 0), radius=0.5, fillColor='black', autoLog=False)
stimCirc = visual.Circle(myWin,
                         pos=(0, 0), radius=0.5, fillColor='white', autoLog=False)
tnumber = visual.TextStim(myWin, pos=(18.5, 10))
tnumber.color = 'white'
tCount = 0
state = 1
run = True
gameTimer = core.Clock()


asyncio.get_event_loop().run_until_complete(main())  # run wristband

ofile.close()
myWin.close()
core.quit()
