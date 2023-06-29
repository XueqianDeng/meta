from psychopy import visual, core, event, monitors  # import some libraries from PsychoPy
from psychopy.visual import circle
from psychopy.hardware import keyboard

import numpy as np
import pandas as pd
import json
import websockets
import time
import asyncio
import pickle
import random
import math
import matplotlib.pyplot as plt
from scipy import signal

vector_ratio = 1
vector_1_angle = np.pi / 4
vector_2_angle = 3 / 4 * np.pi
vector_3_angle = np.pi + np.pi / 4
vector_4_angle = 2 * np.pi - np.pi / 4

def vector_projection(vector, onto):
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector, onto)
    # Calculate the magnitude of the onto vector
    onto_magnitude = np.linalg.norm(onto)
    # Calculate the projection vector
    projection = dot_product / (onto_magnitude**2) * onto
    return projection

def unit_vector(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector
    return vector / magnitude * 1000

class VectorModel:

    def get_Vector(self):
        self.vector_1 = unit_vector(np.loadtxt('data/vector_1.csv', delimiter=','))
        self.vector_2 = unit_vector(np.loadtxt('data/vector_2.csv', delimiter=','))
        self.vector_3 = unit_vector(np.loadtxt('data/vector_3.csv', delimiter=','))
        self.vector_4 = unit_vector(np.loadtxt('data/vector_4.csv', delimiter=',')) ##   correction 1
        print("Vector 1", self.vector_1)
        print("Vector 2", self.vector_2)
        print("Vector 3", self.vector_3)
        print("Vector 4", self.vector_4)

    def predict(self, input_array):
        input_array = unit_vector(input_array)
        vector_1_projection = vector_projection(input_array, self.vector_1)
        vector_2_projection = vector_projection(input_array, self.vector_2)
        vector_3_projection = vector_projection(input_array, self.vector_3)
        vector_4_projection = vector_projection(input_array, self.vector_4)
        x_axis = 0
        y_axis = 0
        x_axis = \
            np.linalg.norm(vector_1_projection) * np.cos(vector_1_angle) + \
            np.linalg.norm(vector_2_projection) * np.cos(vector_2_angle) + \
            np.linalg.norm(vector_3_projection) * np.cos(vector_3_angle) + \
            np.linalg.norm(vector_4_projection) * np.cos(vector_4_angle) #correction 2
        y_axis = \
            np.linalg.norm(vector_1_projection) * np.sin(vector_1_angle) + \
            np.linalg.norm(vector_2_projection) * np.sin(vector_2_angle) + \
            np.linalg.norm(vector_3_projection) * np.sin(vector_3_angle) + \
            np.linalg.norm(vector_4_projection) * np.sin(vector_4_angle) #correction 3
        print("x_axis", x_axis)
        return [x_axis, y_axis]

# function to listen to wristband and update plot
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

        global run, idx2
        result = await ws.recv() # get rid of junk from first call to ws.recv()
        timestamp = 0  # initialize timestamp so it can be used outside while loop
        Nseconds = 10  # number of seconds to run while loop

        initTime = time.time()  # initial time of data collection
        t = initTime
        # while time.time() - initTime < Nseconds:
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
                # print(samples[j]['data'])
                emg_all[idx2, :] = samples[j]['data']
                idx2 += 1

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
        dx = (ex - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        ae = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(ae, ex, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


async def experiment():

    global x, y, idx, run, missing, MAV

    counter = 0
    trialcount = 0

    while run:
        keys = event.getKeys(keyList=['escape'])
        if keys:
            myWin.close()
            core.quit()

        currentTime = time.time()

        if np.sum(np.sum(channel)) != 0:
            if idx >= step + missing - 1:
                idx2 = timeRecord > currentTime - 0.25
                MAV = np.mean(np.abs(channel[:, idx2]), axis=1)

                df = pd.DataFrame(data=MAV.reshape([-1, 16]),
                                  columns=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
                                           'C13', 'C14', 'C15', 'C16'])
                print("DATAFRAME", df)
                INPUT = df.values
                print("Input Array", INPUT)
                pos_predict = model.predict(INPUT)

        else:
            # MAV_all[idx, :] = MAV
            pos_predict = np.array([0, 0])
            missing += 1

        print("Pos", pos_predict)

        x_pos = pos_predict[0]
        y_pos = pos_predict[1]

        if x_pos > 19:
            x_pos = 19
        elif x_pos < -19:
            x_pos = -19

        if y_pos > 10.75:
            y_pos = 10.75
        elif y_pos < -10.75:
            y_pos = -10.75

        if stimCirc.contains(fixSpot.pos):
            counter = counter + 1
            if counter == 5:
                stimCirc.pos = (random.randint(-10, 10), random.randint(-10, 10))
                trialcount = trialcount + 1
                counter = 0
        else:
            counter = 0
        stimCirc.draw()
        fixSpot.setPos((x_pos, y_pos))
        fixSpot.draw()
        tnumber.setText(trialcount)
        tnumber.draw()
        MAV_data = np.append(MAV, [stimCirc.pos[0], stimCirc.pos[1], x_pos, y_pos, currentTime])
        mdata = np.array_str(MAV_data)
        ofile.write(mdata + "\n")

        idx += 1

        myWin.flip()
        await asyncio.sleep(0.001)


async def main():
    await asyncio.gather(listen(), experiment())


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

idx = 0
idx2 = 0
missing = 0

Nchannels = 16
Npoints = 2000  # number of data points to record in history
sensitivity = 0.0025
subjectnum = "Test"
subjecttime = "day1trial1"

channel = np.zeros([Nchannels, Npoints])
timeRecord = np.zeros(Npoints)
emg_all = np.zeros([2000 * 30, Nchannels])

MAV = np.zeros(Nchannels)
MAV_filtered = np.zeros(Nchannels)

ofile = open("data/pilot_data.txt", "w")
ofile.write("Channel Values\n")
ofile.write("Subject: " + subjectnum + subjecttime + "\n")
ofile.write("C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C12, C13, C14, C15, C16, targ_x, targ_y, curs_x, curs_Y, time\n")

# create a window
mon = monitors.Monitor(name='testMonitor', width=38.1)
mon.setSizePix([1920, 1080])
myWin = visual.Window(size=[1920, 1080], fullscr=True, allowGUI=True, monitor=mon, color=[-.6, -.6, -.6],
                      units='cm')

fixSpot = visual.Circle(myWin,
                        pos=(0, 0), radius=0.5, fillColor='black', autoLog=False)
stimCirc = visual.Circle(myWin,
                         pos=(0, 0), radius=0.5, fillColor='white', autoLog=False)

target = circle.Circle(myWin, radius=0.3, lineColor=(1, 1, 1), fillColor=(0.75, 0, 0.75), autoDraw=False, pos=(0, 0))
cursor = circle.Circle(myWin, radius=0.1, lineColor=(1, 1, 1), fillColor=(1, 1, 1), autoDraw=False)
tnumber = visual.TextStim(myWin, pos=(18.5, 10))
tnumber.color = 'white'
tCount = 0
state = 1
run = True
gameTimer = core.Clock()
one_euro_filter = OneEuroFilter(0, 0, min_cutoff=.00004, beta=.7)

#with open('data/pointing_linear.pkl', 'rb') as f:
#    model = pickle.load(f)

model = VectorModel()
model.get_Vector()

Fs = 144
b, a = signal.butter(N=10, Wn=1, btype='lowpass', fs=Fs)
zi = signal.lfilter_zi(b, a)
step = 100

asyncio.get_event_loop().run_until_complete(main())  # run wristband

df = pd.DataFrame(data=emg_all,
                  columns=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
                           'C13', 'C14', 'C15', 'C16'])
df.to_csv('data/emg_sample.csv', index=False)

df = pd.DataFrame(data=emg_all,
                  columns=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
                           'C13', 'C14', 'C15', 'C16', 'x_pos', 'y_pos'])

myWin.close()
core.quit()
