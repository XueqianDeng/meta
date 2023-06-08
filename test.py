# this script visualizes raw emg signals
from psychopy import visual, core, event, monitors  # import some libraries from PsychoPy
from psychopy.visual import circle

import json
import websockets
import time
import asyncio
import numpy as np
import pickle
import pandas as pd


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

        global MAV, direction
        result = await ws.recv() # get rid of junk from first call to ws.recv()
        timestamp = 0  # initialize timestamp so it can be used outside while loop
        Nseconds = 10  # number of seconds to run while loop
        Npoints = 2000  # number of data points to record in history

        channel = np.zeros([Nchannels, Npoints])
        timeRecord = np.zeros(Npoints)

        initTime = time.time()  # initial time of data collection
        t = initTime
        # while time.time() - initTime < Nseconds:
        while True:
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

            currentTime = time.time()
            # update plots every 0.007 seconds (about 144 Hz, which is this laptop's screen refresh rate)
            if currentTime - t > 0.007:

                idx = timeRecord > currentTime - 0.25
                MAV = np.mean(np.abs(channel[:, idx]), axis=1)
                df = pd.DataFrame(data=np.reshape(MAV, [1, 16]),
                                  columns=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
                                           'C13', 'C14', 'C15', 'C16'])

                # y_pred_prob = model.predict_proba(df)
                # y_pred = (np.argmax(y_pred_prob) + 1) % 5
                # if np.max(y_pred) > 0.7:
                #     direction = labels[y_pred]
                direction = model.predict(df)[0]
                t = time.time()  # update plot timer



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

        print(timestamp)  # print last time collected from wristband to check that it matches with program runtime


async def experiment():
    while True:
        keys = event.getKeys(keyList=['escape'])
        if keys:
            myWin.close()
            core.quit()

        newPos = cursor.pos
        if direction == 'left':
            newPos[0] -= step
        elif direction == 'right':
            newPos[0] += step
        elif direction == 'down':
            newPos[1] -= step
        elif direction == 'up':
            newPos[1] += step

        cursor.setPos(newPos=newPos)
        myWin.flip()
        await asyncio.sleep(0.001)


async def main():
    await asyncio.gather(listen(), experiment())


with open('data/model_MAV.pkl', 'rb') as f:
    model = pickle.load(f)
labels = ['up', 'down', 'left', 'right', 'still']

# create a window
mon = monitors.Monitor(name='testMonitor', width=38.1)
mon.setSizePix([1920, 1080])
myWin = visual.Window(size=[1920, 1080], fullscr=True, allowGUI=True, monitor=mon, color=[-.6, -.6, -.6],
                      units='cm')

cursor = circle.Circle(myWin, radius=0.1, lineColor=(1, 1, 1), fillColor=(1, 1, 1), autoDraw=True)
cursor.draw()

Nchannels = 16
MAV = np.zeros([Nchannels])
direction = 'still'

# pc = np.loadtxt("pc.csv", delimiter=',')

step = 0.01
asyncio.get_event_loop().run_until_complete(main())  # run wristband

myWin.close()
core.quit()
