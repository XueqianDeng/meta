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

import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

##  Hyper-parameter:
data_path = ".."
run = False
subject_name = "Hokin"


# function to listen to wristband return data holder object
#
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
        result = await ws.recv()  # get rid of junk from first call to ws.recv()

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
            Nsamples = len(samples)
            channel = np.zeros([Nsamples, 19])
            for j in range(Nsamples):
                channel[j, 0:16] = samples[j]['data']
                channel[j, 16] = j
                channel[j, 17] = samples[j]['timestamp_s']
                channel[j, 18] = samples[j]['produced_timestamp_s']
                #  end data stream from wristband

            # transfer data
            global batchcount, batchindex
            batchindex = batchindex + 1
            batchcount = Nsamples
            data_holder[0:(100 - batchcount), :] = data_holder[batchcount:100, :]
            data_holder[(100 - batchcount):100, :] = channel

        await ws.send(json.dumps({
            "api_version": "0.12",
            "api_request": {
                "request_id": 1,
                "end_stream_request": {
                    "stream_id": "test_stream_id",
                }
            }
        }))

async def experiment():
    global run
    while run:

        # quit button
        keys = event.getKeys(keyList=['escape'])
        if keys:
            core.quit()

        # raw data collection
        mdata = np.array_str(data_holder[(100 - batchcount):100, :])
        raw_data.write("Batch Index" + str(batchindex) + "\n")
        raw_data.write("Batch Count" + str(batchcount) + "\n")
        raw_data.write(mdata + "\n")

        # record time control
        current_time = time.time()


        # synchronization
        await asyncio.sleep(0.001)

async def extraction():
    return False;

async def main():
    global run
    run = True
    await asyncio.gather(listen(), experiment())


#  Setting global things up
data_holder = np.zeros([100, 19])
batchcount = 0
batchindex = 0
start_time = time.time()

# setting up
raw_data = open("data/pilot_data.txt", "w")

asyncio.get_event_loop().run_until_complete(main())  # run wristband
core.quit()
