# this script visualizes raw emg signals

import json
import websockets
import time
import asyncio
import matplotlib.cm as cm
import numpy as np
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

        result = await ws.recv()  # get rid of junk from first call to ws.recv()
        timestamp = 0  # initialize timestamp so it can be used outside while loop
        idx = 0
        idx2 = 0

        initTime = time.time()  # initial time of data collection
        t = initTime
        # while True:
        while time.time() - initTime < Nseconds:
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

            for j in range(Nsamples):
                channel[idx, :] = samples[j]['data']
                timeRecord[idx] = samples[j]['timestamp_s']
                idx += 1

            currentTime = time.time()
            if currentTime - t > 0.007:
                t = currentTime
                window = timeRecord > currentTime - 0.25
                MAV[idx2, :] = np.mean(np.abs(channel[window, :]), axis=0)
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
        data = MAV[:idx2, :]
        df = pd.DataFrame(data, columns=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16'])
        df.to_csv('data/down.csv', index=False)


Nseconds = 10  # number of seconds to run while loop
Npoints = (Nseconds + 1) * 2000  # number of samples to plot on x-axis
Npoints2 = (Nseconds + 1) * 144
Nchannels = 16
channel = np.zeros([Npoints, Nchannels])
timeRecord = np.zeros(Npoints)
MAV = np.zeros([Npoints2, Nchannels])

asyncio.get_event_loop().run_until_complete(listen())  # run wristband
