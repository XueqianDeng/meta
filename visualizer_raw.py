# this script visualizes raw emg signals

from pyqtgraph.Qt import QtWidgets
import pyqtgraph as pg

import json
import websockets
import time
import asyncio
import matplotlib.cm as cm
import numpy as np


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

        result = await ws.recv() # get rid of junk from first call to ws.recv()
        timestamp = 0  # initialize timestamp so it can be used outside while loop
        global ptr
        Nseconds = 60  # number of seconds to run while loop

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
            ptr += Nsamples  # update x position for displaying the curve

            channel[:, :-Nsamples] = channel[:, Nsamples:]  # remove old samples

            for j in range(Nsamples):
                channel[:, -(Nsamples - j)] = samples[j]['data']

            # update plots every 0.007 seconds (about 144 Hz, which is this laptop's screen refresh rate)
            if time.time() - t > 0.007:
                for j in range(Nchannels):
                    plots[j].setData(channel[j])  # update plots
                    plots[j].setPos(ptr, 0)  # set x position in the graph to 0

                QtWidgets.QApplication.processEvents()  # not sure what this does but including this
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

        print(timestamp) # print last time collected from wristband to check that it matches with program runtime
        QtWidgets.QApplication.exec_()  # end widget


### START QtApp #####
app = QtWidgets.QApplication([])            # you MUST do this once (initialize things)
####################

Npoints = 5000  # number of samples to plot on x-axis
Nchannels = 16
sample_period = 20  # sampling period in ms

# preallocate variables
p, plots = [[] for _ in range(Nchannels)], [[] for _ in range(Nchannels)]

channel = np.zeros([Nchannels, Npoints])

win = pg.GraphicsLayoutWidget()  # widget for creating subplots

colors = cm.gist_rainbow(np.linspace(0, 1, Nchannels))*255  # colors for plotting lines

# generate subplots
idx = 0
for a in range(2):  # loop over columns
    for b in range(8): # loop over rows
        myPen = pg.mkPen(color=tuple(colors[idx]))
        p[idx] = win.addPlot(row=b, col=a)
        p[idx].hideAxis('bottom')
        # p[idx].setYRange(-.00008, .0002, padding=0)
        plots[idx] = p[idx].plot(pen=myPen)
        idx += 1

win.show()  # display plots

windowWidth = 500  # I don't know what this does
ptr = -windowWidth  # set first x position

asyncio.get_event_loop().run_until_complete(listen())  # run wristband
