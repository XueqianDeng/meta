from psychopy import visual, core, event, monitors  # import some libraries from PsychoPy
from psychopy.visual import circle
from psychopy.hardware import keyboard
import os
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
import shutil
numpy.set_printoptions(threshold=sys.maxsize)

#Note from Francis July/2023:
#the function is runnning while getting the starting time
#
#



##  Hyper-parameter:
subject_name = "Francis_Horizontal_July_30_test_1"
data_path = "data/" + subject_name

#overwrite the old file
if os.path.exists(data_path):
    shutil.rmtree(data_path)

os.mkdir(data_path)
run = False
section_number = 10
section_data_path = data_path + "/Section_Data"
os.mkdir(section_data_path)


#  Setting global things up
data_holder = np.zeros([100, 19])
batchcount = 0
batchindex = 0
start_time = time.time()

# setting up
raw_data_path = data_path + "/raw_data.txt"
raw_data = open(raw_data_path, "w")

asyncio.get_event_loop().run_until_complete(main())  # run wristband
core.quit()

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
        global testison
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
            if testison:
                print("TEMP this is")
                print(temp)
                print("Channel this is")
                print(channel)
            testison = False


            # transfer data
            global batchcount, batchindex
            batchindex = batchindex + 1
            batchcount = Nsamples
            data_holder[0:(100 - batchcount), :] = data_holder[batchcount:100, :]
            data_holder[(100 - batchcount):100, :] = channel

            mdata = np.array_str(data_holder[(100 - batchcount):100, :])
            raw_data.write("Batch Index" + str(batchindex) + "\n")
            raw_data.write("Batch Count" + str(batchcount) + "\n")
            raw_data.write(mdata + "\n")


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
        #print("Batch Index" + str(batchindex) + "\n")
        #print("Batch Count" + str(batchcount) + "\n")
        #print(mdata + "\n")
        # record time control
        # for this_section in range(section_number):
        #    this_section_start_time = time.time()
            # Create Data Structure for this Section
        #   this_section_data_path = section_data_path + "\section_number" + str(this_section)
        #    os.mkdir(this_section_data_path)



        # synchronization
        await asyncio.sleep(0.001)

async def extraction():
    return False;


async def print_messages():
    #t = 0
    for section_num in range(2):
        print(f"Section number: {section_num}")
        print("3 ready to OPEN")
        await asyncio.sleep(1)
        print("2")
        await asyncio.sleep(1)
        print("1, start opening")
        await asyncio.sleep(1)
        print("Open")

        #take data for 1 s
        await asyncio.sleep(1) #open
        print("Rest")

        await asyncio.sleep(2)
        print("3 ready to CLOSE")

        #take data now for 1s for rest
        await asyncio.sleep(1)
        print("2")

        await asyncio.sleep(1)
        print("1 start to closing")
        await asyncio.sleep(1)
        print("CLOSE")

        #take data for 1s
        await asyncio.sleep(1)
        print("Rest")
        await asyncio.sleep(2)
    core.quit()


async def main():
    global run
    run = True

    global testison
    testison = True

    global initTime
    initTime = time.time()
    print("this is time")
    print(initTime)
    await asyncio.gather(listen(),print_messages())




