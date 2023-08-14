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
import queue
from copy import deepcopy
numpy.set_printoptions(threshold=sys.maxsize)

##  Hyper-parameter
subject_name = "Hokin_Aug"
data_path = "data/" + subject_name
if os.path.exists(data_path):
    shutil.rmtree(data_path)
os.mkdir(data_path)

##
# Experiment Structure
## 

# define how many sections to collect data
section_nums = 10 

# initialize both section and phase into 0
global current_phase = 0
global current_section = 0

############################################################

async def wait_until_i_larger_than_j(i, j, t):
    while i <= j:
        # print("i is {}, j is {}".format(i,j))
        await asyncio.sleep(t)

############################################################



async def wait_until_i_larger_than_j(i,j,t):

    while i <= j:
        #print("i is {}, j is {}".format(i,j))
        await asyncio.sleep(t)

# function to listen to wristband return data holder object
#
async def listen():

    global q
    url = 'ws://127.0.0.1:9999'
    global listen_num
    global concatenating

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
                channel[j, 17] = samples[j]['timestamp_s'] #signal time
                channel[j, 18] = samples[j]['produced_timestamp_s'] # Batch time
                #  end data stream from wristband

            #delete later
            if testison:
                print("TEMP this is")
                print(temp)
                print("Channel this is")
                print(channel)
            testison = False

            q.put(deepcopy(channel))
            listen_num = listen_num + 1
            if q.qsize() > 4:
                print("--------------------warning, q size is {}------------------------".format(q.qsize()))

            #delete later
            #print("Listen finished {} times, queue size: {}".format(listen_num, q.qsize()))
            print("Listen finished {} times, queue size: {} it has {} number".format(listen_num, q.qsize(),Nsamples))

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
    global q
    global listen_num , data_holder
    global experiment_num
    global concatenating
    global run

    while run:

        while listen_num <= experiment_num:
            # print("i is {}, j is {}".format(i,j))
            await asyncio.sleep(0.0005)

        experiment_num = experiment_num + 1
        
        # quit button
        keys = event.getKeys(keyList=['escape'])
        if keys:
            core.quit()
            
        write_start_time = time.time()

        while q.qsize() == 0:
            await asyncio.sleep(0.0005)
            
        mdata = q.get()
        batchtestsize = len(mdata)
        mdata = np.array_str(mdata)
        #raw_data.write("Batch Index"+"\n")
        #raw_data.write("Batch Count" + "\n")
        #midtime = time.time()

        raw_data.write(mdata + "\n")
        #delete later
        print("Experiment finished {} times, queue size: {}, batch size is {}".format(experiment_num, q.qsize(), batchtestsize))

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

# while i is not larger than j, the system would keep waiting

async def main():

    global q
    q = queue.Queue()

    global run
    run = True

    global testison
    testison = True

    global initTime
    initTime = time.time()

    global listen_num
    global experiment_num
    listen_num = 0
    experiment_num = 0

    global concatenating

    #await asyncio.gather(listen(),print_messages(),experiment())
    await asyncio.gather(listen(), experiment())

#  Setting global things up
start_time = time.time()

# setting up
raw_data_path = data_path + "/raw_data.txt"
raw_data = open(raw_data_path, "w")

asyncio.get_event_loop().run_until_complete(main())  # run wristband
core.quit()
