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

# Note from Francis Aug/2023:
# This code is just a testing code. We changed from using self-created buffer to use queue:
# We create a queue that deal with the problem of getting 2000hz data and store them
# Listen will get the data and put them into queue, and experiment gets element in queue and write it into file.
# We give experiment a time to wait so that it knows if there are elements in the queue.

##  Hyper-parameter:

csv_subject_name = 'CSV_aug14testing.csv'
csv_data_path = "data/" + csv_subject_name



# overwrite the old file


if os.path.exists(csv_data_path):
    shutil.rmtree(csv_data_path)

os.mkdir(csv_data_path)
section_data_path = csv_data_path + "/Section_Data"
os.mkdir(section_data_path)

csv_raw_data_path = section_data_path + "/raw_data.csv"

async def wait_until_i_larger_than_j(i, j, t):
    while i <= j:
        # print("i is {}, j is {}".format(i,j))
        await asyncio.sleep(t)


# function to listen to wristband return data holder object
#
async def listen():
    url = 'ws://127.0.0.1:9999'
    global q
    global listen_num
    global concatenating
    global instruction

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
            instruction_curr = instruction
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
            channel = np.zeros([Nsamples, 21])


            for j in range(Nsamples):
                channel[j, 3:19] = samples[j]['data']
                channel[j, 2] = instruction_curr
                channel[j, 0] = samples[j]['timestamp_s']-initTime  # signal time
                channel[j, 1] = samples[j]['produced_timestamp_s']-initTime  # Batch time
                #channel[j, 19] =
                #channel[j, 20] =

            if listen_num > 1:
                batch_start_time = samples[0]['timestamp_s']
                time_between = batch_start_time - batch_finished_time
                if time_between > 0.0006:
                    print("dataloss in listen")
            batch_finished_time = samples[Nsamples - 1]['timestamp_s']

            # delete later
            if testison:
                print("TEMP this is")
                print(temp)
                print("Channel this is")
                print(channel)
            testison = False

            q.put(deepcopy(channel))
            listen_num = listen_num + 1
            if q.qsize() > 7:
                print("--------------------warning, q size is {}------------------------".format(q.qsize()))

            # delete later
            # print("Listen finished {} times, queue size: {}".format(listen_num, q.qsize()))
            #print("Listen finished {} times, queue size: {} it has {} number".format(listen_num, q.qsize(), Nsamples))

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
    global listen_num, data_holder
    global experiment_num
    global concatenating
    global run
    global csv_output
    while run:

        while listen_num <= experiment_num:
            await asyncio.sleep(0.0005)

        experiment_num = experiment_num + 1

        # quit button
        keys = event.getKeys(keyList=['escape'])
        if keys:
            core.quit()

        while q.qsize() == 0:
            await asyncio.sleep(0.0005)

        mdata = q.get()
        curr_instruction = mdata[0][2]
        if curr_instruction != -1:
            np_data = np.array(mdata)

            df = pd.DataFrame(data = np_data, columns = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
                               'C13', 'C14', 'C15', 'C16', 'Instruction', 'Signal_Time', 'Batch_time','X','Y'])

            df.to_csv(csv_raw_data_path,mode='a',header=False, index=False)

async def print_messages():
    global instruction
    # t = 0
    for section_num in range(2):
        print(f"Section number: {section_num}")
        print("3 ready to OPEN")
        await asyncio.sleep(1)
        print("2")
        await asyncio.sleep(1)
        print("1, start opening")
        await asyncio.sleep(1)
        print("Open")
        instruction = 2
        # take data for 1 s
        await asyncio.sleep(1)  # open
        print("Rest")
        instruction = -1
        await asyncio.sleep(2)
        print("3 ready to CLOSE")

        # take data now for 1s for rest
        await asyncio.sleep(1)
        print("2")
        instruction = 1
        await asyncio.sleep(1)
        instruction = -1
        print("1 start to closing")
        await asyncio.sleep(1)
        print("CLOSE")
        instruction = 0
        # take data for 1s
        await asyncio.sleep(1)
        print("Rest")
        instruction = -1
        await asyncio.sleep(2)
    core.quit()


async def main():
    global listen_num
    global experiment_num
    listen_num = 0
    experiment_num = 0

    global instruction
    instruction = -1

    global q
    q = queue.Queue()

    global run
    run = True

    global testison
    testison = True


    global initTime
    initTime = time.time()

    await asyncio.gather(listen(), print_messages(), experiment())




asyncio.get_event_loop().run_until_complete(main())  # run wristband
core.quit()
