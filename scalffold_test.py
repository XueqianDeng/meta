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

##Buffer Creating
#
#
##


##  Hyper-parameter:
subject_name = "Francis_Horizontal_Aug_3_test"
data_path = "data/" + subject_name

#overwrite the old file
if os.path.exists(data_path):
    shutil.rmtree(data_path)

os.mkdir(data_path)
run = False
section_number = 10
section_data_path = data_path + "/Section_Data"
os.mkdir(section_data_path)




async def wait_until_i_larger_than_j(i,j,t):

    while i <= j:
        #print("i is {}, j is {}".format(i,j))
        await asyncio.sleep(t)

# function to listen to wristband return data holder object
#
async def listen():
    url = 'ws://127.0.0.1:9999'
    global listen_num
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
            listen_num = listen_num + 1
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

            #This section is used to test out if Listen() is missing anything

            if listen_num != 1:
                batch_start_time = samples[0]['timestamp_s']
                time_between = batch_start_time - batch_finished_time
                """
                print(batch_finished_time)
                print(batch_start_time)
                print(time_between)
                print()
                """
                if time_between > 0.0006:
                    print("dataloss in listen")
            batch_finished_time = samples[Nsamples - 1]['timestamp_s']


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
            print("listen finished{}".format(listen_num))



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
    global listen_num
    global experiment_num
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



        # raw data collection
        write_start_time = time.time()

        mdata = np.array_str(data_holder[(100 - batchcount):100, :])
        #midtime = time.time()
        raw_data.write("Batch Index" + str(batchindex) + "\n")
        raw_data.write("Batch Count" + str(batchcount) + "\n")
        raw_data.write(mdata + "\n")
        print("experiment finished{}".format(experiment_num))

        if experiment_num < (listen_num - 1):
            experiment_num = listen_num - 1
            print("dataloss: experiment is slower than listen")

        #write_end_time = time.time()
        #write_time_spent = write_end_time-write_start_time

        #if batchindex != 0:
        #    print("midtime{}".format((midtime - write_start_time)/batchcount))

        #    print("this is batch{}".format(batchindex))
        #    print("this batch has {} index".format(batchcount))
        #    print("total time spent{}".format(write_time_spent))
        #    average_time = write_time_spent/batchcount
        #    print("Average time spent{}".format(average_time))
        #    print()

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
        #await asyncio.sleep(0.001)

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
    global run
    run = True

    global testison
    testison = True

    global initTime
    initTime = time.time()
    print("this is time")
    print(initTime)

    global listen_num
    global experiment_num
    listen_num = 0
    experiment_num = 0


    #await asyncio.gather(listen(),print_messages(),experiment())
    await asyncio.gather(listen(),experiment())


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
