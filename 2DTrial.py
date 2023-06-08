from psychopy import visual, core, event, monitors  # import some libraries from PsychoPy
from psychopy.hardware import keyboard
from websocket import create_connection
import json
import time
import random

# ws = create_connection("ws://127.0.0.1:9999")
#
# ws.send(json.dumps({
#     "api_version": "0.12",
#     "api_request": {
#         "request_id": 1,
#         "start_stream_request": {
#             "stream_id": "test_stream_id",
#             "app_id": "my-application",
#             "raw_emg": {}
#         }
#     }
# }))
#
# initTime = time.time()
# result = ws.recv()
#
# while time.time() - initTime < 10:
#     result = ws.recv()
#     temp = json.loads(result)
#     data = temp['stream_batch']['raw_emg']['samples'][0]['data'][0]
# ws.close()

# create a window
mon = monitors.Monitor(name='testMonitor', width=38.1)
mon.setSizePix([1920, 1080])
mywin = visual.Window(size=[1920, 1080], fullscr=True, allowGUI=True, monitor=mon, color=[-.6, -.6, -.6],
                      units='cm')
#ifile = open("some input file", "r")
ofile = open("2DExperiment.txt", "w")
ofile.write("Channel Values\n")

# create some stimuli
fixSpot = visual.Circle(mywin,
                             pos=(0, 0), radius=0.5, fillColor='black', autoLog=False)
stimCirc = visual.Circle(mywin,
                             pos=(0, 0), radius=0.5, fillColor='white', autoLog=False)
myMouse = event.Mouse()  # will use win by default
counter = 0
trial = 1
temp = 2
gloclock = 0
kb = keyboard.Keyboard()
dfile = open("2Ddata" + str(trial) + ".txt", "w")

# draw the stimuli and update the window
while True:  # this will need to change to until end of file
    # get key events
    if temp == trial:
        dfile = open("2Ddata" + str(trial) + ".txt", "w")
        temp = temp + 1
    else:
        dfile.write(str(fixSpot.pos) + " " + str(stimCirc.pos) + " " + str(gloclock) + "\n")
    keys = kb.getKeys(['w', 'a', 's', 'd', 'escape'])
    gloclock = gloclock + 1
    for thisKey in keys:  # use emg controls after it's been set up
        if thisKey == 'escape':  # it is equivalent to the string 'q'
            ofile.close()
            mywin.close()
            core.quit()
    if stimCirc.contains(fixSpot.pos):
        counter = counter + 1
        if counter == 75:
            #ifile.readline()
            stimCirc.pos = (random.randint(-10, 10), random.randint(-10, 10))

            ofile.write(str(stimCirc.pos) + " " + str(gloclock) + "\n")

            counter = 0
            gloclock = 0
            trial = trial + 1
            dfile.close()
    else:
        counter = 0
    stimCirc.draw()
    mouse_dX, mouse_dY = myMouse.getRel()
    fixSpot.setPos(myMouse.getPos())

    # get rid of other, unprocessed events
    event.clearEvents()

    fixSpot.draw()
    mywin.flip()

# cleanup
#ifile.close()
ofile.close()
mywin.close()
core.quit()