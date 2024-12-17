import cv2
from picamera2 import Picamera2,Preview
from ultralytics import YOLO
import sys
import os
picdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pic')
libdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)

import logging    
import time
import traceback
from waveshare_OLED import OLED_1in51
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import subprocess
from dsp import bright_contrast
import RPi.GPIO as GPIO
import threading as th
import numpy as np
import argostranslate.package
import argostranslate.translate

logging.basicConfig(level=logging.DEBUG)
# Initialize the Picamera2


picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
# picam2.start_preview(Preview.QTGL)
picam2.start()


# Load the YOLO11 model
model = YOLO("yolo11n.pt")
disp = OLED_1in51.OLED_1in51()

# Initialize library.
disp.Init()
disp.clear()

# Initialize GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(29, GPIO.OUT)
GPIO.setup(31, GPIO.IN)
GPIO.output(29, GPIO.HIGH)

state = ""

if GPIO.input(31):
    state = "Chinese"
else: 
    state = "English"
    
def doNothing():
    return 0

def FSM():
    
    global state
    
    while True:
        
        print(f"===== New state: {state} =====")
        
        if state == "English":
            GPIO.wait_for_edge(31, GPIO.RISING)
            state = "Chinese_hold"
            th.Thread(target=makeDisplay, args=(), daemon=True).start()
            
        elif state == "Chinese_hold":
            p = th.Timer(0.5, doNothing, args=())
            p.start()
            p.join()
            if GPIO.input(31) == 0: 
                state = "English"
                th.Thread(target=makeDisplay, args=(), daemon=True).start()
            else:
                state = "Chinese"
            
            
        elif state == "Chinese":
            GPIO.wait_for_edge(31, GPIO.FALLING)
            state = "English_hold"
            th.Thread(target=makeDisplay, args=(), daemon=True).start()
            
        elif state == "English_hold":
            p = th.Timer(0.5, doNothing, args=())
            p.start()
            p.join()
            if GPIO.input(31) == 0: 
                state = "English"
            else:
                state = "Chinese"
                th.Thread(target=makeDisplay, args=(), daemon=True).start()
        
        
current_object = []

disp_lock = th.Lock()

def makeDisplay():
    # Create blank image for drawing.
    font1 = ImageFont.truetype('/home/pi/yolov5/Font.ttc', 18)
    image1 = Image.new('1', (disp.width, disp.height), "WHITE")
    draw = ImageDraw.Draw(image1)
    draw.line([(0,0),(127,0)], fill = 0)
    draw.line([(0,0),(0,63)], fill = 0)
    draw.line([(0,63),(127,63)], fill = 0)
    draw.line([(127,0),(127,63)], fill = 0)
    
    if state in ["English", "English_hold"]:
        for i in range(min(3,len(current_object))):
           logging.info(current_object[i])
           draw.text((20,i*20), current_object[i], font = font1, fill = 0)
    elif state in ["Chinese", "Chinese_hold"]:
        for i in range(min(3,len(current_object))):
           logging.info(current_object[i])
           draw.text((20,i*20), translate_to_chinese(current_object[i]), font = font1, fill = 0)
    image1 = image1.rotate(180)
    
    disp_lock.acquire()
    disp.clear()
    disp.ShowImage(disp.getbuffer(image1))
    disp_lock.release()

en_to_zh = dict()
def translate_to_chinese(eng_str):
    if eng_str not in en_to_zh.keys():
        en_to_zh[eng_str] = argostranslate.translate.translate(eng_str, 'en', 'zh')
    return en_to_zh[eng_str]

            

count = 0

try: 
    th.Thread(target=FSM, args=(), daemon=True).start()
    while True:
        # Capture frame-by-frame
        frame = picam2.capture_array() # frame = cv2.imread("sequence-1.jpg") 
        frame = np.rot90(frame, k=3)
        frame = bright_contrast(frame, brightness=0, contrast=2.0)
        # Run YOLO11 inference on the frame
        results = model(frame)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the resulting frame
        
        detect_objects = []
        for result in results:
            for box in result.boxes:
                class_id=int(box.cls[0])
                name = result.names[class_id]
                detect_objects.append(f"{name}")
                    
        current_object = detect_objects
        th.Thread(target=makeDisplay, args=(), daemon=True).start()

        for i in range(min(6,len(detect_objects))):
            box = results[0].boxes[i]
            data = box.data.tolist()[0]
            accuracy = data[4]
            label = result.names[int(box.cls[0])]
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            y = ymin-15 if ymin - 15 > 15 else ymin + 15
            cv2.putText(frame, "{} {:.2f}".format(label, float(accuracy)), (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        
        
        cv2.imwrite("temp.jpg", frame)
        display_process = subprocess.Popen(['python', 'imviewer.py', str(count)])
        count += 1
        
        
        
        time.sleep(1)

except KeyboardInterrupt:
	print("done")
	GPIO.cleanup()

# Release resources and close windows    source .venv/bin/activate
cv2.destroyAllWindows()

