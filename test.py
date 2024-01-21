import json
import os
import time
import subprocess
import threading
import numpy as np
import pyautogui
import pydirectinput
import math
import cv2





def trypydirectinput():
    time.sleep(5)
    pydirectinput.moveTo(200, 300)
    pydirectinput.click()
    pydirectinput.doubleClick()
    pydirectinput.keyDown('a')

if __name__ == "__main__":
    t = threading.Thread(target=trypydirectinput,args=())
    t.start()