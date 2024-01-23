
import os
import torch
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Dataset:
    
    filelist = []
    video_dir = 'datasets/mp4/'
    json_dir = 'datasets/data/'
    label_dir = 'datasets/cfg/'
    list_file = 'datafiles.txt'

    # 定义可以扩写：可以改变地址
    def __init__(self):
        pass

    
    def get_filelist(self, list_file=''):
        if list_file == '':
            return filelist
        else:
            with open(list_file,'r') as file:
                line = file.readline()
                filelist = line.split(',')
            return filelist
        
    def cap_frames(filename):
        cap = cv.VideoCapture(self.video_dir + filename +'.mp4')
        
        motion = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            motion.append(img)
            
        cap.release()


    def load_data(list_file):
        Dataset.get_filelist(list_file)
        
        