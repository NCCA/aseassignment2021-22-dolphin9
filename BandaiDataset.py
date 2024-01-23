
import os
import torch
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class BandaiDataset(Dataset):
    
    filelist = []
    video_dir = 'datasets/mp4/'
    json_dir = 'datasets/data/'
    label_dir = 'datasets/cfg/'
    list_file = 'datafiles.txt'

    # 定义可以扩写：可以改变地址
    def __init__(self, filepath):
        super().__init__()
        with open(filepath,'r') as file:
            line = file.readline()
            self.filelist = line.split(',')


    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, i):
        return self.filelist.iloc[i]
        
    def cap_frames(self,filename):

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
        return motion


    def load_data(list_file):
        BandaiDataset.get_filelist(list_file) 
        
        