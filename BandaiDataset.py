
import os
import torch
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Motion:
    pose_list = []

    def __init__(self):
        pass

    def cap_frames(self,video_dir,filename):

        cap = cv.VideoCapture(video_dir + filename +'.mp4')
        self.pose_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(filename + ": stream end")
                break
            img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            self.pose_list.append(img)
            
        cap.release()
        return self.pose_list




class BandaiDataset(Dataset):
    
    filelist = []
    motion_list = []
    label_list = []
    num_of_files = 0
    video_dir = 'datasets/mp4/'
    json_dir = 'datasets/data/'
    label_dir = 'datasets/cfg/'
    list_file = 'datafiles.txt'

    def __init__(self, filepath = ''):
        super().__init__()

        if filepath == '':
            filepath = self.list_file
        else:
            self.list_file = filepath
        
                                    
    def __len__(self):
        return self.num_of_files
    
    def __getitem__(self, i):
        return self.motion_list[i]
    
    def load(self):
        self.get_filenames(self.list_file)
        # count the number of files as cap frames from vedios
        self.num_of_files = 0
        for filename in self.filelist:
            motion = Motion()
            self.motion_list.append(motion.cap_frames(self.video_dir,filename))
            #self.label_list.append(pd.read_json(self.json_dir+filename))
            self.num_of_files += 1

    
    def get_filenames(self, filepath=''):
        if filepath == '':
            filepath = self.list_file

        if self.filelist == []:
            # get all vedio filename
            with open(filepath,'r') as file:
                line = file.readline()
                self.filelist = line.split(',')
        
        return self.filelist
        


        
        