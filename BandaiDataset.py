import os
import torch
import cv2 as cv
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Motion:
    pose_list = []
    label = -1
    frame_num = -1

    def __init__(self):
        pass

    # aquire pose images from vedio using OpenCV
    def cap_frames(self,video_dir,json_dir,filename):
        
        if filename!='':
            self.read_label(json_dir+filename+".json")

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
        self.frame_num = len(self.pose_list)

        return self
    
    # draw certain pose in motion
    def draw_pose(self, i):
        plt.imshow(self.pose_list[i],cmap='gray')

    def get_pose(self, i):
        return self.pose_list[i]
    
    def read_label(self,label_path):
        with open(label_path) as f:
            jsondata = json.loads(f.readline())
        self.label = jsondata['content']

    def adjust(self,frame):
        if self.frame_num < frame:
            pose = self.pose_list[-1]
            for i in range(0, frame - self.frame_num):
                self.pose_list.append(pose)
            self.frame_num = frame



    

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
            motion = Motion().cap_frames(self.video_dir,self.json_dir,filename)
            self.motion_list.append(motion)
            self.num_of_files += 1

    
    def get_filenames(self, filepath=''):
        if filepath == '':
            filepath = self.list_file

        if self.filelist == []:
            # get all vedio filename
            content = []
            with open(filepath,'r') as file:
                
                line = file.read()
                while line:
                    content.append(line)
                    line = file.read()
                self.filelist = content[0].split('\n')
        
        return self.filelist
    
    def normalize(self):
        max_frame = -1
        for motion in self.motion_list:
            max_frame = max(motion.frame_number,max_frame)
        
        for i in range(0,self.motion_list):
            motion.adjust(max_frame)
            
        


        
        