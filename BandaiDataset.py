import os
import torch
import cv2 as cv
import numpy as np
import pandas as pd
import json
import copy
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Motion:


    def __init__(self):
        self.pose_list = []
        self.label = -1
        self.frame_num = 0

    # aquire pose images from vedio using OpenCV
    def input_motion(self,video_dir,json_dir,filename):
        
        if filename == '':
            return False
            
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
            self.frame_num += 1
            
        cap.release()

        return True
    
    def get_pose(self, i):
        return self.pose_list[i]
    
    # draw certain pose in motion
    def draw_pose(self, i):
        plt.imshow(self.pose_list[i],cmap='gray')

    def read_label(self,label_path):
        with open(label_path) as f:
            jsondata = json.loads(f.readline())
        self.label = jsondata['content']

    def adjust(self,max_frame):
        if self.frame_num < max_frame:
            pose = self.pose_list[-1]
            for i in range(0, max_frame - self.frame_num):
                self.pose_list.append(pose)
            self.frame_num = max_frame



    

class BandaiDataset(Dataset):
    
    filelist = []
    motion_list = []
    label_list = []
    num_of_files = 0
    max_frame = -1
    min_frame = 10000
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

        self.get_filenames()
        
                                    
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
            flag = motion.input_motion(self.video_dir,self.json_dir,filename)
            if flag:
                self.motion_list.append(copy.deepcopy(motion))
                print(self.motion_list[-1].frame_num)
                self.num_of_files += 1
                

    
    def get_filenames(self, filepath=''):
        if filepath == '':
            filepath = self.list_file
        else:
            self.filelist = []

        if self.filelist == []:
            # get all vedio filename
            content = []
            with open(filepath,'r') as file:
                
                line = file.read()
                while line:
                    content.append(line)
                    line = file.read()
                self.filelist = content[0].split('\n')
                self.num_of_files = len(self.filelist)
        
        return self.filelist
    
    def normalize(self):
        for motion in self.motion_list:
            self.max_frame = max(motion.frame_num,self.max_frame)
            self.min_frame = min(motion.frame_num,self.min_frame)
        
        for i in range(0,len(self.motion_list)):
            self.motion_list[i].adjust(self.max_frame)
            
        


        
        