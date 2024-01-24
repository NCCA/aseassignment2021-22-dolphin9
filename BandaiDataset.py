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
from PIL import Image

class Motion:


    def __init__(self):
        self.pose_list = []
        self.label = -1
        self.frame_num = 0
        

    # aquire pose images from vedio using OpenCV
    def input_motion(self,video_dir,filename, json_dir = '',):
        
        if filename == '':
            return False
               
        self.read_label(json_dir+filename+".json")

        cap = cv.VideoCapture(video_dir + filename +'.mp4')
        self.pose_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                #print(filename + ": stream end")
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

    ##### here , transfer motions in to same frames.
    def adjust(self, frame_lenth):  
        if len(self.pose_list) == frame_lenth:
            return

        new_pose_list = copy.deepcopy(self.pose_list)
        less_flag = False
        if len(new_pose_list)< frame_lenth:
            new_pose_list.extend(copy.deepcopy(self.pose_list))
            less_flag = True
            while len(new_pose_list) < frame_lenth:
                new_pose_list.extend(copy.deepcopy(self.pose_list))
        
        if len(new_pose_list) > frame_lenth:
            mid_frame = len(new_pose_list)//2
            start_frame = mid_frame - (frame_lenth//2)
            end_frame = start_frame + frame_lenth
            new_pose_list = copy.deepcopy(new_pose_list[start_frame:end_frame])

        #print(len(new_pose_list))
        #print(frame_lenth)
        if frame_lenth == len(new_pose_list):
            self.pose_list = copy.deepcopy(new_pose_list)
            self.frame_num = frame_lenth
        else:
            print('Err: len(new_pose_lise) is not equal to frame lenth!')



    
    def get_motion_tensor(self, set_frame):
        if self.frame_num != set_frame:
            self.adjust(frame_lenth=set_frame)


        ######################
        motion = np.array(self.pose_list)
        motion_tensor = torch.from_numpy(motion)
        return motion_tensor
    
    def write_vedio(self,filename):
        fourcc = cv.VideoWriter_fourcc(*'MP4V')
        out = cv.VideoWriter(filename, fourcc, 20.0, (640,480))
        for i in range(0,self.frame_num):
            out.write(self.pose_list[i])
        out.release()



    

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
        for filename in self.filelist:
            #
            motion = Motion()
            flag = motion.input_motion(video_dir=self.video_dir,json_dir=self.json_dir,filename=filename)
            if flag:
                self.motion_list.append(copy.deepcopy(motion))
                #print(self.motion_list[-1].frame_num)
            

    
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
 
                self.filelist = list(filter(None,content[0].split('\n')))
                self.num_of_files = len(self.filelist)
        
        return self.filelist
    
    def preprocessing(self):
        for motion in self.motion_list:
            self.max_frame = max(motion.frame_num,self.max_frame)
            self.min_frame = min(motion.frame_num,self.min_frame)
        
        #for i in range(0,len(self.motion_list)):
        #    self.motion_list[i].adjust(self.max_frame)
    
        


        
        