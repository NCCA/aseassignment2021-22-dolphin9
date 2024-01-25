import os
import torch
import cv2 as cv
import numpy as np
import json
import copy
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

"""
There are two classes are contained in the file
Motion class is for the motion data, which is the dataframe for BandaiDataset

"""


class Motion:

    def __init__(self):
        self.pose_list = []
        self.label = -1
        self.frame_num = 0
        

    # aquire pose images from vedio using OpenCV
    def input_motion(self, video_dir:str, filename:str, json_dir = '',):
        """
        * notice : filename dosen't contain its suffix 
        This function is to read .mp4 files by OpenCV.VideoCaputre
        """
        
        if filename == '':
            return False
               
        self.input_label(json_dir+filename+".json")

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
    
    def get_pose(self, i:int) ->list:
        return self.pose_list[i]
    
    def draw_pose(self, i:int) -> None:
        plt.imshow(self.pose_list[i],cmap='gray')

    def input_label(self,label_path:str) -> None:

        with open(label_path) as f:
            jsondata = json.loads(f.readline())
        self.label = jsondata['content']

    def adjust(self, frame_lenth: int) -> None :
        """
            tansfer motion into certain frames
            Two strategies are used in this funcion
                1. for the motion which the number of frame is less than we expect
                    copy the whole sequence in the end 
                2. for the those which the number of frame is more than we expect
                    remain the mid part as they are more representative 
                 
        """
        if len(self.pose_list) == frame_lenth:
            return

        new_pose_list = copy.deepcopy(self.pose_list)
        if len(new_pose_list)< frame_lenth:
            new_pose_list.extend(copy.deepcopy(self.pose_list))
            while len(new_pose_list) < frame_lenth:
                new_pose_list.extend(copy.deepcopy(self.pose_list))
        
        if len(new_pose_list) > frame_lenth:
            mid_frame = len(new_pose_list)//2
            start_frame = mid_frame - (frame_lenth//2)
            end_frame = start_frame + frame_lenth
            new_pose_list = copy.deepcopy(new_pose_list[start_frame:end_frame])

        if frame_lenth == len(new_pose_list):
            self.pose_list = copy.deepcopy(new_pose_list)
            self.frame_num = frame_lenth
        else:
            print('Err: len(new_pose_lise) is not equal to frame lenth!')


    def get_motion_tensor(self, set_frame:int) -> torch.Tensor:

        if self.frame_num != set_frame:
            self.adjust(frame_lenth=set_frame)

        motion = np.array(self.pose_list)
        motion_tensor = torch.from_numpy(motion)
        return motion_tensor
    
    def write_vedio(self,filename:str):
        '''
            to be fixed!
        ''' 
        fourcc = cv.VideoWriter_fourcc('M','P','4','V')
        out = cv.VideoWriter(filename, fourcc, 20.0, (640,480))
        for i in range(0,self.frame_num):
            out.write(self.pose_list[i])
        out.release()



    

class BandaiDataset(Dataset):
    """
        BandaiDataset inherits from torch.utils.data.Dataset
    """

    motion_list = []

    filelist = []
    label_list = []

    num_of_files = 0
    max_frame = -1
    min_frame = 10000

    VIDEO_DIR = 'datasets/mp4/'
    JSON_DIR = 'datasets/data/'
    LABEL_DIR = 'datasets/cfg/'
    LIST_FILE = 'datafiles.txt'




    def __init__(self, filepath:str = LIST_FILE):
        """
            a filepath should be specfied to save the list of filename for data
            defult filepath is 'datafiles.txt' in the root directory
        """
        super().__init__()

        if filepath == '':
            filepath = self.LIST_FILE
        else:
            self.LIST_FILE = filepath

        self.get_filenames()
        
                                    
    def __len__(self):
        return self.num_of_files
    
    def __getitem__(self, i):
        return self.motion_list[i]
    
    def load(self, list_file:str = LIST_FILE):
        """
            load data 
        """

        self.get_filenames(list_file)
        for filename in self.filelist:
            motion = Motion()
            flag = motion.input_motion(video_dir=self.VIDEO_DIR,json_dir=self.JSON_DIR,filename=filename)
            if flag:
                self.motion_list.append(copy.deepcopy(motion))
            

    
    def get_filenames(self, filepath:str = LIST_FILE) -> list:

        if filepath == '' or filepath == self.LIST_FILE:
            pass
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
        '''
            is dropped
        '''
        for motion in self.motion_list:
            self.max_frame = max(motion.frame_num,self.max_frame)
            self.min_frame = min(motion.frame_num,self.min_frame)
        
        #for i in range(0,len(self.motion_list)):
        #    self.motion_list[i].adjust(self.max_frame)
    
        


        
        