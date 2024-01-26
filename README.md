# Motion Classification Project

## Abstract
In this repository, I used Pytorch to implement a CNN network and a to classify video-based character motion classifier. After appropriate tuning, the model is capable of effectively distinguishing between different types of actions. Furthermore, I trained the dataset using the ResNet model provided in the class with the same set of parameters for comparison. This project utilized the BandaiMotion dataset, with the data being subjected to visualization and filtering. Based on this dataset, I designed a well-structured structure to store action data.

## 1. Introduction
This project is an assignment for software engineering; thus, the focus is primarily on implementing a complete AI training workflow by using a series of tools like PyTorch. In addition to achieving good training results, I have made efforts to ensure that my program possesses good scalability.

In this project, implementing the training process includes: data preparation, data processing, model setup, training, testing, validation, unit testing, deployment, checkpointing, and logging.

## 2 Data
### 2.1 Dataset description
 BandaiMotion dataset offers BVH data based on skeletons, which is a data structure relying on temporal offsets for joints. This significantly reduces both the space occupied by the data and the time required to download it.

A typical BVH data consists of two parts: the skeleton structure and the temporal offset records for joints.However, the BandaiMotion dataset provides a Python script based on Blender to facilitate the conversion of BVH to MP4 functionality.This enables us to visualize the changes in actions more intuitively. At the same time, it cleverly categorizes its action data into 17 classes, such as Walk and Run, making classification possible.

### 2.2 Data preprocessing
Firstly, utilize the Python script provided by Bandai to convert BVH data into MP4 video data using Blender.

[]

And then, check the labels and the corresponding labels for actions, filtering out data classes with fewer items.Before filtering, there were 177 motion data points with 17 content labels. After filtering, there are now 165 motion data points and 10 content labels.
[]

### 2.3 BandaiDataset Class

BandaiDataset inherits from torch.utils.data.Dataset.The two necessary functions are: ```__getitem__``` and ```__len__``` These two functions respectively return the length of the dataset's dataframe and the data of a single dataframe. Therefore, the next step is to define a type for storing the dataframe.

### 2.4 Motion Class

'Motion' class is defined for storing motiondata,In this class.

'input_motion' function uses OpenCV to input MP3 data. This data is stored as a list of pictures. This data is stored as a list of grayscale images named ```pose_list[]```. To show the image using plt, just simply use ```draw_pose``` function in the class

In order to adapt to the model training, the function ```adjust``` works to process ```pose_list``` of different lengths into the same number of frames.


```

```




## 3. Models

## 4. Train


## 5. Results & Discussion

## Reference