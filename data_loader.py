# -*- coding: utf-8
import os

import pandas as pd
from PIL import Image

from torchvision import transforms, models

import torch
from torch.utils import data
import cv2
import numpy as np
import random
import json
from torchvision import transforms


class VideoDataset_NR(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, json_path, transform, size, is_train):
        super(VideoDataset_NR, self).__init__()
        with open(json_path, 'r') as f:
            mos_file_content = json.loads(f.read())
            if is_train:
                self.video_names = mos_file_content['train']['dis']
                self.score = mos_file_content['train']['mos']
            else:
                self.video_names = mos_file_content['test']['dis']
                self.score = mos_file_content['test']['mos']

        self.videos_dir = data_dir
        self.transform = transform
        self.size = size
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_score = torch.FloatTensor(np.array(float(self.score[idx])))
        filename=os.path.join(self.videos_dir, video_name.replace('.yuv', '.mp4'))

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap=cv2.VideoCapture(filename)

        video_channel = 3

        video_height_crop = self.size
        video_width_crop = self.size

        # video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    
        video_length_read = int(video_length*2/video_frame_rate)

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
                
        video_read_index = 0
        frame_idx = 0
                
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length_read) and ((frame_idx*2) % video_frame_rate == 0):

                    read_frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    read_frame = self.transform(read_frame)
                    transformed_video[video_read_index] = read_frame
                    video_read_index += 1

                frame_idx += 1

        if video_read_index < video_length_read:
            for i in range(video_read_index, video_length_read):
                transformed_video[i] = transformed_video[video_read_index - 1]

        video_capture.release()

        return transformed_video, video_score, video_name

class VideoDataset_FR(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, json_path, transform, size ,is_train):
        super(VideoDataset_FR, self).__init__()

        with open(json_path, 'r') as f:
            mos_file_content = json.loads(f.read())
            if is_train:
                self.video_names_ref = mos_file_content['train']['ref']
                self.video_names_dis = mos_file_content['train']['dis']
                self.score = mos_file_content['train']['mos']
            else:
                self.video_names_ref = mos_file_content['test']['ref']
                self.video_names_dis = mos_file_content['test']['dis']
                self.score = mos_file_content['test']['mos']

        self.videos_dir = data_dir
        self.transform = transform
        self.size = size
        self.length = len(self.score)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video = {}
        video_type = ['ref', 'dis']
        
        video_score = torch.FloatTensor(np.array(float(self.score[idx])))
        for i_type in video_type:
            if i_type == 'ref':
                video_name = self.video_names_ref[idx]
            else:
                video_name = self.video_names_dis[idx]
                video_name_dis = video_name

            video_score = torch.FloatTensor(np.array(float(self.score[idx])))

            filename=os.path.join(self.videos_dir, video_name.replace('.yuv', '.mp4'))

            video_capture = cv2.VideoCapture()
            video_capture.open(filename)
            cap=cv2.VideoCapture(filename)

            video_channel = 3

            video_height_crop = self.size
            video_width_crop = self.size

            # video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
        
            video_length_read = int(video_length*2/video_frame_rate)

            transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
                    
            video_read_index = 0
            frame_idx = 0
                    
            for i in range(video_length):
                has_frames, frame = video_capture.read()
                if has_frames:
                    # key frame
                    if (video_read_index < video_length_read) and ((frame_idx*2) % video_frame_rate == 0):

                        read_frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                        read_frame = self.transform(read_frame)
                        transformed_video[video_read_index] = read_frame
                        video_read_index += 1

                    frame_idx += 1

            if video_read_index < video_length_read:
                for i in range(video_read_index, video_length_read):
                    transformed_video[i] = transformed_video[video_read_index - 1]
    
            video_capture.release()
            video[i_type] = transformed_video

        return video['ref'], video['dis'], video_score, video_name_dis
