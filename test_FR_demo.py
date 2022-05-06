# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch
import torch.nn
import UGCVQA_FR_model
import cv2
from PIL import Image

from torchvision import transforms

def video_processing(ref, dist):
    video = {}
    video_type = ['ref', 'dist']
    
    
    for i_type in video_type:
        if i_type == 'ref':
            video_name = ref
        else:
            video_name = dist
            video_name_dis = video_name

        video_capture = cv2.VideoCapture()
        video_capture.open(video_name)
        cap=cv2.VideoCapture(video_name)

        video_channel = 3

        video_height_crop = 448
        video_width_crop = 448

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    
        video_length_read = int(video_length/video_frame_rate)

        transformations = transforms.Compose([transforms.Resize(520),transforms.CenterCrop(448),transforms.ToTensor(),\
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

        transformed_video = torch.zeros([video_length_read, video_channel,  video_height_crop, video_width_crop])

        video_read_index = 0
        frame_idx = 0
                
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:

                # key frame
                if (video_read_index < video_length_read) and (frame_idx % video_frame_rate == 0):

                    read_frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    read_frame = transformations(read_frame)
                    transformed_video[video_read_index] = read_frame
                    video_read_index += 1

                frame_idx += 1

        if video_read_index < video_length_read:
            for i in range(video_read_index, video_length_read):
                transformed_video[i] = transformed_video[video_read_index - 1]
 
        video_capture.release()
        video[i_type] = transformed_video


    return video['ref'], video['dist'], video_name_dis










def video_processing_multi_scale(ref, dist):
    video1 = {}
    video2 = {}
    video3 = {}
    video_type = ['ref', 'dist']
    
    
    for i_type in video_type:
        if i_type == 'ref':
            video_name = ref
        else:
            video_name = dist
            video_name_dis = video_name

        video_capture = cv2.VideoCapture()
        video_capture.open(video_name)
        cap=cv2.VideoCapture(video_name)


        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if video_height > video_width:
            video_width_resize = 540
            video_height_resize = int(video_width_resize/video_width*video_height)
        else:
            video_height_resize = 540
            video_width_resize = int(video_height_resize/video_height*video_width)

        dim1 = (video_height_resize, video_width_resize)

        if video_height > video_width:
            video_width_resize = 720
            video_height_resize = int(video_width_resize/video_width*video_height)
        else:
            video_height_resize = 720
            video_width_resize = int(video_height_resize/video_height*video_width)

        dim2 = (video_height_resize, video_width_resize)


        if video_height > video_width:
            video_width_resize = 1080
            video_height_resize = int(video_width_resize/video_width*video_height)
        else:
            video_height_resize = 1080
            video_width_resize = int(video_height_resize/video_height*video_width)

        dim3 = (video_height_resize, video_width_resize)



        video_channel = 3

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    
        video_length_read = int(video_length/video_frame_rate)

        transformed_video1 = torch.zeros([video_length_read, video_channel, dim1[0], dim1[1]])
        transformed_video2 = torch.zeros([video_length_read, video_channel, dim2[0], dim2[1]])
        transformed_video3 = torch.zeros([video_length_read, video_channel, dim3[0], dim3[1]])

        transformations1 = transforms.Compose([transforms.Resize(dim1), transforms.ToTensor(),\
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        transformations2 = transforms.Compose([transforms.Resize(dim2), transforms.ToTensor(),\
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        transformations3 = transforms.Compose([transforms.Resize(dim3), transforms.ToTensor(),\
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])        

        video_read_index = 0
        frame_idx = 0
                
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:

                # key frame
                if (video_read_index < video_length_read) and (frame_idx % video_frame_rate == 0):

                    read_frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

                    read_frame1 = transformations1(read_frame)
                    transformed_video1[video_read_index] = read_frame1

                    read_frame2 = transformations2(read_frame)
                    transformed_video2[video_read_index] = read_frame2

                    read_frame3 = transformations3(read_frame)
                    transformed_video3[video_read_index] = read_frame3
                    video_read_index += 1

                frame_idx += 1

        if video_read_index < video_length_read:
            for i in range(video_read_index, video_length_read):
                transformed_video1[i] = transformed_video1[video_read_index - 1]
                transformed_video2[i] = transformed_video2[video_read_index - 1]
                transformed_video3[i] = transformed_video3[video_read_index - 1]
 
        video_capture.release()
        video1[i_type] = transformed_video1
        video2[i_type] = transformed_video2
        video3[i_type] = transformed_video3


    return video1['ref'], video1['dist'], video2['ref'], video2['dist'], video3['ref'], video3['dist'], video_name_dis


def main(config):

    device = torch.device('cuda' if config.is_gpu else 'cpu')
    print('using ' + str(device))

    model = UGCVQA_FR_model.ResNet50()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('ckpts/UGCVQA_FR_model.pth', map_location=device))
    
    if config.method_name == 'single-scale':
        print('using the single scale method')
        video_ref, video_dist, video_name = video_processing(config.ref, config.dist)

        with torch.no_grad():
            model.eval()

            video_ref = video_ref.to(device)
            video_dist = video_dist.to(device)

            video_ref = video_ref.unsqueeze(dim=0)
            video_dist = video_dist.unsqueeze(dim=0)

            outputs = model(video_ref, video_dist)
            
            y_val = outputs.item()

            print('The video name: ' + video_name)
            print('The quality socre: {:.4f}'.format(y_val))

    if config.method_name == 'multi-scale':
        print('using the multi scale method')
        video_ref1, video_dist1, video_ref2, video_dist2, video_ref3, video_dist3, video_name = video_processing_multi_scale(config.ref, config.dist)

        with torch.no_grad():
            model.eval()

            video_ref1 = video_ref1.to(device)
            video_dist1 = video_dist1.to(device)

            video_ref1 = video_ref1.unsqueeze(dim=0)
            video_dist1 = video_dist1.unsqueeze(dim=0)

            outputs1 = model(video_ref1, video_dist1)
            
            y_val1 = outputs1.item()

            video_ref2 = video_ref2.to(device)
            video_dist2 = video_dist2.to(device)

            video_ref2 = video_ref2.unsqueeze(dim=0)
            video_dist2 = video_dist2.unsqueeze(dim=0)

            outputs2 = model(video_ref2, video_dist2)
            
            y_val2 = outputs2.item()

            video_ref3 = video_ref3.to(device)
            video_dist3 = video_dist3.to(device)

            video_ref3 = video_ref3.unsqueeze(dim=0)
            video_dist3 = video_dist3.unsqueeze(dim=0)

            outputs3 = model(video_ref3, video_dist3)
            
            y_val3 = outputs3.item()

            w1_csf = 0.8317
            w2_csf = 0.0939
            w3_csf = 0.0745

            y_val = pow(y_val1, w1_csf) * pow(y_val2, w2_csf) * pow(y_val3, w3_csf)

            print('The video name: ' + video_name)
            print('The quality socre: {:.4f}'.format(y_val))



    output_name = config.output

    if not os.path.exists(output_name):
        os.system(r"touch {}".format(output_name))

    f = open(output_name,'w')
    f.write(video_name)
    f.write(',')
    f.write(str(y_val))
    f.write('\n')

    f.close()


        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--method_name', type=str, default='single-scale')
    parser.add_argument('--ref', type=str, default='videos/UGC0034_1280x720_30_crf_00.mp4')
    parser.add_argument('--dist', type=str, default='videos/UGC0034_1280x720_30_crf_22.mp4')
    parser.add_argument('--output', type=str, default='result.txt')
    parser.add_argument('--is_gpu', action='store_true')
  

    config = parser.parse_args()

    main(config)