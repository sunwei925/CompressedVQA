# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch
import torch.nn
import UGCVQA_FR_model

from data_loader import VideoDataset_FR
from torchvision import transforms


from scipy import stats
import pandas as pd


def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UGCVQA_FR_model.ResNet50()

    model = model.to(device)

    # load the trained model
    print('loading the trained model')
    model.load_state_dict(torch.load(config.trained_model))
    

    if config.database == 'UGCCompressed':

        transformations_test = transforms.Compose([transforms.Resize(520),transforms.CenterCrop(448),transforms.ToTensor(),\
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])        

        datainfo_test = config.datainfo_test
        videos_dir_test = config.videos_dir_test

        
        valset = VideoDataset_FR(videos_dir_test, datainfo_test, transformations_test, is_train = False)

    ## dataloader

    val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)


    print('Starting testing:')

    with torch.no_grad():
        model.eval()
        label=np.zeros([len(valset)])
        y_val=np.zeros([len(valset)])


        videos_name = []
        for i, (video_ref, video_dis, dmos, video_name) in enumerate(val_loader):
            print(video_name[0])
            videos_name.append(video_name)
            video_ref = video_ref.to(device)
            video_dis = video_dis.to(device)

            label[i] = dmos.item()
            outputs = model(video_ref, video_dis)
            
            y_val[i] = outputs.item()
            print(y_val[i])
        
        val_PLCC = stats.pearsonr(y_val, label)[0]
        val_SRCC = stats.spearmanr(y_val, label)[0]
        val_KRCC = stats.stats.kendalltau(y_val, label)[0]
        val_RMSE = np.sqrt(((y_val-label) ** 2).mean())
        
        print('SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(val_SRCC, val_KRCC, val_PLCC, val_RMSE))


        output_name = config.output_name

        if not os.path.exists(output_name):
            os.system(r"touch {}".format(output_name))

        f = open(output_name,'w')
        for i in range(len(valset)):
            f.write(videos_name[i][0])
            f.write(',')
            f.write(str(y_val[i]))
            f.write('\n')

        f.close()


        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--datainfo_test', type=str, default='json_files/ugcset_dmos.json')
    parser.add_argument('--videos_dir_test', type=str, default=None)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--trained_model', type=str, default=None)
    parser.add_argument('--output_name', type=str, default='FR_output.txt')


    
    config = parser.parse_args()

    main(config)