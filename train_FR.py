# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import VideoDataset_FR
import UGCVQA_FR_model

from torchvision import transforms

import time

from scipy import stats
from scipy.optimize import curve_fit

import time

# import GPUtil

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic


def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UGCVQA_FR_model.ResNet50()

    if config.multi_gpu:
        model = torch.nn.DataParallel(model)
        model = model.to(device)
    else:
        model = model.to(device)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = config.conv_base_lr, weight_decay = 0.0000001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
    criterion = nn.MSELoss().to(device)

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    videos_dir = config.videos_dir
    datainfo = config.datainfo

    transformations_train = transforms.Compose([transforms.Resize(520), transforms.RandomCrop(448), transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    transformations_test = transforms.Compose([transforms.Resize(520),transforms.CenterCrop(448),transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    trainset = VideoDataset_FR(videos_dir, datainfo, transformations_train, 448, is_train = True)
    testset = VideoDataset_FR(videos_dir, datainfo, transformations_test, 448, is_train = False)

    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)


    best_test_criterion = -1  # SROCC min
    best_test = []


    print('Starting training:')

    old_save_name = None

    for epoch in range(config.epochs):
        
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        for i, (video_ref, video_dis, dmos, _) in enumerate(train_loader):
            

            video_ref = video_ref.to(device)
            video_dis = video_dis.to(device)

            labels = dmos.to(device).float()
            
            outputs = model(video_ref, video_dis)
            optimizer.zero_grad()
            
            loss = criterion(labels, outputs)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            loss.backward()
            
            optimizer.step()

            if (i+1) % (config.print_samples//config.train_batch_size) == 0:

                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples//config.train_batch_size)
                print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                    (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, \
                        avg_loss_epoch))
                batch_losses_each_disp = []
                print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                session_start_time = time.time()

        avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        scheduler.step()
        lr = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr[0]))

        with torch.no_grad():
            model.eval()
            label = np.zeros([len(testset)])
            y_output = np.zeros([len(testset)])
            for i, (video_ref, video_dis, dmos, _) in enumerate(test_loader):
                
                video_ref = video_ref.to(device)
                video_dis = video_dis.to(device)

                label[i] = dmos.item()
                outputs = model(video_ref, video_dis)

                y_output[i] = outputs.item()
            
            y_output_logistic = fit_function(label, y_output)
            test_PLCC = stats.pearsonr(y_output_logistic, label)[0]
            test_SRCC = stats.spearmanr(y_output, label)[0]
            test_KRCC = stats.stats.kendalltau(y_output, label)[0]
            test_RMSE = np.sqrt(((y_output_logistic-label) ** 2).mean())
            
            print('Epoch {} completed. The result on the test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(epoch + 1, \
                test_SRCC, test_KRCC, test_PLCC, test_RMSE))


            if test_SRCC > best_test_criterion:
                print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                best_test_criterion = test_SRCC
                best_test = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]
                print('Saving model...')
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)
                if epoch > 0:
                    if os.path.exists(old_save_name):
                        os.remove(old_save_name)
                save_model_name = os.path.join(config.ckpt_path, config.model_name + '_' + config.database + '_NR_v'+ str(config.exp_version) + '_epoch_%d_SRCC_%f.pth' % (epoch + 1, test_SRCC))
                torch.save(model.state_dict(), save_model_name)
                old_save_name = save_model_name


    print('Training completed.')
    print('The best training result on the test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
        best_test[0], best_test[1], best_test[2], best_test[3]))

    np.save(os.path.join(config.results_path, config.model_name + '_' + config.database + '_FR_v'+ str(config.exp_version)), best_test)

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)

    parser.add_argument('--conv_base_lr', type=float)
    parser.add_argument('--datainfo', type=str, default='json_files/ugcset_dmos.json')
    parser.add_argument('--videos_dir', type=str)
    parser.add_argument('--decay_ratio', type=float)
    parser.add_argument('--decay_interval', type=int)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int)
    parser.add_argument('--print_samples', type=int)
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--epochs', type=int)

    # misc
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--reults_path', type=str)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)

    config = parser.parse_args()

    main(config)