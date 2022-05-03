import numpy as np
import os,sys
import torch
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F

def TP(q, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l

class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()

class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet_pretrained_features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,3):
            self.stage1.add_module(str(x), resnet_pretrained_features[x])

        for x in range(3,5):
            self.stage2.add_module(str(x), resnet_pretrained_features[x])

        self.stage3.add_module(str(5), resnet_pretrained_features[5])

        self.stage4.add_module(str(6), resnet_pretrained_features[6])

        self.stage5.add_module(str(7), resnet_pretrained_features[7])

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [3,64,256,512,1024,2048]

        self.quality = self.quality_regression(sum(self.chns)*2,128,1)
        
    def forward_once(self, h):
        # h = (x-self.mean)/self.std
        x = h*self.std + self.mean
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def quality_regression(self,in_channels,middle_channels,out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x, y, require_grad=False, batch_average=False):
        x_size = x.shape
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        y = y.view(-1, x_size[2], x_size[3], x_size[4])

        feats0 = self.forward_once(x)
        feats1 = self.forward_once(y) 

        c1 = 1e-6
        c2 = 1e-6

        S = []

        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2,3], keepdim=True)
            y_mean = feats1[k].mean([2,3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            S.append(S1)

            x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            S.append(S2)

        feats = torch.cat(S, dim = 1).squeeze()
        qs = self.quality(feats)
        qs = qs.view(x_size[0],x_size[1])

        score = torch.zeros(x_size[0], device=qs.device)  #
        for i in range(x_size[0]):  #
            qi = qs[i, :x_size[1]].unsqueeze(-1)
            # print(qi.shape)
            qi = TP(qi)
            score[i] = torch.mean(qi)  # video overall quality
            
        return score

