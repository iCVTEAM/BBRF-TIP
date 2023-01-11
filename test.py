#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from BBNet import BBNet
import logging as logger
TAG = "DSSv1"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="test_%s.log"%(TAG), filemode="w")

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)
	return dn

def hsigmoid(x):
    x = 0.2*x+0.5
    x[x>1]=1
    x[x<0]=0
    return x

def clabel(x):
    bx = bbox(x)
    res = torch.rand(bx.shape[0],1).float().cuda()
    for i in range(bx.shape[0]):
        res[i] = ((bx[i][2]-bx[i][0])*(bx[i][3]-bx[i][1]))/(x.shape[2]*x.shape[3])
    res[res<0]=-res[res<0]
    return res

def select(p1,p2,p3,c):
   # if c[0][0]>=0.8:
   #     return p1
    return p1

class Test(object):
    def __init__(self, Dataset, Network, path, model):
        ## dataset
        self.model  = model
        self.cfg    = Dataset.Config(datapath=path, snapshot=model, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        ## network
        self.net    = Network(self.cfg)
        self.net.load_state_dict(torch.load(self.cfg.snapshot))
        self.net.train(False)
        self.net.cuda()

    def accuracy(self):
        with torch.no_grad():
            mae, fscore, cnt, number   = 0, 0, 0, 256
            mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)
            cost_time = 0
            for image, mask, (H, W), maskpath in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()

                start_time = time.perf_counter()
                p= self.net(image)       
                pred = torch.sigmoid(p[0,0])       
                torch.cuda.synchronize()
                end_time = time.perf_counter()

                cost_time += end_time - start_time
                ## MAE
                cnt += 1
                mae += (pred-mask).abs().mean()
                ## F-Score
                precision = torch.zeros(number)
                recall    = torch.zeros(number)
                for i in range(number):
                    temp         = (pred >= threshod[i]).float()
                    precision[i] = (temp*mask).sum()/(temp.sum()+1e-12)
                    recall[i]    = (temp*mask).sum()/(mask.sum()+1e-12)
                mean_pr += precision
                mean_re += recall
                fscore   = mean_pr*mean_re*(1+0.3)/(0.3*mean_pr+mean_re+1e-12)
            fps = len(self.loader.dataset) / cost_time
            msg = '%s MAE=%.6f, F-score=%.6f, len(imgs)=%s, fps=%.4f'%(self.cfg.datapath, mae/cnt, fscore.max()/cnt, len(self.loader.dataset), fps)
            print(msg)	
            logger.info(msg)
    def save(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                p = self.net(image, shape=shape)
                out   = torch.sigmoid(p[0,0])
                pred  = (out*255).cpu().numpy()
                head  = '../out/'+self.model+'/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #for path in ['../data/ECSSD']:
    for path in ['../data']:
	    for model in ['model-47']:
        	t = Test(dataset,BBNet, path,'../backbone/'+model)
       		t.save()
    #os.system('python ../eval/main.py')
