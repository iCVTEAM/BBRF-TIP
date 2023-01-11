 #!/usr/bin/python3
# coding=utf-8

import sys
import datetime
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from BBNet import BBNet
from apex import amp
import random
def validate(model, val_loader, nums):
    model.train(False)
    avg_mae = 0.0
    model.cfg.mode='test'
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            pred = model(image)
            pred = torch.sigmoid(pred[0, 0])
            avg_mae += torch.abs(pred - mask[0]).mean()

    model.train(True)
    model.cfg.mode='train'
    # print('validating MAE:', (avg_mae / nums).item())
    return (avg_mae / nums).item()

def structure_loss(pred, mask,weit=1):
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def bce_iou_loss(pred, mask):
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def train(Dataset, Network):

    # dataset
    cfg = Dataset.Config(datapath='../data/DUTS-TR', savepath='./res/res50-BBNet/',mode='train', batch=26,lr=0.04, momen=0.9,
                         decay=5e-4, epoch=64)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True,
                        num_workers=8)

    # val dataloader
    val_cfg1 = Dataset.Config(datapath='../data/ECSSD', mode='test')
    val_data1 = Dataset.Data(val_cfg1)
    val_loader1 = DataLoader(val_data1, batch_size=1, shuffle=False, num_workers=8)

    # val mse
    # min_mse1 = 1.0
    min_mse2 = 1.0
    best_epoch = cfg.epoch

    # network
    net = Network(cfg)
    net.train(True)

    # apex
    net.cuda()
    # device = torch.device('cuda')
    # net = nn.DataParallel(net)
    # net.to(device)

    # net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name or 'resnet' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    # sw = SummaryWriter(cfg.savepath)
    global_step = 0

    max_itr = cfg.epoch * len(loader)

    for epoch in range(cfg.epoch):
        if epoch < 32:
            optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1
            optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (32 + 1) * 2 - 1)) * cfg.lr
        else:
            if epoch%2 == 0:
                optimizer.param_groups[0]['lr'] = (1 - abs((30 + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1
                optimizer.param_groups[1]['lr'] = (1 - abs((30 + 1) / (32 + 1) * 2 - 1)) * cfg.lr
            else:
                optimizer.param_groups[0]['lr'] = (1 - abs((31 + 1) / (32 + 1) * 2 - 1)) * cfg.lr * 0.1
                optimizer.param_groups[1]['lr'] = (1 - abs((31 + 1) / (32 + 1) * 2 - 1)) * cfg.lr 

        for step, (image, mask) in enumerate(loader):
            image, mask = image.float().cuda(), mask.float().cuda()      	
            p ,p1,p2,p3,w = net(image,mask=mask)
            #p = net(image)
            loss = structure_loss(p, mask,w)+structure_loss(p1, mask,w)+structure_loss(p2, mask,w)+structure_loss(p3, mask,w)
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
               scale_loss.backward()
            optimizer.step()

            global_step += 1
            if step % 120 == 0:
                print('%s | step:%d/%d | lr=%.6f  loss1=%.6f' % (datetime.datetime.now(), epoch + 1, cfg.epoch, optimizer.param_groups[1]['lr'],loss.item()))

        if epoch >= 5:
            # mse1 = validate(net, val_loader1, 485)
            mse2 = validate(net, val_loader1, 1000)
            # mse3 = validate(net, val_loader3, 300)
            print('ECSSD:%s' % (mse2))
            if mse2 <= min_mse2:
                min_mse2 = mse2
                best_epoch = epoch + 1
            print('best epoch is ', best_epoch, min_mse2)
            if (epoch >= 29 and best_epoch == epoch + 1) or epoch == 31 or epoch == 63:
                torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))
            # torch.cuda.empty_cache()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    train(dataset, BBNet)
