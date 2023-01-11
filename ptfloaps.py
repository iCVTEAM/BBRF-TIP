import torch.nn as nn
import dataset
import torch
from ptflops import get_model_complexity_info
from Res import resnet18
from BBNet import BBNet
import os
cfg = dataset.Config(mode='test')

#model1 = SwinTransformer(560)

with torch.cuda.device(0):
  net = BBNet(cfg)
  flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
  print('Flops:  ' + flops)
  print('Params: ' + params)
