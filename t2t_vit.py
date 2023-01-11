# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
import torch
import torch.nn as nn
from Res import resnet18
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from token_transformer import Token_transformer
from token_performer import Token_performer
from transformer_block import Block, get_sinusoid_encoding
import torch.nn.functional as F
from timm.models import load_checkpoint


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
}


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            # self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            # self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(dim=in_chans*7*7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # x [B, 56*56, 147=7*7*3]
        # iteration1: restricturization/reconstruction
        x_1_4 = self.attention1(x)
        B, new_HW, C = x_1_4.shape
        x = x_1_4.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: restricturization/reconstruction
        x_1_8 = self.attention2(x)
        B, new_HW, C = x_1_8.shape
        x = x_1_8.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x, x_1_8, x_1_4


class T2T_ViT(nn.Module):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, x_1_8, x_1_4 = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # T2T-ViT backbone
        cnt = 0
        out = []
        for blk in self.blocks:
            x = blk(x)
            cnt = cnt+1
            if cnt == len(self.blocks)//2:
                out.append(x)

        #x = self.norm(x)
        # return x[:, 0]
        return out[0],x[:, 1:, :], x_1_8, x_1_4

    def forward(self, x):
        b = x.shape[0]
        xout, x, x_1_8, x_1_4 = self.forward_features(x)
        # x = self.head(x)
        xout = x.view(b, int((xout.shape[1])**0.5),int((xout.shape[1])**(0.5)), -1).permute(0, 3, 1, 2).contiguous()
        x = x.view(b, int((x.shape[1])**0.5),int((x.shape[1])**(0.5)), -1).permute(0, 3, 1, 2).contiguous()
        x_1_4 = x_1_4.view(b, int((x_1_4.shape[1])**0.5),int((x_1_4.shape[1])**(0.5)), -1).permute(0, 3, 1, 2).contiguous()
        x_1_8 = x_1_8.view(b, int((x_1_8.shape[1]) ** 0.5), int((x_1_8.shape[1]) ** (0.5)), -1).permute(0, 3, 1, 2).contiguous()
        return x,xout, x_1_8, x_1_4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

        self.squeeze4 = nn.Sequential(nn.Conv2d(384, 64,kernel_size=1,stride=1,padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True)) # 7*7
        self.squeeze3 = nn.Sequential(nn.Conv2d(384, 64,kernel_size=1,stride=1,padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))  # 14*14
        self.squeeze2 = nn.Sequential(nn.Conv2d(64, 64,kernel_size=1,stride=1,padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))  # 28*28
        self.squeeze1 = nn.Sequential(nn.Conv2d(64, 64,kernel_size=1,stride=1,padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))  # 56*56

        self.sq4 = nn.Sequential(nn.Conv2d(512, 64,kernel_size=1,stride=1,padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True)) # 7*7
        self.sq3 = nn.Sequential(nn.Conv2d(256, 64,kernel_size=1,stride=1,padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True)) # 7*7
        self.sq2 = nn.Sequential(nn.Conv2d(128, 64,kernel_size=1,stride=1,padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))  # 14*14
        self.sq1 = nn.Sequential(nn.Conv2d(64, 64,kernel_size=1,stride=1,padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))  # 28*28

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,s1,s2,s3,s4,x1,x2,x3,x4):

        s4=self.squeeze4(s4)
        s3=self.squeeze3(s3)
        s2=self.squeeze2(s2)
        s1=self.squeeze1(s1)

        x4=self.sq4(x4)
        x3=self.sq3(x3)
        x2=self.sq2(x2)
        x1=self.sq1(x1)

        s4= F.interpolate(s4, size=x4.size()[2:],mode='bilinear',align_corners=True)+x4
        s3= F.interpolate(s3, size=x3.size()[2:],mode='bilinear',align_corners=True)+x3
        s2= F.interpolate(s2, size=x2.size()[2:],mode='bilinear',align_corners=True)+x2
        s1= F.interpolate(s1, size=x1.size()[2:],mode='bilinear',align_corners=True)+x1

        s4=self.relu(self.bn1(self.conv1(s4)))
        s4= F.interpolate(s4, size=s3.size()[2:],mode='bilinear',align_corners=True)

        s3=self.relu(self.bn2(self.conv2(s4+s3)))
        s3= F.interpolate(s3, size=s2.size()[2:],mode='bilinear',align_corners=True)

        s2=self.relu(self.bn3(self.conv3(s3+s2)))
        s2= F.interpolate(s2, size=s1.size()[2:],mode='bilinear',align_corners=True)

        s1=self.relu(self.bn4(self.conv4(s2+s1)))
        return s1

class BBNet(nn.Module):
    def __init__(self, cfg=None):
        super(BBNet, self).__init__()
        self.cfg      = cfg
        self.bkbone   = T2T_ViT(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.)
        self.resnet    = resnet18()
        load_checkpoint(self.bkbone, '../model/80.7_T2T_ViT_t_14.pth.tar', use_ema=True)
        self.resnet.load_state_dict(torch.load('../model/resnet18.pth'),strict=False)
        self.decode   = Decoder()
        self.linear1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_88 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_80 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_72 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_64 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr1_56 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x,shape=None):
        shape = x.size()[2:] if shape is None else shape
        y = F.interpolate(x, size=(224,224), mode='bilinear',align_corners=True)
        x1,x2,x3,x4 = self.resnet(x)
        s4,s3,s2,s1 = self.bkbone(y)
        pred1 = self.decode(s1,s2,s3,s4,x1,x2,x3,x4)
        if self.cfg.mode == 'train':
            if pred1.shape[2] == 176:
                pred1 = F.interpolate(self.linearr1_88(pred1), size=shape, mode='bilinear')
            if pred1.shape[2] == 160:
                pred1 = F.interpolate(self.linearr1_80(pred1), size=shape, mode='bilinear')
            if pred1.shape[2] == 144:
                pred1 = F.interpolate(self.linearr1_72(pred1), size=shape, mode='bilinear')
            if pred1.shape[2] == 128:
                pred1 = F.interpolate(self.linearr1_64(pred1), size=shape, mode='bilinear')
            if pred1.shape[2] == 112:
                pred1 = F.interpolate(self.linearr1_56(pred1), size=shape, mode='bilinear')
        else:
            pred_88 = F.interpolate(pred1, size=[176, 176], mode='bilinear')
            pred_88 = F.interpolate(self.linearr1_88(pred_88), size=shape, mode='bilinear')
            pred_80 = F.interpolate(pred1, size=[160, 160], mode='bilinear')
            pred_80 = F.interpolate(self.linearr1_80(pred_80), size=shape, mode='bilinear')
            pred_72 = F.interpolate(pred1, size=[144, 144], mode='bilinear')
            pred_72 = F.interpolate(self.linearr1_72(pred_72), size=shape, mode='bilinear')
            pred_64 = F.interpolate(pred1, size=[128, 128], mode='bilinear')
            pred_64 = F.interpolate(self.linearr1_64(pred_64), size=shape, mode='bilinear')
            pred_56 = F.interpolate(pred1, size=[112, 112], mode='bilinear')
            pred_56 = F.interpolate(self.linearr1_56(pred_56), size=shape, mode='bilinear')
            pred1 = 1*pred_88 + 0.25*pred_80 + 0.25*pred_72 + 0.25*pred_64 + 0.25*pred_56
        return pred1
