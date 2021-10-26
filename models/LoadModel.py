import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels
from efficientnet import EfficientNet as EfficientNet_old
from config import pretrained_model

###引入的模型
from .pytorch_image_model.timm.models.nfnet import *
from .pytorch_image_model.timm.models.efficientnet import *
from .pytorch_image_model.timm.models.vision_transformer import *
from .pytorch_image_model.timm.models.resnest import *

import pdb


class MainModel(nn.Module):
    def __init__(self, config):
        super(MainModel, self).__init__()
        self.use_dcl = config.use_dcl
        self.num_classes = config.numcls
        self.backbone_arch = config.backbone
        self.use_Asoftmax = config.use_Asoftmax
        print(self.backbone_arch)
        if 'efficientnet' in self.backbone_arch:
            if self.backbone_arch[-1] == '3':
                self.model_chs = 1536
            elif self.backbone_arch[-1] == '4':
                self.model_chs = 1792
            elif self.backbone_arch[-1] == '0':
                self.model_chs = 1792
            elif self.backbone_arch[-1] in ['m', 'l', 's']:
                self.model_chs = 1280
            elif self.backbone_arch.endswith('rw_t'):
                self.model_chs = 1024
        elif self.backbone_arch.startswith('RepVGG'):
            if self.backbone_arch.endswith('B3g4'):
                self.model_chs = 2560
        elif 'nfnet' in self.backbone_arch:
            self.model_chs = 3072
        elif 'vit' in self.backbone_arch:
            self.model_chs = 1024
        elif 'deit' in self.backbone_arch:
            self.model_chs = 1024
        elif 'resnest' in self.backbone_arch:
            self.model_chs = 2048
        else:
            self.model_chs = 2048

        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)()
            if self.backbone_arch in pretrained_model:
                self.model.load_state_dict(
                    torch.load(pretrained_model[self.backbone_arch]))
        elif self.backbone_arch in pretrainedmodels.__dict__:
            self.model = pretrainedmodels.__dict__[self.backbone_arch](
                num_classes=1000)
        elif 'efficientnetv2' in self.backbone_arch:
            self.model = eval(self.backbone_arch)(pretrained=True)
            self.model.global_pool = nn.Identity()
            self.model.classifier = nn.Identity()
        elif self.backbone_arch.startswith('efficientnet'):
            self.model = EfficientNet_old.from_pretrained(
                self.backbone_arch,
                num_classes=self.num_classes,
                include_top=False)
        elif self.backbone_arch.startswith('dm_nfnet'):
            if True:
                self.model = eval(self.backbone_arch)(pretrained=True)
            else:
                self.model = dm_nfnet_f3(pretrained=True)
            self.model.head = nn.Identity()
        elif 'vit' in self.backbone_arch:
            self.model = eval(self.backbone_arch)(pretrained=True)
            self.model.head = nn.Identity()
        elif 'deit' in self.backbone_arch:
            self.model = eval(self.backbone_arch)(pretrained=True)
            #self.model.head = nn.Identity()

        elif 'resnest' in self.backbone_arch:
            self.model = eval(self.backbone_arch)(pretrained=True)
            self.model.global_pool = nn.Identity()
            self.model.fc = nn.Identity()
        else:
            print('Can not find %s net' % self.backbone_arch)

        if self.backbone_arch == 'resnet50' or self.backbone_arch == 'se_resnet50':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'senet154':
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if self.backbone_arch == 'se_resnext101_32x4d':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'se_resnet101':
            self.model = nn.Sequential(*list(self.model.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(self.model_chs,
                                    self.num_classes,
                                    bias=False)
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        if self.use_dcl:
            if config.cls_2:
                self.classifier_swap = nn.Linear(self.model_chs, 2, bias=False)
            if config.cls_2xmul:
                self.classifier_swap = nn.Linear(self.model_chs,
                                                 2 * self.num_classes,
                                                 bias=False)
            self.Convmask = nn.Conv2d(self.model_chs,
                                      1,
                                      1,
                                      stride=1,
                                      padding=0,
                                      bias=True)
            self.avgpool2 = nn.AvgPool2d(2, stride=2)

        if self.use_Asoftmax:
            self.Aclassifier = AngleLinear(self.model_chs,
                                           self.num_classes,
                                           bias=False)

    def forward(self, x, last_cont=None):
        x = self.model(x)
        if self.use_dcl:
            mask = self.Convmask(x)
            mask = self.avgpool2(mask)
            mask = self.dropout(mask)
            mask = torch.tanh(mask)
            mask = mask.view(mask.size(0), -1)
        if not 'vit' in self.backbone_arch and not 'deit' in self.backbone_arch:
            x = self.avgpool(x)
            x = self.dropout2(x)
            x = x.view(x.size(0), -1)
        out = []
        #return x
        if not 'deit' in self.backbone_arch:
            out.append(self.classifier(x))
        else:
            out.append(x)

        if self.use_dcl:
            out.append(self.classifier_swap(x))
            out.append(mask)

        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.model(last_cont)
                last_x = self.avgpool(last_x)
                last_x = last_x.view(last_x.size(0), -1)
                out.append(self.Aclassifier(last_x))

        return out
