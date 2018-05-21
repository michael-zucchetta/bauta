import torchvision.models as models

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch

import cv2
import numpy as np

from bauta.utils.CudaUtils import CudaUtils
from bauta.utils.ModelUtils import ModelUtils

class Backbone(nn.Module):

    def __init__(self, ):
        super(Backbone, self).__init__()
        model = models.resnet18(pretrained=True)
        model.eval()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        embeding_low_raw = self.layer1(x)
        embeding_mid_raw = self.layer2(embeding_low_raw)
        embeding_high_raw = self.layer3(embeding_mid_raw)        
        embeding_low = F.adaptive_max_pool2d(embeding_low_raw, (embeding_high_raw.size()[2], embeding_high_raw.size()[3]))
        embeding_mid = F.adaptive_max_pool2d(embeding_mid_raw, (embeding_high_raw.size()[2], embeding_high_raw.size()[3]))
        return torch.cat([embeding_high_raw, embeding_low, embeding_mid], 1), embeding_low_raw
