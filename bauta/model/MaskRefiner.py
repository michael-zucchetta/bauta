from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import cv2
import numpy as np

from bauta.utils.CudaUtils import CudaUtils
from bauta.utils.ModelUtils import ModelUtils
from bauta.utils.ImageUtils import ImageUtils

class MaskRefiner(nn.Module):

    def __init__(self):
        super(MaskRefiner, self).__init__()
        self.convolutional_16_reducer = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(92, 15, 13, 1)        
        self.convolutional_16_merger = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(16, 16, 13, 1)
        self.convolutional_8 = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(32, 16, 13, 1)
        self.convolutional_4 = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(24, 16, 13, 1)

        self.convolutional_2 = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(20, 1, 13, 1, False)
        ModelUtils.xavier(self.convolutional_2)
        #logits for initial output near 0.01, useful as most of targets are backgrounds
        self.convolutional_2.weight.data = self.convolutional_2.weight.data.abs() * -4.0

    def forward(self, input):
        input_image_size, predicted_masks_crop, embeddings_crop, embeddings_3_crop, embeddings_2_crop, embeddings_1_crop = input

        embeddings = F.relu(self.convolutional_16_reducer(embeddings_crop))
        embeddings = F.relu(self.convolutional_16_merger(torch.cat([embeddings, predicted_masks_crop], 1))) # 1:16

        embeddings_upsampled = F.upsample(embeddings, size=(embeddings_3_crop.size()[2], embeddings_3_crop.size()[3]), mode='bilinear') # 1:8
        embeddings = F.relu(self.convolutional_8(torch.cat([embeddings_upsampled, embeddings_3_crop], 1)))
        
        embeddings_upsampled = F.upsample(embeddings, size=(embeddings_2_crop.size()[2], embeddings_2_crop.size()[3]), mode='bilinear') # 1:4
        embeddings = F.relu(self.convolutional_4(torch.cat([embeddings_upsampled, embeddings_2_crop], 1)))
        
        embeddings_upsampled = F.upsample(embeddings, size=(embeddings_1_crop.size()[2], embeddings_1_crop.size()[3]), mode='bilinear') # 1:2
        embeddings = self.convolutional_2(torch.cat([embeddings_upsampled, embeddings_1_crop], 1))
        
        refined_mask = F.sigmoid(F.upsample(embeddings, size=(input_image_size[2], input_image_size[3]), mode='bilinear')) # 1:1
        return refined_mask
