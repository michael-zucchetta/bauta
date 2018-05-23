from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
from bauta.utils.ModelUtils import ModelUtils
from bauta.model.MaskDetector import MaskDetector

class MaskRefiners(nn.Module):

    def __init__(self, classes):
        super(MaskRefiners, self).__init__()
        self.classes = classes
        mask_refiners = []
        for mask_refiner_index in range(self.classes):
            mask_refiners.append(MaskReiner())
        self.mask_refiners = nn.ModuleList(mask_refiners)

    def forward(self, embeddings):
        outputs = []
        predicted_masks, embeddings, embeddings_1_raw, embeddings_2_raw, embeddings_3_raw = input
        for class_index in range(self.classes):
            output = self.mask_refiners[class_index](\
                [
                input_image_size, 
                predicted_masks_crop[:,class_index:class_index+1,:,:], 
                embeddings_crop, 
                embeddings_3_crop, 
                embeddings_2_crop, 
                embeddings_1_crop])
            outputs.append(output)
        return torch.cat(outputs, 1)
