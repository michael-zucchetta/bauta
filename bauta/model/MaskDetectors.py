from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
from bauta.utils.ModelUtils import ModelUtils
from bauta.model.MaskDetector import MaskDetector

class MaskDetectors(nn.Module):

    def __init__(self, classes, filter_banks, filter_size):
        super(MaskDetectors, self).__init__()
        self.classes = classes
        mask_detectors = []
        for mask_detector_index in range(self.classes):
            mask_detectors.append(MaskDetector(filter_banks, filter_size))
        self.mask_detectors = nn.ModuleList(mask_detectors)

    def forward(self, embeddings):
        masks = []
        masks_embeddings = []
        for class_index in range(self.classes):
            mask, mask_embeddings = self.mask_detectors[class_index](embeddings)
            masks.append(mask)
            mask_embeddings = mask_embeddings.unsqueeze(1)
            masks_embeddings.append(mask_embeddings)
        masks_embeddings = torch.cat(masks_embeddings, 1)
        return torch.cat(masks, 1), masks_embeddings
