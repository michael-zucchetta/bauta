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
        outputs = []
        for class_index in range(self.classes):
            output = self.mask_detectors[class_index](embeddings)
            outputs.append(output)
        return torch.cat(outputs, 1)
