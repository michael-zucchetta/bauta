from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
from bauta.utils.ModelUtils import ModelUtils

class MaskDetector(nn.Module):

    def __init__(self, filter_banks, filter_size):
        super(MaskDetector, self).__init__()
        self.dilations_count = 4
        self.dilation_blocks_count = 6
        self.initial_filter = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(92, filter_banks, filter_size, 1)

        dilation_blocks = []
        bottlenecks = []
        for dilation_block in range(self.dilation_blocks_count):
            dilations = []
            for dilation_index in range(self.dilations_count):
                dilation  = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(filter_banks, filter_banks, filter_size, 2**dilation_index)
                dilations.append(dilation)
            dilation_blocks.insert(len(dilation_blocks), nn.ModuleList(dilations))
            bottleneck = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(filter_banks * self.dilations_count, filter_banks, filter_size, 1)
            bottlenecks.append(bottleneck)

        self.dilation_blocks = nn.ModuleList(dilation_blocks)
        self.bottlenecks = nn.ModuleList(bottlenecks)

        self.last_layer = nn.Conv2d(filter_banks, 1, 1)
        ModelUtils.xavier(self.last_layer)
        ModelUtils.xavier(self.last_layer)
        self.last_layer.weight.data = self.last_layer.weight.data.abs() * -4.0 #logits for initial output near 0.01, useful as most of targets are backgrounds

    def forwardDilations(self, input):
        output = input
        outputs_sum = input
        for dilation_block_index in range(self.dilation_blocks_count):
            output = self.forwardInDilationGroup(outputs_sum, dilation_block_index)
            outputs_sum = outputs_sum + output
        return output

    def forwardInDilationGroup(self, inputs_sum, dilation_block_index):
        outputs = []
        for dilation_index in range(self.dilations_count):
            output = F.relu(self.dilation_blocks[dilation_block_index][dilation_index](inputs_sum))
            outputs.append(output)
        return F.relu(self.bottlenecks[dilation_block_index](torch.cat(outputs, 1)))

    def forward(self, input):
        x = F.relu(self.initial_filter(input))
        x = self.forwardDilations(x)
        x = F.sigmoid(self.last_layer(x))
        return x.squeeze(2).squeeze(2)
