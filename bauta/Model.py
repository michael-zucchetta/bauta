from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
from bauta.BoundingBoxExtractor import BoundingBoxExtractor
from roi_align.roi_align import RoIAlign

class Model(nn.Module):

    def __init__(self, classes, filter_banks, filter_size, scale):
        super(Model, self).__init__()

        self.classes = classes + 1 # +1 class for background
        self.scale = scale
        self.dilations_count = 5
        self.dilation_blocks_count = 5
        self.initial_filter  = self.createDilatedConvolutionPreservingSpatialDimensions(3, filter_banks, filter_size, 1)

        self.pool = nn.MaxPool2d(self.scale, self.scale, return_indices=True)

        dilation_blocks = []
        bottlenecks = []
        for dilation_block in range(self.dilation_blocks_count):
            dilations = []
            for dilation_index in range(self.dilations_count):
                dilation  = self.createDilatedConvolutionPreservingSpatialDimensions(filter_banks, filter_banks, filter_size, 2**dilation_index)
                dilations.append(dilation)
            dilation_blocks.insert(len(dilation_blocks), nn.ModuleList(dilations))
            bottleneck = self.createDilatedConvolutionPreservingSpatialDimensions(filter_banks * 5, filter_banks, filter_size, 1)
            bottlenecks.append(bottleneck)

        self.dilation_blocks = nn.ModuleList(dilation_blocks)
        self.bottlenecks = nn.ModuleList(bottlenecks)

        self.fully_connected_1 = self.createDilatedConvolutionPreservingSpatialDimensions(filter_banks, filter_banks * 5, filter_size, 1)
        self.fully_connected_2 = self.createDilatedConvolutionPreservingSpatialDimensions(filter_banks * 5, self.classes, filter_size, 1)

        self.refine = self.createDilatedConvolutionPreservingSpatialDimensions(((filter_banks * 2) * (classes + 1)) + (classes + 1), filter_banks, filter_size, 1)

        self.upsample = nn.Upsample(scale_factor=self.scale, mode='nearest')


    def forwardDilations(self, inupt):
        output = inupt
        outputs_sum = inupt
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
        input, use_bounding_box = input
        initial_filtering = F.relu(self.initial_filter(input))
        input_size = initial_filtering.size()
        initial_filterin_pooled, pool_indices = self.pool(initial_filtering)
        embeddings = self.forwardDilations(initial_filterin_pooled)
        fully_connected_output_1 = F.relu(self.fully_connected_1(embeddings))
        mask_scaled = F.sigmoid(self.fully_connected_2(fully_connected_output_1))
        bounding_boxes_scaled = None
        bounding_boxes = None
        object_found = None
        input_width = input.size()[3]
        input_height = input.size()[2]
        bounding_box_extractor = BoundingBoxExtractor(input_width, input_height, self.scale)
        if isinstance(input.data, (torch.cuda.FloatTensor)):
            bounding_box_extractor.cuda(input.data.get_device())
        if use_bounding_box:
            object_found, bounding_boxes_scaled, bounding_boxes = bounding_box_extractor(mask_scaled)
        else:
            batch_size = input.size()[0]
            zeros = input.data.new(batch_size).zero_().view(-1, 1)
            ones = (input.data.new(batch_size).zero_() + 1).view(-1, 1)
            bounding_boxes = Variable(torch.cat((zeros, zeros, ones * (input_width - 1), ones * (input_height - 1)), 1))
            bounding_boxes_scaled = Variable(torch.cat((zeros, zeros, ones * ((input_width / self.scale) - 1), ones * ((input_height / self.scale) - 1)), 1))

        roi_align_scaled = RoIAlign(bounding_box_extractor.scaled_input_height, bounding_box_extractor.scaled_input_width)
        roi_align = RoIAlign(bounding_box_extractor.input_height, bounding_box_extractor.input_width)
        boxes_index = Variable(torch.from_numpy(np.arange(bounding_boxes_scaled.size()[0])).int(), requires_grad=False)
        boxes_index_all_masks = Variable(torch.from_numpy(np.arange(bounding_boxes_scaled.size()[0] * bounding_boxes_scaled.size()[1])).int(), requires_grad=False)
        if isinstance(input.data, (torch.cuda.FloatTensor)):
            boxes_index = boxes_index.cuda(input.data.get_device())
            boxes_index_all_masks = boxes_index_all_masks.cuda(input.data.get_device())
            roi_align_scaled = roi_align_scaled.cuda(input.data.get_device())
            roi_align = roi_align.cuda(input.data.get_device())
            bounding_boxes = bounding_boxes.cuda(input.data.get_device())
            bounding_boxes_scaled = bounding_boxes_scaled.cuda(input.data.get_device())
        embeddings_upsampled = self.upsample(embeddings)
        mask_scaled_upsampled = self.upsample(mask_scaled)
        embeddings_bounding_box_upsampled = Model.applyRoiAlignCombined(roi_align, embeddings_upsampled, bounding_boxes, boxes_index)
        initial_filtering_bounding_box =  Model.applyRoiAlignCombined(roi_align, initial_filtering, bounding_boxes, boxes_index)
        mask_scaled_bounding_box_upsampled = Model.applyRoiAlign(roi_align, mask_scaled_upsampled, bounding_boxes, boxes_index_all_masks)
        embeddings = F.relu(self.refine(torch.cat([embeddings_bounding_box_upsampled, initial_filtering_bounding_box, mask_scaled_bounding_box_upsampled], 1)))
        embeddings = F.relu(self.fully_connected_1(embeddings))
        mask = F.sigmoid(self.fully_connected_2(embeddings))

        return object_found, mask_scaled, mask, roi_align_scaled, roi_align, bounding_boxes, bounding_boxes_scaled, boxes_index, boxes_index_all_masks

    def applyRoiAlignCombined(roi_align, input, bounding_boxes, boxes_index):
        bounding_boxes_per_object = torch.split(bounding_boxes, 1, dim=1)
        rois = [roi_align(input, bounding_box_per_object.contiguous().view(-1, 4), boxes_index) for bounding_box_per_object in bounding_boxes_per_object]
        return torch.cat(rois, 1)

    def applyRoiAlign(roi_align, input, bounding_boxes, boxes_index):
        input_redim = input.view(-1, 1, input.size()[2] ,input.size()[3])
        bounding_boxes_redim = bounding_boxes.contiguous().view(-1, 4)
        output = roi_align(input_redim, bounding_boxes_redim, boxes_index)
        output = output.view(input.size()[0], input.size()[1], output.size()[2], output.size()[3])
        return output

    def createDilatedConvolutionPreservingSpatialDimensions(self, input_filter_banks, output_filter_banks, filter_size, dilation_to_use):
        dilated_filter_size = filter_size + ( (filter_size - 1) * (dilation_to_use - 1) )
        if ((dilated_filter_size - 1) % 2 != 0):
            error_message = f"Filter size {filter_size} and dilation {dilation_to_use} cannot be used as the spatial dimensions cannot be preserved"
            raise Exception(error_message)
        padding_to_use = int((dilated_filter_size - 1) / 2)
        dilated_convolution = nn.Sequential(
                    nn.Conv2d(input_filter_banks, output_filter_banks, filter_size, padding = padding_to_use, dilation = dilation_to_use),
                    nn.BatchNorm2d(output_filter_banks)
                )
        return dilated_convolution
