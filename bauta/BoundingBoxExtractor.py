from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch

import cv2
import numpy as np

class BoundingBoxExtractor(nn.Module):
    """Extracts the bounding box given a mask using a 'lookup layer'-style
    using coordinates.
    That means that it takes the mask pixel coordinates of top-left and
    bottom-right by first coverting the mask to coordinates (using lookup) and
    then taking the maximum and the minimum."""

    def __init__(self, input_width, input_height, scale):
        super(BoundingBoxExtractor, self).__init__()
        self.scale = scale
        self.input_width = input_width
        self.input_height = input_height
        self.scaled_input_width = int(self.input_width / self.scale)
        self.scaled_input_height = int(self.input_height / self.scale)
        self.y_coordinate_extractor = self.createYCoordinateExtractor(self.scaled_input_height)
        self.x_coordinate_extractor = self.createXCoordinateExtractor(self.scaled_input_width)

    def getBoundingBoxes(self, mask, threshold=0.5):
        mask_binary = (mask >= threshold).float()
        mask_binary = torch.split(mask_binary, 1, dim=1)
        y_coordinates = torch.cat([self.y_coordinate_extractor(mask_binary[edge]).view(-1, 1, self.scaled_input_height * self.scaled_input_width) for edge in range(len(mask_binary))], 1)
        x_coordinates = torch.cat([self.x_coordinate_extractor(mask_binary[edge]).view(-1, 1, self.scaled_input_height * self.scaled_input_width) for edge in range(len(mask_binary))], 1)
        x_min = torch.min(x_coordinates, -1)[0].unsqueeze(2)
        y_min = torch.min(y_coordinates, -1)[0].unsqueeze(2)
        x_max = torch.max(x_coordinates, -1)[0].unsqueeze(2)
        y_max = torch.max(y_coordinates, -1)[0].unsqueeze(2)
        object_found = ( (y_max > y_min) * (x_max > x_min) ).int()
        bounding_boxes_scaled = torch.cat((x_min, y_min, x_max, y_max), 2)
        bounding_boxes = bounding_boxes_scaled * self.scale
        return object_found, bounding_boxes_scaled, bounding_boxes, mask_binary

    def forward(self, mask):
        object_found, bounding_boxes_scaled, bounding_boxes, mask_binary = self.getBoundingBoxes(mask)
        return object_found, bounding_boxes_scaled, bounding_boxes

    def createXCoordinateExtractor(self, columns):
        y_coordinate_extractor = nn.Conv2d(1, columns, (1, columns))
        zeros = torch.FloatTensor(columns).zero_()
        y_coordinate_extractor.bias = Parameter(zeros, requires_grad=False)
        y_weight = torch.FloatTensor(columns, columns).zero_()
        for column in range(columns):
            y_weight[column][column] = column + 1
        y_coordinate_extractor.weight = Parameter(y_weight.view(columns, 1, 1, columns), requires_grad=False)
        return y_coordinate_extractor

    def createYCoordinateExtractor(self, rows):
        x_coordinate_extractor = nn.Conv2d(1, rows, (rows, 1))
        x_coordinate_extractor.bias = Parameter(torch.zeros(rows), requires_grad=False)
        x_weight = torch.FloatTensor(rows, rows).zero_()
        for row in range(rows):
            x_weight[row][row] = row + 1
        x_coordinate_extractor.weight = Parameter(x_weight.view(rows, 1, rows, 1), requires_grad=False)
        return x_coordinate_extractor
