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
        edges = mask_binary
        edges = (edges > 0.5).float()
        y_coordinates = self.y_coordinate_extractor(edges).view(-1, self.scaled_input_height * self.scaled_input_width) - 1
        x_coordinates = self.x_coordinate_extractor(edges).view(-1, self.scaled_input_height * self.scaled_input_width) - 1
        y_max, _ = torch.max(y_coordinates, -1)
        x_max, _ = torch.max(x_coordinates, -1)
        x_max = x_max + (x_max == -1).float()
        y_max = y_max + (y_max == -1).float()
        y_max_expanded = y_max.view(-1, 1).expand(y_max.size()[0], self.scaled_input_height * self.scaled_input_width)
        y_set_max_to_disabled = torch.mul((y_coordinates == -1).float(), y_max_expanded)
        x_max_expanded = x_max.view(-1, 1).expand(x_max.size()[0], self.scaled_input_height * self.scaled_input_width)
        x_set_max_to_disabled = torch.mul((x_coordinates == -1).float(), x_max_expanded)
        y_min, _ = torch.min(y_coordinates + y_set_max_to_disabled, -1)
        x_min, _ = torch.min(x_coordinates + x_set_max_to_disabled, -1)
        bounding_boxes_scaled = torch.cat((x_min.view(-1,1), y_min.view(-1,1), x_max.view(-1,1), y_max.view(-1,1)), 1)
        bounding_boxes = torch.cat((x_min.view(-1,1) * self.scale, y_min.view(-1,1) * self.scale, x_max.view(-1,1) * self.scale, y_max.view(-1,1) * self.scale), 1)
        return bounding_boxes_scaled, bounding_boxes, mask_binary, edges

    def forward(self, mask):
        bounding_boxes_scaled, bounding_boxes, mask_binary, edges = self.getBoundingBoxes(mask)
        return bounding_boxes_scaled, bounding_boxes

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
