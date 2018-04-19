from PIL import Image
import sys
import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from bauta.DatasetConfiguration import DatasetConfiguration
from bauta.BoundingBox import BoundingBox
from bauta.Constants import constants
from bauta.utils.ImageUtils import ImageUtils
from bauta.ImageInfo import ImageInfo
from bauta.InferenceResult import InferenceResult

class InferenceUtils():

    def __init__(self, data_path, visual_logging):
        self.loader = transforms.Compose([transforms.ToTensor()])
        self.image_utils = ImageUtils()
        self.config = DatasetConfiguration(False, data_path)
        self.visual_logging = visual_logging

    def maskScale(self, input_image):
        return self.image_utils.paddingScale(input_image, constants.input_width, constants.input_height)

    def toOriginalCoordinateFromNetworkInput(self, x, y, mask_width, mask_height, network_input_width, network_input_height):
        width_factor_input = mask_width / network_input_width
        height_factor_input = mask_height/ network_input_height
        x = x * width_factor_input
        y = y * height_factor_input
        return x, y

    def toOriginalCoordinate(self, x, y, mask_width, mask_height, network_input_width, network_input_height, bounding_box):
        bounding_box_width = bounding_box[2] - bounding_box[0]
        bounding_box_height = bounding_box[3] - bounding_box[1]
        width_factor_box = bounding_box_width / network_input_width
        height_factor_box = bounding_box_height / network_input_height
        x = x * width_factor_box
        y = y * height_factor_box
        x, y = self.toOriginalCoordinateFromNetworkInput(x, y, mask_width, mask_height, network_input_width, network_input_height)
        return x, y

    def logMasks(self, mask, object_found):
        if self.visual_logging:
            current_found_index = 0
            for current_index in range(object_found.size()[0]):
                for current_class in range(len(self.config.classes)):
                    if object_found[current_index][current_class].data[0] == 1:
                        cv2.imshow(f'Output Found Mask {current_index} for "{self.config.classes[current_class]}".', self.image_utils.toNumpy(mask.data[current_found_index]))
                        current_found_index = current_found_index + 1
            cv2.waitKey(0)

    def extractMasks(self, mask_detector, input_image):
        image_info = ImageInfo(input_image)
        input_image_scaled, network_input_height, network_input_width = self.maskScale(input_image)
        result = torch.FloatTensor(network_input_height, network_input_width).zero_()
        input_image_scaled_loaded = self.loader(input_image_scaled)
        input_image_scaled_loaded = Variable(input_image_scaled_loaded).unsqueeze(0)
        object_found, _, mask, _, bounding_boxes = mask_detector([input_image_scaled_loaded, False])
        self.logMasks(mask, object_found)
        inference_results = []
        mask_found_index = 0
        for class_index in range(object_found.size()[1]):
            if(object_found[0][class_index].data[0] == 1):
                bounding_box = bounding_boxes[0][class_index].data
                width, height = self.toOriginalCoordinate(input_image_scaled.shape[1], input_image_scaled.shape[0], image_info.width, image_info.height, network_input_width, network_input_height, bounding_box)
                x_min, y_min = self.toOriginalCoordinateFromNetworkInput(bounding_box[0], bounding_box[1], image_info.width, image_info.height, network_input_width, network_input_height)
                x_max, y_max = int(x_min + width - 1), int(y_min + height - 1)
                current_mask_as_numpy = self.image_utils.toNumpy(mask[mask_found_index].data)
                current_mask_as_numpy = cv2.resize(current_mask_as_numpy, (int(width),int(height)))
                bounding_box = BoundingBox(y_min, x_min, y_max, x_max)
                inference_result = InferenceResult( \
                    class_label = self.config.classes[class_index],
                    bounding_box  = bounding_box,
                    mask = current_mask_as_numpy )
                mask_found_index = mask_found_index + 1
                inference_results.append(inference_result)
                assert current_mask_as_numpy.shape[0] == bounding_box.height
                assert current_mask_as_numpy.shape[1] == bounding_box.width
        return inference_results

    def extractConnectedComponents(self, inference_results):
        new_inference_results = []
        for index, inference_result in enumerate(inference_results):
            mask = (inference_result.mask * 255).astype(np.uint8)
            original_bounding_box = inference_result.bounding_box
            width, height = mask.shape
            image_area = (width + 1) * (height + 1)
            _, contour, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
            bounding_boxes_from_connected_components, contour_areas = BoundingBox.fromOpenCVConnectedComponentsImage(mask, min_threshold=64, max_threshold=255)
            for boxes_index in range(len(bounding_boxes_from_connected_components)):
                bounding_box = bounding_boxes_from_connected_components[boxes_index]
                if (bounding_box.area / image_area) >= 0.095 and (len(bounding_boxes_from_connected_components) == 1 or (contour_areas[boxes_index] / image_area) < 1.0):
                    bounding_box_in_image = BoundingBox(bounding_box.top + original_bounding_box.top,
                        bounding_box.left + original_bounding_box.left,
                        bounding_box.top  + original_bounding_box.top  + bounding_box.height - 1,
                        bounding_box.left + original_bounding_box.left + bounding_box.width  - 1)
                    cropped_mask = mask[bounding_box.top : bounding_box.top + bounding_box.height,
                        bounding_box.left : bounding_box.left + bounding_box.width]
                    assert cropped_mask.shape[0] == bounding_box_in_image.height
                    assert cropped_mask.shape[1] == bounding_box_in_image.width
                    inference_result = InferenceResult( \
                        class_label = inference_result.class_label,
                        bounding_box  = bounding_box_in_image,
                        mask = cropped_mask,
                        contour_area = contour_areas[boxes_index])
                    new_inference_results.append(inference_result)
        return new_inference_results

    def extractObjects(self, inference_results, original_image):
        new_inference_results = []
        for inference_result in inference_results:
            image = original_image.copy()
            bounding_box = inference_result.bounding_box
            mask = inference_result.mask
            image_height, image_width, _ = image.shape
            mask_height, mask_width = mask.shape
            image_cropped = image[bounding_box.top : bounding_box.bottom + 1,
                                    bounding_box.left : bounding_box.right + 1, :]
            for channel in range(3):
                image_cropped[:,:, channel] = image_cropped[:,:, channel] * (mask[:,:] / 255)
            blue_channel, green_channel, red_channel  = cv2.split(image_cropped)
            image_cropped_with_alpha_channel = cv2.merge((blue_channel, \
                green_channel, red_channel, mask))
            inference_result = InferenceResult(\
                class_label = inference_result.class_label,
                bounding_box  = inference_result.bounding_box,
                mask = inference_result.mask,
                contour_area = inference_result.contour_area,
                image = image_cropped_with_alpha_channel)
            new_inference_results.append(inference_result)
        return new_inference_results
