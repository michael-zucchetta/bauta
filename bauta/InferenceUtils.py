from PIL import Image
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from bauta.MaskDetectorDataset import MaskDetectorDataset
import sys
import cv2
import numpy as np
from bauta.BoundingBox import BoundingBox
from bauta.ImageUtils import ImageUtils
from bauta.Environment import Environment
import torch.nn.functional as F
from torch.autograd import Variable

class InferenceUtils():
    """Utility methods for inference"""

    def __init__(self):
        self.environment = Environment()
        self.loader = transforms.Compose([transforms.ToTensor()])
        self.image_utils = ImageUtils()

    def maskScale(self, input_image):
        return self.image_utils.paddingScale(input_image, self.environment.input_width, self.environment.input_height)

    def toOriginalCoordinateFromNetworkInput(self, x, y, original_width, original_height, network_input_width, network_input_height):
        width_factor_input = original_width / network_input_width
        height_factor_input = original_height/ network_input_height
        x = x * width_factor_input
        y = y * height_factor_input
        return x, y

    def toOriginalCoordinate(self, x, y, original_width, original_height, network_input_width, network_input_height, bounding_box):
        bounding_box_width = bounding_box[2] - bounding_box[0]
        bounding_box_height = bounding_box[3] - bounding_box[1]
        width_factor_box = bounding_box_width / network_input_width
        height_factor_box = bounding_box_height / network_input_height
        x = x * width_factor_box
        y = y * height_factor_box
        x, y = self.toOriginalCoordinateFromNetworkInput(x, y, original_width, original_height, network_input_width, network_input_height)
        return x, y

    def maskBinaryMatrix(self, mask_detector, input_image):
        original_height = input_image.shape[0]
        original_width  = input_image.shape[1]
        input_image_scaled, network_input_height, network_input_width = self.maskScale(input_image)
        result = torch.FloatTensor(network_input_height, network_input_width).zero_()
        input_image_scaled_loaded = self.loader(input_image_scaled)
        input_image_scaled_loaded = Variable(input_image_scaled_loaded).unsqueeze(0)
        mask_scaled, mask, roi_align_scaled, roi_align, bounding_boxes, bounding_boxes_scaled, boxes_index = mask_detector([input_image_scaled_loaded, True])
        input_image_scaled_loaded = roi_align(input_image_scaled_loaded, bounding_boxes, boxes_index)[0]
        probabilities = mask[0][0].data.numpy()
        input_image_scaled = input_image_scaled_loaded.data.transpose(0,2).transpose(0,1).numpy()
        width, height = self.toOriginalCoordinate(input_image_scaled.shape[1], input_image_scaled.shape[0], original_width, original_height, network_input_width, network_input_height, bounding_boxes.data[0])
        x_min, y_min = self.toOriginalCoordinateFromNetworkInput(bounding_boxes.data[0][0], bounding_boxes.data[0][1], original_width, original_height, network_input_width, network_input_height)
        x_max, y_max = int(x_min + width), int(y_min + height)
        probabilities = cv2.resize(probabilities, (int(width),int(height)))
        input_image_bounding_box = np.zeros((int(y_max) - int(y_min), int(x_max) - int(x_min), 3), dtype=np.uint8)
        for channel in range(3):
            input_image_bounding_box[:, :, channel] = input_image[int(y_min):int(y_max), int(x_min):int(x_max), channel]
        return probabilities, input_image_bounding_box

    def scaleTo255AndThresholdAboveStandardDeviation(self, mask_location_matrix, threshold_multiplicative_factor=1.0):
        standard_deviation = mask_location_matrix.std()
        mask_location_matrix[(mask_location_matrix > (standard_deviation * threshold_multiplicative_factor))] = 1
        mask_location_matrix[(mask_location_matrix < 1)] = 0
        mask_location_matrix[:] = mask_location_matrix[:] * 255
        return mask_location_matrix

    def maskSegments(self, mask_location_matrix, show_results=False):
        mask_location_matrix = mask_location_matrix.astype(np.uint8)
        width, height = mask_location_matrix.shape
        image_area = (width + 1) * (height + 1)
        if show_results:
            im2, contours, hierarchy = cv2.findContours(mask_location_matrix, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
            mask_location_matrix_rgb = np.ones( (mask_location_matrix.shape[0], mask_location_matrix.shape[1], 3), dtype=np.uint8)
            mask_location_matrix_rgb[:,:, 0] = mask_location_matrix.copy()[:,:]
            cv2.drawContours(mask_location_matrix_rgb, contours, -1, (0, 0, 255), 3)
            cv2.imshow(f'mask_location_matrix contour', mask_location_matrix_rgb)
        bounding_boxes = []
        bounding_boxes_from_connected_components, contour_areas = BoundingBox.fromOpenCVConnectedComponentsImage(mask_location_matrix, min_threshold=64, max_threshold=255)
        for boxes_index in range(len(bounding_boxes_from_connected_components)):
            bounding_box = bounding_boxes_from_connected_components[boxes_index]
            # these two constants (0.095 and 1.0) shall be parametrized
            if (bounding_box.area / image_area) >= 0.095 and (len(bounding_boxes_from_connected_components) == 1 or (contour_areas[boxes_index] / image_area) < 1.0):
                bounding_boxes.append(bounding_box)
        return bounding_boxes, mask_location_matrix.copy()

    def generateIndividualDressesWithWhiteBackground(self, bounding_boxes, mask_location_matrix, image):
        """Returns
            list of images generated by:
            for each bounding box in 'bounding_boxes', crop 'image', then remove the
            background using 'mask_location_matrix'.
         """
        # Resize mask_location_matrix to the image
        image = self.image_utils.addAlphaChannelToImage(image)
        height, width, _ = image.shape
        original_height, original_width = mask_location_matrix.shape
        mask_location_matrix = cv2.resize(mask_location_matrix, (width, height))
        images = []
        images_with_background = []
        for bounding_box in bounding_boxes:
            # Resize Bounding Boxes
            bounding_box = bounding_box.resize(original_width, original_height, width, height)
            # Crop Original Image
            image_cropped = image[bounding_box.top: bounding_box.bottom + 1,
                                    bounding_box.left: bounding_box.right + 1,:]
            images_with_background.append(image_cropped)
            # Crop Location Matrix
            mask_location_matrix_cropped = mask_location_matrix[bounding_box.top: bounding_box.bottom + 1,
                                                                    bounding_box.left: bounding_box.right + 1]
            red_channel, green_channel, blue_channel, _ = cv2.split(image_cropped)
            mask_location_matrix_cropped = mask_location_matrix_cropped.astype(np.uint8)
            image_cropped_with_alpha_channel = cv2.merge((red_channel.astype(np.float), green_channel.astype(np.float), blue_channel.astype(np.float), mask_location_matrix_cropped.astype(np.float)))
            # Mege location matrix and original image crop
            mask_with_white_background = np.ones( (bounding_box.height, bounding_box.width, 4), dtype=np.uint8) * 255
            mask_with_white_background = self.image_utils.composeImages(mask_with_white_background, image_cropped_with_alpha_channel, 0, 0)
            images.append(mask_with_white_background)
        return images, images_with_background
