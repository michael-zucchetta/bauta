from bauta.utils.ImageUtils import ImageUtils
from bauta.Constants import constants
from bauta.ImageInfo import ImageInfo

import cv2
import numpy as np
import os
import torch

class DatasetUtils(object):
    def __init__(self, dataset_configuration):
       self.config = dataset_configuration

    def extractConnectedComponents(self, class_index, mask):
        connected_component = None
        image_utils = ImageUtils()
        image_info = ImageInfo(mask)
        if mask.sum() > 10.00:
            mask = (mask * 255).astype(np.uint8)
            _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            return torch.IntTensor([class_index, x, y, x + w, y + h])

    def importImageWithMasks(self, image_path):
        main_file_path = f'{image_path}/{constants.dataset_input_filename}'
        if os.path.exists(main_file_path):
          main_image = cv2.imread(main_file_path, cv2.IMREAD_UNCHANGED)
          main_image = cv2.resize(main_image, ( constants.input_width, constants.input_height ))
          cv2.imwrite(main_file_path, main_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
          objects_in_path = list(set(os.listdir(image_path)) - {constants.dataset_input_filename})
          print(f'{objects_in_path} {constants.dataset_input_filename}')
          # bounding_boxes = torch.zeros((len(objects_in_path), 5)).int()
          bounding_boxes = torch.zeros(50, 5).int()
          original_object_areas = torch.zeros(len(self.config.classes))
          for (index, file_in_path) in enumerate(objects_in_path):
            class_name = file_in_path.replace(constants.dataset_mask_prefix, '').replace(constants.object_ext, '')
            class_index = self.config.classes.index(class_name)
            if class_index is not -1:
              mask = cv2.imread(f'{image_path}{file_in_path}', cv2.IMREAD_UNCHANGED)
              mask = cv2.resize(mask, ( constants.input_width, constants.input_height ))
              cv2.imwrite(f'{image_path}{file_in_path}', mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
              bounding_boxes[index:index+1, :] = self.extractConnectedComponents(class_index, mask)
              original_object_areas[class_index] = original_object_areas[class_index] + mask.sum()
          torch.save(original_object_areas.float(), f'{image_path}/{constants.dataset_original_object_areas_filename}')
          torch.save(bounding_boxes.cpu(), f'{image_path}/{constants.bounding_boxes_filename}')
        else:
          raise ValueError('Path {image_path} does not contain the image input.png') 
