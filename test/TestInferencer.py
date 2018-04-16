import unittest
import sys, os
import random
import cv2
import shutil
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import transforms, utils

from roi_align.roi_align import RoIAlign

from bauta.DatasetGenerator import DatasetGenerator
from bauta.utils.SystemUtils import SystemUtils
from bauta.Inferencer import Inferencer
from bauta.Constants import constants
from bauta.DatasetConfiguration import DatasetConfiguration

class TestInferencer(unittest.TestCase):

    def createDataset():
        system_utils = SystemUtils()
        images_path = f'/tmp/{random.randint(0, 100000)}'
        data_path = f'/tmp/{random.randint(0, 100000)}'
        system_utils.makeDirIfNotExists(images_path)
        squares_image_path = f'{images_path}/square.txt'
        with open(f'{images_path}/square.txt','w') as file:
            file.write('./test/data/images/square/square_1.png\n./test/data/images/square/square_2.png')
        circles_image_path = f'{images_path}/circle.txt'
        with open(f'{images_path}/circle.txt','w') as file:
            file.write('./test/data/images/circle/circle_1.png\n./test/data/images/circle/circle_2.png')
        backgrounds_image_path = f'{images_path}/background.txt'
        with open(f'{images_path}/background.txt','w') as file:
            file.write('./test/data/images/background/background_1.png\n./test/data/images/background/background_2.png')
        dataset_generator = DatasetGenerator(data_path)
        datasets_with_attributes = dataset_generator.generateDatasetFromListOfImages(images_path, 0.5, 5)
        return images_path, data_path

    def removeDataset(images_path, data_path):
        shutil.rmtree(images_path)
        shutil.rmtree(data_path)

    def test_inferencer(self):
        images_path, data_path = TestInferencer.createDataset()
        config = DatasetConfiguration(True, data_path)
        background_image_path = config.objects[constants.background_label][0]
        input_image = cv2.imread(background_image_path)
        inferencer = Inferencer(data_path, visual_logging=False)
        def mask_detector(input):
            local_input_image, only_masks = input
            # background only found
            object_found = torch.zeros(1, 3, 1)
            object_found[0][0][0] = 1
            # whole patch as background
            mask = Variable(torch.ones(1, local_input_image.size()[2], local_input_image.size()[3]))
            roi_align = roi_align = RoIAlign(local_input_image.size()[2], local_input_image.size()[3])
            # everything
            bounding_boxes = torch.zeros(1, 3, 4)
            bounding_boxes[0][0][0] = 0
            bounding_boxes[0][0][1] = 0
            bounding_boxes[0][0][2] = local_input_image.size()[3]
            bounding_boxes[0][0][3] = local_input_image.size()[2]
            return Variable(object_found), None, mask, roi_align, Variable(bounding_boxes)
        objects = inferencer.inferenceOnImage(mask_detector, input_image)
        TestInferencer.removeDataset(images_path, data_path)
        self.assertTrue(np.mean(np.abs(objects[0].image[:,:,0:3] - input_image)) < 1.0)

if __name__ == '__main__':
    unittest.main()
