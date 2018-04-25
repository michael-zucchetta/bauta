import unittest
from unittest import mock
import cv2
import itertools
import numpy as np
import random
import shutil
import itertools
import sys, os

from bauta.DatasetGenerator import DatasetGenerator
from bauta.utils.SystemUtils import SystemUtils
from bauta.DataAugmentationDataset import DataAugmentationDataset 
from bauta.utils.ImageDistortions import ImageDistortions 
import random

send_image_out_of_mask_y = 1000
rotation_angle = 20
scale = 1.0
random_mocked_numbers = [0, 0, 0, 0, 0, rotation_angle, 0, scale, 0, scale, 0]

class TestDataAugmentationDataset(unittest.TestCase):
    def random_uniform(lower_bound, upper_bound):
        if len(random_mocked_numbers) == 0:
          return lower_bound + random.random() / upper_bound
        else:
          return random_mocked_numbers.pop()

    def random_int(lower_bound, upper_bound):
        return 1

    @mock.patch('random.uniform', random_uniform)
    @mock.patch('random.randint', random_int)
    def testGenerateImageWithOverlappingElements(self):
        system_utils = SystemUtils()
        images_path = f'/tmp/{random.random() * 100000}'
        data_path = f'/tmp/{random.random() * 100000}'
        system_utils.makeDirIfNotExists(images_path)
        squares_image_path = f'{images_path}/square.txt'
        with open(f'{images_path}/square.txt','w') as file:
            file.write('./test/data/images/square/square_1.png\n./test/data/images/square/square_2.png')
        circles_image_path = f'{images_path}/circle.txt'
        with open(f'{images_path}/circle.txt','w') as file:
            file.write('./test/data/images/circle/circle_1.png')
        backgrounds_image_path = f'{images_path}/background.txt'
        with open(f'{images_path}/background.txt','w') as file:
            file.write('./test/data/images/background/background_1.png')
        dataset_generator = DatasetGenerator(data_path)
        datasets_with_attributes = dataset_generator.generateDatasetFromListOfImages(images_path, 0.25, 5)
        data_augmentation_dataset = DataAugmentationDataset(True, data_path)
        class_index = 0
        base_path_img1 = os.path.join(data_path, 'dataset/augmentation/train/circle')
        base_path_img2 = os.path.join(data_path, 'dataset/augmentation/train/square')
        img1_path = os.listdir(base_path_img1)[0]
        img2_path = os.listdir(base_path_img2)[0]
        img3_path = os.listdir(base_path_img2)[1]
        mocked_return_values = [
            (1, cv2.imread(os.path.join(base_path_img2, img2_path), cv2.IMREAD_UNCHANGED)),
            (2, cv2.imread(os.path.join(base_path_img1, img1_path), cv2.IMREAD_UNCHANGED)),
            (1, cv2.imread(os.path.join(base_path_img2, img3_path), cv2.IMREAD_UNCHANGED))
        ]
        
        data_augmentation_dataset.randomObject = lambda index: mocked_return_values.pop()
        dataset_index = 0
        data_augmentation_dataset.generateAugmentedImage(dataset_index)
        
        mask_background_name = '0_mask_background.png'
        generated_background_mask = cv2.imread(os.path.join(data_path, 'dataset/train/', str(dataset_index), mask_background_name))
        expected_background_mask = cv2.imread(os.path.join('./test/data/images/generated/augmented_image_with_overlappings', mask_background_name))
        difference_of_masks = cv2.subtract(expected_background_mask, generated_background_mask)
        difference_of_masks_inverse = cv2.subtract(generated_background_mask, expected_background_mask)
        self.assertTrue(np.count_nonzero(difference_of_masks + difference_of_masks_inverse > 10) < 10)

        shutil.rmtree(images_path)
        shutil.rmtree(data_path)

if __name__ == '__main__':
    unittest.main()
