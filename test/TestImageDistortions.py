import unittest
import mock
import cv2
import numpy as np
import sys, os
import random
import shutil
import itertools

from bauta.utils.ImageDistortions import ImageDistortions
from bauta.utils.ImageUtils import ImageUtils
from bauta.ImageInfo import ImageInfo
# 30 is the angle in the distorsion and 1.0 is the scale
random_rotation_mocked_numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 1.0, 0, 1.0, 0]

random_contrast_mocked_numbers = [50, 0, 1.5, 0, 0, 0, 0, 0, 0, 00, 0, 1.0, 0, 1.0, 0]

class TestImageDistorsions(unittest.TestCase):

    def random_uniform_rotation(lower_bound, upper_bound):
        if len(random_rotation_mocked_numbers) == 0:
            return lower_bound + random.random() / upper_bound
        else:
            return random_rotation_mocked_numbers.pop()

    @mock.patch('random.uniform', random_uniform_rotation)
    def testDistorsion(self):
        image_utils = ImageUtils()
        image_distorsions = ImageDistortions()
        image_base = cv2.imread('./test/data/images/square/square_1.png')
        image_info = ImageInfo(image_base)
        expected_rotated_image = cv2.imread('./test/data/images/square/distortions/square_1_rotated.png', cv2.IMREAD_UNCHANGED)
        image_base_with_alpha_channel = image_utils.addAlphaChannelToImage(image_base)
        rotated_image = image_distorsions.distortImage(image_base_with_alpha_channel)
        difference_of_images = cv2.subtract(rotated_image, expected_rotated_image)

        self.assertTrue(np.count_nonzero(difference_of_images > 10) < 10)

    def random_uniform_contrast(lower_bound, upper_bound):
        if len(random_contrast_mocked_numbers) == 0:
            return lower_bound + random.random() / upper_bound
        else:
            return random_contrast_mocked_numbers.pop()

    @mock.patch('random.uniform', random_uniform_contrast)
    def testContrastBrightnessDistorsion(self):
        image_utils = ImageUtils()
        image_distorsions = ImageDistortions()
        image_base = cv2.imread('./test/data/images/square/square_1.png')
        image_info = ImageInfo(image_base)
        expected_brigthened_image  = cv2.imread('./test/data/images/square/distortions/square_1_brightness.png', cv2.IMREAD_UNCHANGED)
        image_base_with_alpha_channel = image_utils.addAlphaChannelToImage(image_base)
        brigthened_image  = image_distorsions.distortImage(image_base_with_alpha_channel)
        difference_of_images = cv2.subtract(brigthened_image, expected_brigthened_image)

        self.assertTrue(np.count_nonzero(difference_of_images > 10) < 10)


if __name__ == '__main__':
    unittest.main()
