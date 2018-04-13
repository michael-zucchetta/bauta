import unittest
import cv2
import numpy as np
import sys, os
import random
import shutil
import itertools

from bauta.utils.ImageDistortions import ImageDistortions
from bauta.utils.ImageUtils import ImageUtils

class TestImageDistorsions(unittest.TestCase):

    def testRotationDistorsion(self):
        image_utils = ImageUtils()
        image_distorsions = ImageDistortions()
        image_base = cv2.imread('./test/data/images/square/square_1.png')
        expected_rotated_image = cv2.imread('./test/data/images/square/distorsions/square_1_rotated.png')
        image_base_with_alpha_channel = image_utils.addAlphaChannelToImage(image_base)
        rotated_image = image_distorsions.applyRotationDistortion(image_base_with_alpha_channel, 10)

        difference_of_images = cv2.subtract(rotated_image, image_base_with_alpha_channel)
        
        self.assertTrue(np.count_nonzero(difference_of_images > 10) < 10)

if __name__ == '__main__':
    unittest.main()
