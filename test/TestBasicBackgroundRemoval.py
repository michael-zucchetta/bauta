import unittest
import cv2
import numpy as np
import sys, os
import random
import shutil
import itertools

from bauta.utils.BasicBackgroundRemover import BasicBackgroundRemover
from bauta.utils.ImageUtils import ImageUtils

class TestBasicBackgroundRemoval(unittest.TestCase):

    def testBasicBackgroundRemoval(self):
        image_utils = ImageUtils()
        image_base = cv2.imread('test/data/images/generated/augmented_image_with_overlappings/2_mask_circle.png')
        basic_background_remover = BasicBackgroundRemover()
        image_base_with_alpha_channel = image_utils.addAlphaChannelToImage(image_base)
        image_without_background = basic_background_remover.removeFlatBackgroundFromRGB(image_base)

        difference_of_images = cv2.subtract(image_without_background, image_base_with_alpha_channel)
        
        self.assertTrue(np.count_nonzero(difference_of_images > 10) < 10)

if __name__ == '__main__':
    unittest.main()
