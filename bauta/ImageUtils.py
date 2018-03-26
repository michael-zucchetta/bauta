from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

import math
import os, random
from PIL import Image
import sys
import numpy as np
import cv2
import traceback

from bauta.Environment import Environment
from bauta.BoundingBox import BoundingBox
from bauta.ImageInfo import ImageInfo

class ImageUtils():

    def pasteRGBAimageIntoRGBimage(self, rgba_image, rgb_image, x_offset, y_offset):
        image_info = ImageInfo(rgba_image)
        object_rgb  = rgba_image[:, :, 0:3]
        object_mask = rgba_image[:, :, 3]
        for channel_index in range(3):
            rgb_image[y_offset : image_info.height + y_offset, x_offset : image_info.width + x_offset, channel_index] = \
                (object_rgb[:, :, channel_index] * (object_mask / 255)) + \
                (rgb_image[y_offset : image_info.height + y_offset, x_offset : image_info.width + x_offset, channel_index] * (1 - (object_mask / 255)))
        image_info_rgb = ImageInfo(rgb_image)
        mask = np.zeros( (image_info_rgb.height, image_info_rgb.width, 1), dtype=np.uint8)
        mask[y_offset : image_info.height + y_offset, x_offset : image_info.width + x_offset, 0] = object_mask[:, :]
        return rgb_image, mask

    def toNumpy(self, tensor):
        return tensor.transpose(0, 2).transpose(0, 1).numpy()

    def addWhitePaddingToKeepAspectRatioOfOriginalImageAndScaleToNeworkInputField(self, input_image_scaled, input_width, input_height, x_offset, y_offset):
        colours = len(cv2.split(input_image_scaled))
        input_image_scaled_padding = np.ones((input_height, input_width, colours), dtype=np.uint8) * 255
        if(len(input_image_scaled.shape) == 2):
            input_image_scaled = np.expand_dims(input_image_scaled, axis=2)
        if(colours == 4):
            input_image_scaled = self.removeAlphaChannelFromImage(input_image_scaled)
        input_image_scaled = self.composeImages(input_image_scaled_padding, input_image_scaled, \
            x_offset , y_offset, use_image_to_add_alpha_channel=False, skip_alpha_channel=True)
        return input_image_scaled

    def paddingScale(self, input_image, input_width, input_height):
        image_info = ImageInfo(input_image)
        environment = Environment()
        # image scaled to input field keeping aspect ratio
        if environment.input_width / environment.input_height <= image_info.aspect_ratio:
            new_height = environment.input_height
            new_width  = int(environment.input_height * image_info.aspect_ratio)
            input_image_scaled = cv2.resize(input_image, (new_width, new_height))
            return input_image_scaled, new_height, new_width
        else:
            new_width = environment.input_width
            new_height = int(environment.input_width * image_info.aspect_ratio)
            input_image_scaled = cv2.resize(input_image, (new_width, new_height))
            return input_image_scaled, new_height, new_width

    def blankImage(self, width, height, channels):
        blank_image = np.zeros( (height, width, channels), dtype=np.uint8)
        return blank_image

    def getAlphaChannel(self, image):
        splits = cv2.split(image)
        if(len(splits) == 4):
            return splits[3]
        else:
            return np.ones_like(splits[0]) * 255

    def addAlphaChannelToImage(self, image):
        splits = cv2.split(image)
        if(len(splits) == 4):
            return image
        if(len(splits) != 3):
            raise Exception(f"Image is not RGB as it has {len(splits)} channels")
        red_channel, green_channel, blue_channel = splits
        alpha_channel = np.ones(blue_channel.shape, dtype=blue_channel.dtype) * 255
        image_with_alpha_channel = cv2.merge((red_channel, green_channel, blue_channel, alpha_channel))
        return image_with_alpha_channel

    def removeAlphaChannelFromImage(self, image):
        splits = cv2.split(image)
        if(len(splits) == 3):
            return image
        if(len(splits) != 4):
            raise Exception(f"Image is not RGBA as it has {len(splits)} channels")
        blue_channel, green_channel, red_channel, alpha_channel = splits
        image_with_alpha_channel = cv2.merge((blue_channel, green_channel, red_channel))
        return image_with_alpha_channel
