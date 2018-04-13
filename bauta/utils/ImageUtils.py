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

from bauta.BoundingBox import BoundingBox
from bauta.ImageInfo import ImageInfo
from bauta.Constants import constants

class ImageUtils():

    def _pasteImage(self, pasted_image, pasted_image_mask, image_background, x_offset, y_offset):
        pasted_image_info = ImageInfo(pasted_image)
        image_background_info = ImageInfo(image_background)
        image_background_bbox = BoundingBox(top=0, left=0, bottom=image_background_info.height, right=image_background_info.width)
        pasted_image_bbox = BoundingBox(top=y_offset, left=x_offset, bottom=y_offset + pasted_image_info.height, right=x_offset + pasted_image_info.width)
        intersection_bbox = pasted_image_bbox.intersect(image_background_bbox)

        pasted_image_y_offset_from, pasted_image_x_offset_from, pasted_image_y_offset_to, pasted_image_x_offset_to = 0, 0, 0, 0

        if image_background_bbox.left > pasted_image_bbox.left:
            pasted_image_x_offset_from = -pasted_image_bbox.left
        if image_background_bbox.top > pasted_image_bbox.top:
            pasted_image_y_offset_from = -pasted_image_bbox.top
        pasted_image_x_offset_to = min(pasted_image_x_offset_from + intersection_bbox.width - 1, pasted_image_info.width)
        pasted_image_y_offset_to = min(pasted_image_y_offset_from + intersection_bbox.height - 1, pasted_image_info.height)

        image_background_y_offset_from, image_background_x_offset_from, image_background_y_offset_to, image_background_x_offset_to = 0, 0, 0, 0
        if x_offset >= 0:
            image_background_x_offset_from = x_offset
        if y_offset >= 0:
            image_background_y_offset_from = y_offset
        image_background_x_offset_to = image_background_x_offset_from + pasted_image_x_offset_to - pasted_image_x_offset_from
        image_background_y_offset_to = image_background_y_offset_from + pasted_image_y_offset_to - pasted_image_y_offset_from

        for channel_index in range(3):
            image_background[image_background_y_offset_from : image_background_y_offset_to, image_background_x_offset_from : image_background_x_offset_to, channel_index] = \
                    (pasted_image[pasted_image_y_offset_from : pasted_image_y_offset_to, pasted_image_x_offset_from : pasted_image_x_offset_to, channel_index] * (pasted_image_mask[pasted_image_y_offset_from : pasted_image_y_offset_to, pasted_image_x_offset_from : pasted_image_x_offset_to] / 255)) + \
                    (image_background[image_background_y_offset_from : image_background_y_offset_to, image_background_x_offset_from : image_background_x_offset_to, channel_index] * (1 - (pasted_image_mask[pasted_image_y_offset_from : pasted_image_y_offset_to, pasted_image_x_offset_from : pasted_image_x_offset_to] / 255)))
        return image_background

    def pasteRGBAimageIntoRGBimage(self, rgba_image, rgb_image, x_offset, y_offset, include_alpha_channel=False):
        image_info = ImageInfo(rgba_image)
        object_rgb  = rgba_image[:, :, 0:3]
        object_mask = rgba_image[:, :, 3]
        rgb_image = self._pasteImage(object_rgb, object_mask, rgb_image, x_offset, y_offset)

        image_info_rgb = ImageInfo(rgb_image)
        mask = self.blankImage(image_info_rgb.width, image_info_rgb.height, 1)
        mask[y_offset : image_info.height + y_offset, x_offset : image_info.width + x_offset, 0] = object_mask[:, :]
        return rgb_image, mask

    def toNumpy(self, tensor):
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.transpose(0, 2).transpose(0, 1).cpu().numpy()

    def addWhitePaddingToKeepAspectRatioOfOriginalImageAndScaleToNeworkInputField(self, input_image_scaled, input_width, input_height, x_offset, y_offset):
        colors = len(cv2.split(input_image_scaled))
        input_image_scaled_padding = np.ones((input_height, input_width, colors), dtype=np.uint8) * 255
        if(len(input_image_scaled.shape) == 2):
            input_image_scaled = np.expand_dims(input_image_scaled, axis=2)
        if(colors == 4):
            input_image_scaled = self.removeAlphaChannelFromImage(input_image_scaled)
        input_image_scaled = self.composeImages(input_image_scaled_padding, input_image_scaled, \
            x_offset , y_offset, use_image_to_add_alpha_channel=False, skip_alpha_channel=True)
        return input_image_scaled

    def paddingScale(self, input_image, input_width, input_height):
        image_info = ImageInfo(input_image)
        # image scaled to input field keeping aspect ratio
        if constants.input_width / constants.input_height <= image_info.aspect_ratio:
            new_height = constants.input_height
            new_width  = int(constants.input_height * image_info.aspect_ratio)
            input_image_scaled = cv2.resize(input_image, (new_width, new_height))
            return input_image_scaled, new_height, new_width
        else:
            new_width = constants.input_width
            new_height = int(constants.input_width * image_info.aspect_ratio)
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
