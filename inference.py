import click
import os
import cv2
import sys

from bauta.Inferencer import Inferencer
from bauta.utils.EnvironmentUtils import EnvironmentUtils
from bauta.utils.SystemUtils import SystemUtils

class Inference():

    def __init__(self, data_path, path, inferencer, result_folder, visual_logging):
        self.result_folder = result_folder
        self.visual_logging = visual_logging
        self.inferencer = inferencer
        self.path = path
        self.is_folder = os.path.isdir(self.path)
        self.environment = EnvironmentUtils(data_path)
        self.mask_detector = self.environment.loadModel(self.environment.best_model_file)
        self.system_utils = SystemUtils()

    def inferenceOnFile(self, full_image_path):
        input_image = cv2.imread(full_image_path)
        if input_image is None:
            sys.stderr.write(f"Error reading image\n")
            sys.exit(-1)
        else:
            if self.visual_logging:
                cv2.imshow(f'input_image', input_image)
            objects = self.inferencer.inferenceOnImage(self.mask_detector, input_image)
            self.displayResults(objects, input_image, full_image_path)
            if self.visual_logging:
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def inference(self, path):
        if self.is_folder:
            self.file_names = self.system_utils.imagesInFolder(self.path)
            for image_name in self.file_names:
                self.inferenceOnFile(os.path.join(self.path, image_name))
        else:
            self.inferenceOnFile(self.path)

    def displayResults(self, objects, input_image, image_name):
        folder_path = f"{self.result_folder}/{os.path.basename(image_name)}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        cv2.imwrite(f"{folder_path}/input.png", input_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        index = 1
        for object_index, object in enumerate(objects):
            image = object.image
            cv2.imwrite(f"{folder_path}/output_{object_index}_{object.class_label}.png", object.image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            if self.visual_logging:
                cv2.imshow(f'{object.class_label} as object {object_index}', image)

@click.command()
@click.option('--data_path', default=f'{os.getcwd()}', help='Data path.')
@click.option('--path', default="", help='Path to the folder or image that contain input image(s).')
@click.option('--result_folder', default="~", help='File where segmented images are stored.')
@click.option('--visual_logging', default=False, help='Diplay in a window the intermediate and final inference results.')
def inference(data_path, path, result_folder, visual_logging):
    inferencer = Inferencer(data_path, visual_logging)
    inference = Inference(data_path, path, inferencer, result_folder, visual_logging)
    inference.inference(path)

if __name__ == '__main__':
    inference()
