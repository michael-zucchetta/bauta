import click
import os
import cv2
import sys

from bauta.Inferencer import Inferencer
from bauta.utils.EnvironmentUtils import EnvironmentUtils
from bauta.utils.SystemUtils import SystemUtils
from bauta.utils.ImageUtils import ImageUtils
from bauta.utils.CudaUtils import CudaUtils
from bauta.Constants import constants
from bauta.ImageInfo import ImageInfo

class Inference():

    def __init__(self, data_path, path, inferencer, result_folder, visual_logging, gpu):
        self.result_folder = result_folder
        self.visual_logging = visual_logging
        self.inferencer = inferencer
        self.path = path
        self.is_folder = os.path.isdir(self.path)
        self.environment = EnvironmentUtils(data_path)
        self.cuda_utils = CudaUtils()
        self.gpu = gpu
        self.model = self.cuda_utils.cudify([self.environment.loadModel(self.environment.best_model_file)], self.gpu)[0]
        self.system_utils = SystemUtils()
        self.image_utils = ImageUtils()

    def inferenceOnFile(self, full_image_path):
        input_image = cv2.imread(full_image_path)
        image_info = ImageInfo(input_image)
        if image_info.channels == 3:
            if input_image is None:
                sys.stderr.write(f"Error reading image\n")
                sys.exit(-1)
            else:
                if self.visual_logging:
                    cv2.imshow(f'input_image', input_image)
                objects = self.inferencer.inferenceOnImage(self.model, input_image)
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
        for object_index, current_object in enumerate(objects):
            image = current_object.image
            cv2.imwrite(f"{folder_path}/output_{object_index}_{current_object.class_label}.png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            if self.visual_logging:
                cv2.imshow(f'{current_object.class_label} as object {object_index}', image)

@click.command()
@click.option('--data_path', default=f'{os.getcwd()}', help='Data path.')
@click.option('--path', default="", help='Path to the folder or image that contain input image(s).')
@click.option('--result_folder', default="~", help='File where segmented images are stored.')
@click.option('--visual_logging', default=False, help='Diplay in a window the intermediate and final inference results.')
@click.option('--gpu', default=0, help='GPU index')
def inference(data_path, path, result_folder, visual_logging, gpu):
    inferencer = Inferencer(data_path, visual_logging, gpu)
    inference = Inference(data_path, path, inferencer, result_folder, visual_logging, gpu)
    inference.inference(path)

if __name__ == '__main__':
    inference()
