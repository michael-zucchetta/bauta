import click
from bauta.MaskDetectorInferencer import MaskDetectorInferencer

@click.command()
@click.option('--file_name', default="", help='Image filename.')
@click.option('--show_results', default=False, help='Diplay in a window the results.')
@click.option('--save_result', default=False, help='Whether to store the segmentation results in a file pointed by --result_folder')
@click.option('--result_folder', default="~", help='File where segmented images are stored.')
@click.option('--folder_name', default=False, help='Images folder.')
def inference(file_name, show_results, save_result, result_folder, folder_name):
    self.mask_detector_inferencer = MaskDetectorInferencer()
    self.mask_detector_inferencer.inference()

if __name__ == '__main__':
    inference()
