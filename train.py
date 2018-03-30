
from bauta.Trainer import Trainer
import click
import os

@click.command()
@click.option('--data_path', default=f'{os.getcwd()}', help='Data path.')
@click.option('--visual_logging', default=False, help='Display additional logging using images (only using desktop). Do not use it in a server, it requires a desktop environment.')
@click.option('--reset_model', default=False, help='Reset model (start from scratch).')
@click.option('--num_epochs', default=10000, help='Number of epochs.')
@click.option('--batch_size', default=16, help='Batch size.')
@click.option('--learning_rate', default=0.0001, help='Learning rate')
@click.option('--only_mask', default=False, help='Only learn masks without bounding boxes. Suitable for initial training.')
@click.option('--gpu', default=0, help='GPU index')
def train(data_path, visual_logging, reset_model, num_epochs, batch_size, learning_rate, gpu, only_mask):
    if not reset_model:
        reset_model_classes = None
    if only_mask:
        loss_scaled_weight = 1.0
        loss_unscaled_weight = 0.0
        bounding_box_loss_weight = 0.0
        loss_objects_found_weight = 0.0
    else:
        loss_scaled_weight = 0.4
        loss_unscaled_weight = 0.5
        bounding_box_loss_weight = 0.05
        loss_objects_found_weight = 0.05
    trainer = Trainer(data_path, visual_logging, reset_model, num_epochs, batch_size, learning_rate, gpu, \
        loss_scaled_weight, loss_unscaled_weight, bounding_box_loss_weight, loss_objects_found_weight)
    trainer.train()

if __name__ == '__main__':
    train()
