import torch.nn as nn

class ModelUtils():

    def createDilatedConvolutionPreservingSpatialDimensions(input_filter_banks, output_filter_banks, filter_size, dilation_to_use, add_batch_norm=True):
        dilated_filter_size = filter_size + ( (filter_size - 1) * (dilation_to_use - 1) )
        if ((dilated_filter_size - 1) % 2 != 0):
            error_message = f"Filter size {filter_size} and dilation {dilation_to_use} cannot be used as the spatial dimensions cannot be preserved"
            raise Exception(error_message)
        padding_to_use = int((dilated_filter_size - 1) / 2)
        convolution_layer = nn.Conv2d(input_filter_banks, output_filter_banks, filter_size, padding = padding_to_use, dilation = dilation_to_use)
        if add_batch_norm:
            return nn.Sequential(
                        convolution_layer,
                        nn.BatchNorm2d(output_filter_banks)
                    )
        else:
            return convolution_layer


    def xavier(convolution_layer):
        nn.init.xavier_uniform(convolution_layer.weight)
        convolution_layer.bias.data.zero_()
