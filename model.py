import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kargs):
        super.__init__()
        self.conv = nn.Conv2d







def get_yolov3_structure():
    """ 
    Information about architecture config:
    Tuple is structured by (filters, kernel_size, stride) 
    Every conv is a same convolution. 
    List is structured by "B" indicating a residual block followed by the number of repeats
    "S" is for scale prediction block and computing the yolo loss
    "U" is for upsampling the feature map and concatenating with a previous layer
    """

    return [
        (32, 3, 1),
        (64, 3, 2),
        ["B", 1],
        (128, 3, 2),
        ["B", 2],
        (256, 3, 2),
        ["B", 8],
        (512, 3, 2),
        ["B", 8],
        (1024, 3, 2),
        ["B", 4],  # To this point is Darknet-53
        (512, 1, 1),
        (1024, 3, 1),
        "S",
        (256, 1, 1),
        "U",
        (256, 1, 1),
        (512, 3, 1),
        "S",
        (128, 1, 1),
        "U",
        (128, 1, 1),
        (256, 3, 1),
        "S",
    ]