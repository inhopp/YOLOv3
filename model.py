import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kargs):
        super.__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            out = self.conv(x)
            out = self.bn(out)
            out = self.leaky(out)

        else:
            out = self.conv(x)

        return out
 

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels//2, kernel_size=1),
                    CNNBlock(channels//2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x += layer(x)
            else:
                x = layer(x)
            
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1)
            CNNBlock(2*in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        out = self.pred(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
        
        return out


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = [] # for each scale
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
            
        return outputs

    
    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        structure = get_yolov3_structure()

        for module in structure:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1 if kernel_size==3 else 0)
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels//2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3
        
        return layers



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