import torch
import torch.nn as nn
from torch import Tensor


###############################
## Example of ECG Classifier ##
##############################

class ConvInstanceNormReLU_1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, kernel_size: int, padding: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d( in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding = padding ),
            nn.InstanceNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
    
class ECG_Classifier(nn.Module):

    def __init__(self, in_channels=12, 
                 features=[16,32,64,128], kernels = [15, 11, 7, 5], 
                 strides=[1, 2, 2, 1], padding=[7, 5, 3, 2],
                 n_classes = 1, name = 'ECG_Classifier', ):
        """
        Input:          [B, 12, 350]
        
        Initial conv:   [B, 16, 350]   stride 1, kernel 15, padding 7
        Block 1:        [B, 32, 175]   stride 2, kernel 11, padding 5
        Block 2:        [B, 64, 88]    stride 2, kernel 7, padding 3
        Block 3:        [B, 128, 88]   stride 1, kernel 5, padding 2
        Last conv:      [B, 1, 88]     stride 1, kernel 3,  padding 1
        Flatten:        [B, 88]
        Linear:         [B, 1]
        
        Note if multilabel: 
        loss is BCEwith Logitsloss
        if Multiclass loss is CrossEntropyLoss
        """
        
        super().__init__()
        self.name = name
        self.padding = padding
        self.strides = strides
        
        self.initial_layer =nn.Sequential( #features = 16
            nn.Conv1d( in_channels, features[0], kernel_size=kernels[0], stride=self.strides[0], padding=self.padding[0], bias=True, ),
            nn.ReLU(inplace=True),
        )
        
        layers = []
        in_channels = features[0]
        
        for idx, feature in enumerate(features[1:]): #for features 32, 64
            kernel_size = kernels[idx+1]
            stride = self.strides[idx+1]
            pad = self.padding[idx+1]
            layers.append(ConvInstanceNormReLU_1D(in_channels, feature, kernel_size=kernel_size, 
                                                  stride=stride, padding=pad))
            in_channels = feature
            
        self.conv_blocks = nn.Sequential(*layers)
        
        
        self.last_conv = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=3, stride=1, padding="same"),)
        
        self.last_linear = nn.Sequential(nn.Flatten(),
                                        nn.LazyLinear(n_classes))
        
        
    def forward(self, x):
        
        x = self.initial_layer(x)
        x = self.conv_blocks(x)
        x = self.last_conv(x)
        x = self.last_linear(x)
        return x