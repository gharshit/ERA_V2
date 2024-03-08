import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    
    '''
    This defines the structure of the NN.
    '''

    def __init__(self):
        super(Net, self).__init__()  # Initialize the superclass (nn.Module) constructor.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # First convolutional layer: 1 input channel, 32 output channels, 3x3 kernel.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel.
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  # Third convolutional layer: 64 input channels, 128 output channels, 3x3 kernel.
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)  # Fourth convolutional layer: 128 input channels, 256 output channels, 3x3 kernel.
        self.fc1 = nn.Linear(4096, 50)  # First fully connected layer: 4*4*256 input features, 50 output features; this takes the flatten output of conv4 so 320 is wrong dimension
        self.fc2 = nn.Linear(50, 10)  # Second fully connected layer: 50 input features, 10 output features (e.g., for 10 classes).



    def forward(self, x):
        # r_in:1, n_in:28, j_in:1, s:1, r_out:3, n_out:26, j_out:1, padding = 0
        x = F.relu(self.conv1(x))  # Apply ReLU activation to the first convolutional layer's output [kernel size 3x3x1x32, output size 26x26x32]

        # r_in:3, n_in:26, j_in:1, s:1, r_out:5, n_out:24, j_out:1, padding = 0
        # r_in:5, n_in:24, j_in:1, s:2, r_out:6, n_out:12, j_out:2, padding = 0
        # [kernel size 3x3x32x64, output size 24x24x64  -->MP-->  output size 12x12x64]
        # Relu should be applied before max pooling layer, the ReLU activation function first introduces non-linearity to the feature maps by setting all negative values to zero, which can help in enhancing the representation of features. Then, the max pooling step is used to reduce the spatial dimensions of these activated feature maps.
        # activation function is a part of neuron (like in brain) so it must be applied just after the computation of convolution of it. and hence max-pooling on the activated features only makes sense as this is just for channel size reduction
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # Apply ReLU activation to the output of the second convolutional layer, then apply max pooling. 2 for max-pool with kernel size 2x2 and torch whill automatically take stride equal to kernal size

        # r_in:6, n_in:12, j_in:2, s:1, r_out:10, n_out:10, j_out:2, padding = 0
        # [kernel size 3x3x64x128, output size 10x10x128]
        x = F.relu(self.conv3(x))  # Apply ReLU activation to the third convolutional layer's output


        # r_in:10, n_in:10, j_in:2, s:1, r_out:14, n_out:8, j_out:2, padding = 0
        # r_in:14, n_in:8, j_in:2, s:2, r_out:16, n_out:4, j_out:4, padding = 0
        # [kernel size 3x3x128x256, output size 8x8x256  -->MP-->  output size 4x4x256]
        # Relu should be applied before max pooling layer, the ReLU activation function first introduces non-linearity to the feature maps by setting all negative values to zero, which can help in enhancing the representation of features. Then, the max pooling step is used to reduce the spatial dimensions of these activated feature maps.
        # activation function is a part of neuron (like in brain) so it must be applied just after the computation of convolution of it. and hence max-pooling on the activated features only makes sense as this is just for channel size reduction
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)  #  # Apply ReLU activation to the output of the second convolutional layer, then apply max pooling. 2 for max-pool with kernel size 2x2 and torch whill automatically take stride equal to kernal size

        #Apply GAP to convert 4x4x256 tensore into 1x1x256
        # x = F.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(-1, 4096)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first fully connected layer's output with 256 as input features
        x = self.fc2(x)  # Get the output from the second fully connected layer.
        return F.log_softmax(x, dim=1)  # Apply log_softmax activation function for the output layer.
