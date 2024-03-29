import torch
import torch.nn as nn
import torch.nn.functional as F



######################################################## Model for Assignment 9 ########################################################

#Define the model
class Model(nn.Module):
    def __init__(self,dropout_value = 0.01):
        super(Model, self).__init__()


        # Specify normalization technique
        self.xnorm = lambda inp: nn.BatchNorm2d(inp)


        ################ CONVOLUTION BLOCK 1 STARTS #################


        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            self.xnorm(16),
            nn.Dropout(dropout_value)
        )  # n_in = 32, n_out = 30, r_in = 1, r_out = 3, j_in = 1, s=1, j_out = 1

        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.xnorm(32),
            nn.Dropout(dropout_value)
        ) # n_in = 30, n_out = 30, r_in = 3, r_out = 5, j_in = 1, s=1, j_out = 1


        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), padding=0, bias=False, dilation=2),
            nn.ReLU(),
            self.xnorm(48),
            nn.Dropout(dropout_value)
        ) # n_in = 30, n_out = 26, r_in = 5, r_out = 9, j_in = 1, s=1, j_out = 1



        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, stride=2),
            nn.ReLU(),
            self.xnorm(32),
            nn.Dropout(dropout_value)
        ) # n_in = 26, n_out = 13, r_in = 9, r_out = 11, j_in = 1, s=2, j_out = 2


        ################ CONVOLUTION BLOCK 1 ENDS #################
        


        ################ CONVOLUTION BLOCK 2 STARTS #################

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.xnorm(64),
            nn.Dropout(dropout_value)
        ) # n_in = 13, n_out = 13, r_in = 11, r_out = 15, j_in = 2, s=1, j_out = 2


        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.xnorm(96),
            nn.Dropout(dropout_value)
        ) # n_in = 13, n_out = 13, r_in = 15, r_out = 19, j_in = 2, s=1, j_out = 2


        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
        ) # n_in = 13, n_out = 13, r_in = 19, r_out = 19, j_in = 2, s=1, j_out = 2

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, stride=2),
            nn.ReLU(),
            self.xnorm(32),
            nn.Dropout(dropout_value)
        ) # n_in = 13, n_out = 7, r_in = 19, r_out = 23, j_in = 2, s=2, j_out = 4

        ################ CONVOLUTION BLOCK 2 ENDS #################




        ################ CONVOLUTION BLOCK 3 STARTS #################

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False,groups=32),
            nn.ReLU(),
            self.xnorm(64),
            nn.Dropout(dropout_value)
        ) # n_in = 7, n_out = 7, r_in = 23, r_out = 31, j_in = 4, s=1, j_out = 4

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False, groups=64),
            nn.ReLU(),
            self.xnorm(128),
            nn.Dropout(dropout_value)
        ) # n_in = 7, n_out = 7, r_in = 31, r_out = 39, j_in = 4, s=1, j_out = 4


        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
        ) # n_in = 7, n_out = 7, r_in = 39, r_out = 39, j_in = 4, s=1, j_out = 4

        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, stride=2),
            nn.ReLU(),
            self.xnorm(64),
            nn.Dropout(dropout_value)
        ) # n_in = 7, n_out = 4, r_in = 39, r_out = 47, j_in = 4, s=2, j_out = 8

        ################ CONVOLUTION BLOCK 3 ENDS #################



        
        ################ CONVOLUTION BLOCK 4 STARTS #################
        self.convblock13 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False,groups=64),
            nn.ReLU(),
            self.xnorm(128),
            nn.Dropout(dropout_value)
        ) # n_in = 4, n_out = 4, r_in = 47, r_out = 63, j_in = 8, s=1, j_out = 8

        self.convblock14 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False,groups=128),
            nn.ReLU(),
            self.xnorm(256),
            nn.Dropout(dropout_value)
        ) # n_in = 4, n_out = 4, r_in = 63, r_out = 79, j_in = 8, s=1, j_out = 8

        self.convblock15 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, bias=False,groups=256),
            nn.ReLU(),
            self.xnorm(256),
            nn.Dropout(dropout_value)
        ) # n_in = 4, n_out = 4, r_in = 79, r_out = 95, j_in = 8, s=1, j_out = 8


        self.convblock16 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=52, kernel_size=(1, 1), padding=0, bias=False),
        ) # n_in = 4, n_out = 4, r_in = 95, r_out = 95, j_in = 8, s=1, j_out = 8

        ################ CONVOLUTION BLOCK 4 ENDS #################


        ####### GAP LAYER
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP reduces spatial dimensions to 1x1

        ####### FC LAYER
        self.convblock17 = nn.Sequential(
            nn.Conv2d(in_channels=52, out_channels=10, kernel_size=(1, 1), padding=0, bias=True),

        )



    def forward(self, x):

        # CB1
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)


        # CB2
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)

        
        #CB3
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.convblock11(x)
        x = self.convblock12(x)

        #CB4
        x = self.convblock13(x)
        x = self.convblock14(x)
        x = self.convblock15(x)
        x = self.convblock16(x)

        # GAP
        x = self.global_avg_pool(x)     # Apply GAP

        #FC
        x = self.convblock17(x)         # Get the output from the fully connected layer

        x = x.view(x.size(0),-1)        # Flatten for softmax
        return F.log_softmax(x,dim=1)   # Apply log_softmax activation function on final output values




