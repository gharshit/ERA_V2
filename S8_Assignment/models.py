import torch
import torch.nn as nn
import torch.nn.functional as F



######################################################## Model for Assignment 8 ########################################################

## All three models are made in Modelx with conditional function to handle normalization criteria


#Define the model
class Modelx(nn.Module):
    def __init__(self,dropout_value = 0.01,norm_form = "bn",num_group=4):
        super(Modelx, self).__init__()
        
        #state the normalization technique
        if norm_form == 'bn':
            self.xnorm = lambda inp: nn.BatchNorm2d(inp)
        elif norm_form == "gn":
            self.xnorm = lambda inp: nn.GroupNorm(num_groups = num_group,num_channels=inp)
        elif norm_form == "ln":
            self.xnorm = lambda inp: nn.GroupNorm(num_groups = 1,num_channels=inp)
        else:
            raise ValueError('Incorrect normalization technique string. Available options: BN, GN and LN')
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            self.xnorm(16),
            # nn.Dropout(dropout_value)
        )  # n_in = 32, n_out = 30, r_in = 1, r_out = 3, j_in = 1, s=1, j_out = 1

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            self.xnorm(32),
            nn.Dropout(dropout_value)
        ) # n_in = 30, n_out = 28, r_in = 3, r_out = 5, j_in = 1, s=1, j_out = 1



        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # n_in = 28, n_out = 28, r_in = 5, r_out = 5, j_in = 1, s=1, j_out = 1
        self.pool1 = nn.MaxPool2d(2, 2) # n_in = 28, n_out = 14, r_in = 5, r_out = 6, j_in = 1, s=2, j_out = 2


        
        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.xnorm(24),
            nn.Dropout(dropout_value)
        ) # n_in = 14, n_out = 14, r_in = 6, r_out = 10, j_in = 2, s=1, j_out = 2

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.xnorm(32),
            nn.Dropout(dropout_value)
        ) # n_in = 14, n_out = 14, r_in = 10, r_out = 14, j_in = 2, s=1, j_out = 2
        
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=40, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            self.xnorm(40),
            nn.Dropout(dropout_value)
        ) # n_in = 14, n_out = 12, r_in = 14, r_out = 18, j_in = 2, s=1, j_out = 2


        
        # TRANSITION BLOCK 2
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        )  # n_in = 12, n_out = 12, r_in = 18, r_out = 18, j_in = 2, s=1, j_out = 2
        self.pool2 = nn.MaxPool2d(2, 2) # n_in = 12, n_out = 6, r_in = 18, r_out = 20, j_in = 2, s=2, j_out = 4



        # CONVOLUTION BLOCK 3 
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.xnorm(24),
            nn.Dropout(dropout_value)
        ) # n_in = 6, n_out = 6, r_in = 20, r_out = 28, j_in = 4, s=1, j_out = 4

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.xnorm(32),
            nn.Dropout(dropout_value)
        ) # n_in = 6, n_out = 6, r_in = 28, r_out = 36, j_in = 4, s=1, j_out = 4
        
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            self.xnorm(32),
            nn.Dropout(dropout_value)
        ) # n_in = 6, n_out = 4, r_in = 36, r_out = 44, j_in = 4, s=1, j_out = 4
        
        

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP reduces spatial dimensions to 1x1

        # define fully connected layer
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),

        ) 



    def forward(self, x):
        x = self.convblock1(x)

        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)

        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        
        x = self.convblock7(x)
        x = self.pool2(x)
        
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.global_avg_pool(x)     # Apply GAP
        
        x = self.convblock11(x)         # Get the output from the fully connected layer
        x = x.view(x.size(0),-1)        # Flatten for softmax
        return F.log_softmax(x,dim=1)   # Apply log_softmax activation function on final output values


    
    
    



######################################################## Model from Assignment 7 #########################################################
    
########### Step1 Model: Setup, Basic Skeleton, Lighter Model, Batch Normalization ###############



#Define the model

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )  # n_in = 28, n_out = 26, r_in = 1, r_out = 3, j_in = 1, s=1, j_out = 1

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=21, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(21),
        ) # n_in = 26, n_out = 24, r_in = 3, r_out = 5, j_in = 1, s=1, j_out = 1



        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=21, out_channels=9, kernel_size=(1, 1), padding=0, bias=False),
        ) # n_in = 24, n_out = 24, r_in = 5, r_out = 5, j_in = 1, s=1, j_out = 1
        self.pool1 = nn.MaxPool2d(2, 2) # n_in = 24, n_out = 12, r_in = 5, r_out = 6, j_in = 1, s=2, j_out = 2


        
        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),

        ) # n_in = 12, n_out = 10, r_in = 6, r_out = 10, j_in = 2, s=1, j_out = 2

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        ) # n_in = 10, n_out = 8, r_in = 10, r_out = 14, j_in = 2, s=1, j_out = 2


        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2) # n_in = 8, n_out = 4, r_in = 14, r_out = 16, j_in = 2, s=2, j_out = 4


        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(4, 4), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )   # n_in = 4, n_out = 1, r_in = 16, r_out = 24, j_in = 4, s=1, j_out = 4




    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)

        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool2(x)

        x = self.convblock6(x)
        x = x.view(x.size(0),-1)        # Flatten 
    
        return F.log_softmax(x,dim=1)   # Apply log_softmax activation function for the output layer.





############### Step2 Model: Step1 +  Regularization, GAP , Increasing Capacity, Correct Max Pooling Layer #################
    


#Define the model
dropout_value = 0.05

class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )  # n_in = 28, n_out = 26, r_in = 1, r_out = 3, j_in = 1, s=1, j_out = 1

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=21, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(21),
            nn.Dropout(dropout_value)
        ) # n_in = 26, n_out = 24, r_in = 3, r_out = 5, j_in = 1, s=1, j_out = 1



        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=21, out_channels=9, kernel_size=(1, 1), padding=0, bias=False),
        ) # n_in = 24, n_out = 24, r_in = 5, r_out = 5, j_in = 1, s=1, j_out = 1
        self.pool1 = nn.MaxPool2d(2, 2) # n_in = 24, n_out = 12, r_in = 5, r_out = 6, j_in = 1, s=2, j_out = 2


        
        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # n_in = 12, n_out = 10, r_in = 6, r_out = 10, j_in = 2, s=1, j_out = 2

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # n_in = 10, n_out = 8, r_in = 10, r_out = 14, j_in = 2, s=1, j_out = 2


        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2) # n_in = 8, n_out = 4, r_in = 14, r_out = 16, j_in = 2, s=2, j_out = 4


        # CONVOLUTION BLOCK3 : ADDING CAPACITY
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # n_in = 4, n_out = 2, r_in = 16, r_out = 24, j_in = 4, s=1, j_out = 4


        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  #GAP reduces spatial dimensions to 1x1


        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(1, 1), padding=0, bias=True),
            nn.ReLU(),
            # nn.BatchNorm2d(16),
            # nn.Dropout(dropout_value)
        )

        self.convblock7b = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=True),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )




    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)

        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool2(x)

        x = self.convblock6(x)          # Apply 
        x = self.global_avg_pool(x)

        x = self.convblock7(x)
        x = self.convblock7b(x)
        x = x.view(x.size(0),-1)        # Flatten
    
        return F.log_softmax(x,dim=1)   # Apply log_softmax activation function for the output layer.
    





######################################################## Model from Assignment 6 ########################################################
    

#Define the model
dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        )  # n_in = 28, n_out = 26, r_in = 1, r_out = 3, j_in = 1, s=1, j_out = 1

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # n_in = 26, n_out = 24, r_in = 3, r_out = 5, j_in = 1, s=1, j_out = 1



        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
        ) # n_in = 24, n_out = 24, r_in = 5, r_out = 5, j_in = 1, s=1, j_out = 1
        self.pool1 = nn.MaxPool2d(2, 2) # n_in = 24, n_out = 12, r_in = 5, r_out = 6, j_in = 1, s=2, j_out = 2


        
        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(20,momentum=0.2),
            nn.Dropout(dropout_value)
        ) # n_in = 12, n_out = 10, r_in = 6, r_out = 10, j_in = 2, s=1, j_out = 2

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32,momentum=0.2),
            nn.Dropout(dropout_value)
        ) # n_in = 10, n_out = 10, r_in = 10, r_out = 14, j_in = 2, s=1, j_out = 2


        # TRANSITION BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        )  # n_in = 10, n_out = 10, r_in = 14, r_out = 14, j_in = 2, s=1, j_out = 2
        self.pool2 = nn.MaxPool2d(2, 2) # n_in = 10, n_out = 5, r_in = 14, r_out = 16, j_in = 2, s=2, j_out = 4



        # CONVOLUTION BLOCK 3 
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16,momentum=0.2),
            nn.Dropout(dropout_value)
        ) # n_in = 5, n_out = 5, r_in = 16, r_out = 24, j_in = 4, s=1, j_out = 4

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32,momentum=0.2),
            nn.Dropout(dropout_value)
        ) # n_in = 5, n_out = 5, r_in = 24, r_out = 32, j_in = 4, s=1, j_out = 4
        
        

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP reduces spatial dimensions to 1x1

        # define fully connected layer
        self.fc = nn.Linear(32, 10)



    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)

        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.global_avg_pool(x)     # Apply GAP
        x = x.view(x.size(0),-1)        # Flatten for fully connected layer
        x = self.fc(x)                  # Get the output from the fully connected layer
        return F.log_softmax(x,dim=1)   # Apply log_softmax activation function for the output layer.