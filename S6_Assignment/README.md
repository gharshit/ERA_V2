[Click for Part2](#-------------------------------------------------------------------------------------------------------)

<br>


# Part 1 : (Excel: S6_Assignment/Part1/Class BP File_HG.xlsx)

## Overview

[Neural Networks](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)) can be broken down into three main components which are

1. **Forward Propagation** : is a process which takes the input data to generate the output by feeding the data from one layer to another until it reaches the output layer. At each layer, the input(output from previous layer) is used to calculate the output(input for next layer) using the computation involving weights, biases and activation functions.

2. **Loss Calculation**: is a process of calculating the inefficiency of the neural network in determing the required output. We aim to reduce this loss by altering the underlying weights & biases which are used in forward propagation to predict the output.

3. **Backward Propagation** : is a process of calculating gradients for the weights and biases used in the forward pass with an aim of reducing the loss value. This gradients of loss Value with respect to multiple layer's weights and biases are calculated using chain rule, which are then used to update the current network. The underlying principle is that of gradient descent.

<br>

### Example

Below is an example of a basic fully connected neural network.


![Example](https://github.com/gharshit/ERA_V2/assets/19635712/1231dc8a-80de-430e-bdad-3d8fd6c26d58)



We will now use the above example to understand how we can calculate the three components which we defined in the overview.

<br>


## Excel Screenshot (Refer to sheet_main in excel) 
![Excel Screenshot](https://github.com/gharshit/ERA_V2/assets/19635712/6bc6c52b-b028-4705-8d7d-b27eecbcec94)

The image is screenshot of excel filled with formulae and values for backpropagation.

<br>

## Calculation

- ### **Forward Propagation**

   ![Forward Pass](https://github.com/gharshit/ERA_V2/assets/19635712/3ae6ac05-5ce6-45e1-aaed-4c6a2e0b1e20)


<br>

- ### **Loss Calculation**

   ![Loss Calculation](https://github.com/gharshit/ERA_V2/assets/19635712/730a6d96-2c67-4047-90bb-7518f300cc9f)


<br>

- ### **Backward Propagation**
    We will calculate the partial differential of the loss function with respect to weights we want to update. The main concept used in this process is of [chain rule.](https://en.wikipedia.org/wiki/Chain_rule)

    <br>

   - First we will calculate the gradients of weights w5, w6, w7 and w8

     ![Gradients w5tow8](https://github.com/gharshit/ERA_V2/assets/19635712/2fc9d014-cf3a-4543-a147-373df15855ed)


     <br>


   - Calculating the gradients of weights w1, w2, w3 and w4

     ![Gradients w1tow4](https://github.com/gharshit/ERA_V2/assets/19635712/60627b05-3ef8-44a4-9ce2-ec57e409e905)




<br>

Using the above calculated gradients, we will update the weights. Then in an iterative manner till we reach loss threhold or iteration count:
  - Forward pass variables will be calculated on updated weights and we get a_o1 and a_o2.
  - Loss value (E_total) will be calculated.
  - Calculate gradients and update the weights.


<br>

## Change in Loss Value vs Learning Rate (Refer to sheet_LRtrend in excel) 
![LR Change](https://github.com/gharshit/ERA_V2/assets/19635712/5decbb38-3ec7-49d2-9353-9f5d0401d9d6)

As we are increasing the Learning Rate from 0.1 to 2, the rate at which loss value drops/changes is increasing. So for larger LR value, the neural network is coverging faster to the optimal solution ***for the above given example and values***.


<br>



## -------------------------------------------------------------------------------------------------------


<br>

# Part 2 : [Notebook](./Part2/ERA_V2_Session_6_HG.ipynb)

<br>

### *Languages/Framework : [Python](https://www.python.org/) and [Pytorch](https://pytorch.org/)

### *Algorithm : [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)

#### About Dataset

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets. The creators felt that since NIST's training dataset was taken from American Census Bureau employees, while the testing dataset was taken from American high school students, it was not well-suited for machine learning experiments. Furthermore, the black and white images from NIST were normalized to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.<br>
For more information refer to this [link](https://en.wikipedia.org/wiki/MNIST_database).


#### Overall Context

This part contains python code to train a convolutional neural network to classify images of handwritten digits (MNIST dataset). For the training, Stochastic Gradient Descent is used with loss function as Negative Log Likelihood. This code is developed to achieve a certain validation accuracy with contraints on number of parameters and epochs. To achieve such a network, different techniques such as batch-normalization, regularization etc are implemented. Detailed information on the network can be found below.

<br>

### Constraints:

Below are the contraints which are to be considered while developing the architecture
   - 99.4% validation accuracy :white_check_mark:
   - Less than 20k model parameters :white_check_mark:
   - Less than 20 epochs :white_check_mark:
   - Have used BN, dropout :white_check_mark:
   - GAP followed by Fully Connected Layer :white_check_mark:

<br>


### Network Structure:

   ![NN structure](https://github.com/gharshit/ERA_V2/assets/19635712/2deb76d4-4df0-4523-811b-6eca249a711c)

<br>

   For better understanding, structured summary of the network is as follows:
   
1. `Input Block`:
   - Convolution: 1 input channel -> 12 output channels, kernel size (3x3), no padding
   - Activation: ReLU
   - Normalization: Batch normalization
   - Regularization: Dropout (p=0.1)

2. `Convolution Block 1`:
   - Convolution: 12 input channels -> 24 output channels, kernel size (3x3), no padding
   - Activation: ReLU
   - Normalization: Batch normalization
   - Regularization: Dropout (p=0.1)

3. `Transition Block 1`:
   - Convolution: 24 input channels -> 12 output channels, kernel size (1x1), no padding
   - Pooling: Max pooling (2x2), stride 2

4. `Convolution Block 2`:
   - Convolution 1: 12 input channels -> 20 output channels, kernel size (3x3), no padding
   - Activation: ReLU
   - Normalization: Batch normalization (momentum=0.2)
   - Regularization: Dropout (p=0.1)
   - Convolution 2: 20 input channels -> 32 output channels, kernel size (3x3), padding 1
   - Activation: ReLU
   - Normalization: Batch normalization (momentum=0.2)
   - Regularization: Dropout (p=0.1)

5. `Transition Block 2`:
   - Convolution: 32 input channels -> 16 output channels, kernel size (1x1), no padding
   - Pooling: Max pooling (2x2), stride 2

6. `Convolution Block 3`:
   - Convolution 1: 16 input channels -> 16 output channels, kernel size (3x3), padding 1
   - Activation: ReLU
   - Normalization: Batch normalization (momentum=0.2)
   - Regularization: Dropout (p=0.1)
   - Convolution 2: 16 input channels -> 32 output channels, kernel size (3x3), padding 1
   - Activation: ReLU
   - Normalization: Batch normalization (momentum=0.2)
   - Regularization: Dropout (p=0.1)

7. `Global Average Pooling`:
   - AdaptiveAvgPool2d: Reduce spatial dimensions to 1x1

8. `Fully Connected Layer`:
   - Linear: 32 input features -> 10 output classes

<br>

### Training Logs (snapshot is of last 10 epochs, for more refer to notebook)

   - Number of Epochs: 19 
   - Final training accuracy: 99.04%
   - Final validation accuracy: 99.47%

   ![Logs](https://github.com/gharshit/ERA_V2/assets/19635712/4093ad38-801c-49cb-8136-9d95288b6747)


<br>

### Accuracy and Loss Plots:

   ![Accuracy and Loss](https://github.com/gharshit/ERA_V2/assets/19635712/6e241cde-3989-4da8-be4a-11ea93417dfb)









