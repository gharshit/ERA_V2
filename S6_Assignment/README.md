[Click for Part2](#part-2--neural-network-with-constrains)

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
![Excel Screenshot](https://github.com/gharshit/ERA_V2/assets/19635712/983e17e7-eebe-4216-b9d7-b0bb843c35d7)

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

---

<br>


# Part 2 : (Neural Network with constrains:S6_Assignment/Part2/ERA_V2_Session_6_HG.ipynb)

<br>

### Constraints:

   ![constraints](https://github.com/gharshit/ERA_V2/assets/19635712/b9d28a28-9190-4bb7-b49d-3b2a3437625a)

<br>


### Network Structure:

   ![structure](https://github.com/gharshit/ERA_V2/assets/19635712/41b6eb64-7c2f-4d33-86b0-66a84480e985)

<br>

   For better understanding, structured summary of the network is as follows:
1. **`conv1`**:
    - Input: 1 channel (grayscale image)
    - Output: 16 channels
    - Layers:
        - 3x3 Convolution (16 filters)
        - ReLU activation
        - Batch normalization
        - 3x3 Convolution (16 filters)
        - ReLU activation
        - Batch normalization
        - 2x2 Max Pooling (stride 2)
        - Dropout (with a rate of 0.025)

2. **`conv2`**:
    - Input: 16 channels
    - Output: 16 channels
    - Layers:
        - 3x3 Convolution (24 filters)
        - ReLU activation
        - Batch normalization
        - 3x3 Convolution (32 filters)
        - ReLU activation
        - Batch normalization
        - 2x2 Max Pooling (stride 2)
        - Dropout (with a rate of 0.025)
        - 1x1 Convolution (16 filters)   

3. **`conv3`**:
    - Input: 16 channels
    - Output: 32 channels
    - Layers:
        - 3x3 Convolution with padding (32 filters)
        - ReLU activation
        - Batch normalization

4. **`Global Average Pooling (GAP)`**:
    - Reduces spatial dimensions to 1x1 (averages feature maps across spatial dimensions)

5. **`Fully Connected Layer (Linear)`**:
    - Input: 32 features (from GAP)
    - Output: 10 classes (for classification)
    - Applies log softmax activation for the output layer

<br>

### Training Logs (last 10 epochs, for more refer to notebook)

   ![Logs](https://github.com/gharshit/ERA_V2/assets/19635712/863e7f8d-fe88-4c45-9cca-b3ae1b15ff7a)

<br>

### Accuracy and Loss Plots:

   ![Accuracy and Loss](https://github.com/gharshit/ERA_V2/assets/19635712/8b27bca1-e9d8-4345-98be-d20dc59e5b13)








