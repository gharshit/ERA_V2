[Click for Part2](#Part2)  

<br>

# Part 1



## Overview

[Neural Networks](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)) can be broken down into three main components which are

1. **Forward Propagation** : is a process which takes the input data to generate the output by feeding the data from one layer to another until it reaches the output layer. At each layer, the input(output from previous layer) is used to calculate the output(input for next layer) using the computation involving weights, biases and activation functions.

2. **Loss Calculation**: is a process of calculating the inefficiency of the neural network in determing the required output. We aim to reduce this loss by altering the underlying weights & biases which are used in forward propagation to predict the output.

3. **Backward Propagation** : is a process of calculating gradients for the weights and biases used in the forward pass with an aim of reducing the loss value. This gradients of loss Value with respect to multiple layer's weights and biases are calculated using chain rule, which are then used to update the current network. The underlying principle is that of gradient descent.

<br>

### Example

Below is an example of a basic fully connected neural network.


![Example](image.png)


We will now use the above example to understand how we can calculate the three components which we defined in the overview.

<br>


## Excel Screenshot (sheetname:main) 
![Excel Screenshot](image-7.png)
The image is screenshot of excel filled with formulae and values for backpropagation.

<br>

## Calculation

- ### **Forward Propagation**

   ![Forward Pass](image-1.png)

<br>

- ### **Loss Calculation**

   ![Loss Calculation](image-2.png)

<br>

- ### **Backward Propagation**
    We will calculate the partial differential of the loss function with respect to weights we want to update. The main concept used in this process is of [chain rule.](https://en.wikipedia.org/wiki/Chain_rule)

    <br>

   - First we will calculate the gradients of weights w5, w6, w7 and w8

     ![Gradients w5tow8](image-4.png)

     <br>


   - Calculating the gradients of weights w1, w2, w3 and w4

     ![Gradients w1tow4](image-5.png)



<br>

Using the above calculated gradients, we will update the weights. Then in an iterative manner till we reach loss threhold or iteration count:
  - Forward pass variables will be calculated on updated weights and we get a_o1 and a_o2.
  - Loss value (E_total) will be calculated.
  - Calculate gradients and update the weights.


<br>



