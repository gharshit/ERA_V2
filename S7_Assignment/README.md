# Repository for Image Classification on MNIST dataset

### *Languages/Framework : [Python](https://www.python.org/) and [Pytorch](https://pytorch.org/)

### *Algorithm : [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)

#### About Dataset

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets. The creators felt that since NIST's training dataset was taken from American Census Bureau employees, while the testing dataset was taken from American high school students, it was not well-suited for machine learning experiments. Furthermore, the black and white images from NIST were normalized to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.<br>
For more information refer to this [link](https://en.wikipedia.org/wiki/MNIST_database).

<br>

#### Overall Context

This repo contains python code to train a convolutional neural network to classify images of handwritten digits (MNIST dataset). The neural net used contains 4 convolution layer followed by 2 fully connected layers as the underlying architecture. For the training, Stochastic Gradient Descent is used with loss fucntion as Negative Log Likelihood.  

<br>


---
##  **Training Metrics** 

![S5 Training Metrics](https://github.com/gharshit/ERA_S5_Assignment/assets/19635712/da38ad30-65e1-4cfc-b90d-0703da90fa20)



---
<br>

## Code Structure

####  1. **S5.ipynb**
<br>**Code Block 1**: Contains the necessary libraries and hyperparamters.
<br>**Code Block 2**: Checks the availability of GPU or else the processing happens on CPU.
<br>**Code Block 3**: Get MNIST dataset and pass it to the loader for the model to train/text on in batch manner.
<br>**Code Block 4**: Get sample dataset and display it.
<br>**Code Block 5**: Load to device and get summary of the model.
<br>**Code Block 6**: Setup optimizer, scheduler, loss function and call function to train the neural network on the dataset in batches.
<br>**Code Block 7**: Plot the train/test accurcy and loss values as epoch increases.


*Above code imports the model struture from model.py and other helper functions from utils.py to execute the code efficiently.


<br>

---

#### 2. **model.py** 
This file contains the definition of a convolutional neural network (CNN) implemented in PyTorch for image classification tasks. The neural network architecture, named Net, consists of four convolutional layers followed by two fully connected layers. The CNN is designed to take 1-channel images as input and output probabilities for 10 classes. Each convolutional layer is followed by rectified linear unit (ReLU) activation, and max-pooling is applied to reduce spatial dimensions.

*Network Architecture:*<br>

Conv1: 32 filters with a 3x3 kernel.<br>
Conv2: 64 filters with a 3x3 kernel, followed by max-pooling.<br>
Conv3: 128 filters with a 3x3 kernel.<br>
Conv4: 256 filters with a 3x3 kernel, followed by max-pooling.<br>
FC1: Linear layer with 4096 input features and 50 output features.<br>
FC2: Linear layer with 50 input features and 10 output features.<br>

<br>

##### **Model Summary**

![Summary](https://github.com/gharshit/ERA_S5_Assignment/assets/19635712/a3bcd62c-7391-497b-914e-0b05b05b6290)

<br>

The forward method defines the forward pass of the neural network, including the application of activation functions (ReLU) and the log-softmax function for the output layer.

---

<br>
<br>

#### 3. **utils.py** 
The file contains a Python script for training and evaluating a neural network model on the MNIST dataset using PyTorch. It includes functions for defining data transformations, setting up data loaders, implementing the training and evaluation process, and visualizing training metrics. Key functionalities include:

<br>

- **Data Transformations**:

The script provides functions to define data transformations for both training and test datasets. These transformations include random cropping, rotation, resizing, and normalization. These transformations are crucial for augmenting the dataset and preparing it for training a neural network.

<br>

- **Data Loader Setup**:

It includes a function to create data loaders for both the training and test datasets. These data loaders are essential for efficiently loading the data in batches during training and evaluation processes. The script ensures that the specified transformations are applied to the datasets during loading.


<br>

- **Helper Functions**:
The script contains several helper functions:<br>

    - *GetCorrectPredCount*: Computes the number of correct predictions by comparing predicted labels against true labels.<br>

    - *train*: Performs the training process for each epoch, including forward and backward passes, optimization, and tracking of training metrics such as loss and accuracy.<br>

    - *test*: Evaluates the model on the test dataset, calculating test loss and accuracy without performing gradient updates.<br>

    - *get_optimizer*: Initializes the optimizer (Stochastic Gradient Descent) with specified learning rate and momentum.<br>

    - *get_scheduler*: Configures a learning rate scheduler that adjusts the learning rate during training.<br>

    - *get_loss*: Retrieves the specified loss function (negative log likelihood loss in this case).<br>


<br>

- **Visualization**:
The script offers functions to visualize the dataset and training metrics:<br>

    - *post_display*: Displays sample images from the training dataset along with their corresponding labels.<br>

    - *post_accuracyplots*: Plots the training and test losses, as well as training and test accuracies, providing insights into model performance over epochs.<br>

<br>




