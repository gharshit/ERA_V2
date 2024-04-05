# Repository for Image Classification on CIFAR10 dataset

### *Languages/Framework : [Python](https://www.python.org/) and [Pytorch](https://pytorch.org/)

### *Algorithm : [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)

#### About Dataset

The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.<br>
For more information refer to this [link](https://en.wikipedia.org/wiki/CIFAR-10).

<br>

#### Overall Context

This repo contains python code to train a convolutional neural network to classify images of `CIFAR-10` dataset. The aim of this experiments execute the different set of `Data Augmentation`, different convolutions like `dilated convolution`, `depthwise separable convolutions` and grouped convolutions. Following is the objective and constraints of the assignment:

![Assignment09](https://github.com/gharshit/ERA_V2/assets/19635712/b6213ea0-f52a-4d0f-b46b-212621519996)

<br>

#### Solution Context

        1. Number of Epochs used: 80
        2. Total number of parameters: 199,810
        3. Total RF of the architecture used is 95 (>44).
        4. Training Accuracy Achieved: 85.39%
        5. Testing Accuracy Achieved: 84.59%
        6. Data Augmenation Used: 
        
            # Apply horizontal flip
            A.HorizontalFlip(p=0.1)

            # Apply shift, scale and rotate
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.1)
        
            # Apply removal of box regions from the image to introduce regularization
            A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=[q*255 for q in mean_list],         mask_fill_value = None,p=0.3)

        7. Dilated convolution is being in first convolution block- 3rd layer. [Conv2d-13]
        8. First 3 Convolutional Blocks end with the convolution of stride 2. [Conv2d-13, Conv2d-26, Conv2d-39]
        9. Depthwise Separable Convolutions are being used in the last 2 convolutions blocks. [Conv2d-30, Conv2d-34 & Conv2d-43, Conv2d-47, Conv2d-51]


#### Plots

![A&Lplots](https://github.com/gharshit/ERA_V2/assets/19635712/c95a3c2a-faec-42d1-8963-89981e57901a)

<br>

---

### Code Structure

####  1. **train_S9.ipynb** [link](./train_S9.ipynb)
The driver code has the following code flow:

    Code Block 1: Contains the necessary libraries and hyperparamters.

    Code Block 2: Checks the availability of GPU or else the processing happens on CPU.

    Code Block 3: Get CIFAR-10 dataset and pass it to the loader for the model to train/text on in batch manner.

    Code Block 4: Get sample dataset and display it.

    Code Block 5: Load to device and get summary of the model.

    Code Block 6: Setup optimizer, scheduler, loss function and call function to train the neural network on the dataset in batches.

    Code Block 7: Plot the train/test accurcy and loss values as epoch increases, as well as a sample of misclassified images.


*Above code imports the model struture from model.py and other helper functions from utils.py to execute the code efficiently.


<br>


#### 2. **models.py** [link](./models.py)
This file contains the definition of a convolutional neural network (CNN) implemented in PyTorch for image classification tasks. 

The model architecture of the assignment is into four blocks such as C1, C2, C3 and C4. Except C4, all convolution blocks end with the convolution of stride 2. Refer to the code in models.py for more info

Below is the architecture with BatchNorm normalization: <br>
![skeleton](https://github.com/gharshit/ERA_V2/assets/19635712/1d084f97-bb8b-47a6-860d-59e1b6b9d785)


<br>


#### 3. **utils.py** [link](./utils.py)
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
