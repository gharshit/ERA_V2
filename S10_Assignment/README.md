# Repository for Image Classification on CIFAR10 dataset

### *Languages/Framework : [Python](https://www.python.org/) and [Pytorch](https://pytorch.org/)

### *Algorithm : [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)

#### About Dataset

The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.<br>
For more information refer to this [link](https://en.wikipedia.org/wiki/CIFAR-10).

<br>

#### Overall Context

This repo contains python code to train a convolutional neural network to classify images of `CIFAR-10` dataset. The aim of these experiments is to assess the outcome of different normalization techniques such as  `Batch Normalization` , `Group Normalization` and `Layer Normalization`. The architecture in each of the experiment is kept same with same number of parameters so that normalization techniques can be assessed on the same scale.

    Architecture Skeleton: C1 > C2 > **c3** > *P1* > C4 > C5 > C6 > **c7** > *P2* > C8 > C9 > C10 > GAP > **c11**

    Number of Parameters: 48,448

    Max Epoch: 20


*What is the importance of normalization of layers in CNN?*

Normalization in Convolutional Neural Networks (CNNs) plays a crucial role in improving the performance of the model. Here are some key points:

1. **Stabilizing the Gradient Descent**: Normalization helps to stabilize the gradient descent step, which allows us to use larger learning rates or helps models converge faster for a given learning rate.

2. **Features on a Similar Scale**: Normalization ensures that the different features are on a similar scale. This is important because it helps to balance the contribution of different features to the model.

3. **Reducing Internal Covariate Shift**: Normalization reduces the amount of change in the distribution of the input of layers, providing a more robust ground for the deeper layers during the learning process.

4. **Accelerating Training**: Normalization can accelerate the training of neural networks by allowing each layer of the network to learn more independently of other layers.

5. **Regularization**: Normalization also acts as a form of regularization, reducing the risk of overfitting.
---
<br>


### Accuracy Details

Normalization Technique | Notebook Link | Training Accuracy | Testing Accuracy
--- | --- | --- | ---
Batch Normalization | [link](./ERA_Session8_BatchNormalization.ipynb) | 81.10% | 78.86%
Group Normalization | [link](./ERA_Session8_GroupNormalization.ipynb) | 80.30% | 77.88%
Layer Normalization | [link](./ERA_Session8_LayerNormalization.ipynb) | 79.91% | 76.94%

<br>

---

### Findings from the experiments

1. Batch Normalization achieves highest accruracy in both training and testing w.r.t other techniques.
2. Above is due to the fact that BN works with one channel only (or one feature) across the batch which helps the algorithm to <br>
   2.a Capture Internal Covariance Shift which GN/LN lack.<br>
   2.b Normalize across images which serves better regularization than layer(GN/LN).<br>
   2.c Have feature across the batch on same scale which stabizes the gradient descent when calculating backprop gradient for a batch.
3. Group Normalization has better accuracy than Layer Normalization, which can be due to the fact the some features(channel) instead all, share some common properties (are correlated) and hence normalization serves better in groups than whole layer.

<br>

---


### Accuracy and Loss Plots

![A&LPlot](https://github.com/gharshit/ERA_V2/assets/19635712/14038dbd-ef17-4909-801f-a461cd79b2b5)

<br>

---

### Misclassified Images


`Batch Normalization `
![BN](https://github.com/gharshit/ERA_V2/assets/19635712/45a9a09a-fcaa-49ec-bcd9-07cc7d2c266e)

<br>

`Group Normalization'
![GN](https://github.com/gharshit/ERA_V2/assets/19635712/1829dc40-be83-4ad3-8ecf-2dba74c02409)

<br>

`Layer Normalization`
![LN](https://github.com/gharshit/ERA_V2/assets/19635712/a4437a10-7f6e-4507-adb0-2a143cfad422)



<br>

---

### Code Structure

####  1. **ERA_Session8_XNormalization.ipynb**
Each normalization technique (X) experiment has the following code flow:

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
This file contains the definition of a convolutional neural network (CNN) implemented in PyTorch for image classification tasks. Some older models (from S6 & S7) are also included in the file for reference.

The model architecture for this assisgnment is kept same across the three experiment with a conditional function to handle normalization criteria as per passed variable (ln/gn/bn).

    if 'bn': set normalization to BatchNorm(# of channels in that layer)
    if 'gn': set normalization to GroupNorm(num_groups = **4**, # of channels in that layer)
    if 'bn': set normalization to LayerNorm(num_groups = **1**, # of channels in that layer)

Below is the architecture with BatchNorm normalization:

![skeleton](https://github.com/gharshit/ERA_V2/assets/19635712/29fe28ae-4eda-495d-9fd1-5a82bd9bdb7f)

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
