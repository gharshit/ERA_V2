# Image Classification on MNIST 

## Problem Statement

    1. Your new target is:
       a. 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
       b. Less than or equal to 15 Epochs
       c. Less than 8000 Parameters
    2. Do this using your modular code. Every model that you make must be there in the model.py file as Model_1, Model_2, etc.
    3. Do this in exactly 3 steps
    4. Each File must have a "target, result, analysis" TEXT block (either at the start or the end)
    5. You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct. 



### *Languages/Framework : [Python](https://www.python.org/) and [Pytorch](https://pytorch.org/)

### *Algorithm : [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)

#### About Dataset

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets. The creators felt that since NIST's training dataset was taken from American Census Bureau employees, while the testing dataset was taken from American high school students, it was not well-suited for machine learning experiments. Furthermore, the black and white images from NIST were normalized to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.<br>
For more information refer to this [link](https://en.wikipedia.org/wiki/MNIST_database).
<br>

#### *Overall Context*

This repo contains python code to train a convolutional neural network to classify images of handwritten digits (MNIST dataset).  The assignment is divided into 3 steps to achieve the desired accuracy in problem statement.

<br>

## Step-1  &nbsp; [<>Notebook](./Sesson7_Step1.ipynb)

The first step required setting up the basic working code which will train and test the model along with a descent model architecture. Only after we have achieved a descent model in this step, we can further modify like adding regularization, GAP etc. to improve further on accuracy.

The model used in this notebook consists of the 4 different steps collectively as one that were discussed in the session: <br>
   - Setup
   - Basic Skeleton
   - Lighter Model
   - Batch Normalization

The above step model is named as *Model_1* in models.py
<br>

1. **Target**
     * Set the transforms (converting to tensor and normalization) and data loader
     * Set other basic working code to display sample/plots and model summary
     * Set optimizer, training and testing loops.
     * Design the basic skeleton right to have minimal changes afterwards.
     * Make the model lighter in terms of number of parameters.
     * Add Batch-norm to increase model efficiency.


2. **Result**
     * Parameters: 7547
     * Best Training Accuracy: 99.75% (15th epoch)
     * Best Testing Accuracy: 99.31% (13th epoch)


3. **Analysis**
     * The model (Model_1) we have created for *step1* with ~7.6k paramaters is overall a good light model which is able to achieve 99.15% testing accuracy after 15 epochs with best at 99.31% in 13th epoch.
     * From the training and testing accuracy achieved by the model, we can conclude that the model is overfitting the data as from 5th epoch onwards, training accuracy is more testing accuracy respectively *(Regularization is needed).
     * If we push the current model further, it is unlikely to reach the testing accuracy of 99.40% without any overfitting.

---
<br>


## Step-2  &nbsp; [<>Notebook](./Sesson7_Step2.ipynb)

The second step requires adding accuracy improvement techiniques which will generalize the model better.

The model used in this notebook consists of the 4 different steps (on top of *Model_1*) as one that were discussed in the session: <br>
   - Regularization
   - Global Average Pooling
   - Increasing Capacity
   - Correct MaxPooling Location
   - 
The above step model is named as *Model_2* in models.py
<br>

1. **Target**
     * Introduce regularization in the model using DROPOUT at each layer. [This reduces the overfitting in the model as the training accuracy has decreased and the gap between the traininga and testing accuracy has narrowed.]
     * Remove the last big kernel and introduce GAP layer followed by fully connected layer. [Due to reduction in number of parameters, both the accuracies have reduced.]
     * Insert capacity layer at the end before GAP to boost model capacity to learn complex patterns. [This increased the testing and training accuracy significanly.]
     * Add fully connected layers after GAP to increase model performance. [This increased the testing and training accuracy.]
     * Check and Correct(if) position of MaxPooling layer. [Position of MAXPOOL layer is at RF=5 and RF=14]


2. **Result**
     * Parameters: 7873
     * Best Training Accuracy: 99.10% (15th epoch)
     * Best Testing Accuracy: 99.26% (15th epoch)


3. **Analysis**
     * The model (Model_2) we have created for step2 with ~7.9k paramaters is overall a good model which is able to achieve 99.26% testing accuracy in 15 epochs.
     * From the training and testing accuracy achieved by the model, we can conclude that the model is not overfitting at all.
     * If we push the current model further, there is scope left to reach the testing accuracy of 99.4%.
     * We now will need to add some image augmentation and play with lr to reach the goal of 99.4% testing accuracy.

---
<br>



## Step-3  &nbsp; [<>Notebook](./Sesson7_Step3.ipynb)

The third step requires adding image augmentation as testing dataset might have some transformed images and playing with LR scheduler to update the learing rate while training.

The model used in this notebook consists of the 8 different steps from step1 & step2 as one that were discussed. Noow to reach the desired accuracy, we will do the following change: <br>
   - Image Augmentation
   - Playing with Learning Rates

Above changes doesn't require any change in the model architecture but in the working code. <br> 
> *Model_2 configuration which is used in Step2 only has been continued here*.
<br>


1. **Target**
     * Introduce data augmentation like random crop, random rotation and random perspective so that model is able to achive better generalization while training.
     * Try different learning rates along with scheduler to adapt the rate of weights update according to need.


2. **Result**
     * Parameters: 7873
     * Best Training Accuracy: 98.71% (15th epoch)
     * Best Testing Accuracy: 99.45%  (15th epoch)


3. **Analysis**
     * The test accuracy is up as compared to step2, which means our test data had few images that had transformation difference w.r.t. train dataset.
     * From the training and testing accuracy achieved by the model, we can conclude that the model is underfitting the data. This if fine, as we know that we have made training data harder after data augmentation.
     * With a little high LR of 0.3, the training speed is higher as compared to 0.01.
     * But with only data augmentation changes, the model was able to reach 99.30-99.32% but it was not increasing further. The accuracies became stagnated after coming to a certain point as the learning rate was too high for it at that stage.
     * Using LR scheuler which updates the learning rate after 5 epoch has proved to be effective to reach the desired accuracy. This required many iterations of experiment to see which LR scheduler is effective.

---
<br>
<br>


## Code Structure

####  1. **Session7_Stepx.ipynb**
    Code Block 1: Contains the necessary libraries and hyperparamters.
    Code Block 2: Checks the availability of GPU or else the processing happens on CPU.
    Code Block 3: Get MNIST dataset and pass it to the loader for the model to train/text on in batch manner.
    Code Block 4: Get sample dataset and display it.
    Code Block 5: Load to device and get summary of the model.
    Code Block 6: Setup optimizer, scheduler, loss function and call function to train the neural network on the dataset in batches.
    Code Block 7: Plot the train/test accurcy and loss values as epoch increases.


*Above code imports the model struture from model.py and other helper functions from utils.py to execute the code efficiently.
<br>

---

#### 2. **models.py** [link](./models.py)
This file contains the definition of a convolutional neural network (CNN) implemented in PyTorch for image classification tasks. The file has two model structure namely *Model_1* and *Model_2*. 
For step-1 of assignent, Model_1 is used and for step-2/step-3, Model_2 is used in the code. In step-3, the changes required are in the transformation code and lr scheduler so Model_2 is only used to execute it.

<br>

---

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




