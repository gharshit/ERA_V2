# Repository for Image Classification on CIFAR10 dataset

### *Languages/Framework : [Python](https://www.python.org/) and [Pytorch](https://pytorch.org/)

### *Algorithm : [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)

#### About Dataset

The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.<br>
For more information refer to this [link](https://en.wikipedia.org/wiki/CIFAR-10).

<br>

#### Overall Context

This repo contains python code to train a convolutional neural network to classify images of `CIFAR-10` dataset. The aim of this experiments is to make a custom `ResNet` architecture along with the usage of [LRfinder](https://github.com/davidtvs/pytorch-lr-finder) . The strategy is to use the suggested LR for deciding the *MAXLR* and *MINLR* which will be used to implement `One Cycle Policy` as the scheduler.
<br>
Following are the targets which are achieved 

    1. Using the suggested LR from LRfinder, we will select MAXLR as 0.017 and MINLR as 0.0017 to execute One Cycle Policy.
    2. The custom ResNet model architecture is trained on CIFAR10 dataset for 20 epochs.
    3. Training dataset transformation: RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
    4. Batch Size: 512, Optimizer: ADAM, Loss Criteria: Cross Entropy
    5. Learning rate starts from MINLR and increases to achieve MAXLR at Epoch-5, then LR decreases back from there and reached MINLR at Epoch-20.
    6. TRAINING ACCURACY: 93.77% and TESTING ACCURACY: 91.42%

---
<br>

### Custom Resnet architecture implemented

![customResNet](https://github.com/gharshit/ERA_V2/assets/19635712/805a9141-4fdb-4007-99b8-b53c2c690a4d)

<br>



### Results of LRfinder tool with above model:

![LRfinder](https://github.com/gharshit/ERA_V2/assets/19635712/9df51106-9421-4b5d-aaf6-fdbbbedecf14)

<br>



### One Cycle Policy (LR vs Epoch)

![OCP](https://github.com/gharshit/ERA_V2/assets/19635712/ef709d6f-8e43-44bc-9c8d-636b401087ed)

<br>


### Accuracy and Loss Plots

![A&LPlot](https://github.com/gharshit/ERA_V2/assets/19635712/7ff9e7a0-8d97-47a1-9628-56084fbf43c0)

<br>

---

`MASTER CODE REPO`: [Link](https://github.com/gharshit/ERA_MasterCodes)
<br>
`Training Notebook`: [Link](./train_S10.ipynb)
