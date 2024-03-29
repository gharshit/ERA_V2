import torch
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import v2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


mean_list = [0.4914, 0.4822, 0.4465]
std_list  = [0.2471,0.2435, 0.2616]


#################################### Define the data transformations ######################


class getCIFAR10(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
      # Initialize dataset and transform
      self.dataset = dataset
      self.transform = transform

    def __len__(self):
      return len(self.dataset)  # return length of dataset

    def __getitem__(self, idx):
      image, label = self.dataset[idx]

      #convert image to numpy array
      image = np.array(image)

      if self.transform is not None:
        image = self.transform(image=image)["image"]

      return image, label

######### Step1 Transformations Start ###########


def get_train_transforms():

  '''

  Train data transformations

  '''

  return A.Compose([

    # Apply horizontal flip
    A.HorizontalFlip(p=0.1),

    # Apply shift, scale and rotate
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.1),

    # Apply removal of box regions from the image to introduce regularization
    A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=[q*255 for q in mean_list], mask_fill_value = None,p=0.3),

    # Apply normalization; albumentations applies transfomation wrt to the max pixel whose default value is 255
    A.Normalize(mean=mean_list, std = std_list),

    #convert to tensor
    ToTensorV2(),
    ])


def get_test_transforms():

  '''

  Test data transformations

  '''
  return A.Compose([

    # Apply normalization; albumentations applies transfomation wrt to the max pixel whose default value is 255
    A.Normalize(mean=mean_list, std = std_list),

    #convert to tensor
    ToTensorV2(),

    ])


######### Step1 Transformations End   ###########







######## Get image from tensor ##############


def get_image_from_tensor(img_tensor):
    to_pil = v2.ToPILImage()

    # Unnormalize the image tensor
    for t, m, s in zip(img_tensor, mean_list, std_list):
        t.mul_(s).add_(m)

    # Convert the tensor to a PIL Image
    img = to_pil(img_tensor)

    return img






############################# Get CIFAR dataset and pass it to loader ###########################

def get_CIFARdataset_with_loader(datasettype,kwargs):
    '''
    <datasettype> can have values as 'train' or 'test'
    This function loads the CIFAR training and testing dataset.
    The datasets are loaded and then passed on for transformation

    For more information on tranformation execute the following lines:
    get_train_transforms??   # For training data transformation
    get_test_transforms??    # For testing data transformation

    '''

    if datasettype == 'train':
        train_data = getCIFAR10(datasets.CIFAR10('../data', train=True, download=True), transform=get_train_transforms())  # download and load the "training" data of CIFAR and apply test_transform
        print("Training data loaded successfully. Shape of data: ",train_data.dataset.data.shape)
        return train_data.dataset.class_to_idx, torch.utils.data.DataLoader(train_data, **kwargs)    # load train data
    elif datasettype == 'test':
        test_data = getCIFAR10(datasets.CIFAR10('../data', train=False, download=True), transform=get_test_transforms())   # download and load the "test" data of CIFAR and apply test_transform
        print("Testing data loaded successfully. Shape of data: ",test_data.dataset.data.shape)
        return test_data.dataset.class_to_idx, torch.utils.data.DataLoader(test_data, **kwargs)      # load test data
    else:
        raise ValueError('Incorrect dataset type string...pass valid name from available values')






########################## Define helper functions to train, test and measure #####################

def GetCorrectPredCount(pPrediction, pLabels):

  '''

  This function computes the number of correct predictions by comparing
  the predicted labels (with the highest probability) against the true labels.

  '''

  return pPrediction.eq(pLabels).sum().item()


def train(model, device, train_loader, optimizer, criterion, train_losses=[], train_acc=[]):

  '''

  This function trains the neural network using the training data

  '''

  model.train()  # Set the model to training mode
  pbar = tqdm(train_loader)  # Wrap the data loader with tqdm for a progress bar

  train_loss = 0  # Initialize total training loss
  correct = 0  # Initialize total number of correct predictions
  processed = 0  # Initialize total number of processed samples

  for batch_idx, (data, target) in enumerate(pbar):
    # Move the batch data and labels to the specified device (GPU)
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()  # Clear the gradients of all optimized variables
    # Forward pass: compute predicted outputs by passing inputs to the model
    pred = model(data)

    # Compute loss: calculate the batch loss by comparing predicted and true labels
    loss = criterion(pred, target)
    train_loss += loss.item()  # Aggregate the loss

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    optimizer.step()  # Perform a single optimization step (parameter update)

    predicted_ = pred.argmax(dim=1)  # Get the index of the max log-probability
    correct += GetCorrectPredCount(predicted_, target)  # Update total correct predictions for the batch
    processed += len(data)  # Update total processed samples of batch

    # Update progress bar description with current loss and accuracy
    pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  # Calculate and store the average accuracy and loss for this training epoch of training data
  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))


def test(model, device, test_loader, criterion, test_losses=[], test_acc=[]):

  '''

  This function sets the neural network in testing mode for inference

  '''

  model.eval()  # Set the model to evaluation mode
  test_loss = 0  # Initialize total test loss
  correct = 0  # Initialize total number of correct predictions
  misclassified_samples = []

  with torch.no_grad():  # Disable gradient calculation
    for batch_idx, (data, target) in enumerate(test_loader):
      # Move the batch data and labels to the specified device (GPU)
      data, target = data.to(device), target.to(device)

      output = model(data)  # Compute output by passing inputs to the model
      test_loss += criterion(output, target).item()  # Sum up batch loss

      predicted_ = output.argmax(dim=1)  # Get the index of the max log-probability
      correct += GetCorrectPredCount(predicted_, target)  # Update total correct predictions for each batch in test data

      if len(misclassified_samples)<10:
         for i in range(len(target)):
            if predicted_[i] != target[i]:
               misclassified_samples.append((data[i].cpu(), target[i].cpu(), predicted_[i].cpu()))
               if len(misclassified_samples) == 10:
                break

  # Calculate and store the average loss and accuracy for this test run
  test_loss /= len(test_loader.dataset)
  test_acc.append(100. * correct / len(test_loader.dataset))
  test_losses.append(test_loss)

  # Print test results
  print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

  return misclassified_samples




############################ Get optimizer #############################

def get_optimizer( model, optname="SGD", learningrate = 0.01,momentumval = 0.9):

  '''

  Get optimizer object of required algorith with learning rate and momentum. default optimizer = "SGD"

  # Momentum is a technique used to prevent GD from getting stuck in local minima
  Example of how it works internally:
    velocity = momentum * velocity - learning_rate * gradient
    parameters = parameters + velocity


  '''

  if optname == "SGD":
    return optim.SGD(model.parameters(), lr = learningrate, momentum=momentumval)
  else:
    raise ValueError('Incorrect optimizer algo string...pass valid name from available values')



############################ Get scheduler #############################


def get_scheduler(optimizer,stepsize=10,gamma_val = 0.1,verbose_bool = True):

  '''

  initialize scheduler for learning rate to be slower after n steps i.e after stepsize; learning rate updates to gamma*learningrate

  '''

  return optim.lr_scheduler.StepLR(optimizer, step_size= stepsize, gamma=gamma_val, verbose=verbose_bool)



############################ Get loss function #########################

def get_loss(loss_name="nll_loss"):

  '''

  get loss function object of specified loss criteria, default value = "nll_loss"

  '''

  if loss_name=="nll_loss":
    return F.nll_loss
  else:
    raise ValueError('Incorrect loss name string...pass valid name from available values')



################## Display Samples ###########################

def post_display(train_loader,label_map):

  '''

  Display some of the samples from training data

  '''

  # Get a batch of data and labels from the training DataLoader
  batch_data, batch_label = next(iter(train_loader))

    # Create a new figure for plotting
  fig, axes = plt.subplots(4, 4, figsize=(8, 8))



  # Loop through 16 samples in the batch
  for i in range(16):
      # Display the image (convert from tensor to numpy array) in RGB

      img_tensor = batch_data[i]

      img = get_image_from_tensor(img_tensor)

      # Convert the PIL Image to a numpy array and display it
      image = np.array(img)

      axes[i // 4, i % 4].imshow(image)

      # Set the title of the subplot to the corresponding label
      axes[i // 4, i % 4].set_title(label_map[batch_label[i].item()] + f", Label: {batch_label[i].item()}")

      # Remove x and y ticks for cleaner visualization
      axes[i // 4, i % 4].axis("off")

  # Ensure tight layout for better visualization
  plt.tight_layout()

  # Show the entire figure with subplots
  plt.show()




################## Display Accuract & Loss Plots ###########################

def post_accuracyplots(train_losses,test_losses,train_acc,test_acc):

  '''

  Plot Accuracy and Loss plots on training and testing

  '''

  fig, axs = plt.subplots(2,2,figsize=(15,10))

    # Set the title for the entire figure
  fig.suptitle(f"Plots", fontsize=16)

  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")





# ###############################

# def showmisclassifiedsamples(misclassified_samples,label_map,normtype):

#   '''

#   Display some of the samples from training data

#   '''

#     # Create a new figure for plotting
#   fig, axes = plt.subplots(2, 5, figsize=(12, 7))

#   # Set the title for the entire figure
#   fig.suptitle(f"Misclassified Images - {normtype}", fontsize=16)

#   # Loop through 16 samples in the batch
#   for i in range(len(misclassified_samples)):
#       one_sample = misclassified_samples[i]
#       # Display the image (convert from tensor to numpy array) in RGB

#       img_tensor = one_sample[0].clone().detach()
#       img = get_image_from_tensor(img_tensor)

#       # Convert the PIL Image to a numpy array and display it
#       image = np.array(img)

#       axes[i // 5, i % 5].imshow(image)

#       # Set the title of the subplot to the corresponding label
#       axes[i // 5, i % 5].set_title(f"Actual: {label_map[one_sample[1].item()]}"+" , "+f"Predicted: {label_map[one_sample[2].item()]}", fontsize=8)

#       # Remove x and y ticks for cleaner visualization
#       axes[i // 5, i % 5].axis("off")

#   # Ensure tight layout for better visualization
#   plt.tight_layout()

#   # Show the entire figure with subplots
#   plt.show()