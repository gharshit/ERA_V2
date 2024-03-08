import torch
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F



#################################### Define the data transformations ######################
def get_train_transforms():

  '''

  Train data transformations

  '''

  return v2.Compose([
    # Randomly apply CenterCrop(22) with a 10% probability
    # Purpose: Introduce variability during training by randomly cropping images.
    v2.RandomApply([v2.CenterCrop(22), ], p=0.1),


    # Randomly rotate the image by an angle between -15 and +15 degrees
    # fill = 0 to fill with black pixels as background
    # Purpose: Augment the training data by introducing random rotations.
    v2.RandomRotation((-15., 15.), fill=0),


    # Resize the image to 28x28 pixels (works through interpolation)
    # Purpose: Standardize the image size for model input.(To maintain the consistent size of the images, resize should be done after all cropping/rotation which have the possibility of altering the image size)
    v2.Resize((28, 28)),

    # Convert the image to a PyTorch tensor
    # Purpose: Prepare the image data for neural network input.
    # transforms.ToTensor(),  # this works in PytorchV1
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),

    # Normalize the pixel values using mean 0.1307 and standard deviation 0.3081
    # Purpose: Normalize pixel values to improve model convergence and performance.
    v2.Normalize(mean=[0.1307], std = [0.3081]),
    ])


def get_test_transforms():

  '''

  Test data transformations

  '''
  return v2.Compose([
    # Convert the image to a PyTorch tensor
    # Purpose: Prepare the image data for neural network input.
    # transforms.ToTensor(),  # this works in PytorchV1
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),

    # Normalize the pixel values using mean 0.1307 and standard deviation 0.3081
    # Purpose: Normalize pixel values to improve model convergence and performance.
    v2.Normalize(mean=[0.1307], std = [0.3081]),
    ])



############################# Get MNIST dataset and pass it to loader ###########################

def get_MNISTdataset_with_loader(datasettype,kwargs):

  '''
  <datasettype> can have values as 'train' and 'test'
  This function loads the MNIST training and testing dataset.
  The datasets are loaded and then passed on for transformation

  For more information on tranformation execute the following lines:
  get_train_transforms??   # For training data transformation
  get_test_transforms??    # For testing data transformation

  '''

  if datasettype == 'train':
    train_data = datasets.MNIST('../data', train=True, download=True, transform=get_train_transforms())  # download and load the "training" data of MNIST and apply train_transform
    print("Training data loaded successfully...")
    return torch.utils.data.DataLoader(train_data, **kwargs) # load train data
  elif datasettype == 'test':
    test_data = datasets.MNIST('../data', train=False, download=True, transform=get_test_transforms())   # download and load the "test" data of MNIST and apply test_transform
    print("Testing data loaded successfully...")
    return torch.utils.data.DataLoader(test_data, **kwargs) # load test data 
  else:
    raise ValueError('Incorrect dataset type string...pass valid name from available values')






########################## Define helper functions to train, test and measure #####################

def GetCorrectPredCount(pPrediction, pLabels):

  '''

  This function computes the number of correct predictions by comparing
  the predicted labels (with the highest probability) against the true labels.

  '''

  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


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

    correct += GetCorrectPredCount(pred, target)  # Update total correct predictions for the batch
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

  with torch.no_grad():  # Disable gradient calculation
    for batch_idx, (data, target) in enumerate(test_loader):
      # Move the batch data and labels to the specified device (GPU)
      data, target = data.to(device), target.to(device)

      output = model(data)  # Compute output by passing inputs to the model
      test_loss += criterion(output, target).item()  # Sum up batch loss
      correct += GetCorrectPredCount(output, target)  # Update total correct predictions for each batch in test data

  # Calculate and store the average loss and accuracy for this test run
  test_loss /= len(test_loader.dataset)
  test_acc.append(100. * correct / len(test_loader.dataset))
  test_losses.append(test_loss)

  # Print test results
  print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))




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
  
def post_display(train_loader):
  
  '''

  Display some of the samples from training data

  '''

  # Get a batch of data and labels from the training DataLoader
  batch_data, batch_label = next(iter(train_loader))
  
  # Create a new figure for plotting
  fig = plt.figure()

  # Loop through 12 samples in the batch
  for i in range(16):
      # Create a subplot grid (3 rows, 4 columns)
      plt.subplot(4, 4, i + 1)

      # Ensure tight layout for better visualization
      plt.tight_layout()

      # Display the image (convert from tensor to numpy array) in grayscale
      plt.imshow(batch_data[i].squeeze(0), cmap='gray')

      # Set the title of the subplot to the corresponding label
      plt.title(batch_label[i].item())

      # Remove x and y ticks for cleaner visualization
      plt.xticks([])
      plt.yticks([])

  # Show the entire figure with subplots
  plt.show()



################## Display Accuract & Loss Plots ###########################
  
def post_accuracyplots(train_losses,test_losses,train_acc,test_acc):

  '''

  Plot Accuracy and Loss plots on training and testing

  '''

  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")



