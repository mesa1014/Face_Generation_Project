
# Face Generation

In this project, you'll define and train a DCGAN on a dataset of faces. Your goal is to get a generator network to generate *new* images of faces that look as realistic as possible!

The project will be broken down into a series of tasks from **loading in data to defining and training adversarial networks**. At the end of the notebook, you'll be able to visualize the results of your trained Generator to see how it performs; your generated samples should look like fairly realistic faces with small amounts of noise.

### Get the Data

You'll be using the [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to train your adversarial networks.

This dataset is more complex than the number datasets (like MNIST or SVHN) you've been working with, and so, you should prepare to define deeper networks and train them for a longer time to get good results. It is suggested that you utilize a GPU for training.

### Pre-processed Data

Since the project's main focus is on building the GANs, we've done *some* of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. Some sample data is show below.

<img src='notebook_images/processed_face_data.png' width=60% />

> If you are working locally, you can download this data [by clicking here](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip)

This is a zip file that you'll need to extract in the home directory of this notebook for further loading and processing. After extracting the data, you should be left with a directory of data `processed_celeba_small/`


```python
# can comment out after executing
#!unzip processed_celeba_small.zip
```


```python
data_dir = 'processed_celeba_small/'

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import problem_unittests as tests
#import helper

%matplotlib inline
```

## Visualize the CelebA Data

The [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations. Since you're going to be generating faces, you won't need the annotations, you'll only need the images. Note that these are color images with [3 color channels (RGB)](https://en.wikipedia.org/wiki/Channel_(digital_image)#RGB_Images) each.

### Pre-process and Load the Data

Since the project's main focus is on building the GANs, we've done *some* of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. This *pre-processed* dataset is a smaller subset of the very large CelebA data.

> There are a few other steps that you'll need to **transform** this data and create a **DataLoader**.

#### Exercise: Complete the following `get_dataloader` function, such that it satisfies these requirements:

* Your images should be square, Tensor images of size `image_size x image_size` in the x and y dimension.
* Your function should return a DataLoader that shuffles and batches these Tensor images.

#### ImageFolder

To create a dataset given a directory of images, it's recommended that you use PyTorch's [ImageFolder](https://pytorch.org/docs/0.4.0/torchvision/datasets.html#imagefolder) wrapper, with a root directory `processed_celeba_small/` and data transformation passed in.


```python
# necessary imports
import torch
from torchvision import datasets
from torchvision import transforms
```


```python
def get_dataloader(batch_size, image_size, data_dir='processed_celeba_small/'):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param img_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """

    # TODO: Implement function and return a dataloader
    # Tensor transform
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor()])

    # datasets
    dataset = datasets.ImageFolder(data_dir, transform)


    # build DataLoaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


    return data_loader

```

## Create a DataLoader

#### Exercise: Create a DataLoader `celeba_train_loader` with appropriate hyperparameters.

Call the above function and create a dataloader to view images.
* You can decide on any reasonable `batch_size` parameter
* Your `image_size` **must be** `32`. Resizing the data to a smaller size will make for faster training, while still creating convincing images of faces!


```python
# Define function hyperparameters
batch_size = 64
img_size = 32

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Call your function and get a dataloader
celeba_train_loader = get_dataloader(batch_size, img_size)

```

Next, you can view some images! You should seen square images of somewhat-centered faces.

Note: You'll need to convert the Tensor images into a NumPy type and transpose the dimensions to correctly display an image, suggested `imshow` code is below, but it may not be perfect.


```python
# helper display function
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# obtain one batch of training images
dataiter = iter(celeba_train_loader)
images, _ = dataiter.next() # _ for no labels

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(20, 4))
plot_size=20
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
```


![png](/notebook_images/output_9_0.png)


#### Exercise: Pre-process your image data and scale it to a pixel range of -1 to 1

You need to do a bit of pre-processing; you know that the output of a `tanh` activated generator will contain pixel values in a range from -1 to 1, and so, we need to rescale our training images to a range of -1 to 1. (Right now, they are in a range from 0-1.)


```python
# TODO: Complete the scale function
def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1.
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min, max = feature_range
    x = x * (max - min) + min

    return x

```


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# check scaled range
# should be close to -1 to 1
img = images[0]
scaled_img = scale(img)

print('Min: ', scaled_img.min())
print('Max: ', scaled_img.max())
```

    Min:  tensor(-0.9843)
    Max:  tensor(0.9137)


---
# Define the Model

A GAN is comprised of two adversarial networks, a discriminator and a generator.

## Discriminator

Your first task will be to define the discriminator. This is a convolutional classifier like you've built before, only without any maxpooling layers. To deal with this complex data, it's suggested you use a deep network with **normalization**. You are also allowed to create any helper functions that may be useful.

#### Exercise: Complete the Discriminator class
* The inputs to the discriminator are 32x32x3 tensor images
* The output should be a single value that will indicate whether a given image is real or fake



```python
import torch.nn as nn
import torch.nn.functional as F
```


```python
# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels,
                           kernel_size, stride, padding, bias=False)

    # append conv layer
    layers.append(conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))

    # using Sequential container
    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim

        self.conv1 = conv(3, conv_dim, 4, batch_norm=False) # first layer, no batch_norm
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)

        # final, fully-connected layer
        self.fc = nn.Linear(conv_dim*4*4*4, 1)


    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)

        # flatten
        x = x.view(-1, self.conv_dim*4*4*4)

        # final output layer
        x = self.fc(x)  

        return x


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_discriminator(Discriminator)
```

    Tests Passed


## Generator

The generator should upsample an input and generate a *new* image of the same size as our training data `32x32x3`. This should be mostly transpose convolutional layers with normalization applied to the outputs.

#### Exercise: Complete the Generator class
* The inputs to the generator are vectors of some length `z_size`
* The output should be a image of shape `32x32x3`


```python
# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    # create a sequence of transpose + optional batch norm layers
    layers = []
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels,
                                              kernel_size, stride, padding, bias=False)
    # append transpose convolutional layer
    layers.append(transpose_conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)

class Generator(nn.Module):

    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim

        # first, fully-connected layer
        self.fc = nn.Linear(z_size, conv_dim*4*4*4)

        # transpose conv layers
        self.t_conv1 = deconv(conv_dim*4, conv_dim*2, 4)
        self.t_conv2 = deconv(conv_dim*2, conv_dim, 4)
        self.t_conv3 = deconv(conv_dim, 3, 4, batch_norm=False)


    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        x = self.fc(x)
        x = x.view(-1, self.conv_dim*4, 4, 4) # (batch_size, depth, 4, 4)

        # hidden transpose conv layers + relu
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))

        # last layer + tanh activation
        x = self.t_conv3(x)
        x = F.tanh(x)

        return x

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_generator(Generator)
```

    Tests Passed


## Initialize the weights of your networks

To help your models converge, you should initialize the weights of the convolutional and linear layers in your model. From reading the [original DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf), they say:
> All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02.

So, your next task will be to define a weight initialization function that does just this!

You can refer back to the lesson on weight initialization or even consult existing model code, such as that from [the `networks.py` file in CycleGAN Github repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py) to help you complete this function.

#### Exercise: Complete the weight initialization function

* This should initialize only **convolutional** and **linear** layers
* Initialize the weights to a normal distribution, centered around 0, with a standard deviation of 0.02.
* The bias terms, if they exist, may be left alone or set to 0.


```python
def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__

    # TODO: Apply initial weights to convolutional and linear layers
    if classname.find('Linear') != -1:
        # get the number of the inputs
        m.weight.data.normal_(0, 0.02)
        m.bias.data.fill_(0)


```

## Build complete network

Define your models' hyperparameters and instantiate the discriminator and generator from the classes defined above. Make sure you've passed in the correct input arguments.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    print(D)
    print()
    print(G)

    return D, G

```

#### Exercise: Define model hyperparameters


```python
# Define model hyperparams
d_conv_dim = 32
g_conv_dim = 32
z_size = 100

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
D, G = build_network(d_conv_dim, g_conv_dim, z_size)
```

    Discriminator(
      (conv1): Sequential(
        (0): Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      )
      (conv2): Sequential(
        (0): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Sequential(
        (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (fc): Linear(in_features=2048, out_features=1, bias=True)
    )

    Generator(
      (fc): Linear(in_features=100, out_features=2048, bias=True)
      (t_conv1): Sequential(
        (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (t_conv2): Sequential(
        (0): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (t_conv3): Sequential(
        (0): ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      )
    )


### Training on GPU

Check if you can train on GPU. Here, we'll set this as a boolean variable `train_on_gpu`. Later, you'll be responsible for making sure that
>* Models,
* Model inputs, and
* Loss function arguments

Are moved to GPU, where appropriate.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import torch

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')
```

    Training on GPU!


---
## Discriminator and Generator Losses

Now we need to calculate the losses for both types of adversarial networks.

### Discriminator Losses

> * For the discriminator, the total loss is the sum of the losses for real and fake images, `d_loss = d_real_loss + d_fake_loss`.
* Remember that we want the discriminator to output 1 for real images and 0 for fake images, so we need to set up the losses to reflect that.


### Generator Loss

The generator loss will look similar only with flipped labels. The generator's goal is to get the discriminator to *think* its generated images are *real*.

#### Exercise: Complete real and fake loss functions

**You may choose to use either cross entropy or a least squares error loss to complete the following `real_loss` and `fake_loss` functions.**


```python
def real_loss(D_out):
    '''Calculates how close discriminator outputs are to being real.
       param, D_out: discriminator logits
       return: real loss'''
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size) # real labels = 1
    # move labels to GPU if available     
    if train_on_gpu:
        labels = labels.cuda()
    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    '''Calculates how close discriminator outputs are to being fake.
       param, D_out: discriminator logits
       return: fake loss'''
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    # move labels to GPU if available     
    if train_on_gpu:
        labels = labels.cuda()
    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss
```

## Optimizers

#### Exercise: Define optimizers for your Discriminator (D) and Generator (G)

Define optimizers for your models with appropriate hyperparameters.


```python
import torch.optim as optim

# Create optimizers for the discriminator D and generator G

# params
lr = 0.0001
beta1=0.5
beta2=0.999 # default value

# Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])
```

---
## Training

Training will involve alternating between training the discriminator and the generator. You'll use your functions `real_loss` and `fake_loss` to help you calculate the discriminator losses.

* You should train the discriminator by alternating on real and fake images
* Then the generator, which tries to trick the discriminator and should have an opposing loss function


#### Saving Samples

You've been given some code to print out some loss statistics and save some generated "fake" samples.

#### Exercise: Complete the training function

Keep in mind that, if you've moved your models to GPU, you'll also have to move any model inputs to GPU.


```python
def train(D, G, n_epochs, print_every=50):
    '''Trains adversarial networks for some number of epochs
       param, D: the discriminator network
       param, G: the generator network
       param, n_epochs: number of epochs to train for
       param, print_every: when to print and record the models' losses
       return: D and G losses'''

    # move models to GPU
    if train_on_gpu:
        D.cuda()
        G.cuda()

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(celeba_train_loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)

            # ===============================================
            #         YOUR CODE HERE: TRAIN THE NETWORKS
            # ===============================================

            # 1. Train the discriminator on real and fake images
            d_optimizer.zero_grad()

            # 1. Train with real images

            # Compute the discriminator losses on real images
            if train_on_gpu:
                real_images = real_images.cuda()

            D_real = D(real_images)
            d_real_loss = real_loss(D_real)

            # 2. Train with fake images

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            # move x to GPU, if available
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)

            # Compute the discriminator losses on fake images            
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)

            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # 2. Train the generator with an adversarial loss
            g_optimizer.zero_grad()

            # 1. Train with fake images and flipped labels

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)

            # Compute the discriminator losses on fake images
            # using flipped labels!
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake) # use real loss to flip labels

            # perform backprop
            g_loss.backward()
            g_optimizer.step()


            # ===============================================
            #              END OF YOUR CODE
            # ===============================================

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))


        ## AFTER EACH EPOCH##    
        # this code assumes your generator is named G, feel free to change the name
        # generate and save sample, fake images
        G.eval() # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train() # back to training mode

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    # finally return losses
    return losses
```

Set your number of training epochs and train your GAN!


```python
# set number of epochs
n_epochs = 25


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# call training function
losses = train(D, G, n_epochs=n_epochs)
```

    Epoch [    1/   25] | d_loss: 1.4161 | g_loss: 0.6742
    Epoch [    1/   25] | d_loss: 0.3868 | g_loss: 1.7630
    Epoch [    1/   25] | d_loss: 0.3072 | g_loss: 2.1768
    Epoch [    1/   25] | d_loss: 0.2832 | g_loss: 2.3627
    Epoch [    1/   25] | d_loss: 0.3249 | g_loss: 2.4847
    Epoch [    1/   25] | d_loss: 0.4415 | g_loss: 2.3509
    Epoch [    1/   25] | d_loss: 0.5082 | g_loss: 1.8528
    Epoch [    1/   25] | d_loss: 0.7061 | g_loss: 1.6492
    Epoch [    1/   25] | d_loss: 0.8360 | g_loss: 1.6600
    Epoch [    1/   25] | d_loss: 0.6129 | g_loss: 1.9572
    Epoch [    1/   25] | d_loss: 0.7325 | g_loss: 1.5320
    Epoch [    1/   25] | d_loss: 0.8651 | g_loss: 1.5865
    Epoch [    1/   25] | d_loss: 0.7887 | g_loss: 1.3625
    Epoch [    1/   25] | d_loss: 0.7523 | g_loss: 1.4845
    Epoch [    1/   25] | d_loss: 0.8214 | g_loss: 1.9528
    Epoch [    1/   25] | d_loss: 0.9039 | g_loss: 1.5470
    Epoch [    1/   25] | d_loss: 0.8675 | g_loss: 1.3767
    Epoch [    1/   25] | d_loss: 0.9191 | g_loss: 1.4801
    Epoch [    1/   25] | d_loss: 0.9372 | g_loss: 0.9653
    Epoch [    1/   25] | d_loss: 0.9632 | g_loss: 0.8673
    Epoch [    1/   25] | d_loss: 1.1384 | g_loss: 1.6962
    Epoch [    1/   25] | d_loss: 0.9776 | g_loss: 1.0979
    Epoch [    1/   25] | d_loss: 0.9837 | g_loss: 0.9046
    Epoch [    1/   25] | d_loss: 0.9928 | g_loss: 1.0274
    Epoch [    1/   25] | d_loss: 0.9588 | g_loss: 1.0078
    Epoch [    1/   25] | d_loss: 0.9655 | g_loss: 1.2063
    Epoch [    1/   25] | d_loss: 0.8350 | g_loss: 1.2380
    Epoch [    1/   25] | d_loss: 1.0200 | g_loss: 0.9606
    Epoch [    1/   25] | d_loss: 0.8394 | g_loss: 1.5851
    Epoch [    2/   25] | d_loss: 0.9360 | g_loss: 1.7330
    Epoch [    2/   25] | d_loss: 1.0013 | g_loss: 0.7645
    Epoch [    2/   25] | d_loss: 0.9924 | g_loss: 1.0625
    Epoch [    2/   25] | d_loss: 0.9966 | g_loss: 1.1266
    Epoch [    2/   25] | d_loss: 1.0943 | g_loss: 0.9484
    Epoch [    2/   25] | d_loss: 0.9682 | g_loss: 1.3340
    Epoch [    2/   25] | d_loss: 0.8346 | g_loss: 1.3421
    Epoch [    2/   25] | d_loss: 1.0859 | g_loss: 1.2440
    Epoch [    2/   25] | d_loss: 1.0835 | g_loss: 1.0085
    Epoch [    2/   25] | d_loss: 1.2788 | g_loss: 1.7171
    Epoch [    2/   25] | d_loss: 1.2005 | g_loss: 0.7756
    Epoch [    2/   25] | d_loss: 1.1420 | g_loss: 0.9110
    Epoch [    2/   25] | d_loss: 0.9820 | g_loss: 1.1315
    Epoch [    2/   25] | d_loss: 0.9737 | g_loss: 0.9385
    Epoch [    2/   25] | d_loss: 0.9066 | g_loss: 1.4482
    Epoch [    2/   25] | d_loss: 1.0415 | g_loss: 0.9935
    Epoch [    2/   25] | d_loss: 1.3469 | g_loss: 0.7348
    Epoch [    2/   25] | d_loss: 1.0636 | g_loss: 1.4137
    Epoch [    2/   25] | d_loss: 0.9227 | g_loss: 1.2535
    Epoch [    2/   25] | d_loss: 1.0712 | g_loss: 1.1009
    Epoch [    2/   25] | d_loss: 0.8652 | g_loss: 1.5568
    Epoch [    2/   25] | d_loss: 1.1326 | g_loss: 1.1381
    Epoch [    2/   25] | d_loss: 1.1338 | g_loss: 0.9074
    Epoch [    2/   25] | d_loss: 1.0637 | g_loss: 0.7581
    Epoch [    2/   25] | d_loss: 0.9489 | g_loss: 0.9380
    Epoch [    2/   25] | d_loss: 1.1110 | g_loss: 0.8606
    Epoch [    2/   25] | d_loss: 0.8257 | g_loss: 1.4913
    Epoch [    2/   25] | d_loss: 1.1122 | g_loss: 0.9420
    Epoch [    2/   25] | d_loss: 1.0878 | g_loss: 0.9247
    Epoch [    3/   25] | d_loss: 1.2118 | g_loss: 1.2268
    Epoch [    3/   25] | d_loss: 1.3281 | g_loss: 0.7496
    Epoch [    3/   25] | d_loss: 1.1388 | g_loss: 1.0303
    Epoch [    3/   25] | d_loss: 1.1830 | g_loss: 0.9915
    Epoch [    3/   25] | d_loss: 0.9893 | g_loss: 0.5572
    Epoch [    3/   25] | d_loss: 1.0269 | g_loss: 1.2588
    Epoch [    3/   25] | d_loss: 1.0492 | g_loss: 1.1510
    Epoch [    3/   25] | d_loss: 1.2025 | g_loss: 0.8900
    Epoch [    3/   25] | d_loss: 1.0266 | g_loss: 1.0273
    Epoch [    3/   25] | d_loss: 1.0828 | g_loss: 1.0645
    Epoch [    3/   25] | d_loss: 0.8842 | g_loss: 1.0730
    Epoch [    3/   25] | d_loss: 1.0424 | g_loss: 1.2264
    Epoch [    3/   25] | d_loss: 1.1318 | g_loss: 1.0557
    Epoch [    3/   25] | d_loss: 1.1179 | g_loss: 1.2485
    Epoch [    3/   25] | d_loss: 1.1370 | g_loss: 0.6591
    Epoch [    3/   25] | d_loss: 0.8790 | g_loss: 1.0498
    Epoch [    3/   25] | d_loss: 0.8662 | g_loss: 1.3828
    Epoch [    3/   25] | d_loss: 1.2219 | g_loss: 0.8693
    Epoch [    3/   25] | d_loss: 1.0331 | g_loss: 1.0340
    Epoch [    3/   25] | d_loss: 1.2096 | g_loss: 1.4265
    Epoch [    3/   25] | d_loss: 0.9340 | g_loss: 0.7650
    Epoch [    3/   25] | d_loss: 1.2985 | g_loss: 0.8368
    Epoch [    3/   25] | d_loss: 1.1435 | g_loss: 0.9407
    Epoch [    3/   25] | d_loss: 1.0624 | g_loss: 0.8047
    Epoch [    3/   25] | d_loss: 1.2683 | g_loss: 1.1476
    Epoch [    3/   25] | d_loss: 1.0606 | g_loss: 1.0145
    Epoch [    3/   25] | d_loss: 1.2336 | g_loss: 1.1633
    Epoch [    3/   25] | d_loss: 1.2031 | g_loss: 0.7202
    Epoch [    3/   25] | d_loss: 1.0275 | g_loss: 1.1428
    Epoch [    4/   25] | d_loss: 1.2200 | g_loss: 1.0999
    Epoch [    4/   25] | d_loss: 1.0789 | g_loss: 1.0184
    Epoch [    4/   25] | d_loss: 1.1118 | g_loss: 0.9404
    Epoch [    4/   25] | d_loss: 0.9789 | g_loss: 0.9856
    Epoch [    4/   25] | d_loss: 1.1056 | g_loss: 0.7834
    Epoch [    4/   25] | d_loss: 1.0126 | g_loss: 1.0337
    Epoch [    4/   25] | d_loss: 1.1729 | g_loss: 0.7003
    Epoch [    4/   25] | d_loss: 0.8950 | g_loss: 1.0926
    Epoch [    4/   25] | d_loss: 1.2494 | g_loss: 1.0229
    Epoch [    4/   25] | d_loss: 1.1081 | g_loss: 0.7922
    Epoch [    4/   25] | d_loss: 1.1762 | g_loss: 1.4731
    Epoch [    4/   25] | d_loss: 1.0714 | g_loss: 1.0264
    Epoch [    4/   25] | d_loss: 0.8883 | g_loss: 0.9962
    Epoch [    4/   25] | d_loss: 1.0070 | g_loss: 1.1817
    Epoch [    4/   25] | d_loss: 0.9634 | g_loss: 0.9232
    Epoch [    4/   25] | d_loss: 1.1466 | g_loss: 1.0064
    Epoch [    4/   25] | d_loss: 1.0967 | g_loss: 1.2883
    Epoch [    4/   25] | d_loss: 0.8545 | g_loss: 1.1549
    Epoch [    4/   25] | d_loss: 0.8445 | g_loss: 1.4967
    Epoch [    4/   25] | d_loss: 1.1559 | g_loss: 0.7060
    Epoch [    4/   25] | d_loss: 0.9501 | g_loss: 1.2408
    Epoch [    4/   25] | d_loss: 1.2104 | g_loss: 0.9369
    Epoch [    4/   25] | d_loss: 1.1245 | g_loss: 0.8736
    Epoch [    4/   25] | d_loss: 1.3882 | g_loss: 1.1966
    Epoch [    4/   25] | d_loss: 1.0839 | g_loss: 0.9391
    Epoch [    4/   25] | d_loss: 1.1642 | g_loss: 0.9419
    Epoch [    4/   25] | d_loss: 1.2151 | g_loss: 0.8490
    Epoch [    4/   25] | d_loss: 0.9797 | g_loss: 1.1378
    Epoch [    4/   25] | d_loss: 1.0442 | g_loss: 1.0468
    Epoch [    5/   25] | d_loss: 1.2317 | g_loss: 0.8917
    Epoch [    5/   25] | d_loss: 1.1855 | g_loss: 1.1086
    Epoch [    5/   25] | d_loss: 1.0429 | g_loss: 1.0303
    Epoch [    5/   25] | d_loss: 0.9110 | g_loss: 1.3952
    Epoch [    5/   25] | d_loss: 1.0077 | g_loss: 1.1267
    Epoch [    5/   25] | d_loss: 1.2128 | g_loss: 0.9828
    Epoch [    5/   25] | d_loss: 0.9124 | g_loss: 1.3280
    Epoch [    5/   25] | d_loss: 1.0790 | g_loss: 0.7074
    Epoch [    5/   25] | d_loss: 1.0149 | g_loss: 1.0249
    Epoch [    5/   25] | d_loss: 0.8526 | g_loss: 1.0633
    Epoch [    5/   25] | d_loss: 1.1903 | g_loss: 0.9956
    Epoch [    5/   25] | d_loss: 1.0376 | g_loss: 1.0037
    Epoch [    5/   25] | d_loss: 1.0428 | g_loss: 0.9766
    Epoch [    5/   25] | d_loss: 1.0491 | g_loss: 0.9334
    Epoch [    5/   25] | d_loss: 1.2537 | g_loss: 0.8560
    Epoch [    5/   25] | d_loss: 1.1087 | g_loss: 0.9173
    Epoch [    5/   25] | d_loss: 1.1157 | g_loss: 1.3411
    Epoch [    5/   25] | d_loss: 1.3934 | g_loss: 0.7130
    Epoch [    5/   25] | d_loss: 0.9262 | g_loss: 0.9615
    Epoch [    5/   25] | d_loss: 1.1437 | g_loss: 1.0072
    Epoch [    5/   25] | d_loss: 1.0579 | g_loss: 1.1305
    Epoch [    5/   25] | d_loss: 1.0181 | g_loss: 0.9799
    Epoch [    5/   25] | d_loss: 0.9706 | g_loss: 0.8196
    Epoch [    5/   25] | d_loss: 1.0973 | g_loss: 0.9720
    Epoch [    5/   25] | d_loss: 0.9488 | g_loss: 1.0006
    Epoch [    5/   25] | d_loss: 0.9505 | g_loss: 1.0618
    Epoch [    5/   25] | d_loss: 1.0930 | g_loss: 1.1455
    Epoch [    5/   25] | d_loss: 1.0264 | g_loss: 1.1381
    Epoch [    5/   25] | d_loss: 1.1099 | g_loss: 0.8651
    Epoch [    6/   25] | d_loss: 1.1121 | g_loss: 1.0911
    Epoch [    6/   25] | d_loss: 1.0698 | g_loss: 1.3098
    Epoch [    6/   25] | d_loss: 1.0049 | g_loss: 0.9020
    Epoch [    6/   25] | d_loss: 0.7109 | g_loss: 1.2741
    Epoch [    6/   25] | d_loss: 0.9985 | g_loss: 1.3052
    Epoch [    6/   25] | d_loss: 0.9896 | g_loss: 1.4528
    Epoch [    6/   25] | d_loss: 0.9242 | g_loss: 0.9478
    Epoch [    6/   25] | d_loss: 1.2713 | g_loss: 1.2507
    Epoch [    6/   25] | d_loss: 0.7514 | g_loss: 0.6859
    Epoch [    6/   25] | d_loss: 0.9191 | g_loss: 1.2297
    Epoch [    6/   25] | d_loss: 0.8704 | g_loss: 1.3748
    Epoch [    6/   25] | d_loss: 1.0682 | g_loss: 1.1086
    Epoch [    6/   25] | d_loss: 1.0891 | g_loss: 0.9718
    Epoch [    6/   25] | d_loss: 1.0033 | g_loss: 0.8543
    Epoch [    6/   25] | d_loss: 0.9542 | g_loss: 1.0871
    Epoch [    6/   25] | d_loss: 0.9793 | g_loss: 1.2942
    Epoch [    6/   25] | d_loss: 1.1603 | g_loss: 1.1420
    Epoch [    6/   25] | d_loss: 1.0006 | g_loss: 1.2282
    Epoch [    6/   25] | d_loss: 1.0701 | g_loss: 1.1932
    Epoch [    6/   25] | d_loss: 0.8777 | g_loss: 1.1318
    Epoch [    6/   25] | d_loss: 0.7627 | g_loss: 1.2192
    Epoch [    6/   25] | d_loss: 1.0900 | g_loss: 1.2272
    Epoch [    6/   25] | d_loss: 0.9998 | g_loss: 1.3268
    Epoch [    6/   25] | d_loss: 0.9069 | g_loss: 1.4363
    Epoch [    6/   25] | d_loss: 1.0406 | g_loss: 0.9714
    Epoch [    6/   25] | d_loss: 0.9453 | g_loss: 0.8800
    Epoch [    6/   25] | d_loss: 1.0022 | g_loss: 1.2016
    Epoch [    6/   25] | d_loss: 1.1604 | g_loss: 0.8363
    Epoch [    6/   25] | d_loss: 0.9898 | g_loss: 0.8795
    Epoch [    7/   25] | d_loss: 1.0820 | g_loss: 0.8488
    Epoch [    7/   25] | d_loss: 1.0176 | g_loss: 1.1208
    Epoch [    7/   25] | d_loss: 0.9258 | g_loss: 1.3274
    Epoch [    7/   25] | d_loss: 0.9213 | g_loss: 0.9510
    Epoch [    7/   25] | d_loss: 0.9758 | g_loss: 0.7853
    Epoch [    7/   25] | d_loss: 0.7825 | g_loss: 1.2230
    Epoch [    7/   25] | d_loss: 1.0702 | g_loss: 1.2085
    Epoch [    7/   25] | d_loss: 0.8303 | g_loss: 1.0185
    Epoch [    7/   25] | d_loss: 0.9218 | g_loss: 0.9370
    Epoch [    7/   25] | d_loss: 0.8405 | g_loss: 1.2809
    Epoch [    7/   25] | d_loss: 1.0669 | g_loss: 1.2048
    Epoch [    7/   25] | d_loss: 0.8536 | g_loss: 1.4214
    Epoch [    7/   25] | d_loss: 1.0221 | g_loss: 0.7438
    Epoch [    7/   25] | d_loss: 0.8006 | g_loss: 1.4918
    Epoch [    7/   25] | d_loss: 1.3947 | g_loss: 0.7861
    Epoch [    7/   25] | d_loss: 0.8849 | g_loss: 0.9825
    Epoch [    7/   25] | d_loss: 1.0016 | g_loss: 1.2536
    Epoch [    7/   25] | d_loss: 1.1330 | g_loss: 0.8181
    Epoch [    7/   25] | d_loss: 1.0452 | g_loss: 1.1186
    Epoch [    7/   25] | d_loss: 0.8886 | g_loss: 1.1311
    Epoch [    7/   25] | d_loss: 0.9994 | g_loss: 0.7332
    Epoch [    7/   25] | d_loss: 0.8674 | g_loss: 1.2930
    Epoch [    7/   25] | d_loss: 0.9091 | g_loss: 1.0381
    Epoch [    7/   25] | d_loss: 1.1422 | g_loss: 0.7729
    Epoch [    7/   25] | d_loss: 1.0084 | g_loss: 1.4699
    Epoch [    7/   25] | d_loss: 1.1678 | g_loss: 1.1567
    Epoch [    7/   25] | d_loss: 0.8390 | g_loss: 1.6388
    Epoch [    7/   25] | d_loss: 0.8753 | g_loss: 1.4353
    Epoch [    7/   25] | d_loss: 0.8649 | g_loss: 1.4344
    Epoch [    8/   25] | d_loss: 0.9673 | g_loss: 1.2410
    Epoch [    8/   25] | d_loss: 0.8553 | g_loss: 0.9731
    Epoch [    8/   25] | d_loss: 1.0092 | g_loss: 1.2696
    Epoch [    8/   25] | d_loss: 0.8753 | g_loss: 0.6726
    Epoch [    8/   25] | d_loss: 0.7527 | g_loss: 1.2533
    Epoch [    8/   25] | d_loss: 0.8731 | g_loss: 0.9865
    Epoch [    8/   25] | d_loss: 0.9388 | g_loss: 2.1057
    Epoch [    8/   25] | d_loss: 0.6800 | g_loss: 1.9668
    Epoch [    8/   25] | d_loss: 1.1867 | g_loss: 0.8265
    Epoch [    8/   25] | d_loss: 0.9176 | g_loss: 1.1113
    Epoch [    8/   25] | d_loss: 0.8666 | g_loss: 1.0052
    Epoch [    8/   25] | d_loss: 0.8002 | g_loss: 1.1204
    Epoch [    8/   25] | d_loss: 0.8824 | g_loss: 1.4182
    Epoch [    8/   25] | d_loss: 0.8136 | g_loss: 1.4255
    Epoch [    8/   25] | d_loss: 0.9524 | g_loss: 1.1338
    Epoch [    8/   25] | d_loss: 0.9392 | g_loss: 1.1928
    Epoch [    8/   25] | d_loss: 1.0125 | g_loss: 1.1310
    Epoch [    8/   25] | d_loss: 0.9796 | g_loss: 1.4749
    Epoch [    8/   25] | d_loss: 1.0668 | g_loss: 1.4210
    Epoch [    8/   25] | d_loss: 0.9672 | g_loss: 1.5199
    Epoch [    8/   25] | d_loss: 0.9761 | g_loss: 1.0375
    Epoch [    8/   25] | d_loss: 0.8495 | g_loss: 1.2817
    Epoch [    8/   25] | d_loss: 1.1192 | g_loss: 1.2998
    Epoch [    8/   25] | d_loss: 0.8639 | g_loss: 0.7721
    Epoch [    8/   25] | d_loss: 0.7687 | g_loss: 1.6451
    Epoch [    8/   25] | d_loss: 0.6508 | g_loss: 1.5002
    Epoch [    8/   25] | d_loss: 0.7929 | g_loss: 1.8002
    Epoch [    8/   25] | d_loss: 0.8456 | g_loss: 1.0076
    Epoch [    8/   25] | d_loss: 0.7957 | g_loss: 1.1281
    Epoch [    9/   25] | d_loss: 0.8511 | g_loss: 1.4324
    Epoch [    9/   25] | d_loss: 1.0662 | g_loss: 1.1980
    Epoch [    9/   25] | d_loss: 0.8644 | g_loss: 0.7941
    Epoch [    9/   25] | d_loss: 0.8326 | g_loss: 0.9982
    Epoch [    9/   25] | d_loss: 1.3206 | g_loss: 1.4236
    Epoch [    9/   25] | d_loss: 1.0092 | g_loss: 1.4953
    Epoch [    9/   25] | d_loss: 0.8699 | g_loss: 1.4487
    Epoch [    9/   25] | d_loss: 1.0477 | g_loss: 1.1256
    Epoch [    9/   25] | d_loss: 0.7673 | g_loss: 1.5568
    Epoch [    9/   25] | d_loss: 0.9003 | g_loss: 1.2093
    Epoch [    9/   25] | d_loss: 0.8868 | g_loss: 1.1328
    Epoch [    9/   25] | d_loss: 0.8861 | g_loss: 1.5058
    Epoch [    9/   25] | d_loss: 0.8691 | g_loss: 1.0486
    Epoch [    9/   25] | d_loss: 1.0697 | g_loss: 1.3348
    Epoch [    9/   25] | d_loss: 0.8124 | g_loss: 1.3251
    Epoch [    9/   25] | d_loss: 0.7805 | g_loss: 1.4413
    Epoch [    9/   25] | d_loss: 0.8562 | g_loss: 0.8686
    Epoch [    9/   25] | d_loss: 0.8654 | g_loss: 1.9168
    Epoch [    9/   25] | d_loss: 1.3441 | g_loss: 1.1808
    Epoch [    9/   25] | d_loss: 0.9939 | g_loss: 0.8467
    Epoch [    9/   25] | d_loss: 0.8034 | g_loss: 1.6517
    Epoch [    9/   25] | d_loss: 0.9639 | g_loss: 0.9718
    Epoch [    9/   25] | d_loss: 0.7616 | g_loss: 1.2553
    Epoch [    9/   25] | d_loss: 0.8224 | g_loss: 1.5117
    Epoch [    9/   25] | d_loss: 0.9310 | g_loss: 1.3161
    Epoch [    9/   25] | d_loss: 0.7914 | g_loss: 1.0428
    Epoch [    9/   25] | d_loss: 1.0304 | g_loss: 1.2216
    Epoch [    9/   25] | d_loss: 0.9303 | g_loss: 0.8622
    Epoch [    9/   25] | d_loss: 1.1304 | g_loss: 0.6747
    Epoch [   10/   25] | d_loss: 0.7900 | g_loss: 1.3079
    Epoch [   10/   25] | d_loss: 0.9204 | g_loss: 1.1460
    Epoch [   10/   25] | d_loss: 0.7996 | g_loss: 1.2664
    Epoch [   10/   25] | d_loss: 0.8905 | g_loss: 1.2435
    Epoch [   10/   25] | d_loss: 0.7559 | g_loss: 1.2487
    Epoch [   10/   25] | d_loss: 0.8270 | g_loss: 1.3159
    Epoch [   10/   25] | d_loss: 0.9432 | g_loss: 1.1373
    Epoch [   10/   25] | d_loss: 0.9184 | g_loss: 1.0852
    Epoch [   10/   25] | d_loss: 1.1061 | g_loss: 1.3165
    Epoch [   10/   25] | d_loss: 0.8450 | g_loss: 1.1750
    Epoch [   10/   25] | d_loss: 0.6587 | g_loss: 0.8898
    Epoch [   10/   25] | d_loss: 1.0173 | g_loss: 0.7879
    Epoch [   10/   25] | d_loss: 0.7752 | g_loss: 1.1903
    Epoch [   10/   25] | d_loss: 0.8288 | g_loss: 0.9333
    Epoch [   10/   25] | d_loss: 0.8637 | g_loss: 1.3672
    Epoch [   10/   25] | d_loss: 0.7904 | g_loss: 1.4388
    Epoch [   10/   25] | d_loss: 0.8556 | g_loss: 1.2855
    Epoch [   10/   25] | d_loss: 0.8637 | g_loss: 1.5560
    Epoch [   10/   25] | d_loss: 1.0442 | g_loss: 0.8482
    Epoch [   10/   25] | d_loss: 0.8287 | g_loss: 1.5463
    Epoch [   10/   25] | d_loss: 0.8467 | g_loss: 1.3416
    Epoch [   10/   25] | d_loss: 0.9573 | g_loss: 1.4403
    Epoch [   10/   25] | d_loss: 0.9948 | g_loss: 1.4526
    Epoch [   10/   25] | d_loss: 0.8266 | g_loss: 1.2312
    Epoch [   10/   25] | d_loss: 1.0911 | g_loss: 1.1638
    Epoch [   10/   25] | d_loss: 0.8931 | g_loss: 1.1397
    Epoch [   10/   25] | d_loss: 0.8698 | g_loss: 1.6702
    Epoch [   10/   25] | d_loss: 0.6137 | g_loss: 1.8999
    Epoch [   10/   25] | d_loss: 0.9515 | g_loss: 1.2501
    Epoch [   11/   25] | d_loss: 1.1625 | g_loss: 1.9464
    Epoch [   11/   25] | d_loss: 0.8327 | g_loss: 1.1964
    Epoch [   11/   25] | d_loss: 0.9940 | g_loss: 0.6765
    Epoch [   11/   25] | d_loss: 0.7792 | g_loss: 1.1438
    Epoch [   11/   25] | d_loss: 0.8068 | g_loss: 1.0910
    Epoch [   11/   25] | d_loss: 0.8679 | g_loss: 1.1351
    Epoch [   11/   25] | d_loss: 0.7849 | g_loss: 1.0542
    Epoch [   11/   25] | d_loss: 0.8238 | g_loss: 0.9972
    Epoch [   11/   25] | d_loss: 0.7973 | g_loss: 0.9202
    Epoch [   11/   25] | d_loss: 0.8758 | g_loss: 1.1791
    Epoch [   11/   25] | d_loss: 0.9558 | g_loss: 0.7820
    Epoch [   11/   25] | d_loss: 0.8757 | g_loss: 1.0010
    Epoch [   11/   25] | d_loss: 1.0134 | g_loss: 1.1307
    Epoch [   11/   25] | d_loss: 0.9474 | g_loss: 1.4200
    Epoch [   11/   25] | d_loss: 0.8433 | g_loss: 1.8908
    Epoch [   11/   25] | d_loss: 1.0319 | g_loss: 0.8235
    Epoch [   11/   25] | d_loss: 0.8984 | g_loss: 1.2924
    Epoch [   11/   25] | d_loss: 0.7783 | g_loss: 1.0289
    Epoch [   11/   25] | d_loss: 0.8108 | g_loss: 1.1361
    Epoch [   11/   25] | d_loss: 0.8260 | g_loss: 1.2925
    Epoch [   11/   25] | d_loss: 0.7850 | g_loss: 1.1467
    Epoch [   11/   25] | d_loss: 0.6541 | g_loss: 2.1554
    Epoch [   11/   25] | d_loss: 0.8579 | g_loss: 0.7930
    Epoch [   11/   25] | d_loss: 1.2166 | g_loss: 0.8411
    Epoch [   11/   25] | d_loss: 0.8765 | g_loss: 1.8028
    Epoch [   11/   25] | d_loss: 0.9479 | g_loss: 0.8582
    Epoch [   11/   25] | d_loss: 0.8970 | g_loss: 1.1148
    Epoch [   11/   25] | d_loss: 0.9452 | g_loss: 0.8489
    Epoch [   11/   25] | d_loss: 0.8305 | g_loss: 1.1818
    Epoch [   12/   25] | d_loss: 0.6459 | g_loss: 1.3116
    Epoch [   12/   25] | d_loss: 0.9282 | g_loss: 1.7538
    Epoch [   12/   25] | d_loss: 0.8758 | g_loss: 1.5223
    Epoch [   12/   25] | d_loss: 1.1533 | g_loss: 1.5480
    Epoch [   12/   25] | d_loss: 0.7960 | g_loss: 1.5866
    Epoch [   12/   25] | d_loss: 0.6951 | g_loss: 1.2104
    Epoch [   12/   25] | d_loss: 1.1835 | g_loss: 1.1527
    Epoch [   12/   25] | d_loss: 0.9414 | g_loss: 1.2842
    Epoch [   12/   25] | d_loss: 0.6333 | g_loss: 1.6379
    Epoch [   12/   25] | d_loss: 0.8127 | g_loss: 1.3589
    Epoch [   12/   25] | d_loss: 0.9534 | g_loss: 1.3213
    Epoch [   12/   25] | d_loss: 0.9936 | g_loss: 1.7651
    Epoch [   12/   25] | d_loss: 0.8404 | g_loss: 1.1476
    Epoch [   12/   25] | d_loss: 0.8894 | g_loss: 1.8481
    Epoch [   12/   25] | d_loss: 1.0540 | g_loss: 1.2139
    Epoch [   12/   25] | d_loss: 0.8186 | g_loss: 0.9896
    Epoch [   12/   25] | d_loss: 0.7847 | g_loss: 1.4864
    Epoch [   12/   25] | d_loss: 0.8546 | g_loss: 1.3949
    Epoch [   12/   25] | d_loss: 0.9161 | g_loss: 1.2015
    Epoch [   12/   25] | d_loss: 0.8213 | g_loss: 1.1410
    Epoch [   12/   25] | d_loss: 0.8130 | g_loss: 0.9844
    Epoch [   12/   25] | d_loss: 0.6878 | g_loss: 1.6874
    Epoch [   12/   25] | d_loss: 0.9647 | g_loss: 1.1736
    Epoch [   12/   25] | d_loss: 0.8884 | g_loss: 1.2806
    Epoch [   12/   25] | d_loss: 0.9759 | g_loss: 1.0083
    Epoch [   12/   25] | d_loss: 0.7296 | g_loss: 1.5743
    Epoch [   12/   25] | d_loss: 0.6803 | g_loss: 1.6710
    Epoch [   12/   25] | d_loss: 0.7752 | g_loss: 1.4931
    Epoch [   12/   25] | d_loss: 1.0242 | g_loss: 0.8904
    Epoch [   13/   25] | d_loss: 0.8242 | g_loss: 1.4923
    Epoch [   13/   25] | d_loss: 0.8289 | g_loss: 1.8566
    Epoch [   13/   25] | d_loss: 0.6669 | g_loss: 1.5910
    Epoch [   13/   25] | d_loss: 0.6840 | g_loss: 1.5274
    Epoch [   13/   25] | d_loss: 0.9531 | g_loss: 1.8493
    Epoch [   13/   25] | d_loss: 0.8259 | g_loss: 1.0923
    Epoch [   13/   25] | d_loss: 0.7733 | g_loss: 1.6929
    Epoch [   13/   25] | d_loss: 0.8102 | g_loss: 1.4062
    Epoch [   13/   25] | d_loss: 0.8927 | g_loss: 1.3642
    Epoch [   13/   25] | d_loss: 0.8037 | g_loss: 1.4584
    Epoch [   13/   25] | d_loss: 0.7607 | g_loss: 1.4944
    Epoch [   13/   25] | d_loss: 0.8640 | g_loss: 1.3984
    Epoch [   13/   25] | d_loss: 1.0389 | g_loss: 1.7728
    Epoch [   13/   25] | d_loss: 0.7382 | g_loss: 0.9539
    Epoch [   13/   25] | d_loss: 0.8244 | g_loss: 2.0306
    Epoch [   13/   25] | d_loss: 0.7413 | g_loss: 1.8363
    Epoch [   13/   25] | d_loss: 0.8739 | g_loss: 1.9030
    Epoch [   13/   25] | d_loss: 0.7685 | g_loss: 1.7616
    Epoch [   13/   25] | d_loss: 0.6871 | g_loss: 1.8121
    Epoch [   13/   25] | d_loss: 0.7710 | g_loss: 1.0613
    Epoch [   13/   25] | d_loss: 0.7193 | g_loss: 1.9879
    Epoch [   13/   25] | d_loss: 0.9082 | g_loss: 1.9215
    Epoch [   13/   25] | d_loss: 0.8428 | g_loss: 1.0723
    Epoch [   13/   25] | d_loss: 1.0081 | g_loss: 0.9607
    Epoch [   13/   25] | d_loss: 0.7742 | g_loss: 1.0507
    Epoch [   13/   25] | d_loss: 0.6990 | g_loss: 1.5100
    Epoch [   13/   25] | d_loss: 0.8065 | g_loss: 2.2429
    Epoch [   13/   25] | d_loss: 0.6531 | g_loss: 1.2834
    Epoch [   13/   25] | d_loss: 1.0285 | g_loss: 1.5473
    Epoch [   14/   25] | d_loss: 0.7480 | g_loss: 1.2829
    Epoch [   14/   25] | d_loss: 1.0969 | g_loss: 1.4045
    Epoch [   14/   25] | d_loss: 0.9491 | g_loss: 1.2400
    Epoch [   14/   25] | d_loss: 0.8357 | g_loss: 1.3591
    Epoch [   14/   25] | d_loss: 0.8715 | g_loss: 1.1936
    Epoch [   14/   25] | d_loss: 0.7721 | g_loss: 2.2012
    Epoch [   14/   25] | d_loss: 0.6536 | g_loss: 1.0201
    Epoch [   14/   25] | d_loss: 0.8275 | g_loss: 1.1746
    Epoch [   14/   25] | d_loss: 0.8530 | g_loss: 1.2383
    Epoch [   14/   25] | d_loss: 0.8046 | g_loss: 1.8550
    Epoch [   14/   25] | d_loss: 0.7930 | g_loss: 1.3912
    Epoch [   14/   25] | d_loss: 0.7841 | g_loss: 1.2870
    Epoch [   14/   25] | d_loss: 0.7877 | g_loss: 1.5063
    Epoch [   14/   25] | d_loss: 0.5005 | g_loss: 2.2351
    Epoch [   14/   25] | d_loss: 0.7967 | g_loss: 1.1716
    Epoch [   14/   25] | d_loss: 0.8103 | g_loss: 1.6185
    Epoch [   14/   25] | d_loss: 0.6628 | g_loss: 2.2009
    Epoch [   14/   25] | d_loss: 0.9896 | g_loss: 0.8251
    Epoch [   14/   25] | d_loss: 0.6791 | g_loss: 1.5095
    Epoch [   14/   25] | d_loss: 0.7124 | g_loss: 2.4318
    Epoch [   14/   25] | d_loss: 0.7786 | g_loss: 1.3323
    Epoch [   14/   25] | d_loss: 0.9156 | g_loss: 1.2961
    Epoch [   14/   25] | d_loss: 0.7697 | g_loss: 2.0068
    Epoch [   14/   25] | d_loss: 0.8621 | g_loss: 1.5440
    Epoch [   14/   25] | d_loss: 0.6332 | g_loss: 1.5118
    Epoch [   14/   25] | d_loss: 0.9725 | g_loss: 1.0634
    Epoch [   14/   25] | d_loss: 0.7702 | g_loss: 1.6564
    Epoch [   14/   25] | d_loss: 1.0131 | g_loss: 1.6845
    Epoch [   14/   25] | d_loss: 0.7300 | g_loss: 1.7566
    Epoch [   15/   25] | d_loss: 0.8360 | g_loss: 1.6959
    Epoch [   15/   25] | d_loss: 0.6025 | g_loss: 2.8627
    Epoch [   15/   25] | d_loss: 0.7624 | g_loss: 0.8848
    Epoch [   15/   25] | d_loss: 0.7978 | g_loss: 1.9487
    Epoch [   15/   25] | d_loss: 0.6404 | g_loss: 1.7013
    Epoch [   15/   25] | d_loss: 0.8054 | g_loss: 1.5378
    Epoch [   15/   25] | d_loss: 0.5563 | g_loss: 1.4239
    Epoch [   15/   25] | d_loss: 0.6995 | g_loss: 1.1944
    Epoch [   15/   25] | d_loss: 1.0296 | g_loss: 2.2035
    Epoch [   15/   25] | d_loss: 0.7986 | g_loss: 1.2384
    Epoch [   15/   25] | d_loss: 0.8583 | g_loss: 1.1653
    Epoch [   15/   25] | d_loss: 0.7649 | g_loss: 1.1897
    Epoch [   15/   25] | d_loss: 0.5735 | g_loss: 1.5428
    Epoch [   15/   25] | d_loss: 0.9108 | g_loss: 2.2530
    Epoch [   15/   25] | d_loss: 0.6071 | g_loss: 2.0654
    Epoch [   15/   25] | d_loss: 0.4718 | g_loss: 1.7847
    Epoch [   15/   25] | d_loss: 0.8360 | g_loss: 1.8086
    Epoch [   15/   25] | d_loss: 0.6566 | g_loss: 1.6045
    Epoch [   15/   25] | d_loss: 0.8139 | g_loss: 1.6889
    Epoch [   15/   25] | d_loss: 1.0009 | g_loss: 0.9059
    Epoch [   15/   25] | d_loss: 0.7081 | g_loss: 1.9544
    Epoch [   15/   25] | d_loss: 0.5741 | g_loss: 1.6378
    Epoch [   15/   25] | d_loss: 0.8095 | g_loss: 1.3413
    Epoch [   15/   25] | d_loss: 0.5605 | g_loss: 1.5415
    Epoch [   15/   25] | d_loss: 0.5528 | g_loss: 1.4141
    Epoch [   15/   25] | d_loss: 0.6288 | g_loss: 1.9488
    Epoch [   15/   25] | d_loss: 0.9617 | g_loss: 1.3599
    Epoch [   15/   25] | d_loss: 0.9448 | g_loss: 1.3477
    Epoch [   15/   25] | d_loss: 0.5499 | g_loss: 1.7566
    Epoch [   16/   25] | d_loss: 0.6633 | g_loss: 1.6409
    Epoch [   16/   25] | d_loss: 0.6286 | g_loss: 1.2211
    Epoch [   16/   25] | d_loss: 0.4422 | g_loss: 2.5264
    Epoch [   16/   25] | d_loss: 0.6341 | g_loss: 1.4506
    Epoch [   16/   25] | d_loss: 0.6431 | g_loss: 1.2447
    Epoch [   16/   25] | d_loss: 0.5846 | g_loss: 1.3870
    Epoch [   16/   25] | d_loss: 0.7588 | g_loss: 1.3585
    Epoch [   16/   25] | d_loss: 0.8359 | g_loss: 1.3778
    Epoch [   16/   25] | d_loss: 0.9575 | g_loss: 1.6657
    Epoch [   16/   25] | d_loss: 0.7964 | g_loss: 1.3897
    Epoch [   16/   25] | d_loss: 1.1210 | g_loss: 1.4148
    Epoch [   16/   25] | d_loss: 0.7016 | g_loss: 1.8349
    Epoch [   16/   25] | d_loss: 0.8168 | g_loss: 1.5243
    Epoch [   16/   25] | d_loss: 0.6816 | g_loss: 1.2366
    Epoch [   16/   25] | d_loss: 0.5498 | g_loss: 2.1711
    Epoch [   16/   25] | d_loss: 0.7354 | g_loss: 1.4043
    Epoch [   16/   25] | d_loss: 0.5402 | g_loss: 1.9862
    Epoch [   16/   25] | d_loss: 0.7649 | g_loss: 1.2940
    Epoch [   16/   25] | d_loss: 0.8359 | g_loss: 1.3669
    Epoch [   16/   25] | d_loss: 0.7501 | g_loss: 1.3128
    Epoch [   16/   25] | d_loss: 0.6683 | g_loss: 1.7530
    Epoch [   16/   25] | d_loss: 0.9990 | g_loss: 1.1812
    Epoch [   16/   25] | d_loss: 0.6442 | g_loss: 1.6032
    Epoch [   16/   25] | d_loss: 0.7735 | g_loss: 1.7387
    Epoch [   16/   25] | d_loss: 0.6749 | g_loss: 1.4183
    Epoch [   16/   25] | d_loss: 1.1703 | g_loss: 1.2655
    Epoch [   16/   25] | d_loss: 0.6369 | g_loss: 1.9493
    Epoch [   16/   25] | d_loss: 0.7345 | g_loss: 0.9482
    Epoch [   16/   25] | d_loss: 0.6160 | g_loss: 1.9188
    Epoch [   17/   25] | d_loss: 0.8465 | g_loss: 1.7413
    Epoch [   17/   25] | d_loss: 0.6608 | g_loss: 1.3905
    Epoch [   17/   25] | d_loss: 0.6871 | g_loss: 1.7556
    Epoch [   17/   25] | d_loss: 0.5295 | g_loss: 1.8908
    Epoch [   17/   25] | d_loss: 0.5901 | g_loss: 2.2217
    Epoch [   17/   25] | d_loss: 0.6965 | g_loss: 1.2821
    Epoch [   17/   25] | d_loss: 0.8779 | g_loss: 0.3723
    Epoch [   17/   25] | d_loss: 0.5342 | g_loss: 1.4398
    Epoch [   17/   25] | d_loss: 0.5498 | g_loss: 2.4653
    Epoch [   17/   25] | d_loss: 0.6642 | g_loss: 1.2544
    Epoch [   17/   25] | d_loss: 0.7683 | g_loss: 1.5326
    Epoch [   17/   25] | d_loss: 0.8293 | g_loss: 1.8449
    Epoch [   17/   25] | d_loss: 0.7102 | g_loss: 2.1161
    Epoch [   17/   25] | d_loss: 0.6364 | g_loss: 1.3531
    Epoch [   17/   25] | d_loss: 0.7619 | g_loss: 1.0989
    Epoch [   17/   25] | d_loss: 0.7340 | g_loss: 1.3926
    Epoch [   17/   25] | d_loss: 0.6429 | g_loss: 1.5032
    Epoch [   17/   25] | d_loss: 1.0964 | g_loss: 1.5623
    Epoch [   17/   25] | d_loss: 0.8774 | g_loss: 1.5322
    Epoch [   17/   25] | d_loss: 0.5411 | g_loss: 1.2667
    Epoch [   17/   25] | d_loss: 0.7864 | g_loss: 1.0802
    Epoch [   17/   25] | d_loss: 0.7416 | g_loss: 1.8103
    Epoch [   17/   25] | d_loss: 0.7219 | g_loss: 1.6879
    Epoch [   17/   25] | d_loss: 0.5723 | g_loss: 1.5642
    Epoch [   17/   25] | d_loss: 0.9473 | g_loss: 2.3438
    Epoch [   17/   25] | d_loss: 0.4991 | g_loss: 1.8020
    Epoch [   17/   25] | d_loss: 0.7634 | g_loss: 1.1737
    Epoch [   17/   25] | d_loss: 0.8034 | g_loss: 0.9052
    Epoch [   17/   25] | d_loss: 1.1777 | g_loss: 0.7141
    Epoch [   18/   25] | d_loss: 0.8600 | g_loss: 2.9713
    Epoch [   18/   25] | d_loss: 0.5591 | g_loss: 1.6729
    Epoch [   18/   25] | d_loss: 0.7524 | g_loss: 1.9247
    Epoch [   18/   25] | d_loss: 0.6064 | g_loss: 2.0994
    Epoch [   18/   25] | d_loss: 0.7100 | g_loss: 1.8843
    Epoch [   18/   25] | d_loss: 0.7205 | g_loss: 1.4715
    Epoch [   18/   25] | d_loss: 0.6415 | g_loss: 2.1160
    Epoch [   18/   25] | d_loss: 0.6853 | g_loss: 1.0622
    Epoch [   18/   25] | d_loss: 0.8332 | g_loss: 1.1722
    Epoch [   18/   25] | d_loss: 0.6545 | g_loss: 1.3573
    Epoch [   18/   25] | d_loss: 0.5654 | g_loss: 1.0602
    Epoch [   18/   25] | d_loss: 0.8422 | g_loss: 1.8183
    Epoch [   18/   25] | d_loss: 0.5993 | g_loss: 1.6572
    Epoch [   18/   25] | d_loss: 0.6664 | g_loss: 2.3108
    Epoch [   18/   25] | d_loss: 0.5546 | g_loss: 1.1419
    Epoch [   18/   25] | d_loss: 0.6323 | g_loss: 1.6800
    Epoch [   18/   25] | d_loss: 0.6171 | g_loss: 1.6584
    Epoch [   18/   25] | d_loss: 0.6697 | g_loss: 1.4131
    Epoch [   18/   25] | d_loss: 0.4251 | g_loss: 1.6283
    Epoch [   18/   25] | d_loss: 0.6975 | g_loss: 0.9843
    Epoch [   18/   25] | d_loss: 0.9115 | g_loss: 1.7346
    Epoch [   18/   25] | d_loss: 0.6740 | g_loss: 2.0027
    Epoch [   18/   25] | d_loss: 0.7355 | g_loss: 2.0644
    Epoch [   18/   25] | d_loss: 0.5292 | g_loss: 1.7544
    Epoch [   18/   25] | d_loss: 0.8290 | g_loss: 2.0307
    Epoch [   18/   25] | d_loss: 0.4846 | g_loss: 2.2310
    Epoch [   18/   25] | d_loss: 0.7752 | g_loss: 1.3335
    Epoch [   18/   25] | d_loss: 0.6299 | g_loss: 1.6400
    Epoch [   18/   25] | d_loss: 0.7214 | g_loss: 2.2676
    Epoch [   19/   25] | d_loss: 0.8834 | g_loss: 1.4510
    Epoch [   19/   25] | d_loss: 0.3995 | g_loss: 2.2766
    Epoch [   19/   25] | d_loss: 0.6864 | g_loss: 1.2043
    Epoch [   19/   25] | d_loss: 0.5387 | g_loss: 1.5848
    Epoch [   19/   25] | d_loss: 0.7984 | g_loss: 1.6468
    Epoch [   19/   25] | d_loss: 0.5082 | g_loss: 2.4158
    Epoch [   19/   25] | d_loss: 0.7308 | g_loss: 1.8370
    Epoch [   19/   25] | d_loss: 0.7327 | g_loss: 1.7616
    Epoch [   19/   25] | d_loss: 0.6993 | g_loss: 1.7498
    Epoch [   19/   25] | d_loss: 0.5649 | g_loss: 2.0504
    Epoch [   19/   25] | d_loss: 0.7803 | g_loss: 1.9498
    Epoch [   19/   25] | d_loss: 0.5109 | g_loss: 1.4159
    Epoch [   19/   25] | d_loss: 0.5820 | g_loss: 2.0288
    Epoch [   19/   25] | d_loss: 0.8418 | g_loss: 1.8370
    Epoch [   19/   25] | d_loss: 0.8073 | g_loss: 1.4037
    Epoch [   19/   25] | d_loss: 0.7068 | g_loss: 2.2584
    Epoch [   19/   25] | d_loss: 0.5817 | g_loss: 1.4827
    Epoch [   19/   25] | d_loss: 0.7307 | g_loss: 1.8218
    Epoch [   19/   25] | d_loss: 0.6973 | g_loss: 1.7576
    Epoch [   19/   25] | d_loss: 0.6543 | g_loss: 2.2651
    Epoch [   19/   25] | d_loss: 0.6101 | g_loss: 2.1569
    Epoch [   19/   25] | d_loss: 0.7473 | g_loss: 1.5744
    Epoch [   19/   25] | d_loss: 0.6024 | g_loss: 1.2443
    Epoch [   19/   25] | d_loss: 0.6165 | g_loss: 1.6552
    Epoch [   19/   25] | d_loss: 0.5930 | g_loss: 1.7712
    Epoch [   19/   25] | d_loss: 0.5086 | g_loss: 1.7136
    Epoch [   19/   25] | d_loss: 0.5097 | g_loss: 1.8248
    Epoch [   19/   25] | d_loss: 0.6012 | g_loss: 1.9211
    Epoch [   19/   25] | d_loss: 0.5833 | g_loss: 2.3839
    Epoch [   20/   25] | d_loss: 0.6432 | g_loss: 1.2693
    Epoch [   20/   25] | d_loss: 0.4218 | g_loss: 1.6197
    Epoch [   20/   25] | d_loss: 0.6091 | g_loss: 2.3031
    Epoch [   20/   25] | d_loss: 0.7373 | g_loss: 1.1006
    Epoch [   20/   25] | d_loss: 0.5939 | g_loss: 1.8930
    Epoch [   20/   25] | d_loss: 0.5418 | g_loss: 2.0177
    Epoch [   20/   25] | d_loss: 0.8049 | g_loss: 1.8898
    Epoch [   20/   25] | d_loss: 0.6562 | g_loss: 2.0292
    Epoch [   20/   25] | d_loss: 0.5336 | g_loss: 1.9089
    Epoch [   20/   25] | d_loss: 0.6046 | g_loss: 1.5372
    Epoch [   20/   25] | d_loss: 1.3073 | g_loss: 1.1221
    Epoch [   20/   25] | d_loss: 0.5657 | g_loss: 1.4987
    Epoch [   20/   25] | d_loss: 0.5349 | g_loss: 2.3735
    Epoch [   20/   25] | d_loss: 0.7639 | g_loss: 2.0889
    Epoch [   20/   25] | d_loss: 0.4922 | g_loss: 2.3023
    Epoch [   20/   25] | d_loss: 0.5291 | g_loss: 1.6138
    Epoch [   20/   25] | d_loss: 0.7777 | g_loss: 2.3032
    Epoch [   20/   25] | d_loss: 0.6198 | g_loss: 1.5780
    Epoch [   20/   25] | d_loss: 0.6165 | g_loss: 1.6144
    Epoch [   20/   25] | d_loss: 0.6946 | g_loss: 1.7337
    Epoch [   20/   25] | d_loss: 0.6417 | g_loss: 1.5117
    Epoch [   20/   25] | d_loss: 0.6125 | g_loss: 2.2380
    Epoch [   20/   25] | d_loss: 1.1232 | g_loss: 1.0757
    Epoch [   20/   25] | d_loss: 0.4994 | g_loss: 1.8952
    Epoch [   20/   25] | d_loss: 0.7121 | g_loss: 1.9807
    Epoch [   20/   25] | d_loss: 0.6088 | g_loss: 1.6285
    Epoch [   20/   25] | d_loss: 0.6226 | g_loss: 1.1577
    Epoch [   20/   25] | d_loss: 0.5010 | g_loss: 1.4973
    Epoch [   20/   25] | d_loss: 0.7626 | g_loss: 1.6401
    Epoch [   21/   25] | d_loss: 0.8763 | g_loss: 2.6764
    Epoch [   21/   25] | d_loss: 0.5088 | g_loss: 1.3737
    Epoch [   21/   25] | d_loss: 0.7732 | g_loss: 1.2731
    Epoch [   21/   25] | d_loss: 0.4732 | g_loss: 2.2847
    Epoch [   21/   25] | d_loss: 0.4907 | g_loss: 2.1520
    Epoch [   21/   25] | d_loss: 0.8015 | g_loss: 1.3201
    Epoch [   21/   25] | d_loss: 0.4634 | g_loss: 2.3047
    Epoch [   21/   25] | d_loss: 1.4389 | g_loss: 3.6018
    Epoch [   21/   25] | d_loss: 0.7018 | g_loss: 0.8139
    Epoch [   21/   25] | d_loss: 0.7002 | g_loss: 1.1399
    Epoch [   21/   25] | d_loss: 0.4537 | g_loss: 1.8002
    Epoch [   21/   25] | d_loss: 0.3948 | g_loss: 2.3370
    Epoch [   21/   25] | d_loss: 0.4676 | g_loss: 1.6851
    Epoch [   21/   25] | d_loss: 0.8104 | g_loss: 0.7523
    Epoch [   21/   25] | d_loss: 0.6516 | g_loss: 2.9001
    Epoch [   21/   25] | d_loss: 0.5424 | g_loss: 1.6283
    Epoch [   21/   25] | d_loss: 0.6946 | g_loss: 1.5425
    Epoch [   21/   25] | d_loss: 0.8699 | g_loss: 1.7531
    Epoch [   21/   25] | d_loss: 0.6848 | g_loss: 1.4527
    Epoch [   21/   25] | d_loss: 0.5921 | g_loss: 1.7615
    Epoch [   21/   25] | d_loss: 0.4762 | g_loss: 1.4185
    Epoch [   21/   25] | d_loss: 0.7726 | g_loss: 1.3251
    Epoch [   21/   25] | d_loss: 0.5753 | g_loss: 1.7177
    Epoch [   21/   25] | d_loss: 0.6439 | g_loss: 1.4324
    Epoch [   21/   25] | d_loss: 0.8590 | g_loss: 1.9229
    Epoch [   21/   25] | d_loss: 0.3135 | g_loss: 2.0143
    Epoch [   21/   25] | d_loss: 0.6397 | g_loss: 1.3193
    Epoch [   21/   25] | d_loss: 0.9679 | g_loss: 1.1382
    Epoch [   21/   25] | d_loss: 0.6260 | g_loss: 1.5868
    Epoch [   22/   25] | d_loss: 0.5982 | g_loss: 2.0104
    Epoch [   22/   25] | d_loss: 0.7241 | g_loss: 1.6322
    Epoch [   22/   25] | d_loss: 0.5527 | g_loss: 2.1378
    Epoch [   22/   25] | d_loss: 0.5422 | g_loss: 2.0007
    Epoch [   22/   25] | d_loss: 0.6468 | g_loss: 1.2851
    Epoch [   22/   25] | d_loss: 0.2835 | g_loss: 3.0415
    Epoch [   22/   25] | d_loss: 0.5287 | g_loss: 2.1578
    Epoch [   22/   25] | d_loss: 0.3919 | g_loss: 2.9017
    Epoch [   22/   25] | d_loss: 0.5847 | g_loss: 1.3585
    Epoch [   22/   25] | d_loss: 0.6414 | g_loss: 1.5678
    Epoch [   22/   25] | d_loss: 0.4763 | g_loss: 2.7407
    Epoch [   22/   25] | d_loss: 0.6485 | g_loss: 1.3146
    Epoch [   22/   25] | d_loss: 0.3307 | g_loss: 2.7901
    Epoch [   22/   25] | d_loss: 1.0801 | g_loss: 1.3014
    Epoch [   22/   25] | d_loss: 0.6445 | g_loss: 1.5198
    Epoch [   22/   25] | d_loss: 0.6869 | g_loss: 1.5085
    Epoch [   22/   25] | d_loss: 0.4501 | g_loss: 2.4889
    Epoch [   22/   25] | d_loss: 0.5272 | g_loss: 1.7477
    Epoch [   22/   25] | d_loss: 0.6501 | g_loss: 1.9340
    Epoch [   22/   25] | d_loss: 0.9605 | g_loss: 1.4742
    Epoch [   22/   25] | d_loss: 0.8953 | g_loss: 2.1754
    Epoch [   22/   25] | d_loss: 0.5202 | g_loss: 2.2417
    Epoch [   22/   25] | d_loss: 0.5839 | g_loss: 1.3557
    Epoch [   22/   25] | d_loss: 0.5769 | g_loss: 2.4501
    Epoch [   22/   25] | d_loss: 0.7913 | g_loss: 2.1865
    Epoch [   22/   25] | d_loss: 0.4460 | g_loss: 2.4384
    Epoch [   22/   25] | d_loss: 0.6498 | g_loss: 1.9332
    Epoch [   22/   25] | d_loss: 0.9517 | g_loss: 0.6582
    Epoch [   22/   25] | d_loss: 0.6260 | g_loss: 2.7183
    Epoch [   23/   25] | d_loss: 0.5842 | g_loss: 1.6230
    Epoch [   23/   25] | d_loss: 0.5816 | g_loss: 1.5675
    Epoch [   23/   25] | d_loss: 0.6014 | g_loss: 1.3827
    Epoch [   23/   25] | d_loss: 0.8327 | g_loss: 2.4828
    Epoch [   23/   25] | d_loss: 0.6633 | g_loss: 1.3694
    Epoch [   23/   25] | d_loss: 0.6614 | g_loss: 1.8797
    Epoch [   23/   25] | d_loss: 0.8101 | g_loss: 1.6317
    Epoch [   23/   25] | d_loss: 0.5123 | g_loss: 1.6391
    Epoch [   23/   25] | d_loss: 0.7626 | g_loss: 1.4367
    Epoch [   23/   25] | d_loss: 0.8259 | g_loss: 1.5490
    Epoch [   23/   25] | d_loss: 0.9640 | g_loss: 2.4758
    Epoch [   23/   25] | d_loss: 0.4715 | g_loss: 2.0373
    Epoch [   23/   25] | d_loss: 0.7061 | g_loss: 1.4364
    Epoch [   23/   25] | d_loss: 0.5051 | g_loss: 1.9546
    Epoch [   23/   25] | d_loss: 0.5220 | g_loss: 1.9298
    Epoch [   23/   25] | d_loss: 0.7190 | g_loss: 1.7328
    Epoch [   23/   25] | d_loss: 0.6915 | g_loss: 2.5597
    Epoch [   23/   25] | d_loss: 0.3440 | g_loss: 2.1481
    Epoch [   23/   25] | d_loss: 0.8006 | g_loss: 1.2359
    Epoch [   23/   25] | d_loss: 0.7773 | g_loss: 1.3178
    Epoch [   23/   25] | d_loss: 0.4118 | g_loss: 1.6262
    Epoch [   23/   25] | d_loss: 0.7586 | g_loss: 0.9491
    Epoch [   23/   25] | d_loss: 0.5897 | g_loss: 1.4727
    Epoch [   23/   25] | d_loss: 0.5789 | g_loss: 1.9932
    Epoch [   23/   25] | d_loss: 0.7629 | g_loss: 1.5090
    Epoch [   23/   25] | d_loss: 0.4721 | g_loss: 2.1045
    Epoch [   23/   25] | d_loss: 0.5535 | g_loss: 1.6421
    Epoch [   23/   25] | d_loss: 0.8013 | g_loss: 2.3912
    Epoch [   23/   25] | d_loss: 0.3773 | g_loss: 2.1823
    Epoch [   24/   25] | d_loss: 0.5227 | g_loss: 2.1540
    Epoch [   24/   25] | d_loss: 0.5649 | g_loss: 2.8204
    Epoch [   24/   25] | d_loss: 0.6103 | g_loss: 0.9175
    Epoch [   24/   25] | d_loss: 0.6582 | g_loss: 1.9511
    Epoch [   24/   25] | d_loss: 0.6437 | g_loss: 2.1922
    Epoch [   24/   25] | d_loss: 0.7445 | g_loss: 1.4419
    Epoch [   24/   25] | d_loss: 0.2906 | g_loss: 1.5158
    Epoch [   24/   25] | d_loss: 0.5120 | g_loss: 2.1421
    Epoch [   24/   25] | d_loss: 0.7121 | g_loss: 2.4226
    Epoch [   24/   25] | d_loss: 0.3749 | g_loss: 2.3880
    Epoch [   24/   25] | d_loss: 0.5659 | g_loss: 1.6897
    Epoch [   24/   25] | d_loss: 0.6418 | g_loss: 2.5978
    Epoch [   24/   25] | d_loss: 0.5687 | g_loss: 2.5555
    Epoch [   24/   25] | d_loss: 0.3639 | g_loss: 2.5789
    Epoch [   24/   25] | d_loss: 0.4507 | g_loss: 2.3557
    Epoch [   24/   25] | d_loss: 0.5794 | g_loss: 2.3574
    Epoch [   24/   25] | d_loss: 0.6419 | g_loss: 1.4397
    Epoch [   24/   25] | d_loss: 0.5300 | g_loss: 1.8033
    Epoch [   24/   25] | d_loss: 0.6415 | g_loss: 2.0888
    Epoch [   24/   25] | d_loss: 0.5369 | g_loss: 2.1490
    Epoch [   24/   25] | d_loss: 0.5290 | g_loss: 2.3431
    Epoch [   24/   25] | d_loss: 0.5625 | g_loss: 2.2535
    Epoch [   24/   25] | d_loss: 0.4578 | g_loss: 2.2150
    Epoch [   24/   25] | d_loss: 0.7811 | g_loss: 1.5763
    Epoch [   24/   25] | d_loss: 0.5767 | g_loss: 1.1458
    Epoch [   24/   25] | d_loss: 0.6839 | g_loss: 2.7002
    Epoch [   24/   25] | d_loss: 0.7143 | g_loss: 2.2834
    Epoch [   24/   25] | d_loss: 0.6597 | g_loss: 1.7441
    Epoch [   24/   25] | d_loss: 0.5178 | g_loss: 1.9938
    Epoch [   25/   25] | d_loss: 1.8239 | g_loss: 1.8015
    Epoch [   25/   25] | d_loss: 0.3482 | g_loss: 2.5544
    Epoch [   25/   25] | d_loss: 0.6714 | g_loss: 1.4270
    Epoch [   25/   25] | d_loss: 0.3652 | g_loss: 1.9219
    Epoch [   25/   25] | d_loss: 0.5655 | g_loss: 1.3604
    Epoch [   25/   25] | d_loss: 0.8190 | g_loss: 1.5622
    Epoch [   25/   25] | d_loss: 0.6647 | g_loss: 1.9233
    Epoch [   25/   25] | d_loss: 0.9010 | g_loss: 0.9438
    Epoch [   25/   25] | d_loss: 0.6076 | g_loss: 1.5336
    Epoch [   25/   25] | d_loss: 0.3768 | g_loss: 2.1153
    Epoch [   25/   25] | d_loss: 0.5297 | g_loss: 1.5897
    Epoch [   25/   25] | d_loss: 0.5580 | g_loss: 1.5240
    Epoch [   25/   25] | d_loss: 0.5211 | g_loss: 2.0984
    Epoch [   25/   25] | d_loss: 0.5827 | g_loss: 1.7195
    Epoch [   25/   25] | d_loss: 0.3957 | g_loss: 1.4812
    Epoch [   25/   25] | d_loss: 0.5212 | g_loss: 2.1269
    Epoch [   25/   25] | d_loss: 0.3407 | g_loss: 2.7834
    Epoch [   25/   25] | d_loss: 0.5970 | g_loss: 1.8113
    Epoch [   25/   25] | d_loss: 0.6581 | g_loss: 2.2948
    Epoch [   25/   25] | d_loss: 0.7352 | g_loss: 1.2153
    Epoch [   25/   25] | d_loss: 0.6136 | g_loss: 1.6078
    Epoch [   25/   25] | d_loss: 0.3949 | g_loss: 2.1776
    Epoch [   25/   25] | d_loss: 0.5922 | g_loss: 1.1952
    Epoch [   25/   25] | d_loss: 0.8417 | g_loss: 0.8000
    Epoch [   25/   25] | d_loss: 0.5537 | g_loss: 1.9322
    Epoch [   25/   25] | d_loss: 0.4513 | g_loss: 1.9089
    Epoch [   25/   25] | d_loss: 0.7249 | g_loss: 1.8146
    Epoch [   25/   25] | d_loss: 0.4872 | g_loss: 1.5163
    Epoch [   25/   25] | d_loss: 0.7371 | g_loss: 1.2264


## Training loss

Plot the training losses for the generator and discriminator, recorded after each epoch.


```python
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f76d2f6e2b0>




![png](/notebook_images/output_36_1.png)


## Generator samples from training

View samples of images from the generator, and answer a question about the strengths and weaknesses of your trained models.


```python
# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32,32,3)))
```


```python
# Load samples from generator, taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)
```


```python
_ = view_samples(-1, samples)
```


![png](/notebook_images/output_40_0.png)


### Question: What do you notice about your generated samples and how might you improve this model?
When you answer this question, consider the following factors:
* The dataset is biased; it is made of "celebrity" faces that are mostly white
* Model size; larger models have the opportunity to learn more features in a data feature space
* Optimization strategy; optimizers and number of epochs affect your final result


**Answer:** (Write your answer in this cell)

The dataset is definitely biased. We need to include different ethnicities to make our dataset more diverse. Including different skin colours, eye and lip colours, and shapes can improve the results.

Having a deeper network always helps, but there is a tradeoff between accuracy and computational load. Also, a very deep network may cause the vanishing gradient problem.  

Increasing the number of epochs often leads to a more accurate model, but is computationally expensive. Based on what we learned in our course, I think it's better to use Adam instead of SGD for GANs. 

Our generated images are low resolution. We can generate higher resolution images by training on higher resolution images, but again it is computationally expensive. We also can use larger size images. 

In general, I think our model performed sufficiently. 
