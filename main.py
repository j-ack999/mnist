import os
import requests # this allows python to communicate with websites, we need this to access the data on the website and download it 
import gzip # allows us to unzip files  
import numpy as np
# checking github working

# file names, use of a dictionary to match the label to the file. e.g, training_images = the training images file 
data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "training_labels": "train-labels-idx1-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# create folder for the training and test data 
data_dir = "_data"
os.makedirs(data_dir, exist_ok=True) # making our directory to hold our data. giving it the name we defined on the previous line 

# download files from the website 
base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

for fname in data_sources.values():
    fpath = os.path.join(data_dir, fname)

    if not os.path.exists(fpath):
        print(f"Downloading {fname}...")
        resp = requests.get(base_url + fname, stream=True)
        resp.raise_for_status()

        with open(fpath, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=128):
                fh.write(chunk)

print("Done!")



mnist_dataset = {}

# Images
for key in ("training_images", "test_images"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(
            mnist_file.read(), np.uint8, offset=16 # an offset of 16 means skip the first 16 bytes since this is the metadata, not image data 
        ).reshape(-1, 28 * 28) # shape the data so it is not in one long line of numbers but instead in image form of 28 x 28. -1 since we dont know how many images there are 
# Labels
for key in ("training_labels", "test_labels"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8) # offset is 8 since the metadata is 8 bytes only 
        # additionally, we dont need to reshape sinc ethe labels are just the answers. it is NOT images

# split the data into x and y pairs. x = data (images), y = labels (answers) 

x_train, y_train, x_test, y_test = (
    mnist_dataset["training_images"],
    mnist_dataset["training_labels"],
    mnist_dataset["test_images"],
    mnist_dataset["test_labels"],
)

########################################
# x_train = images for the training data 
# x_test = images for the testing data 

# y_train = labels for the training data 
# y_test = labels for the training data 
########################################

# inspection of images

import matplotlib.pyplot as plt

image_number = 55 # unlike matlab python arrays are zero indexed so this is really the image_number + 1 th image
mnist_image = x_train[image_number, :].reshape(28,28) # take a row and all of the values in the row, now reshape it into the image

fig, ax = plt.subplots()
ax.imshow(mnist_image, cmap="grey") # the colourmap is grey meaning that we get a black background and greyscale image 
# plt.show()
# print("x_train matches to a y_train of: ",  y_train[image_number])


num_examples = 5
seed = 147197952744
rng = np.random.default_rng(seed)

fig, axes = plt.subplots(1, num_examples)
for sample, ax in zip(rng.choice(x_train, size=num_examples, replace=False), axes):
    ax.imshow(sample.reshape(28, 28), cmap="gray")
# plt.show()

# fig is the whole output and axes are the individual sub plots within the figure. like the randomly picked numbers 

# print(x_train.dtype)

# logic here: 
# the data is currently in the form of unsigned integers, meaning a number between 0 and 255. We change the data type to a floating point 
# number and divide it by 255. this gives us values between 0-1. Now we have each pixel represented as an intensity value rather than a 0-255 
# value 

# we will use a subset of training data here and not the full 60,000. i would like to test this at a later date with the full dataset to see if 
# the results are any different purely out of curiousity 

training_sample, test_sample = 1000, 1000 # here is the subset of 1000 images out of the 60,000 and 10,000 in total, respectively

# take the n samples, where n = training_sample, and divide each pixel value by 255 to get the intensity, note that this is what is changing 
# the data type to float64 

training_images = x_train[0:training_sample] / 255
test_images = x_test[0:test_sample] / 255

print(len(training_images)) # 1000
print(training_images.shape) # 1000, 784 

print(training_images[0])


fig, ax = plt.subplots()
ax.imshow(training_images[0].reshape(28,28), cmap="grey")
# plt.show()

# we now know the first image in the dataset looks like a 5.

## New Concept ##

# One Hot Encoding is a way of storing labels as numbers without any label being seen as greater than any other label.
# For example, since in the mnist dataset we have 10 possible labels, 0-9, we can represent a number this way: 
# 1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# 9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

# very useful for the machine learning algorithm to make predictions, since we can have probabilities in this matrix: 

# [0.01, 0.02, 0.05, 0.90, 0.01, 0.00, 0.00, 0.00, 0.01, 0.00] this would be saying that we are 90% sure the number is a three
# we then compare this against the actual label, which might be: 
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] if the number genuinely is a 3 

# the algorithm will compare the prediction to the target and use this to adjust weights later on 

# one hot encoding function 
def one_hot_encoding(labels, dimension=10):
    # Define a one-hot variable for an all-zero vector
    # with 10 dimensions (number labels from 0 to 9).
    one_hot_labels = labels[..., None] == np.arange(dimension)[None]
    # Return one-hot encoded labels.
    return one_hot_labels.astype(np.float64)

# labels = NumPy array of integer labels 
# dimensions = ten possible classes (0-9)

# use one hot encoding on both the training and test labels 

# take the y's since these are the labels matched to the images (the x's)
training_labels = one_hot_encoding(y_train[:training_sample]) # passing in the subset of training labels and hot encoding them 
test_labels = one_hot_encoding(y_test[:training_sample]) # again taking the subset of n = 1000 at the time of writing



# x_train links with y_train as a key value pair, and the same for the other pair 

# print(training_labels[0]) # e.g print the first label in the training set as one hot encoding - we see it is a 5 if we toggle the 6th value (5)

# building the network # 

seed = 884736743
rng = np.random.default_rng(seed)

def relu(x):
    return (x >= 0 ) * x 
    # inner bracket is simply returning zero unless the value is larger than zero
    # e.g if the input is three, 1 is output.

    # the inner bracket is the boolean part, since this comparison will give us a new 
    # array built of true and false
    # then when we multiply the booleans by numbers, where 0 is false and 1 is true 

    # all we are doing is toggling the positives. any positive nonzero stays as is, else its a 0

x = np.array([ -3, 1, 3, 6, 8, 0])

print(x)
print(relu(x))

def relu2deriv(output):
    return output >= 0
    # derivative of the relu function which will return 1 for a positive and zero otherwise (think of it as the gradient of the input in a graphing sense)
    # makes sense intuitively, if the relu is giving us the positives and else its a zero, we are simply seeing which neurones fired
    # differentiate the relu -> did the neurone actually fire? 

    # what is the point of having a function for calculating the derivative of the relu function in the context of a neural network? 
    # -> updating weights so that we can have a more accurate prediction


# neural network architecture
# input layer (784) -> hidden layer (100) -> output layer (10)

#logic? 
# one neurone for each input pixel and one choice is generated at the output layer, that is, 0-9. hidden layer neurone count is simply a sweet spot for the problem, 
# it is not too large nor too small so we over or underfit to the problem. computing power is also considered here. 

# hyperparameters #

learning_rate = 0.005
epochs = 20 # the number of complete passes of data through the network 
hidden_size = 100 # amount of neurones in the hidden layer of the network, not trivial, more a sweet spot 
pixels_per_image  = 784 # amount of pixels in one image, 28 x 28 
num_labels = 10 # number of possible labels for each image (0-9) -> used in the prediction stage of the network 

# ~~~~~~~~~~~~~~ # 

# weights #

# weight matrices have to transform a vector from one layers size into the next layers size, the rules of matrix multiplication still apply here of course 

weights_1 = 0.2 * rng.random((pixels_per_image, hidden_size)) - 0.1 # shape = (784, 100)
weights_2 = 0.2 * rng.random((hidden_size, num_labels)) - 0.1 # shape = (100,10)

# TRAINING PROCESS 

