import os
import requests # this allows python to communicate with websites, we need this to access the data on the website and download it 
import gzip # allows us to unzip files  
import numpy as np

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

# inspection of images

import matplotlib.pyplot as plt

image_number = 55 # unlike matlab python arrays are zero indexed so this is really the image_number + 1 th image
mnist_image = x_train[image_number, :].reshape(28,28) # take a row and all of the values in the row, now reshape it into the image

fig, ax = plt.subplots()
ax.imshow(mnist_image, cmap="grey") # the colourmap is grey meaning that we get a black background and greyscale image 
plt.show()
print("x_train matches to a y_train of: ",  y_train[image_number])


num_examples = 5
seed = 147197952744
rng = np.random.default_rng(seed)

fig, axes = plt.subplots(1, num_examples)
for sample, ax in zip(rng.choice(x_train, size=num_examples, replace=False), axes):
    ax.imshow(sample.reshape(28, 28), cmap="gray")
plt.show()

# fig is the whole output and axes are the individual sub plots within the figure. like the randomly picked numbers 

