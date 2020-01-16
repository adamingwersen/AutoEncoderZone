import cv2
import matplotlib.pyplot as plt
import keras as K
from scipy.spatial.distance import cdist
import numpy as np
import tqdm
import glob
from PIL import Image
import random
from pathlib import Path
import os
import datetime
import heapq
import json
import places_utils as P

""" keras.callbacks """
def get_callbacks(model_name):
    #checkpoint_path = '../keras_checkpoints'+ '/{}-{}'.format(model_name, datetime.datetime.now()) + '-{epoch:02d}.hdf5'
    #checkpointer = K.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose = 1)
    tensorboard = K.callbacks.TensorBoard(log_dir = '../../tensorboard_logs/{}-{}'.format(model_name, datetime.datetime.now()))
    earlystop = K.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto')
    return [tensorboard, earlystop]

""" Image Preprocessing """
def prep_img_clean(file, w, h):
    img = K.preprocessing.image.load_img(file, target_size = (w, h))
    img = K.preprocessing.image.img_to_array(img)
    label = file.split(os.path.sep)[-2]
    return img, label

def read_prep_imgs_clean(path, w, h):
    img_paths = glob.glob(str(Path(path).expanduser() / '**/*.jpg'))
    random.shuffle(img_paths)
    imgs, labels, paths = [], [], []
    for f in tqdm.tqdm(img_paths):
        i, l = prep_img_clean(f, w, h)
        imgs.append(i)
        labels.append(l)
        paths.append(f) # return paths so that the right images can be found when visualizing
    return np.array(imgs, dtype = 'float32')/255.0, np.array(labels), np.array(img_paths)


# Loads a single image with resnet50 preprocessing configuration
def prep_img(file, w, h, weights):
    img = K.preprocessing.image.load_img(file, target_size = (w, h))
    img = K.preprocessing.image.img_to_array(img)
    label = file.split(os.path.sep)[-2]
    if weights == 'imagenet':
        img = K.applications.resnet50.preprocess_input(img)
    if weights == 'places':
        img = P.preprocess_input(img)
    else: 
        img = img / 255.0
    return img, label

# Loads a set of images and labels determined by their directories.
# Follows Keras dataloader structure
def read_prep_imgs(path, w, h, weights):
    img_paths = glob.glob(str(Path(path).expanduser() / '**/*.jpg'))
    random.shuffle(img_paths)
    imgs, labels, paths = [], [], []
    for f in tqdm.tqdm(img_paths):
        i, l = prep_img(f, w, h, weights)
        imgs.append(i)
        labels.append(l)
        paths.append(f) # return paths so that the right images can be found when visualizing
    return np.array(imgs, dtype = 'float32'), np.array(labels), np.array(img_paths)

# Retrieves output layer prediction
def get_feature_vecs(model, input_imgs):
    return model.predict(input_imgs, verbose = 0)

# Flattens nparray/image
def np_flatten(images):
    return images.reshape((len(images), np.prod(images.shape[1:])))

def corrupt_input(vector, noise_factor = 0.5):
    corrupted = vector + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = vector.shape)
    return np.clip(corrupted, 0., 1.)

# Loads single image without preprocessing from path
# Handy for visualizations
def load_base_img(path, w, h):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (w, h))

""" Metrics """
# Returns the most similar index
def min_sim_idx(sim_matrix):
    np.fill_diagonal(sim_matrix, 1)
    return sim_matrix.argmin(axis=0)

# Returns the most similar index
def most_similar_idx(sim_matrix, index):
    column = sim_matrix[:,index]
    return heapq.nsmallest(2, range(len(column)), column.take)[1]

# Calculate top-n most similar indexes
def most_similar_arr(column, n = 5):
    #column = sim_matrix[:,index]
    return heapq.nsmallest(n+1, range(len(column)), column.take)[1:]

# Calculate top-n most similar indexes  - also return cosine similarity coefficient
def most_similar_arr_w_similarity(column, n = 5):
    top5 = heapq.nsmallest(n+1, range(len(column)), column.take)[1:]
    sims = column[top5]
    return list(zip(top5, sims))

# Calculate top-n accuracy
def calc_top_accuracy(sim_matrix, labels):
    bins = []
    for i in range(0, len(sim_matrix)):
        arr = most_similar_arr(sim_matrix[:,i])
        subbins = []
        s_label = labels[i]
        for j in range(0, len(arr)):
            p_label = labels[arr[j]]
            subbins.append(s_label == p_label)
        if True in subbins:
            bins.append(True)
        else:
            bins.append(False)
    return sum(bins)/len(bins)

# Calculate accuracy of a similarity matrix
def calc_accuracy(sim_matrix, labels):
    bins = []
    matrix = min_sim_idx(sim_matrix)
    for i in range(len(matrix)):
        s_label = labels[i]
        p_label = labels[matrix[i]]
        bins.append(s_label == p_label)
    return sum(bins)/len(bins)

# Calculate cosine distance matrix from model predictions
def cosine_sim_matrix(model: K.models.Model, input_vector):
    preds = model.predict(input_vector)
    sim_matrix = cdist(preds, preds, metric = 'cosine')
    return sim_matrix

""" Files """
def is_non_zero_file(filename):  
    return os.path.isfile(filename) and os.path.getsize(filename) > 0

def write_to_results(json_contents, filename):
    f = open(filename, 'a')
    f.write("\n" + json.dumps(json_contents) + ";")
    f.close()

def write_to_distances(filename, sim_matrix):
    np.savetxt(filename, sim_matrix, fmt='%.4f', delimiter = ';')

# For visualization purposes
def read_prep_imgs_noshuffle(path, w, h, weights):
    img_paths = glob.glob(str(Path(path).expanduser() / '**/*.jpg'))
    imgs, labels, paths = [], [], []
    for f in tqdm.tqdm(img_paths):
        i, l = utils.prep_img(f, w, h, weights)
        imgs.append(i)
        labels.append(l)
        paths.append(f) # return paths so that the right images can be found when visualizing
    return np.array(imgs, dtype = 'float32'), np.array(labels), np.array(img_paths)






