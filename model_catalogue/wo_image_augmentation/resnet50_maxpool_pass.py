""" Libraries """
import utils as utils
import numpy as np
import keras as K
import pprint

from vgg16_places_365 import VGG16_Places365

""" Path Params """
base_path  = '../../'
train_path = base_path + 'train/'
test_path  = base_path + 'test/'
val_path   = base_path + 'val/'

results_path    = '../'
weights_path    = results_path + '_weights/'
distances_path  = results_path + '_distances/'
results_path    = results_path + 'results.json'

""" Model Params """
img_size = (224, 224)
img_width, img_height = img_size
weights = 'imagenet'

model_name = 'resnet50_maxpool_pass'


""" Load images, labels and corresponding paths into np.arrays """
val_images, val_labels, val_paths       = utils.read_prep_imgs_clean(val_path, img_width, img_height)
train_images, train_labels, train_paths = utils.read_prep_imgs_clean(train_path, img_width, img_height)
test_images, test_labels, test_paths    = utils.read_prep_imgs_clean(test_path, img_width, img_height)

""" Load ResNet50 pretrained on imagenet data """
resnet50 = K.applications.resnet50.ResNet50(include_top = False,
                                     input_shape = (img_width, img_height, 3),
                                     pooling = 'avg',
                                     weights = weights)


""" Pass data through ResNet50 """
train_feature_vecs  = utils.get_feature_vecs(resnet50, train_images)
val_feature_vecs    = utils.get_feature_vecs(resnet50, val_images)
test_feature_vecs   = utils.get_feature_vecs(resnet50, test_images)

""" Evaluate model accuracy """
matrix_path = distances_path + model_name + "_{}" + ".txt"
test_dmatrix = utils.cosine_sim_matrix(resnet50, test_images)
utils.write_to_distances(matrix_path.format('test'), test_dmatrix)

val_dmatrix = utils.cosine_sim_matrix(resnet50, val_images)
utils.write_to_distances(matrix_path.format('val'), val_dmatrix)

test_accuracy = utils.calc_accuracy(test_dmatrix, test_labels)
val_accuracy = utils.calc_accuracy(val_dmatrix, val_labels)

test_group_accuracy = utils.calc_top_accuracy(test_dmatrix, test_labels)
val_group_accuracy = utils.calc_top_accuracy(val_dmatrix, val_labels)

results_dict = {
    "model_name": model_name,
    "test_acc": test_accuracy,
    "val_acc": val_accuracy,
    "test_group_acc": test_group_accuracy,
    "val_group_acc": val_group_accuracy,
    "regularization": None,
    "noise": None,
    "bottleneck_size": None,
    "trained_epochs": None,
    "batch_size": None 
}

pprint.pprint(results_dict)

utils.write_to_results(results_dict, results_path)












