""" Libraries """
import utils as utils
import numpy as np
import keras as K
import pprint

#from vgg16_places_365 import VGG16_Places365

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
weights = 'places'

model_name = 'resnet50_maxpool_dae'


""" Load images, labels and corresponding paths into np.arrays """
val_images, val_labels, val_paths       = utils.read_prep_imgs(val_path, img_width, img_height, weights)
train_images, train_labels, train_paths = utils.read_prep_imgs(train_path, img_width, img_height, weights)
test_images, test_labels, test_paths    = utils.read_prep_imgs(test_path, img_width, img_height, weights)

""" Load ResNet50 pretrained on imagenet data """
resnet50 = K.applications.resnet50.ResNet50(include_top = False,
                                     input_shape = (img_width, img_height, 3),
                                     pooling = 'avg',
                                     weights = 'imagenet')


""" Pass data through ResNet50 """
train_feature_vecs  = utils.get_feature_vecs(resnet50, train_images)
val_feature_vecs    = utils.get_feature_vecs(resnet50, val_images)
test_feature_vecs   = utils.get_feature_vecs(resnet50, test_images)

""" Corrupt input """
noise_factor = 0.3
train_corrupted = utils.corrupt_input(train_feature_vecs, noise_factor)
val_corrupted = utils.corrupt_input(val_feature_vecs, noise_factor)


""" Extend ResNet50 w. Sparse Denoising Autoencoder """
# Model hyperparams
regularization = None #K.regularizers.l1(1e-4)
bottleneck_size = 512
n_epochs = 2000
batch_size = 128

# Model definition
autoencoder_input = resnet50.output_shape[1]

input_layer = K.layers.Input(shape = (autoencoder_input,))

encoded = K.layers.Dense(2048, activation = 'relu')(input_layer)
encoded = K.layers.Dense(bottleneck_size, activation = 'relu')(encoded)

decoded = K.layers.Dense(2048, activation = 'relu')(encoded)
decoded = K.layers.Dense(autoencoder_input, activation = 'sigmoid')(decoded)

resnet50_maxpool_dae = K.models.Model(input_layer, decoded)
encoder = K.models.Model(input_layer, encoded)

resnet50_maxpool_dae.compile(optimizer = 'adadelta', loss = 'mse') 

callbacks = utils.get_callbacks(model_name)

resnet50_maxpool_dae.fit(train_corrupted, train_feature_vecs,
                epochs = n_epochs, batch_size = batch_size,
                validation_data = (val_corrupted, val_feature_vecs),
                callbacks = callbacks)

last_epoch = callbacks[1].stopped_epoch

""" Save the Model """
resnet50_maxpool_dae.save(weights_path + model_name + '.hdf5')

""" Evaluate model accuracy """
matrix_path = distances_path + model_name + "_{}" + ".txt"
test_dmatrix = utils.cosine_sim_matrix(encoder, test_feature_vecs)
utils.write_to_distances(matrix_path.format('test'), test_dmatrix)

val_dmatrix = utils.cosine_sim_matrix(encoder, val_feature_vecs)
utils.write_to_distances(matrix_path.format('val'), val_dmatrix)

test_accuracy = utils.calc_accuracy(test_dmatrix, test_labels)
val_accuracy = utils.calc_accuracy(val_dmatrix, val_labels)

test_group_accuracy = utils.calc_top_accuracy(test_dmatrix, test_labels)
val_group_accuracy = utils.calc_top_accuracy(val_dmatrix, val_labels)

results_dict = {
    "model_name": model_name,
    "preprocess": weights,
    "test_acc": test_accuracy,
    "val_acc": val_accuracy,
    "test_group_acc": test_group_accuracy,
    "val_group_acc": val_group_accuracy,
    "regularization": None, #str(regularization.get_config()),
    "noise": noise_factor,
    "bottleneck_size": bottleneck_size,
    "trained_epochs": last_epoch,
    "batch_size": batch_size 
}

pprint.pprint(results_dict)

utils.write_to_results(results_dict, results_path)
















