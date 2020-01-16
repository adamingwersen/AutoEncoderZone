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
weights = 'places'

model_name = 'vgg16_fc1_deep_ae'


""" Load images, labels and corresponding paths into np.arrays """
val_images, val_labels, val_paths       = utils.read_prep_imgs_clean(val_path, img_width, img_height)
train_images, train_labels, train_paths = utils.read_prep_imgs_clean(train_path, img_width, img_height)
test_images, test_labels, test_paths    = utils.read_prep_imgs_clean(test_path, img_width, img_height)

""" Load VGG16 pretrained on places365 data """
places_365_full = VGG16_Places365(weights = 'places', include_top = True) # include_top = True is faulty. Does not import lower layers

# Forward pass through fully connected layer (fc1)
places365_fc1 = K.models.Model(places_365_full.input, places_365_full.get_layer('fc1').output)

""" Pass data through VGG16 """
train_feature_vecs  = utils.get_feature_vecs(places365_fc1, train_images)
val_feature_vecs    = utils.get_feature_vecs(places365_fc1, val_images)
test_feature_vecs   = utils.get_feature_vecs(places365_fc1, test_images)


""" Extend VGG16 w. Sparse Denoising Autoencoder """
# Model hyperparams
noise_factor = None
regularization = None
bottleneck_size = 1028
n_epochs = 2000
batch_size = 128

# Model definition
autoencoder_input = places365_fc1.output_shape[1]

input_layer = K.layers.Input(shape = (autoencoder_input,))

encoded = K.layers.Dense(4096, activation = 'relu')(input_layer)
encoded = K.layers.Dense(2048, activation = 'relu')(encoded)
encoded = K.layers.Dense(bottleneck_size, activation = 'relu')(encoded)

decoded = K.layers.Dense(2048, activation = 'relu')(encoded)
decoded = K.layers.Dense(4096, activation = 'relu')(decoded)
decoded = K.layers.Dense(autoencoder_input, activation = 'sigmoid')(decoded)

vgg16_fc1_deep_ae = K.models.Model(input_layer, decoded)
encoder = K.models.Model(input_layer, encoded)

vgg16_fc1_deep_ae.compile(optimizer = 'adadelta', loss = 'mse')

callbacks = utils.get_callbacks(model_name)

vgg16_fc1_deep_ae.fit(train_feature_vecs, train_feature_vecs,
                epochs = n_epochs, batch_size = batch_size,
                validation_data = (val_feature_vecs, val_feature_vecs),
                callbacks = callbacks)

last_epoch = callbacks[1].stopped_epoch

""" Save the Model """
vgg16_fc1_deep_ae.save(weights_path + model_name + '.hdf5')

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
    "test_acc": test_accuracy,
    "val_acc": val_accuracy,
    "test_group_acc": test_group_accuracy,
    "val_group_acc": val_group_accuracy,
    "regularization": regularization,
    "noise": noise_factor,
    "bottleneck_size": bottleneck_size,
    "trained_epochs": last_epoch,
    "batch_size": batch_size 
}

pprint.pprint(results_dict)

utils.write_to_results(results_dict, results_path)












