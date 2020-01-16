from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 

import utils as utils
import numpy as np
import keras as K

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
val_images, val_labels, val_paths       = utils.read_prep_imgs(val_path, img_width, img_height, weights)
train_images, train_labels, train_paths = utils.read_prep_imgs(train_path, img_width, img_height, weights)
test_images, test_labels, test_paths    = utils.read_prep_imgs(test_path, img_width, img_height, weights)


""" Get VGG Preds """
places_365_full = VGG16_Places365(weights = 'places', include_top = True)
places365_fc1 = K.models.Model(places_365_full.input, places_365_full.get_layer('fc1').output)

train_feature_vecs  = utils.get_feature_vecs(places365_fc1, train_images)
val_feature_vecs    = utils.get_feature_vecs(places365_fc1, val_images)
test_feature_vecs   = utils.get_feature_vecs(places365_fc1, test_images)

""" Get VGG + DAE Preds """
ae = K.models.load_model('../_weights/vgg16_fc1_dae_with_preprocess.hdf5')
encoder = K.models.Model(ae.input, ae.get_layer('dense_2').output)

encoded_train = utils.get_feature_vecs(encoder, train_feature_vecs)
encoded_test = utils.get_feature_vecs(encoder, test_feature_vecs)
encoded_val = utils.get_feature_vecs(encoder, val_feature_vecs)

""" Prepare for comparison """
encoded_data = [encoded_train,encoded_test,encoded_val]
passed_data = [train_feature_vecs, test_feature_vecs, val_feature_vecs]
labels = [train_labels, test_labels, val_labels]

doms = ['train', 'test', 'val']
perps = [5, 20, 50]


def run_tsne_comparison(vgg_dae, vgg, labels, perplexity):
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=10000, learning_rate=1)
    tsne_results = tsne.fit_transform(vgg_dae)
    tsne_results2 = tsne.fit_transform(vgg)
    
    df_tsne = pd.DataFrame(tsne_results, columns=['t-SNE_1', 't-SNE_2'])
    df_tsne['Label'] = labels
    df_tsne2 = pd.DataFrame(tsne_results2, columns=['t-SNE_1', 't-SNE_2'])
    df_tsne2['Label'] = labels
    
    return df_tsne, df_tsne2

def plot_tsne(df, model, perplexity, dom):
    f = sns.lmplot(x='t-SNE_1', y='t-SNE_2', data=df, hue='Label', fit_reg=False)
    f = plt.gca()
    f = t1.get_figure()
    f.savefig('tsne_{}_{}_{}.png'.format(model, perplexity, dom))

for i, (e, p) in enumerate(zip(encoded_data, passed_data)):
dom = doms[i]
lab = labels[i]
for perp in perps:
    df1, df2 = run_tsne_comparison(e, p, lab, perp)
    plot_tsne(df1, 'vggdae', perp, dom)
    plot_tsne(df2, 'vgg', perp, dom)