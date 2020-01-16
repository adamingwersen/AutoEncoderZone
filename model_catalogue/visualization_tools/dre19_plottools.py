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
import utils 

def plot_list_similar(dist_matrix, idxs, labels, paths):
    #idxs = [298, 563, 111, 236, 837]
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fig, axs = plt.subplots(len(idxs), 2, figsize=(20, 30))
    for i in range(0, len(idxs)):
        fig.tight_layout()
        image_l = load_base_img(paths[idxs[i]], 224, 224)
        image_l = cv2.putText(img = image_l, text = '{}'.format(paths[idxs[i]].split('/')[2]), 
                              org = (0,20), fontFace=font, fontScale=0.5, color=(255,0,0), thickness=1)
        image_r_idx = most_similar_idx(dist_matrix, idxs[i])
        image_r = load_base_img(paths[image_r_idx], 224, 224)
        image_r = cv2.putText(img = image_r, text = '{}'.format(paths[image_r_idx].split('/')[2]), 
                              org = (0,20), fontFace=font, fontScale=0.5, color=(255,0,0), thickness=1)
        axs[i,0].imshow(image_l)
        axs[i,1].imshow(image_r)

def plot_list_similar_sideways(dist_matrix, idxs, labels, paths, save = None):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fig, axs = plt.subplots(2, len(idxs), figsize=(16, 6))
    fig.tight_layout()
    for i in range(0, len(idxs)):
        image_l = utils.load_base_img(paths[idxs[i]], 224, 224)
        
        image_r_idx = utils.most_similar_idx(dist_matrix, idxs[i])
        image_r = utils.load_base_img(paths[image_r_idx], 224, 224)
        
        axs[0,i].imshow(image_l)
        axs[0,i].axis('off')
        axs[0,i].set_title('query: {}'.format(paths[idxs[i]].split('/')[3]))
        
        axs[1,i].imshow(image_r)
        axs[1,i].axis('off')
        axs[1,i].set_title('prediction: {}'.format(paths[image_r_idx].split('/')[3]))
        if save != None:
            fig.savefig('{}.png'.format(save))



def plot_accuracies(df, title, save = None): 
""" EXAMPLE Input
results_resnet = json.loads("""
{"model_name": ["resnet50_maxpool_pass", "resnet50_maxpool_pass", "resnet50_maxpool_pass"], 
 "preprocess": ["Simple", "ImageNet", "Places"],
 "test_acc": [0.3333333333333333, 0.34476190476190477, 0.6438095238095238], 
 "val_acc": [0.34213006597549483, 0.35626767200754006, 0.6663524976437323], 
 "test_group_acc": [0.7038095238095238, 0.6933333333333334, 0.9038095238095238], 
 "val_group_acc": [0.6710650329877474, 0.6786050895381716, 0.8982092365692743], 
 "regularization": [null, null, null], 
 "noise": [null, null, null], 
 "bottleneck_size": [null, null, null], 
 "trained_epochs": [null, null, null], 
 "batch_size": [null, null, null]}
""")

df_resnet = pd.DataFrame.from_dict(results_resnet)
"""
    df = df.T
    pos = list(range(4))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10,5))
    
    acc_cols = ["test_acc", "val_acc", "test_group_acc", "val_group_acc"]
    acc_col_titles = ['Test Acc.', 'Val. Acc.', 'Top-5 Test Acc.', 'Top-5 Val. Acc.']

    plt.bar(pos, df[0][acc_cols],
            width, alpha = 0.5, label = df[0]['preprocess'])

    plt.bar([p + width for p in pos], df[1][acc_cols], 
            width, alpha = 0.5, label = df[1]['preprocess'])

    plt.bar([p + width*2 for p in pos], df[2][acc_cols], 
            width, alpha = 0.5, label = df[2]['preprocess'])

    ax.set_xticks([p + 1.5 * width for p in pos])
    ax.set_xticklabels(acc_col_titles)
    
    plt.xlim(min(pos)-width, max(pos)+width*4)
    plt.ylim([0, 1])

    ax.set_ylabel('Accuracy')
    plt.legend(['Simple', 'ImageNet', 'Places'], loc='upper left', title='Preprocessing')
    ax.set_title(title)
    plt.grid()
    plt.show()
    if save != None:
        fig.savefig('{}.png'.format(save), bbox_inches='tight')

def plot_accuracies_model(df, title, save = None):
    df = df.T
    pos = list(range(4))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10,5))
    
    acc_cols = ["test_acc", "val_acc", "test_group_acc", "val_group_acc"]
    acc_col_titles = ['Test Acc.', 'Val. Acc.', 'Top-5 Test Acc.', 'Top-5 Val. Acc.']

    plt.bar(pos, df[0][acc_cols],
            width, alpha = 0.5, label = df[0]['preprocess'])

    plt.bar([p + width for p in pos], df[1][acc_cols], 
            width, alpha = 0.5, label = df[1]['preprocess'])

    plt.bar([p + width*2 for p in pos], df[2][acc_cols], 
            width, alpha = 0.5, label = df[2]['preprocess'])

    ax.set_xticks([p + 1.5 * width for p in pos])
    ax.set_xticklabels(acc_col_titles)
    
    plt.xlim(min(pos)-width, max(pos)+width*4)
    plt.ylim([0, 1])

    ax.set_ylabel('Accuracy')
    plt.legend(['VGG16 FP', 'VGG16 FP + Deep AE', 'VGG16 FB + DAE'], loc='lower right', title='Model')
    ax.set_title(title)
    plt.grid()
    plt.show()
    if save != None:
        fig.savefig('{}.png'.format(save), bbox_inches='tight')


def plot_1_3_similar(query, dmatrix, labels, paths, save = None):
    zipped = utils.most_similar_arr_w_similarity(dmatrix[:,query])
    query_image = utils.load_base_img(paths[query], 224, 224)
    fig = plt.figure(1, figsize=(18, 12))
    fig.tight_layout()
    matplotlib.gridspec.GridSpec(3,3)
    
    #left side
    plt.subplot2grid((3,3), (0,0), colspan = 2, rowspan = 3)
    plt.axis('off')
    plt.title('query image, class={}'.format(labels[query]))
    plt.imshow(query_image)
    
    # right side
    for i in range(0, len(zipped)-2):
        plt.subplot2grid((3,3), (i, 2))
        plt.axis('off')
        t_index = zipped[i][0]
        t_distance = zipped[i][1]
        plt.title('rank={}, distance={}, class={}'.format(i+1, round(t_distance), labels[t_index]))
        result_image = utils.load_base_img(paths[t_index], 224, 224)
        plt.imshow(result_image)
    if save != None:
        fig.savefig('{}.png'.format(save), bbox_inches='tight')

def plot_list_similar_sideways(dist_matrix, idxs, labels, paths, save = None):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fig, axs = plt.subplots(2, len(idxs), figsize=(16, 6))
    fig.tight_layout()
    for i in range(0, len(idxs)):
        image_l = utils.load_base_img(paths[idxs[i]], 224, 224)
        
        image_r_idx = utils.most_similar_idx(dist_matrix, idxs[i])
        image_r = utils.load_base_img(paths[image_r_idx], 224, 224)
        
        axs[0,i].imshow(image_l)
        axs[0,i].axis('off')
        axs[0,i].set_title('query: {}'.format(paths[idxs[i]].split('/')[3]))
        
        axs[1,i].imshow(image_r)
        axs[1,i].axis('off')
        axs[1,i].set_title('prediction: {}'.format(paths[image_r_idx].split('/')[3]))
        if save != None:
            fig.savefig('{}.png'.format(save))