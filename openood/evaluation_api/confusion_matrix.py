import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import torchvision.transforms as tf
import torch
import torch.nn as nn
from utils import anchor_means, gather_outputs
import numpy as np

def unique_labels(y_true, y_pred):
    labels = np.concatenate((y_true, y_pred), axis=0)
    return np.unique(labels)

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Oranges):
    """
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
    # if not title:
    #     if normalize:
    #         title = 'Normalized confusion matrix'
    #     else:
    #         title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
    # print(cm.shape[1])
    # print(cm.shape[0])
    font = FontProperties(family='Times New Roman', size=12)
    # print(len(y_pred),len(y_true))
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    # classes = list(set(y_true)&set(y_pred))
    # print(classes,len(classes))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    else:
        pass
    # print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    im.set_clim(0, 1200)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label'
           )
    ax.set_xlabel('Predicted label', fontproperties='Times New Roman', fontsize=14)
    ax.set_ylabel('True label', fontproperties='Times New Roman', fontsize=14)
    ax.set_ylim(len(classes) - 0.5, -0.5)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font)

    # 设置y轴刻度标签的字体为新罗马字体
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax