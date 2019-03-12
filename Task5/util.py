import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import itertools

# Balanced Multi Class Accuracy
def bmac_score(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


# partition train/test data set
def train_test_partition(x_input, y_labels, valid_size=0.2):
    x_valid = pd.DataFrame()
    x_train = pd.DataFrame()
    y_train = pd.DataFrame()
    y_valid = pd.DataFrame()
    class_values = y_labels['y'].unique()
    for class_value in class_values:
        y_class = y_labels[y_labels['y']==class_value]
        x_class = x_input.loc[y_class.index.values, :]
        x_class_train, x_class_valid, y_class_train, y_class_valid = train_test_split(x_class, y_class, test_size=valid_size, random_state=123)

        x_train = pd.concat([x_train, pd.DataFrame(x_class_train)])
        y_train = pd.concat([y_train, pd.DataFrame(y_class_train)])
        x_valid = pd.concat([x_valid, pd.DataFrame(x_class_valid)])
        y_valid = pd.concat([y_valid, pd.DataFrame(y_class_valid)])

    return x_train, y_train, x_valid, y_valid


def train_test_partition_np(x_input, y_labels, valid_size=0.2):
    x_valid = np.empty((0, 2, 512))
    x_train = np.empty((0, 2, 512))
    y_train = np.empty((0, 1))
    y_valid = np.empty((0, 1))
    class_values = y_labels['y'].unique()
    for class_value in class_values:
        y_class = y_labels[y_labels['y']==class_value]
        x_class = np.take(x_input, y_class.index.values, axis=0)
        x_class_train, x_class_valid, y_class_train, y_class_valid = train_test_split(x_class, y_class, test_size=valid_size, random_state=123)

        x_train = np.concatenate((x_train, x_class_train))
        y_train = np.concatenate((y_train, y_class_train))
        x_valid = np.concatenate((x_valid, x_class_valid))
        y_valid = np.concatenate((y_valid, y_class_valid))

    return x_train, y_train, x_valid, y_valid


def print_n_samples_each_class(labels):
    import numpy as np
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print("{}: {}".format(c, n_samples))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
