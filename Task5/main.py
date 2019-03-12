import numpy as np
from datetime import datetime
from EEGModels import *
from keras.utils import to_categorical
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

from util import *

train_eeg1_filename = 'train_eeg1.csv'
train_eeg2_filename = 'train_eeg2.csv'
train_emg_filename = 'train_emg.csv'
train_lables_filename = 'train_labels.csv'


print('{}: Reading train dataset...'.format(datetime.now().strftime("%H:%M:%S")))
x_train_eeg1 = pd.read_csv(train_eeg1_filename, header=0)
x_train_eeg2 = pd.read_csv(train_eeg2_filename, header=0)
x_train_emg = pd.read_csv(train_emg_filename, header=0)
y_train_labels = pd.read_csv(train_lables_filename, header=0)

class_values = y_train_labels.groupby('y')['Id'].nunique()
print(class_values)

x_train_eeg1 = x_train_eeg1.fillna(0)
x_train_eeg2 = x_train_eeg2.fillna(0)
x_train_emg = x_train_emg.fillna(0)

# x_train_data = pd.DataFrame(data=np.dstack([x_train_eeg1.iloc[:, 1:], x_train_eeg2.iloc[:, 1:]]),
#                             index=x_train_eeg1.Id[0:],
#                             columns=x_train_eeg1.columns[1:])
# x_train_data = data=np.dstack([x_train_eeg1.iloc[:, 1:], x_train_eeg2.iloc[:, 1:]])
x_train_data = pd.concat([x_train_eeg1.iloc[:, 1:], x_train_eeg2.iloc[:, 1:]], axis=1, keys=['eeg1', 'eeg2'])
print('x_train.shape={}'.format(x_train_data.shape))
x_train_data = x_train_data.values.reshape(x_train_data.shape[0], 2, 512)
print('x_train.shape={}'.format(x_train_data.shape))
# x_train_data = x_train_eeg1.iloc[:, 1:]
y_train_data = y_train_labels.iloc[:, 1:]

print('{}: Train/Test split...'.format(datetime.now().strftime("%H:%M:%S")))
x_train, y_train, x_valid, y_valid = train_test_partition_np(x_train_data, y_train_data)
print('{}: Train set: x[{}], y[{}]'.format(datetime.now().strftime("%H:%M:%S"), x_train.shape, y_train.shape))
print_n_samples_each_class(y_train)
print('{}: Validation set: x[{}], y[{}]'.format(datetime.now().strftime("%H:%M:%S"), x_valid.shape, y_valid.shape))
print_n_samples_each_class(y_valid)

print('{}: Fitting...'.format(datetime.now().strftime("%H:%M:%S"), x_valid.shape, y_valid.shape))
# model = DeepConvNet(nb_classes=3,
#                     Chans=2,
#                     Samples=512)
# model = ShallowConvNet(nb_classes=3,
#                     Chans=2,
#                     Samples=512)
model = EEGNet(nb_classes=3,
                    Chans=2,
                    Samples=512,
                    dropoutRate=0.5,
                    F1=8,
                    F2=16)
model.compile(loss='categorical_crossentropy', optimizer='adam')
x_train_reshaped = x_train.reshape((-1, 2, 1, 512))
y_train = np.subtract(y_train, 1)
y_train_one_hot = to_categorical(y_train)
sample_weights = class_weight.compute_sample_weight('balanced', y_train_one_hot)
print(sample_weights)
fitted_model = model.fit(x_train_reshaped, y_train_one_hot, epochs=3, shuffle=True, sample_weight=sample_weights)

model.save('eegnet_eeg12_chans2_epochs3.h5')

print('{}: Predicting...'.format(datetime.now().strftime("%H:%M:%S"), x_valid.shape, y_valid.shape))
x_valid_reshaped = x_valid.reshape((-1, 2, 1, 512))
y_valid_pred = model.predict(x_valid_reshaped)
y_valid = np.subtract(y_valid, 1)
bmac_score = bmac_score(y_valid, y_valid_pred.argmax(axis=1))
print("BMAC score={}".format(bmac_score))

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_valid, y_valid_pred.argmax(axis=1))
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_values, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
