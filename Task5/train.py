import numpy as np
from datetime import datetime
from EEGModels import *
from keras.utils import to_categorical

from util import *



train_eeg1_filename = 'train_eeg1.csv'
train_eeg2_filename = 'train_eeg2.csv'
train_emg_filename = 'train_emg.csv'
train_lables_filename = 'train_labels.csv'
test_eeg1_filename = 'test_eeg1.csv'
test_eeg2_filename = 'test_eeg2.csv'
test_emg_filename = 'test_emg.csv'

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
# x_train_data = pd.concat([x_train_eeg1.iloc[:, 1:], x_train_eeg2.iloc[:, 1:]], axis=1, keys=['eeg1', 'eeg2'])
x_train_data = x_train_eeg1.iloc[:, 1:]
y_train_data = y_train_labels.iloc[:, 1:]

print('{}: Train/Test split...'.format(datetime.now().strftime("%H:%M:%S")))
x_train, y_train, x_valid, y_valid = train_test_partition(x_train_data, y_train_data)
print('{}: Train set: x[{}], y[{}]'.format(datetime.now().strftime("%H:%M:%S"), x_train.shape, y_train.shape))
print_n_samples_each_class(y_train)
print('{}: Validation set: x[{}], y[{}]'.format(datetime.now().strftime("%H:%M:%S"), x_valid.shape, y_valid.shape))
print_n_samples_each_class(y_valid)

print('{}: Fitting...'.format(datetime.now().strftime("%H:%M:%S"), x_valid.shape, y_valid.shape))
model = DeepConvNet(nb_classes=4,
                    Chans=1,
                    Samples=512)
model.compile(loss='categorical_crossentropy', optimizer='adam')
x_train_reshaped = x_train.values.reshape([-1, 1, 1, x_train.shape[1]])
y_train_one_hot = to_categorical(y_train)
fitted_model = model.fit(x_train_reshaped, y_train_one_hot)

model.save('deepconvnet_epoch1.h5')

