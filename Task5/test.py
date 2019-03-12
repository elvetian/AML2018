import numpy as np
from datetime import datetime
import pandas as pd
from keras.models import load_model
from EEGModels import *


test_eeg1_filename = 'test_eeg1.csv'
test_eeg2_filename = 'test_eeg2.csv'
test_emg_filename = 'test_emg.csv'

print('{}: Reading test dataset...'.format(datetime.now().strftime("%H:%M:%S")))
x_test_eeg1 = pd.read_csv(test_eeg1_filename, header=0)
x_test_eeg2 = pd.read_csv(test_eeg2_filename, header=0)
x_test_emg = pd.read_csv(test_emg_filename, header=0)

x_test_data = pd.concat([x_test_eeg1.iloc[:, 1:], x_test_eeg2.iloc[:, 1:]], axis=1, keys=['eeg1', 'eeg2'])
print('x_test.shape={}'.format(x_test_data.shape))
x_test_data = x_test_data.values.reshape(-1, 2, 1, 512)
print('x_test.shape={}'.format(x_test_data.shape))
model_name = 'eegnet_eeg12_chans2_epochs3.h5'
print('{}: Loading model {}'.format(datetime.now().strftime("%H:%M:%S"), model_name))
model = load_model(model_name)

print('{}: Predicting...'.format(datetime.now().strftime("%H:%M:%S")))
y_test_pred = model.predict_on_batch(x_test_data)

print('{}: Writing solution file...'.format(datetime.now().strftime("%H:%M:%S")))
y_test_dataset=pd.DataFrame({'Id': x_test_eeg1.Id.values})
y_test_dataset['y'] = np.add(y_test_pred.argmax(axis=1), 1)
header = ["Id", "y"]
y_test_dataset.to_csv('solution.csv', columns=header, index=False)

