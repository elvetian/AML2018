

class Signal(object):
    def __init__(self, id, x_eeg1, x_eeg2, x_emg, y_labels):
        self.id = id
        self.x_eeg1 = x_eeg1
        self.x_eeg2 = x_eeg2
        self.x_emg = x_emg
        self.y_labels = y_labels

        self.sampling_rate = 128.  # Hz

