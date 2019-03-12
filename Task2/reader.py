

def read_csv():
    print('{}: Reading train dataset...'.format(datetime.now().strftime("%H:%M:%S")))
    x_train_dataset = pd.read_csv('X_train.csv', header=0)
    y_train_dataset = pd.read_csv('y_train.csv', header=0)
