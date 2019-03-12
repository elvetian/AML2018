



print('{}: Predicting...'.format(datetime.now().strftime("%H:%M:%S"), x_valid.shape, y_valid.shape))
x_valid_reshaped = x_valid.values.reshape([-1, 1, 1, x_valid.shape[1]])
y_valid_pred = model.predict(x_valid_reshaped)
y_valid_one_hot = to_categorical(y_valid)
bmac_score = bmac_score(y_valid, y_valid_pred.argmax(axis=1))
print("BMAC score={}".format(bmac_score))
