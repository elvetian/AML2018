import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tf_utils import input_fn_from_dataset, save_tf_record, prob_positive_class_from_prediction
from get_data import get_videos_from_folder,get_target_from_csv

from utils import save_solution

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Echocardiograpyh videos are 100x100 pixels, and have one color channel
    input_layer = tf.reshape(features["video"], [-1, 100, 100, 1])
    input_layer = tf.cast(input_layer, tf.float32)
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 100, 100, 1]
    # Output Tensor Shape: [batch_size, 100, 100, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 100, 100, 32]
    # Output Tensor Shape: [batch_size, 50, 50, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 50, 50, 32]
    # Output Tensor Shape: [batch_size, 50, 50, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 50, 50, 64]
    # Output Tensor Shape: [batch_size, 25, 25, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 25, 25, 64]
    # Output Tensor Shape: [batch_size, 25 * 25 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 25 * 25 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 25 * 25 * 64]
    # Output Tensor Shape: [batch_size, 2]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 1]
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.tile(labels, [212])
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    batchsize_video = 1

    dir_path = os.path.dirname(os.path.realpath(__file__))
    train_folder = os.path.join(dir_path, "train/")
    test_folder = os.path.join(dir_path, "test/")

    train_target = os.path.join(dir_path, 'train_target.csv')
    my_solution_file = os.path.join(dir_path, 'solution.csv')

    tf_record_dir = os.path.join(dir_path, 'tf_records')
    os.makedirs(tf_record_dir, exist_ok=True)

    tf_record_train = os.path.join(tf_record_dir, 'train' + '.tfrecords')
    tf_record_test = os.path.join(tf_record_dir, 'test' + '.tfrecords')

    if not os.path.exists(tf_record_train):
        x_train = get_videos_from_folder(train_folder)
        y_train = get_target_from_csv(train_target)
        save_tf_record(x_train, tf_record_train, y=y_train)

    if not os.path.exists(tf_record_test):
        x_test = get_videos_from_folder(test_folder)
        save_tf_record(x_test, tf_record_test)


    # Create the Estimator
    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="\\tmp\\model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    print('{}: Train'.format(datetime.now().strftime("%H:%M:%S")))
    # Train the model
    classifier.train(input_fn=lambda: input_fn_from_dataset(tf_record_train, batch_size=batchsize_video),
                     max_steps=1,
                     hooks=[logging_hook])

    print('{}: Evaluate'.format(datetime.now().strftime("%H:%M:%S")))
    # Evaluate the model and print results
    #eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=lambda: input_fn_from_dataset(tf_record_test, batch_size=batchsize_video))
    print(eval_results)

    print('{}: Predict'.format(datetime.now().strftime("%H:%M:%S")))
    pred = classifier.predict(input_fn=lambda: input_fn_from_dataset(tf_record_test, batch_size=batchsize_video, num_epochs=1, shuffle = False))


    print('{}: Save solution to {}'.format(datetime.now().strftime("%H:%M:%S"), my_solution_file))
    solution = prob_positive_class_from_prediction(pred)
    save_solution(my_solution_file, solution)

if __name__ == "__main__":
    tf.app.run()
