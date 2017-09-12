import os
import sys

import numpy as np
import scipy.misc
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

import models

MODEL_PATH = 'NYU_ResNet-UpProj.npy'


def load_network(session, model_data_path):
    # Default input size
    batch_size = 1
    channels = 3
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size)

    # Load the converted parameters
    print('Loading the model')
    net.load(model_data_path, session)

    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            session.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    session.run(init_new_vars_op)

    def prediction_func(img):
        return session.run(net.get_output(), feed_dict={input_node: img})

    return prediction_func


height = 228
width = 304


def predict(prediction_func, image_path, channel_to_change=2, visualize=False):
    # Read image
    input_img = Image.open(image_path)
    img = input_img.resize([width, height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis=0)

    # Evalute the network for the given image
    pred = prediction_func(img)

    pred = np.uint8(np.minimum(np.resize(pred, pred.shape[1:3]) * 13, 255))
    pred_resized = scipy.misc.imresize(pred, input_img.size, interp='cubic')
    result_img = np.asarray(input_img)
    result_img.setflags(write=True)
    result_img[:, :, channel_to_change] = pred_resized

    # Plot result
    if visualize:
        fig = plt.figure()
        ii = plt.imshow(pred[:, :], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()

    return result_img


def main():
    info_file_path = sys.argv[1]
    img_paths = (s.split(' ') for s in open(info_file_path).read().split('\n') if s)

    with tf.Session() as session:
        prediction_func = load_network(session, MODEL_PATH)

        # Predict the image
        for img_path, out_path in img_paths:
            modified_image = predict(prediction_func, img_path)
            Image.fromarray(modified_image).save(out_path)

    os._exit(0)


if __name__ == '__main__':
    main()
