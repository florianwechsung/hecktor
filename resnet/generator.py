import os
import threading
from random import shuffle, choice

import numpy as np
import keras.utils as utils

from utils.utils import pickle_dump, pickle_load
from augment import augment_data
from config import config


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def get_training_and_validation_generators(data_file, batch_size, training_keys_file, validation_keys_file,
                                           data_split=0.8, resume_training=False, augment=False,
                                           augment_flip=False, augment_distortion_factor=0.25):
    """
    Creates the training and validation generators that can be used when training the model.
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param augment: If True, training data will be distorted on the fly so as to avoid over-fitting.
    :param labels: List or tuple containing the ordered label values in the image files. The length of the list or tuple
    should be equal to the n_labels value.
    Example: (10, 25, 50)
    The data generator would then return binary truth arrays representing the labels 10, 25, and 30 in that order.
    :param data_file: hdf5 file to load the data from.
    :param batch_size: Size of the batches that the training generator will provide.
    :param n_labels: Number of binary labels.
    :param training_keys_file: Pickle file where the index locations of the training data will be stored.
    :param validation_keys_file: Pickle file where the index locations of the validation data will be stored.
    :param data_split: How the training and validation data will be split. 0 means all the data will be used for
    validation and none of it will be used for training. 1 means that all the data will be used for training and none
    will be used for validation. Default is 0.8 or 80%.
    :param resume_training: If set to False, previous files will be overwritten. The default mode is false, so that the
    training and validation splits won't be overwritten when rerunning model training.
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """
    training_list, validation_list = get_validation_split(data_file, data_split=data_split, resume_training=resume_training,
                                                          training_file=training_keys_file,
                                                          testing_file=validation_keys_file)
    if config["up_sample"]:
        # upsample minority classes in training_list to parity
        training_list = up_sampler(data_file, training_list)
        # validation_list = up_sampler(data_file, validation_list)

    training_generator = data_generator(data_file, training_list, batch_size=batch_size, augment=augment,
                                        augment_flip=augment_flip, augment_distortion_factor=augment_distortion_factor)
    validation_generator = data_generator(
        data_file, validation_list, batch_size=batch_size)
    # Set the number of training and testing samples per epoch correctly
    num_training_steps = len(training_list)//batch_size
    num_validation_steps = len(validation_list)//batch_size
    return training_generator, validation_generator, num_training_steps, num_validation_steps


def get_validation_split(data_file, training_file, testing_file, data_split=0.8, resume_training=False):
    if not resume_training or not os.path.exists(training_file):
        print("Creating validation split...")

        #nb_samples = data_file.root.data.shape[0]
        #sample_list = list(range(nb_samples))
        #training_list, testing_list = split_list(sample_list, split=data_split)

        # if label_type == 'binary':
        positive_indices = [x for x in range(
            len(data_file.root.data)) if data_file.root.truth[x][0][1] == 1]
        negative_indices = [x for x in range(
            len(data_file.root.data)) if data_file.root.truth[x][0][1] == 0]
        #positive_indices = [x for x in range(len(data_file.root.truth)) if int(data_file.root.truth[x])==1]
        #negative_indices = [y for y in range(len(data_file.root.truth)) if int(data_file.root.truth[y])==0]
        shuffle(positive_indices)
        shuffle(negative_indices)
        training_positive, test_positive = split_list(
            positive_indices, split=data_split)
        training_negative, test_negative = split_list(
            negative_indices, split=data_split)
        training_list = training_positive + training_negative
        testing_list = test_positive + test_negative
        shuffle(training_list)
        shuffle(testing_list)

        pickle_dump(training_list, training_file)
        pickle_dump(testing_list, testing_file)
        return training_list, testing_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(testing_file)


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def up_sampler(data_file, training_list):
    num_classes = data_file.root.truth[0].shape[1]
    label_count = np.zeros(num_classes)
    label_array = np.empty((num_classes,), dtype=np.object_)
    label_array.fill([])
    for index in training_list:
        x, y = np.where(data_file.root.truth[index])
        label = int(y)
        label_count[label] += 1
        label_array[label] = np.append(label_array[label], index)

    majority_n = max(label_count)

    for i in range(num_classes):
        if label_count[i] == 0:
            continue
        resample_n = majority_n - label_count[i]
        index_list = label_array[i]
        while resample_n > 0:
            index = int(choice(index_list))
            training_list.append(index)
            resample_n = resample_n - 1
    return training_list


@threadsafe_generator
def data_generator(data_file, index_list, batch_size, augment=False, augment_flip=False, augment_distortion_factor=0.25):

    # if config["label_data_type"] == "categorical":
    #    truth_shape = data_file.root.truth[0].shape[1]
    #    y_list = list()
    #y_list = np.empty((0,truth_shape), int)
    # elif config["label_data_type"] == "continuous":
    #    y_list = list()

    while True:
        x_list = list()
        y_list = list()

        if config["lecun_sampler"]:
            index_list = pair_shuffle(index_list)
        elif config["binary"]:
            index_list = pair_shuffle(index_list)
        else:
            shuffle(index_list)

        for index in index_list:
            add_data(x_list, y_list, data_file, index, augment=augment, augment_flip=augment_flip,
                     augment_distortion_factor=augment_distortion_factor)
            if len(x_list) == batch_size:
                yield convert_data(x_list, y_list)
                x_list = list()
                y_list = list()


def pair_shuffle(index_list):
    pair_index_list = [i for i in range(len(index_list)//2)]
    shuffle(pair_index_list)
    new_index_list = []
    for pair_index in pair_index_list:
        new_index_list.append(index_list[pair_index*2])
        new_index_list.append(index_list[pair_index*2 + 1])

    return new_index_list


def add_data(x_list, y_list, data_file, index, augment=False, augment_flip=False,
             augment_distortion_factor=0.25):
    """
    Adds data from the data file to the given lists of feature and target data
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :return:
    """

    data = data_file.root.data[index]
    truth = data_file.root.truth[index][0]

    if augment:
        data = augment_data(data, data_file.root.affine, flip=augment_flip,
                            scale_deviation=augment_distortion_factor)

    x_list.append(data)
    y_list.append(truth)

    # if config["label_data_type"] == "categorical":
    #    y_list.append(truth)
    #    #y_list = np.append(y_list, truth, axis=0)
    # elif config["label_data_type"] == "continuous":
    #    y_list.append(np.asscalar(truth))

    # return y_list


def convert_data(x_list, y_list):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    return x, y
