import os
import sys
import glob

import tables

from config import config
from generator import get_training_and_validation_generators
from model import unet_model_3d, vgg16_3d, unet_descending, generic_3dcnn
from resnet_model import Resnet3DBuilder
from training import load_old_model, train_model

# os.environ["CUDA_VISIBLE_DEVICES"]="2"


def run_experiment(experiment_name="", resume_training=False, model_type="unet_descending", data_parallelize=False, ngpu=1):

    # setup experiment dir for logging stuff
    experiment_dir = os.path.join(config["logdir"], experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # load file, determine configuration of final layers of model depending on label data type
    hdf5_file_opened = tables.open_file(config["hdf5_file"], "r")
    num_final_nodes = 2
    # if config["label_data_type"] == "categorical":
    #num_final_nodes = hdf5_file_opened.root.truth[0].shape[1]
    #final_activation = "softmax"
    #loss = "categorical_crossentropy"
    # elif config["label_data_type"] == "continuous":
    #num_final_nodes = 1
    #final_activation = False
    #loss = "mean_squared_error"

    # Get model
    if resume_training and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        if config["model_type"] == 'vgg16_3d':
            model = vgg16_3d(input_shape=config["input_shape"], num_final_nodes=num_final_nodes,
                             downsize_filters_factor=config["downsize_nb_filters_factor"],
                             initial_learning_rate=config["initial_learning_rate"])
        elif config["model_type"] == 'unet_3d':
            model = unet_model_3d(input_shape=config["input_shape"], num_final_nodes=num_final_nodes,
                                  downsize_filters_factor=config["downsize_nb_filters_factor"],
                                  initial_learning_rate=config["initial_learning_rate"])
        elif config["model_type"] == 'unet_descending':
            model = unet_descending(input_shape=config["input_shape"], num_final_nodes=num_final_nodes,
                                    downsize_filters_factor=config["downsize_nb_filters_factor"],
                                    initial_learning_rate=config["initial_learning_rate"])
        elif config["model_type"] == 'resnet_3d':
            model = Resnet3DBuilder.build_resnet_50(
                config["input_shape"], num_final_nodes)
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy', metrics=['accuracy'])
        elif config["model_type"] == 'generic_3dcnn':
            model = generic_3dcnn(input_shape=config["input_shape"], num_final_nodes=num_final_nodes,
                                  downsize_filters_factor=config["downsize_nb_filters_factor"],
                                  initial_learning_rate=config["initial_learning_rate"])

    model.summary()
    # get training and testing generators

    train_generator, validation_generator, nb_train_samples, nb_test_samples = get_training_and_validation_generators(
        hdf5_file_opened, batch_size=config["batch_size"], data_split=config[
            "validation_split"], resume_training=resume_training,
        validation_keys_file=config["validation_file"], training_keys_file=config["training_file"],
        augment=config["augment"], augment_flip=True, augment_distortion_factor=None)

    # run training
    train_model(model=model, model_file=config["model_file"], training_generator=train_generator,
                validation_generator=validation_generator, steps_per_epoch=nb_train_samples,
                validation_steps=nb_test_samples, initial_learning_rate=config[
                    "initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_epochs=config["decay_learning_rate_every_x_epochs"], n_epochs=config["n_epochs"],
                logdir=experiment_dir)
    hdf5_file_opened.close()


if __name__ == "__main__":
    run_experiment(
        experiment_name=config["experiment_name"],
        resume_training=config["resume_training"],
        model_type=config["model_type"],
        data_parallelize=config["multi_gpu"],
        ngpu=config["ngpu"])
