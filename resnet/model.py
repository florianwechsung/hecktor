from keras import Model
from keras.layers import Conv3D, Dropout, MaxPooling3D, Activation, Dense, Flatten, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU


def generic_3dcnn(input_shape, num_final_nodes, downsize_filters_factor, initial_learning_rate):
    inputs = Input(input_shape)
    conv1 = Conv3D(int(32/downsize_filters_factor), (3, 3, 3),
                   activation='linear', padding='same')(inputs)
    Normalized1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=.01)(Normalized1)
    conv1 = Conv3D(int(32/downsize_filters_factor), (3, 3, 3),
                   activation='linear', padding='same')(act1)
    Normalized1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=.01)(Normalized1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(act1)

    conv2 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3),
                   activation='linear', padding='same')(pool1)
    Normalized2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=.01)(Normalized2)
    conv2 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3),
                   activation='linear', padding='same')(act2)
    Normalized2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=.01)(Normalized2)
    pool2 = MaxPooling3D(pool_size=(3, 3, 3))(act2)

    conv3 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3),
                   activation='linear', padding='same')(pool2)
    Normalized3 = BatchNormalization()(conv3)
    act3 = LeakyReLU(alpha=.01)(Normalized3)
    conv3 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3),
                   activation='linear', padding='same')(act3)
    Normalized3 = BatchNormalization()(conv3)
    act3 = LeakyReLU(alpha=.01)(Normalized3)
    pool3 = MaxPooling3D(pool_size=(4, 4, 4))(act3)

    flattened = Flatten()(pool3)
    FC1 = Dense(int(512/downsize_filters_factor),
                activation='linear')(flattened)
    FC1_act = LeakyReLU(alpha=0.001)(FC1)
    dropout1 = Dropout(0.4)(FC1_act)
    FC2 = Dense(int(128/downsize_filters_factor),
                activation='linear')(dropout1)
    FC2_act = LeakyReLU(alpha=.01)(FC2)
    dropout2 = Dropout(0.4)(FC2_act)
    FC3 = Dense(num_final_nodes)(dropout2)
    act = Activation('softmax')(FC3)
    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=initial_learning_rate),
                  loss="categorical_crossentropy", metrics=['accuracy'])

    return model
