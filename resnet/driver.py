from resnet_model import Resnet3DBuilder
from pathlib import Path
import SimpleITK as sitk
import pandas as pd
import numpy as np



N = 32
batch_size = 4

# # input_folder = "../data/resampled/"
input_folder = "../data/hecktor_nii/"
input_folder = Path(input_folder)
patient_list = [f.name.split("_")[0] for f in input_folder.rglob("*_ct*")]
patient_images = {}
for p in patient_list[:N]:
    img = sitk.ReadImage("../data/resampled/" + p + "_ct.nii.gz")
    print(img.GetWidth(), img.GetHeight(), img.GetDepth())
    patient_images[p] = sitk.GetArrayViewFromImage(img).reshape((144, 144, 144, 1)).copy()

patient_outcomes = pd.read_csv("../data/hecktor2021_patient_endpoint_training.csv")
patient_outcomes = dict(zip(patient_outcomes["PatientID"][:N], patient_outcomes["Progression"][:N]))


# input_shape = (224, 224, 224, 1)
input_shape = (144, 144, 144, 1)
num_final_nodes = 1
model = Resnet3DBuilder.build_resnet_small(input_shape, num_final_nodes)

from tensorflow.keras.optimizers import Adam
init_lr = 1e-4
model.compile(optimizer=Adam(learning_rate=init_lr), loss="binary_crossentropy", metrics=['accuracy'])


x_train = np.asarray([patient_images[k] for k in patient_images.keys()])
y_train = np.asarray([patient_outcomes[k] for k in patient_outcomes.keys()])
model.fit(x_train, y_train, batch_size=batch_size, verbose=2)

import IPython; IPython.embed()
import sys; sys.exit()

# for k in patient_images.keys():
#     # k = "CHGJ082"
#     print(k)
#     img = patient_images[k]
#     # img = np.random.standard_normal(size=(1,) + input_shape)
#     print(img.shape)
#     # img = img.reshape((1, 144, 144, 144, 1))
#     print(k, img.shape, model(img))
