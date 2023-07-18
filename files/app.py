import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#Configure ImageDataGenerator Class
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = [.99,1.01],
                                   brightness_range = [0.8,1.2],
                                   data_format = "channels_last",
                                   fill_mode = "constant",
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

#Train and Test data
training_set = train_datagen.flow_from_directory('/content/train',
                                                 target_size = (180,180),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('/content/test',
                                                 target_size = (180,180),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')

#Feature Extractor
VGG19 = VGG19(input_shape= [180,180] + [3],weights = 'imagenet',include_top=False)

for layer in VGG19.layers:
    layer.trainable = False

x = Flatten()(VGG19.output)

prediction = Dense(5, activation = 'softmax')(x)

model = Model(inputs = VGG19.input, outputs=prediction)
model.summary