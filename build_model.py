import tensorflow as tf 
from shufflenetv2 import ShufflenetV2
import sys
import numpy as np

def export_VGG16():
    model = tf.keras.applications.vgg16.VGG16()
    tf.saved_model.save(model, "models/savedModel/VGG16/")

def export_Inception():
    model = tf.keras.applications.inception_v3.InceptionV3()
    tf.saved_model.save(model, "models/savedModel/InceptionV3/")

def export_EfficientNetV2():
    model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0()
    tf.saved_model.save(model, "models/savedModel/EfficientNetV2/")

def export_Xception():
    model = tf.keras.applications.xception.Xception()
    tf.saved_model.save(model, "models/savedModel/Xception/")

def export_ShuffleNetV2():
    model = ShufflenetV2(1000, False)(224)
    tf.saved_model.save(model, "models/savedModel/ShuffleNetV2/")

def export_LinearModel():
    inputs = tf.keras.layers.Input(shape=(500,))
    outputs = tf.keras.layers.Dense(500)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    # model = tf.keras.experimental.LinearModel(
    #     units=500,
    #     activation=None,
    #     use_bias=False,
    #     kernel_initializer='random_normal',
    #     bias_initializer='zeros',
    #     kernel_regularizer=None,
    #     bias_regularizer=None,
    # )
    # inputs = tf.keras.layers.Input(shape=(500,))
    # outputs = tf.keras.layers.Dense(500)(inputs)
    # model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    tf.saved_model.save(model, "models/savedModel/LinearModel/")


export_LinearModel()
export_VGG16()
export_Inception()
export_EfficientNetV2()
export_Xception()
export_ShuffleNetV2()

