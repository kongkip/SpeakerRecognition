import tensorflow as tf
from datetime import datetime
from model import speech_model

model = speech_model(16000,
                     num_classes=7)
model.save("models/untrained_model.h5")
# check the latest directory and pass it
checkpoint_path = 'checkpoints/spectrogram_model/20200108-144000/cp.ckpt'
model.load_weights(checkpoint_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tf_lite_model = converter.convert()
open("models/tflite_model.tflite", 'wb').write(tf_lite_model)
