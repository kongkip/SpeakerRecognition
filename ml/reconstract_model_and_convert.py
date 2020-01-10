import tensorflow as tf
import argparse
from model import speech_model

parser = argparse.ArgumentParser(description='set input arguments')

parser.add_argument(
    '-checkpoint_dir',
    action='store',
    type=str,
    default=16000,
    help='model weights checkpoint path')

args = parser.parse_args()

model = speech_model(16000,
                     num_classes=7)
# check the latest directory and pass it
checkpoint_path = args.checkpoint_dir + 'cp.ckpt'
model.load_weights(checkpoint_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tf_lite_model = converter.convert()
open("ml/models/tflite_model.tflite", 'wb').write(tf_lite_model)
