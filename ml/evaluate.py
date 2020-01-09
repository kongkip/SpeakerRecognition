"""
Running
    python train.py -data_dirs ../data/1600_Pcm/
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

import tensorflow as tf
from datetime import datetime
from callbacks import ConfusionMatrixCallback
from model import speech_model, prepare_model_settings
from generator import AudioProcessor, prepare_words_list
from classes import get_classes
from helpers import data_gen

parser = argparse.ArgumentParser(description='set input arguments')

parser.add_argument(
    '-sample_rate',
    action='store',
    dest='sample_rate',
    type=int,
    default=16000,
    help='Sample rate of audio')
parser.add_argument(
    '-batch_size',
    action='store',
    dest='batch_size',
    type=int,
    default=32,
    help='Size of the training batch')
parser.add_argument(
    '-output_representation',
    action='store',
    dest='output_representation',
    type=str,
    default='raw',
    help='raw, spec, mfcc or mfcc_and_raw')
parser.add_argument(
    '-data_dirs',
    '--list',
    dest='data_dirs',
    nargs='+',
    required=True,
    help='<Required> The list of data directories. e.g., data/train')

args = parser.parse_args()
parser.print_help()
print('input args: ', args)

if __name__ == '__main__':
    # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU': 0}))
    data_dirs = args.data_dirs
    output_representation = args.output_representation
    sample_rate = args.sample_rate
    batch_size = args.batch_size
    classes = get_classes(wanted_only=True)
    model_settings = prepare_model_settings(
        label_count=len(prepare_words_list(classes)),
        sample_rate=sample_rate,
        clip_duration_ms=1000,
        output_representation=output_representation)

    print(model_settings)

    ap = AudioProcessor(
        data_dirs=data_dirs,
        wanted_words=classes,
        silence_percentage=13.0,
        unknown_percentage=60.0,
        validation_percentage=10.0,
        testing_percentage=0.0,
        model_settings=model_settings,
        output_representation=output_representation)
    train_gen = data_gen(ap, sess, batch_size=batch_size, mode='training')
    data = next(train_gen)
    print(data[0].shape)
    val_gen = data_gen(ap, sess, batch_size=batch_size, mode='validation')

    model = speech_model(model_settings['desired_samples'],
                         num_classes=model_settings['label_count'],
                         **model_settings)

    checkpoint_path = 'checkpoints/spectrogram_model/20200108-144000/cp.ckpt'
    model.load_weights(checkpoint_path)
    print(model.summary())

    eval_res = model.evaluate_generator(val_gen,
                                        ap.set_size('validation') // batch_size)

    print(eval_res)
