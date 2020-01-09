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
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU': 0}))
    data_dirs = args.data_dirs
    output_representation = args.output_representation
    sample_rate = args.sample_rate
    batch_size = args.batch_size
    classes = get_classes(wanted_only=True)
    wanted_words = prepare_words_list(get_classes(wanted_only=True))
    with open("labels/conv_labels.txt", "w") as labels:
        for label in wanted_words:
            labels.write('%s\n' % label)
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

    # embed()
    checkpoint_path = 'checkpoints/spectrogram_model/' + \
                      datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    callbacks = [
        ConfusionMatrixCallback(
            val_gen,
            ap.set_size('validation') // batch_size,
            wanted_words=prepare_words_list(get_classes(wanted_only=True)),
            all_words=prepare_words_list(classes),
            label2int=ap.word_to_index),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_categorical_accuracy',
            mode='max',
            factor=0.5,
            patience=4,
            verbose=20,
            min_lr=1e-5),
        tf.keras.callbacks.TensorBoard(log_dir='logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            save_weights_only=True,
            monitor='val_categorical_accuracy',
            mode='max')
    ]
    
    print(model.summary())
    model.fit_generator(
        train_gen,
        steps_per_epoch=ap.set_size('training') // batch_size,
        epochs=10,
        verbose=1,
        callbacks=callbacks)

    eval_res = model.evaluate_generator(val_gen,
                                        ap.set_size('validation') // batch_size)

    print(eval_res)
