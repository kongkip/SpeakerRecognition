from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from spela.spectrogram import Spectrogram


def spectrogram_model(input_size=16000, num_classes=36):
    model = tf.keras.Sequential()
    model.add(Spectrogram(n_dft=512, n_hop=256,
                          input_shape=(1, input_size),
                          return_decibel_spectrogram=True, power_spectrogram=2.0,
                          trainable_kernel=False, name='static_stft'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    # model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes
                                    , activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=3e-4),
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.categorical_accuracy])
    return model


def mobile_net_model(input_size=16000, num_classes=36):
    mobile_net = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(256, 63, 1),
                                                                include_top=False, weights=None)
    model = tf.keras.Sequential()
    model.add(Spectrogram(n_dft=512, n_hop=256,
                          input_shape=(1, input_size),
                          return_decibel_spectrogram=True, power_spectrogram=2.0,
                          trainable_kernel=False, name='static_stft'))
    model.add(mobile_net)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes
                                    , activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=3e-4),
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.categorical_accuracy])
    return model


def speech_model(input_size, num_classes=36, *args, **kwargs):
    return spectrogram_model(input_size, num_classes)


def prepare_model_settings(label_count,
                           sample_rate,
                           clip_duration_ms,
                           output_representation='raw'):
    """Calculates common settings needed for all models."""
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    if output_representation == 'raw':
        fingerprint_size = desired_samples
    return {
        'desired_samples': desired_samples,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
    }
