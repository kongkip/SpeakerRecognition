from __future__ import absolute_import, division, print_function, unicode_literals

import hashlib
import math
import os.path
import random
import re

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

from helpers import tf_roll

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185


def prepare_words_list(wanted_words):
    """Prepends common tokens to the custom word list."""
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to."""
    dir_name = os.path.basename(os.path.dirname(filename))
    if dir_name == 'unknown_unknown':
        return 'training'

    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)

    hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1))
                       * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


class AudioProcessor(object):
    """Handles loading, partitioning, and preparing audio training data."""

    def __init__(self,
                 data_dirs,
                 silence_percentage,
                 unknown_percentage,
                 wanted_words,
                 validation_percentage,
                 testing_percentage,
                 model_settings,
                 output_representation=False):
        desired_samples = model_settings['desired_samples']
        self.background_clamp_ = None
        self.background_volume_placeholder_ = tf.compat.v1.placeholder(
            tf.float32, [], name='background_volume')
        self.background_data_placeholder_ = tf.compat.v1.placeholder(
            tf.float32, [desired_samples, 1], name='background_data')
        self.time_shift_placeholder_ = tf.compat.v1.placeholder(tf.int32, name='timeshift')
        self.foreground_volume_placeholder_ = tf.compat.v1.placeholder(
            tf.float32, [], name='foreground_volme')
        self.wav_filename_placeholder_ = tf.compat.v1.placeholder(
            tf.string, [], name='filename')
        self.word_to_index = {}
        self.words_list = prepare_words_list(wanted_words)
        self.background_data = []
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        self.data_dirs = data_dirs
        assert output_representation in {'raw', 'spec', 'mfcc', 'mfcc_and_raw'}
        self.output_representation = output_representation
        self.model_settings = model_settings
        self.prepare_data_index(silence_percentage, unknown_percentage,
                                wanted_words, validation_percentage,
                                testing_percentage)
        self.prepare_background_data()
        self.prepare_processing_graph(model_settings)

    def prepare_data_index(self, silence_percentage, unknown_percentage,
                           wanted_words, validation_percentage,
                           testing_percentage):
        """Prepares a list of the samples organized by set and label"""
        random.seed(RANDOM_SEED)
        wanted_words_index = {}
        for index, wanted_word in enumerate(wanted_words):
            wanted_words_index[wanted_word] = index + 2
        unknown_index = {'validation': [], 'testing': [], 'training': []}
        all_words = {}
        # Look through all the subfolders to find audio samples
        for data_dir in self.data_dirs:
            search_path = os.path.join(data_dir, '*', '*.wav')
            for wav_path in tf.io.gfile.glob(search_path):
                word = re.search('.*/([^/]+)/.*.wav', wav_path).group(1).lower()
                # Treat the '_background_noise_' folder as a special case,
                # since we expect it to contain long audio samples we mix in
                # to improve training.
                if word == BACKGROUND_NOISE_DIR_NAME:
                    continue
                all_words[word] = True
                set_index = which_set(wav_path, validation_percentage,
                                      testing_percentage)
                # If it's a known class, store its detail, otherwise add it to the list
                # we'll use to train the unknown label.
                if word in wanted_words_index:
                    self.data_index[set_index].append({'label': word, 'file': wav_path})
                else:
                    unknown_index[set_index].append({'label': word, 'file': wav_path})
            if not all_words:
                raise Exception('No .wavs found at ' + search_path)
            for index, wanted_word in enumerate(wanted_words):
                if wanted_word not in all_words:
                    raise Exception('Expected to find ' + wanted_word +
                                    ' in labels but only found ' +
                                    ', '.join(all_words.keys()))
        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in ['validation', 'testing', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append({
                    'label': SILENCE_LABEL,
                    'file': silence_wav_path
                })
            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            # not really needed since the indices are chosen by random
            random.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def prepare_background_data(self):
        """Searches a folder for background noise audio, and loads it into memory"""
        background_dir = os.path.join(self.data_dirs[0], BACKGROUND_NOISE_DIR_NAME)
        if not os.path.exists(background_dir):
            return self.background_data
        with tf.compat.v1.Session(graph=tf.compat.v1.Graph()) as sess:
            wav_filename_placeholder = tf.compat.v1.placeholder(tf.compat.v1.string, [])
            wav_loader = tf.io.read_file(wav_filename_placeholder)
            wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
            search_path = os.path.join(self.data_dirs[0], BACKGROUND_NOISE_DIR_NAME,
                                       '*.wav')
            for wav_path in tf.io.gfile.glob(search_path):
                wav_data = sess.run(
                    wav_decoder, feed_dict={
                        wav_filename_placeholder: wav_path
                    }).audio.flatten()
                self.background_data.append(wav_data)
            if not self.background_data:
                raise Exception('No background wav files were found in ' + search_path)

    def prepare_processing_graph(self, model_settings):
        """Builds a TensorFlow graph to apply the input distortions"""
        desired_samples = model_settings['desired_samples']
        wav_loader = tf.io.read_file(self.wav_filename_placeholder_)
        wav_decoder = tf.audio.decode_wav(
            wav_loader, desired_channels=1, desired_samples=desired_samples)
        # Allow the audio sample's volume to be adjusted.
        scaled_foreground = tf.multiply(wav_decoder.audio,
                                        self.foreground_volume_placeholder_)
        # Shift the sample's start position, and pad any gaps with zeros.
        shifted_foreground = tf_roll(scaled_foreground,
                                     self.time_shift_placeholder_)
        # Mix in background noise.
        background_mul = tf.multiply(self.background_data_placeholder_,
                                     self.background_volume_placeholder_)
        background_add = tf.add(background_mul, shifted_foreground)
        # removed clipping: tf.clip_by_value(background_add, -1.0, 1.0)
        self.background_clamp_ = background_add
        self.background_clamp_ = tf.reshape(self.background_clamp_,
                                            (1, 1, model_settings['desired_samples']))

    def set_size(self, mode):
        """Calculates the number of samples in the dataset partition"""
        return len(self.data_index[mode])

    def get_data(self,
                 how_many,
                 offset,
                 background_frequency,
                 background_volume_range,
                 foreground_frequency,
                 foreground_volume_range,
                 time_shift_frequency,
                 time_shift_range,
                 mode,
                 sess,
                 flip_frequency=0.0,
                 silence_volume_range=0.0):
        """Gather samples from the data set, applying transformations as needed"""
        # Pick one of the partitions to choose samples from.
        model_settings = self.model_settings
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
        # Data and labels will be populated and returned.
        if self.output_representation == 'raw':
            data_dim = model_settings['desired_samples']
        elif self.output_representation == 'spec':
            data_dim = model_settings['spectrogram_length'] * model_settings[
                'spectrogram_frequencies']
        elif self.output_representation == 'mfcc':
            data_dim = model_settings['spectrogram_length'] * \
                       model_settings['num_log_mel_features']
        elif self.output_representation == 'mfcc_and_raw':
            data_dim = model_settings['spectrogram_length'] * \
                       model_settings['num_log_mel_features']
            raw_data = np.zeros((sample_count, model_settings['desired_samples']))

        data = np.zeros((sample_count, 1, data_dim))
        labels = np.zeros((sample_count, model_settings['label_count']))
        desired_samples = model_settings['desired_samples']
        use_background = self.background_data and (mode == 'training')
        pick_deterministically = (mode != 'training')
        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in xrange(offset, offset + sample_count):
            # Pick which audio sample to use.
            if how_many == -1 or pick_deterministically:
                sample_index = i
                sample = candidates[sample_index]
            else:
                sample_index = np.random.randint(len(candidates))
                sample = candidates[sample_index]

            # If we're time shifting, set up the offset for this sample.
            if np.random.uniform(0.0, 1.0) < time_shift_frequency:
                time_shift = np.random.randint(time_shift_range[0],
                                               time_shift_range[1] + 1)
            else:
                time_shift = 0
            input_dict = {
                self.wav_filename_placeholder_: sample['file'],
                self.time_shift_placeholder_: time_shift,
            }
            # Choose a section of background noise to mix in.
            if use_background:
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                background_offset = np.random.randint(
                    0,
                    len(background_samples) - model_settings['desired_samples'])
                background_clipped = background_samples[background_offset:(
                        background_offset + desired_samples)]
                background_reshaped = background_clipped.reshape([desired_samples, 1])
                if np.random.uniform(0, 1) < background_frequency:
                    background_volume = np.random.uniform(0, background_volume_range)
                else:
                    background_volume = 0.0
                    # silence class with all zeros is boring!
                    if sample['label'] == SILENCE_LABEL and \
                            np.random.uniform(0, 1) < 0.9:
                        background_volume = np.random.uniform(0, silence_volume_range)
            else:
                background_reshaped = np.zeros([desired_samples, 1])
                background_volume = 0.0
            input_dict[self.background_data_placeholder_] = background_reshaped
            input_dict[self.background_volume_placeholder_] = background_volume
            # If we want silence, mute out the main sample but leave the background.
            if sample['label'] == SILENCE_LABEL:
                input_dict[self.foreground_volume_placeholder_] = 0.0
            else:
                # Turn it up or down
                foreground_volume = 1.0
                if np.random.uniform(0, 1) < foreground_frequency:
                    foreground_volume = 1.0 + np.random.uniform(-foreground_volume_range,
                                                                foreground_volume_range)
                # flip sign
                if np.random.uniform(0, 1) < flip_frequency:
                    foreground_volume *= -1.0
                input_dict[self.foreground_volume_placeholder_] = foreground_volume

            # Run the graph to produce the output audio.
            if self.output_representation == 'raw':
                data[i - offset, :] = sess.run(
                    self.background_clamp_, feed_dict=input_dict).flatten()

            label_index = self.word_to_index[sample['label']]
            labels[i - offset, label_index] = 1

        if self.output_representation == 'raw':
            return data, labels

    def summary(self):
        """Prints a summary of classes and label distributions"""
        set_counts = {}
        print('There are %d classes.' % (len(self.word_to_index)))
        print("1%% <-> %d samples in 'training'" % int(
            self.set_size('training') / 100))
        for set_index in ['training', 'validation', 'testing']:
            counts = {k: 0 for k in sorted(self.word_to_index.keys())}
            num_total = self.set_size(set_index)
            for data_point in self.data_index[set_index]:
                counts[data_point['label']] += (1.0 / num_total) * 100.0
            set_counts[set_index] = counts

        print('%-13s%-6s%-6s%-6s' % ('', 'Train', 'Val', 'Test'))
        for label_name in sorted(
                self.word_to_index.keys(), key=self.word_to_index.get):
            line = '%02d %-12s: ' % (self.word_to_index[label_name], label_name)
            for set_index in ['training', 'validation', 'testing']:
                line += '%.1f%% ' % (set_counts[set_index][label_name])
            print(line)
