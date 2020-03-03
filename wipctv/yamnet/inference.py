# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This file is modified for module usage.
"""Inference demo for YAMNet."""
from __future__ import division, print_function

import os
import sys
import urllib

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf

from .. import params
from . import yamnet as yamnet_model
from ..audio import Audio

directory = os.path.dirname(os.path.abspath(__file__))


def predict(audio: Audio):

    graph = tf.Graph()
    with graph.as_default():
        yamnet = yamnet_model.yamnet_frames_model(params)
        if not os.path.exists('yamnet.h5'):
            url = 'https://storage.googleapis.com/audioset/yamnet.h5'
            urllib.request.urlretrieve(url, f'{directory}/yamnet.h5')
        yamnet.load_weights(f'{directory}/yamnet.h5')

    if not os.path.exists('yamnet_class_map.csv'):
        url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
        urllib.request.urlretrieve(url)
    yamnet_classes = yamnet_model.class_names(
        f'{directory}/yamnet_class_map.csv')

    # Decode the WAV file.
    wav_data, sr = audio.wave.astype(np.int16), audio.sr
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.SAMPLE_RATE:
        waveform = resampy.resample(waveform, sr, params.SAMPLE_RATE)

    # Predict YAMNet classes.
    # Second output is log-mel-spectrogram array (used for visualizations).
    # (steps=1 is a work around for Keras batching limitations.)
    with graph.as_default():
        scores, _ = yamnet.predict(np.reshape(waveform, [1, -1]), steps=1)
    # Scores is a matrix of (time_frames, num_classes) classifier scores.
    # Average them along time to get an overall classifier output for the clip.
    prediction = np.mean(scores, axis=0)
    # Report the highest-scoring classes and their scores.
    top5_i = np.argsort(prediction)[::-1][:5]

    return yamnet_classes[top5_i], prediction[top5_i]