#
# Copyright (c) 2019-2020 StephLin.
#
# This file is part of wipctv
# (see https://gitea.mcl.math.ncu.edu.tw/StephLin/wipctv).
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
"""Reassignment of Short-Time Fourier Transform"""
import numpy as np
import soundfile
import librosa

from .params import SAMPLE_RATE, STFT_WINDOW_SECONDS, STFT_HOP_SECONDS
from . import restft

STFT_WINDOW_SAMPLES = int(SAMPLE_RATE * STFT_WINDOW_SECONDS)
STFT_HOP_SAMPLES = int(SAMPLE_RATE * STFT_HOP_SECONDS)


class Audio:
    """Object for audio expression."""
    def __init__(self,
                 wave=None,
                 sr=SAMPLE_RATE,
                 n_fft=STFT_WINDOW_SAMPLES,
                 hop_length=STFT_HOP_SAMPLES):
        self._wave = wave
        self._sr = sr
        self._raw_wave = wave
        self._raw_sr = sr
        self._n_fft = n_fft
        self._hop_length = hop_length

        self._spectrum = None
        self._fft_frequencies = None
        self._frequency_shift = None

        # phase related operators
        self._e_pc = None
        self._e_ipc = None

    @property
    def x(self) -> np.ndarray:
        return self._wave

    @property
    def wave(self) -> np.ndarray:
        return self._wave

    @property
    def sr(self) -> int:
        return self._sr

    @property
    def raw_wave(self) -> np.ndarray:
        return self._raw_wave

    @property
    def raw_sr(self) -> int:
        return self._raw_sr

    @property
    def n_fft(self) -> int:
        return self._n_fft

    @property
    def hop_length(self) -> int:
        return self._hop_length

    @property
    def spectrum(self) -> np.ndarray:
        return self._spectrum

    @property
    def fft_frequencies(self) -> np.ndarray:
        return self._fft_frequencies

    @property
    def frequency_shift(self) -> float:
        return self._frequency_shift

    @property
    def e_pc(self) -> np.ndarray:
        return self._e_pc

    @property
    def e_ipc(self) -> np.ndarray:
        return self._e_ipc

    @classmethod
    def read_wavfile(cls,
                     wavfile: str,
                     sr: int = SAMPLE_RATE,
                     n_fft: int = STFT_WINDOW_SAMPLES,
                     hop_length: int = STFT_HOP_SAMPLES) -> 'Audio':
        raw_wave, raw_sr = soundfile.read(wavfile, dtype=np.int16)

        if len(raw_wave.shape) > 1:
            raw_wave = np.mean(raw_wave, axis=-1)

        if raw_sr != sr:
            wave = librosa.core.resample(raw_wave, raw_sr, sr)
        else:
            wave = raw_wave.copy()

        return cls.read_wave(wave, sr, n_fft, hop_length)

    @classmethod
    def read_wave(cls,
                  wave: np.ndarray,
                  sr: int = SAMPLE_RATE,
                  n_fft: int = STFT_WINDOW_SAMPLES,
                  hop_length: int = STFT_HOP_SAMPLES) -> 'Audio':
        """Read np.ndarray wave as an Audio instance.

        Args:
            wave: 1-D np.ndarray signal.
            sr: Sampling rate of the target signal.
            n_fft: Number of FFT computation.
            hop_length: Number of hop for STFT.

        Returns:
            An instance of `Audio` with spectrum-related properties.
        """
        audio = cls(wave=wave, sr=sr, n_fft=n_fft, hop_length=hop_length)
        audio._spectrum = restft.stft(wave, n_fft=n_fft, hop_length=hop_length)

        frequencies = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
        shift = frequencies[1] - frequencies[0]
        audio._fft_frequencies = frequencies
        audio._frequency_shift = shift

        shape = audio.spectrum.shape
        e_pc = restft.phase_corrected_operator(shape, sr, hop_length, shift)
        audio._e_pc = e_pc
        e_ipc = restft.instantaneous_phase_operator(wave, audio.spectrum, sr,
                                                    n_fft, hop_length, 'hann',
                                                    shift)
        audio._e_ipc = e_ipc

        return audio
