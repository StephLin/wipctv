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
from typing import Union
import numpy as np
import librosa


def stft(x: np.ndarray,
         n_fft: int,
         hop_length: int,
         window: Union[str, np.ndarray] = 'hann',
         center: bool = False) -> np.ndarray:
    """Short-Time Fourier Transform with prior arguments.

    Args:
        x: 1-D signal.
        n_fft: Number of FFT computation.
        hop_length: Number of audio samples between adjacent STFT columns.
        window: Window function for STFT. Default to `'hann'`.
        center: Argument of `librosa.core.stft`. Default to `False`.

    Returns:
        Complex-valued 2-D spectrum.
    """
    return librosa.core.stft(x,
                             n_fft=n_fft,
                             hop_length=hop_length,
                             window=window,
                             center=center)


def istft(spectrum: np.ndarray,
          hop_length: int,
          window: Union[str, np.ndarray] = 'hann',
          center: bool = False,
          length: int = None) -> np.ndarray:
    """Inverse Short-Time Fourier Transform with prior arguments.

    Args:
        spectrum: Complex-valued 2-D spectrum.
        hop_length: Number of audio samples between adjacent STFT columns.
        window: Window function for STFT. Default to `'hann'`.
        center: Argument of `librosa.core.stft`. Default to `False`.
        length: Length of 1-D wave.

    Returns:
        Real-valued 1-D wave.
    """
    return librosa.core.istft(spectrum,
                              hop_length=hop_length,
                              window=window,
                              center=center,
                              length=length)


def phase_corrected_operator(spectrum_shape: np.ndarray, sr: int,
                             hop_length: int,
                             frequency_shift: float) -> np.ndarray:
    """Phase corrected operator $E_{PC}$ for iPCTV

    Args:
        spectrum_shape: Shape of the target spectrum.
        sr: Sampling rate of the target signal.
        hop_length: Number of audio samples between adjacent STFT columns.
        frequency_shift: Unit of frequency shift, which is based on FFT.

    Returns:
        Complex-valued 2-D phase corrected opreator $E_{PC}$.
    """

    freq_idx, time_idx = [np.arange(n_idx) for n_idx in spectrum_shape]
    exp_term = np.kron(freq_idx, time_idx) * hop_length * frequency_shift / sr
    exp_term = -2j * np.pi * exp_term
    return np.exp(exp_term).reshape(spectrum_shape)


def instantaneous_phase_operator(x: np.ndarray, spectrum: np.ndarray, sr: int,
                                 n_fft, hop_length: int,
                                 window: Union[str, np.ndarray],
                                 frequency_shift: float):
    """Instantaneous phase corrected operator $E_{iPC}$ for Hann-based iPCTV.

    Args:
        x: 1-D signal.
        spectrum: STFT-based spectrum of 1-D signal `x`.
        sr: Sampling rate of the target signal.
        n_fft: Number of FFT computation.
        hop_length: Number of audio samples between adjacent STFT columns.
        window: Window function for STFT. Default to `'hann'`.
        frequency_shift: Unit of frequency shift, which is based on FFT.

    Returns:
        Complex-valued 2-D instantaneous phase corrected opreator $E_{iPC}$.

    Raises:
        ValueError: If shape of `spectrum` does not fit the shape under
                    arguments received.
        NotImplementedError: If window function does not supported.
    """

    if window != 'hann':
        # TODO: General approach for various window function
        raise NotImplementedError("Unsupported window function %r" % window)
    dt_window_x = np.pi * np.arange(n_fft) / (n_fft - 1)
    dt_window = np.sin(2 * dt_window_x) / (2 * n_fft)

    delta_complex = stft(x,
                         n_fft=n_fft,
                         hop_length=hop_length,
                         window=dt_window)

    if delta_complex.shape != spectrum.shape:
        raise ValueError("Mismatch of spectrum shapes %r and %r" %
                         (spectrum.shape, delta_complex.shape))

    delta_complex = np.where(
        np.abs(spectrum) > 1e-10, delta_complex / spectrum, 0)
    delta = -np.imag(delta_complex)

    time_idx = np.arange(delta.shape[1])
    delta_tilde = np.zeros(delta.shape)
    for idx in time_idx[1:]:
        s = delta_tilde[:, idx - 1] + (delta[:, idx] + delta[:, idx - 1]) / 2
        delta_tilde[:, idx] = s

    exp_term = -2j * np.pi * hop_length * delta_tilde
    return np.exp(exp_term).reshape(spectrum.shape)
