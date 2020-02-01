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
"""Audio denoising methods"""
from typing import Union
import numpy as np
from tqdm import tqdm

from .restft import stft, istft
from .audio import Audio

Number = Union[int, float]


class iPCTV:
    """Instantaneous Phase Corrected Total Variation Denoising"""
    def __init__(self, audio: Audio):
        self._audio = audio

        self._wave = audio.wave.copy()

        operator = audio.e_pc * audio.e_ipc
        self._dual_spectrum = self.phi(self._wave, audio.n_fft,
                                       audio.hop_length, 'hann', operator)

        self._lambda = 1e-1
        self._max_iter = 100
        self._sigma_1 = .002
        self._sigma_2 = 1

        self._wave_history = []
        self._dual_history = []
        self._energy_history = []

    @property
    def audio(self) -> Audio:
        return self._audio

    @property
    def wave(self) -> np.ndarray:
        return self._wave

    @property
    def dual_spectrum(self) -> np.ndarray:
        return self._dual_spectrum

    @property
    def length(self) -> int:
        return self.wave.shape[-1]

    @property
    def hop_length(self) -> int:
        return self.audio.hop_length

    @property
    def e_pc(self) -> np.ndarray:
        return self.audio.e_pc

    @property
    def e_ipc(self) -> np.ndarray:
        return self.audio.e_ipc

    @property
    def operator(self) -> np.ndarray:
        return self.e_pc * self.e_ipc

    @property
    def lambda_(self) -> Number:
        return self._lambda

    @lambda_.setter
    def lambda_(self, l: Number) -> None:
        if isinstance(l, float) or isinstance(l, int):
            if l < 0:
                raise ValueError("lambda should be non-negative")
            self._lambda = l
        else:
            raise ValueError("lambda should be an instance of int or float")

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @max_iter.setter
    def max_iter(self, i: int) -> None:
        if not isinstance(i, int) or i < 0:
            raise ValueError("max_iter should >= 0 and should be integer")
        self._max_iter = i

    @property
    def sigma_1(self) -> float:
        return self._sigma_1

    @property
    def sigma_2(self) -> float:
        return self._sigma_2

    @property
    def wave_history(self) -> np.ndarray:
        return np.array(self._wave_history)

    @property
    def dual_history(self) -> np.ndarray:
        return np.array(self._dual_history)

    @property
    def energy_history(self) -> np.ndarray:
        return np.array(self._energy_history)

    @classmethod
    def phi(cls, wave: np.ndarray, n_fft: int, hop_length: int, window: str,
            operator: np.ndarray) -> np.ndarray:
        """Phi operator for instantaneous phase corrected total variation.

        Args:
            wave: 1-D real-valued wave.
            n_fft: Number of FFT computation.
            hop_length: Number of audio samples between adjacent STFT columns.
            window: Window function for STFT.
            operator: phase operator for the instantaneous phase corrected
                      total variation.

        Returns:
            Complex-valued 2-D time directional difference of spectrum.
        """
        spectrum = stft(wave,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        window=window)
        dual_spectrum = np.zeros(spectrum.shape)
        dual_spectrum[:, :-1] = np.diff(spectrum * operator)
        return dual_spectrum

    @classmethod
    def phi_star(cls, dual_spectrum: np.ndarray, operator: np.ndarray,
                 hop_length: int, length: int) -> np.ndarray:
        """Phi star operator for instantaneous phase corrected total variation.

        Args:
            dual_spectrum: Complex-valued 2-D time directional difference of
                           spectrum.
            operator: phase operator for the instantaneous phase corrected
                      total variation.
            hop_length: Number of audio samples between adjacent STFT columns.
            length: Length of wave.

        Returns:
            1-D real-valued wave, which is inverse of `dual_spectrum`.
        """
        spectrum = np.zeros(dual_spectrum.shape)
        spectrum[:, 1:] = -np.diff(dual_spectrum)
        spectrum = spectrum / operator

        return istft(spectrum, hop_length, length=length)

    def _primal_phase(self) -> np.ndarray:
        """Primal part of the primal-dual splitting algorithm.

        Returns:
            Calculated wave data.
        """
        wave = self.phi_star(self.dual_spectrum, self.operator,
                             self.hop_length, self.length)

        return self.wave - self.sigma_1 * (self.wave - self.audio.wave + wave)

    def _dual_phase(self, wave_updated: np.ndarray) -> np.ndarray:
        """Dual part of the primal-dual splitting algorithm.

        Args:
            wave_updated: Updated wave data.

        Returns:
            Calculated spectrum data.
        """
        spectrum_diff = self.sigma_2 * self.phi(2 * wave_updated - self.wave)
        dual_spectrum = self.dual_spectrum + spectrum_diff

        # tao tilde operator
        dual_spectrum = np.where(
            np.abs(dual_spectrum) > self.lambda_,
            dual_spectrum * self.lambda_ / np.abs(dual_spectrum),
            dual_spectrum)

        return dual_spectrum

    def compute(self) -> None:
        """Compute denoised audio using the primal-dual splitting algorithm."""

        for _ in tqdm(range(self.max_iter)):
            wave = self._primal_phase()
            dual_spectrum = self._dual_phase(wave)

            # handling history
            self._wave_history.append(wave)
            self._dual_history.append(dual_spectrum)
            self._energy_history.append(np.sum(np.abs(dual_spectrum)))

            self._wave = wave
            self._dual_spectrum = dual_spectrum
