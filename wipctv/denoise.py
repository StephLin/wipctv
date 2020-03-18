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
import scipy as sp
from tqdm import tqdm

from .restft import stft, istft
from .audio import Audio

Number = Union[int, float]


class iPCTV:
    """Instantaneous Phase Corrected Total Variation Denoising"""
    def __init__(self, audio: Audio):
        self._audio = audio

        self._wave = audio.wave.copy()
        self._window = 'hann'
        self._dual_spectrum = None

        self._lambda = 1e-1
        self._max_iter = 100
        self._sigma_2 = 1e-3

        fourier_norm = np.linalg.norm(sp.linalg.dft(audio.n_fft))**2
        self._sigma_1 = 1 / (.5 + self._sigma_2 * 16 * fourier_norm)

        self._wave_history = []
        self._dual_history = []
        self._energy_history = []

    @property
    def audio(self) -> Audio:
        return self._audio

    @property
    def window(self) -> str:
        return self._window

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
    def n_fft(self) -> int:
        return self.audio.n_fft

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

    @sigma_1.setter
    def sigma_1(self, s: float) -> None:
        if not isinstance(s, float):
            raise ValueError("sigma_1 should be a float value")
        if s <= 0:
            raise ValueError("sigma_1 should greater than 0")
        self._sigma_1 = s

    @sigma_2.setter
    def sigma_2(self, s: float) -> None:
        if not isinstance(s, float):
            raise ValueError("sigma_2 should be a float value")
        if s <= 0:
            raise ValueError("sigma_2 should greater than 0")
        self._sigma_2 = s

    @property
    def wave_history(self) -> np.ndarray:
        return np.array(self._wave_history)

    @property
    def dual_history(self) -> np.ndarray:
        return np.array(self._dual_history)

    @property
    def energy_history(self) -> np.ndarray:
        return np.array(self._energy_history)

    def phi(self, wave: np.ndarray) -> np.ndarray:
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
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        window=self.window)
        dual_spectrum = np.zeros(spectrum.shape).astype(np.complex)
        dual_spectrum[:, :-1] = np.diff(spectrum * self.operator, axis=1)
        dual_spectrum[:, -1] = dual_spectrum[:, -2]
        return dual_spectrum

    def phi_star(self, dual_spectrum: np.ndarray) -> np.ndarray:
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
        spectrum = np.zeros(dual_spectrum.shape).astype(np.complex)
        spectrum[:, 1:] = -1 * np.diff(dual_spectrum, axis=1)
        spectrum = spectrum / self.operator

        return istft(spectrum=spectrum,
                     hop_length=self.hop_length,
                     window=self.window,
                     length=self.length)

    def _primal_phase(self) -> np.ndarray:
        """Primal part of the primal-dual splitting algorithm.

        Returns:
            Calculated wave data.
        """
        dual_wave = self.phi_star(self.dual_spectrum)
        wave_diff = self.sigma_1 * (self.wave - self.audio.wave + dual_wave)

        return self.wave - wave_diff

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

    def compute(self, disable_wave_history=False) -> None:
        """Compute denoised audio using the primal-dual splitting algorithm."""

        self._dual_spectrum = self.phi(self.wave)

        for _ in tqdm(range(self.max_iter)):
            wave = self._primal_phase()
            dual_spectrum = self._dual_phase(wave)

            # handling history
            if not disable_wave_history:
                self._wave_history.append(wave)
                self._dual_history.append(dual_spectrum)
            self._energy_history.append(np.sum(np.abs(self.phi(wave))))

            self._wave = wave
            self._dual_spectrum = dual_spectrum


class WiPCTV(iPCTV):
    def __init__(self, audio: Audio):
        super(WiPCTV, self).__init__(audio)
        self._weight = np.ones(audio.n_fft)

    @property
    def weight(self) -> np.ndarray:
        return self._weight

    @weight.setter
    def weight(self, w: np.ndarray) -> None:
        w = np.array(w).flatten().astype(np.float)
        if w.shape[0] != self.audio.fft_frequencies.shape[0]:
            raise ValueError("Invalid shape of weight")
        self._weight = w

    def phi(self, wave: np.ndarray) -> np.ndarray:
        dual_spectrum = super().phi(wave)
        return np.multiply(dual_spectrum.T, self.weight).T

    def phi_star(self, dual_spectrum: np.ndarray) -> np.ndarray:
        weight = np.where(np.abs(self.weight) < 1e-10, 1, self.weight)
        dual_spectrum = np.multiply(dual_spectrum.T, 1 / weight).T
        return super().phi_star(dual_spectrum)

    def compute(self) -> None:
        """Compute denoised audio using the primal-dual splitting algorithm."""

        self._dual_spectrum = self.phi(self.wave)

        for _ in tqdm(range(self.max_iter)):
            wave = self._primal_phase()
            dual_spectrum = self._dual_phase(wave)

            # handling history
            self._wave_history.append(wave)
            self._dual_history.append(dual_spectrum)
            self._energy_history.append(np.sum(np.abs(super().phi(wave))))

            self._wave = wave
            self._dual_spectrum = dual_spectrum
