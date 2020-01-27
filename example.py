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
import numpy as np
import matplotlib.pyplot as plt

from wipctv import restft

f0 = 440  # Compute the STFT of a 440 Hz sinusoid
fs = 16000  # sampled at 8 kHz (sample rate)
T = 5  # lasting 5 seconds
framesz = 0.050  # with a frame size of 50 milliseconds
hop = 0.025  # and hop size of 25 milliseconds.

t = np.linspace(0, T, T * fs, endpoint=False)
x = np.sin(2 * np.pi * f0 * t + np.pi) * 1
f, t, X = restft.stft(x, fs, framesz, hop)

plt.pcolormesh(t, f, np.abs(X.T), cmap='binary')
plt.ylim(200, 680)
plt.show()

rx = restft.istft(X, fs, T, hop)
plt.plot(t[:100], rx[:100])
plt.show()
