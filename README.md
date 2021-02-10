# Weighted instantaneous Phase Corrected Total Variation

Weighted instantaneous phase corrected total variation denoising in application
to audio denoising.

This is the 2020 winter project for exploring an adaptive method to audio
denoising issue based on the
[iPCTV](http://contents.acoust.ias.sci.waseda.ac.jp/publications/IEEE/2018/icassp-yatabe2-2018apr.pdf)
and
[YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet).

## References

1. I. Bayram and M. E. Kamasak, “[A simple prior for audio signals](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6457419), ” IEEE Trans. Audio, Speech, Language Process, vol. 21, no. 6, pp. 1190–1200, 2013.
2. K. Yatabe and Y. Oikawa, “[Phase corrected total variation for audio signals](http://contents.acoust.ias.sci.waseda.ac.jp/publications/IEEE/2018/icassp-yatabe2-2018apr.pdf),” ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings, vol. 2018-April, pp. 656–660, 2018.
3. L. Condat, “[A primal–dual splitting method for convex optimization involving lipschitzian, proximable and linear composite terms](https://hal.archives-ouvertes.fr/hal-00609728v5/document),” Journal of Optimization Theory and Applications, vol. 158, 08 2013.
4. Manoj Plakal and Dan Ellis, “YAMNet.” https://github.com/tensorflow/models/tree/master/research/audioset/yamnet, 2019.
5. J. F. Gemmeke, D. P. W. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, “[Audio set: An ontology and human-labeled dataset for audio events](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45857.pdf),” in 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 776–780, March 2017.
6. A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, “[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf),” arXiv e-prints, p. arXiv:1704.04861, Apr 2017.

## Development Guide

> ### Limitation
> This project only supports **Python 3**. Any version of Python 2 may not work!

You need the following packages:

* `numpy>=1.15.0`
* `librosa>=0.7.2`
* `SoundFile>=0.10`
* `tensorflow>=2.0.0` (Probably `>=1.15.0` may also work. See [this commit](https://github.com/tensorflow/models/commit/831281cedfc8a4a0ad7c0c37173963fafb99da37) for details)
* `tqdm>=4.41.1`
* `resampy>=0.2.2`
* `pypesq>=1.2.4`

They are also available in `requirements.txt`. So you can simply use `pip`
command to install them.

## Example

This is a sample code using iPCTV and WiPCTV with score and spectrum representation.

```python
import numpy as np
import matplotlib.pyplot as plt
# from terminalplot import plot
import soundfile as sf

import wipctv.audio as audio_module
import wipctv.denoise as denoise_module

audio_ref = audio_module.Audio.read_wavfile('./test.wav')

# add pseudo noise ~ N(0, .01**2)
wave = audio_ref.wave + np.random.normal(0, .02, audio_ref.wave.shape[0])

audio = audio_module.Audio.read_wave(wave)
sf.write('test_n.wav', audio.wave, audio.sr)

plt.figure()
audio.specshow()
plt.ylim(0, 10000)
plt.savefig('test_n.png')
plt.close()

print('Noised PESQ score: {}'.format(audio.pesq_score(audio_ref)))
print('Noised SNR  score: {}'.format(audio.snr(audio_ref)))

print('iPCTV')

denoise = denoise_module.iPCTV(audio)
denoise.max_iter = 4000

denoise.compute()

# plot(list(range(denoise.max_iter)),
#      list(denoise.energy_history),
#      rows=20,
#      columns=50)

denoised_audio = denoise.export_audio()

print('iPCTV PESQ score: {}'.format(denoised_audio.pesq_score(audio_ref)))
print('iPCTV SNR  score: {}'.format(denoised_audio.snr(audio_ref)))

sf.write('test_ipctv.wav', denoise.wave, audio.sr)

plt.figure()
denoised_audio.specshow()
plt.ylim(0, 10000)
plt.savefig('test_ipctv.png')
plt.close()

# calculate weight of audio
weight = np.sum(np.abs(audio.spectrum), axis=1)
weight /= np.max(weight) / .5
weight = 1 - weight

w_denoise = denoise_module.WiPCTV(audio)
w_denoise.weight = weight
w_denoise.max_iter = 4000

sf.write('test_ipctv.wav', audio.wave, audio.sr)

w_denoise.compute()

# plot(list(range(w_denoise.max_iter)),
#      list(w_denoise.energy_history),
#      rows=20,
#      columns=50)

w_denoised_audio = w_denoise.export_audio()

print('WiPCTV PESQ score: {}'.format(w_denoised_audio.pesq_score(audio_ref)))
print('WiPCTV SNR  score: {}'.format(w_denoised_audio.snr(audio_ref)))

sf.write('test_wipctv.wav', w_denoise.wave, audio.sr)

plt.figure()
w_denoised_audio.specshow()
plt.ylim(0, 10000)
plt.savefig('test_wipctv.png')
plt.close()
```

Sample output as follow:

```
Noised PESQ score: 2.4734928607940674
Noised SNR  score: 9.133245615520154
iPCTV
100%|█████████████████████████████████████████████████████| 4000/4000 [00:14<00:00, 277.60it/s]
iPCTV PESQ score: 2.6278393268585205
iPCTV SNR  score: 13.057250439988852
100%|█████████████████████████████████████████████████████| 4000/4000 [00:15<00:00, 257.98it/s]
WiPCTV PESQ score: 2.6453375816345215
WiPCTV SNR  score: 13.38203305912749
```
