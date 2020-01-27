# Weighted instantaneous Phase Corrected Total Variation

Weighted instantaneous phase corrected total variation denoising in application
to audio denoising.

This is the 2020 winter project for exploring an adaptive method to audio
denoising issue based on the
[iPCTV](http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0000656.pdf)
and
[YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet).

## References

1. I. Bayram and M. E. Kamasak, “[A simple prior for audio signals](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6457419), ” IEEE Trans. Audio, Speech, Language Process, vol. 21, no. 6, pp. 1190–1200, 2013.
2. K. Yatabe and Y. Oikawa, “[Phase corrected total variation for audio signals](http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0000656.pdf),” ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings, vol. 2018-April, pp. 656–660, 2018.
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

They are also available in `requirements.txt`. So you can simply use `pip`
command to install them.
