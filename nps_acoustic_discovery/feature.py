__author__ = "Cameron Summers"

from python_speech_features import mfcc
import numpy as np


class FeatureExtractor(object):
    """
    Feature extraction on the audio.
    """

    def __init__(self, feature_config):
        self.fconfig = feature_config

    def process(self, audio_data, sample_rate):
        mfcc_feat = mfcc(audio_data, sample_rate,
                         winstep=self.fconfig['hop_size'],
                         numcep=self.fconfig['num_cepstral_coeffs'],
                         nfilt=self.fconfig['num_filters'],
                         nfft=self.fconfig['nfft'],
                         lowfreq=self.fconfig['low_freq'],
                         highfreq=self.fconfig['high_freq'])

        # deltas
        N = 2
        mfcc_delta = np.zeros(shape=mfcc_feat.shape)
        for t in range(mfcc_feat.shape[0]):
            try:
                numer = np.sum([n * (mfcc_feat[t + n, :] - mfcc_feat[t - n, :]) for n in range(1, N + 1)])
                denom = 2 * np.sum([n ** 2 for n in range(N)])
                mfcc_delta[t, :] = np.divide(numer, denom)
            except IndexError:
                mfcc_delta[t, :] = mfcc_feat[t, :]

        # delta deltas
        mfcc_delta_delta = np.zeros(shape=mfcc_delta.shape)
        for t in range(mfcc_delta.shape[0]):
            try:
                numer = np.sum([n * (mfcc_delta[t + n, :] - mfcc_delta[t - n, :]) for n in range(1, N + 1)])
                denom = 2 * np.sum([n ** 2 for n in range(N)])
                mfcc_delta_delta[t, :] = np.divide(numer, denom)
            except IndexError:
                mfcc_delta_delta[t, :] = mfcc_delta[t, :]

        return np.hstack((mfcc_feat, mfcc_delta, mfcc_delta_delta))

