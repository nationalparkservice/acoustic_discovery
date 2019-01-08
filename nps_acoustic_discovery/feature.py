__author__ = "Cameron Summers"

from python_speech_features import mfcc, fbank
import numpy as np


class FeatureExtractor(object):
    """
    Feature extraction on the audio.
    """

    def __init__(self, feature_config):
        self.fconfig = feature_config

    def process(self, audio_data, sample_rate):
        """
        Process the audio and return features.

        Args:
            audio_data (ndarray): incoming audio data
            sample_rate (int): audio sample rate

        Returns:
            ndarray: feature matrix (time x features)
        """
        if self.fconfig['num_cepstral_coeffs'] is not None:
            # For models that use only mfccs

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

            feats = np.hstack((mfcc_feat, mfcc_delta, mfcc_delta_delta))

        else:
            # For models that use only mel bands

            mel_feat, energy = fbank(audio_data, sample_rate,
                                     winstep=self.fconfig['hop_size'],
                                     nfilt=self.fconfig['num_filters'],
                                     nfft=self.fconfig['nfft'],
                                     lowfreq=self.fconfig['low_freq'],
                                     highfreq=self.fconfig['high_freq'],
                                     winfunc=lambda x: np.hamming(x))
            feats = np.log(mel_feat)

        return feats
