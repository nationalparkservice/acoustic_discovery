__author__ = 'Cameron Summers'

"""
Command line utility for running audio event detection for the National Park Service.
"""

import logging
from collections import defaultdict
import os

import argparse

import scipy.io.wavfile as wav
import numpy as np

from nps_acoustic_discovery.output import probs_to_pandas, probs_to_raven_detections
from nps_acoustic_discovery.feature import FeatureExtractor
from nps_acoustic_discovery.model import EventModel

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class AcousticDetector(object):
    """
    A class for handling detections with various models.
    """

    def __init__(self, model_paths, thresholds):
        """
        Args:
            model_paths (list): Which models to use for detection
            thresholds (list): Thresholds for the models, expected to match order
        """
        if len(thresholds) != len(model_paths):
            raise Exception('Expected same number of models and thresholds. '
                            'Instead got {} models and {} thresholds'.format(len(model_paths), len(thresholds)))

        self.models = dict()

        last_feature_config = None
        for i, model_path in enumerate(model_paths):
            model = EventModel(model_path)
            model.set_threshold(thresholds[i])
            self.models[model.model_id] = model

            # Excpect all models to have same feature config
            if last_feature_config is None:
                last_feature_config = model.fconfig
            else:
                if last_feature_config != model.fconfig:
                    raise Exception('Feature configurations for models must match.')

            self.fconfig = last_feature_config
        self.fextractor = FeatureExtractor(last_feature_config)

    def iter_feature_vector(self, audio_data, sample_rate):
        """
        Provide a feature vector for the models to process.

        Args:
            audio_data (ndarray): audio signal
            sample_rate (float): audio sample rate

        Returns:
            ndarray: features of feature windows
        """

        logging.debug('Processing features...')
        X = self.fextractor.process(audio_data, sample_rate)
        logging.debug('Input vector shape: {}'.format(X.shape))
        window_size_frames = int(
            self.fconfig['window_size_sec'] / self.fconfig['hop_size'])  # sec / (sec / frame) -> frame

        windows = []
        for i in range(X.shape[0]):
            start_frame = i
            end_frame = i + window_size_frames
            window_mean = np.mean(X[start_frame:end_frame, :], axis=0)
            window_std = np.std(X[start_frame:end_frame, :], axis=0)
            feature_vector = np.hstack((window_mean, window_std))
            windows.append(feature_vector)

        X_win = np.vstack(tuple(windows))
        return X_win

    def process(self, audio_filepath):
        """
        Get raw probabilities of events for the audio data.

        Args:
            audio_filepath (str): path to audio

        Returns:
            dict: model obj to detection probabilities
        """
        try:
            (sample_rate, sig) = wav.read(audio_filepath)
        except Exception as e:
            logging.error('Could not read wav file: {}'.format(audio_filepath))
            raise e

        model_probabilities = defaultdict(list)
        # for time_stamp, fvec in self.iter_feature_vector(sig, sample_rate):
        X_win = self.iter_feature_vector(sig, sample_rate)
        for model_id, model in self.models.items():
            feat = np.copy(X_win)
            prob = model.process(feat)
            model_probabilities[model].append(prob)

        for model, probs in model_probabilities.items():
            probs = np.concatenate(tuple(probs), axis=0)
            model_probabilities[model] = probs

        return model_probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Audio event detection for the National Park Service')

    parser.add_argument('audio_path',
                        help='Path to audio file on which to run the classifier')
    parser.add_argument('save_dir',
                        help='Directory in which to save the output.')
    parser.add_argument('-m', '--model_dir_path',
                        action='append',
                        required=True,
                        help='Path to model(s) directories for classification')
    parser.add_argument('-t', '--threshold',
                        type=float,
                        action='append',
                        required=True,
                        help='If outputing detections, the threshold for a positive detection')
    parser.add_argument('-o', '--output',
                        choices=['probs', 'detections'],
                        default='probs',
                        help='Type of output, probabilities or detections at a threshold')

    args = parser.parse_args()

    thresholds = args.threshold
    model_dir_paths = args.model_dir_path
    audio_path = args.audio_path
    save_dir = args.save_dir
    output_type = args.output

    detector = AcousticDetector(model_dir_paths, thresholds)

    model_prob_map = detector.process(audio_path)
    model_prob_df_map = probs_to_pandas(model_prob_map)

    logging.debug('Saving output...')

    audio_filename = os.path.basename(audio_path)
    audio_name = os.path.splitext(audio_filename)[0]
    if output_type == 'probs':
        for model, df in model_prob_df_map.items():
            df.to_csv(os.path.join(save_dir, '{}_{}_{}_probs_df.tsv'.format(model.event_code,
                                                                            model.model_id,
                                                                           os.path.basename(audio_name))),
                      sep='\t',
                      float_format='%0.4f',
                      index=False
                      )
    elif output_type == 'detections':
        model_raven_df_map = probs_to_raven_detections(model_prob_df_map)
        for model, raven_df in model_raven_df_map.items():
            if len(raven_df) == 0:
                logging.info(
                    'No detections at threshold {} for model id {} on code {}'.format(model.detection_threshold,
                                                                                      model.model_id,
                                                                                      model.event_code))
            else:
                header = ['Selection', 'Begin Time (s)', 'End Time (s)', 'Species']
                raven_df[header].to_csv(
                    os.path.join(save_dir, '{}_{}_{}_th{}_selection_table.txt'.format(model.event_code,
                                                                                      model.model_id,
                                                                                      os.path.basename(audio_name),
                                                                                      model.detection_threshold)),
                    sep='\t',
                    float_format='%.1f',
                    index=False
                )
