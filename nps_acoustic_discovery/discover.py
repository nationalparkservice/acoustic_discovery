__author__='Cameron Summers'

"""
This is the command line utility for running audio event detection for the National Park Service.
"""

import logging
from collections import defaultdict
import pdb

import argparse

import scipy.io.wavfile as wav
import numpy as np
import pandas as pd

from output import probs_to_pandas, probs_to_raven_detections
from feature import FeatureExtractor
from model import EventModel

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class AcousticDetector(object):
    """
    A class for handling detections with various models.
    """
    def __init__(self, model_paths, hop_size):
        """
        Args:
            model_paths (list): Which models to use for detection
            hop_size (float): The hop size for detections
        """
        self.hop_size = float(hop_size)
        self.models = dict()

        last_feature_config = None
        for model_path in model_paths:
            model = EventModel(model_path)
            self.models[model.model_id] = model

            # Excpect all models to have same feature config
            if last_feature_config is None:
                last_feature_config = model.fconfig
            else:
                if last_feature_config != model.fconfig:
                    raise Exception('Feature configurations for models must match.')

	self.fconfig = last_feature_config
        self.fextractor = FeatureExtractor(last_feature_config, hop_size)

    def iter_feature_vector(self, audio_data, sample_rate):
        """
        Provide a feature vector for the models to process.

        Args:
            audio_data (ndarray): audio signal
            sample_rate (float): audio sample rate

        Yields:
            float: the time in the audio for the feature vector
            ndarray: the feature vector
        """ 

	logging.debug('Processing features...')
	X = self.fextractor.process(audio_data, sample_rate)
	logging.debug('Input vector shape: {}'.format( X.shape))
	window_size_frames = int(self.fconfig['window_size_sec'] / self.hop_size)  # sec / (sec / frame) -> frame

	for i in range(X.shape[0]):
	    start_frame = i 
	    end_frame = i + window_size_frames
 	    window_mean = np.mean(X[start_frame:end_frame, :], axis=0)
 	    window_std = np.std(X[start_frame:end_frame, :], axis=0)
	    time_secs = i * self.hop_size
	    feature_vector = np.hstack((window_mean, window_std))
	    yield time_secs, feature_vector[np.newaxis, :]

    def process(self, audio_filepath):
        """
        Get raw probabilities of events for the audio data.

        Args:
            audio_filepath (str): path to audio

        Returns:
            dict: model obj to detection probabilities
        """
        #TODO Handle mp3 file inputs and decode

	try:
	    (sample_rate, sig) = wav.read(audio_filepath)
	except Exception as e:
	    logging.error('Could not read wav file: {}'.format(audio_filepath))
	    raise e

        model_probabilities = defaultdict(list)
        for time_stamp, fvec in self.iter_feature_vector(sig, sample_rate):
            for model_id, model in self.models.iteritems():
                prob = model.process(fvec)
                model_probabilities[model].append(prob)

	for model, probs in model_probabilities.iteritems():
	    probs = np.concatenate(tuple(probs), axis=0) 
	    model_probabilities[model] = probs

    	return model_probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Audio event detection for the National Park Service')

    parser.add_argument('audio_path',
			help='Path to audio file on which to run the classifier')
    parser.add_argument('save_dir',
			help='Directory in which to save the output.')
    parser.add_argument('--model_dir_path', 
			nargs='+',
			required=True,
			help='Path to model(s) directories for classification')
    parser.add_argument('--hop_size_sec', 
			type=float, 
			default=0.5,
			help='Size in seconds of classification window hop')
    parser.add_argument('--threshold',
			type=float,
			default=0.8,
			help='If outputing detections, the threshold for a positive detection')
    parser.add_argument('--output', 
			choices=['probs', 'detections'], 
			default='probs', 
			help='Type of output, probabilities or detections at a threshold')

    args = parser.parse_args()

    detector = AcousticDetector(args.model_dir_path, args.hop_size_sec)

    probabilities = detector.process(args.audio_path)

    logging.debug('Saving output...')

    if args.output == 'probs':
	probs_to_pandas(probabilities, args.save_dir)
    elif args.output == 'detections':
	probs_to_raven_detections(probabilities, args.threshold)
