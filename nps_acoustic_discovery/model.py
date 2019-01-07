__author__ = "Cameron Summers"

import logging
import json
import os
import argparse

import keras
import numpy as np


class EventModel(object):
    """
    A class to encapsulate detection models
    """

    def __init__(self, model_path):
        """
        Args:
            model_path (str): path to model directory
        """
        try:
            event_config = json.load(open(os.path.join(model_path, 'config.json')))
        except Exception as e:
            logging.error('Could not load model config at: {}'.format(model_path))
            raise e

        try:
            model_params_path = os.path.join(model_path, 'model_params.h5')
            keras_model = keras.models.load_model(model_params_path)
        except Exception as e:
            logging.error('Could not load model params at: {}'.format(model_params_path))
            raise e

        try:
            scaler_mean_path = os.path.join(model_path, 'scaler_mean.npy')
            scaler_mean = np.load(scaler_mean_path)
        except Exception as e:
            logging.error('Could not load scaler means at: {}'.format(scaler_mean_path))
            raise e

        try:
            scaler_var_path = os.path.join(model_path, 'scaler_var.npy')
            scaler_var = np.load(scaler_var_path)
        except Exception as e:
            logging.error('Could not load scaler vars at: {}'.format(scaler_var_path))
            raise e

        self.event_type = event_config['event_type']
        self.event_code = event_config['code']
        self.fconfig = event_config['feature_config']
        self.model_id = event_config['model_id']
        self.keras_model = keras_model
        self.scaler_mean = scaler_mean
        self.scaler_std = np.sqrt(scaler_var)
        self.model_path = model_path
        self.detection_threshold = None

    def process(self, feature_vector):
        """
        Get the probability of this event for the feature vector.

        Args:
            feature_vector (ndarray): feature matrix (time x features)
        """
        feature_vector -= self.scaler_mean
        feature_vector /= self.scaler_std
        return self.keras_model.predict_proba(feature_vector, verbose=0)

    def set_threshold(self, threshold):
        """
        Set the threshold if model is used for detections.

        Args:
            threshold (float): threshold for a detection
        """
        if not 0.0 <= threshold <= 1.0:
            raise Exception('Threshold {} must be between 0.0 and 1.0')

        self.detection_threshold = threshold


def get_available_models(models_path='./models/'):
    """
    Return a dictionary of species code to model path for available models.

    Args:
        models_path (str): path to models

    Returns:
        dict: species code to model path
    """

    models_dict = dict()
    for model_dir in os.listdir(models_path):
        if os.path.isfile(os.path.join(models_path, model_dir, 'model_params.h5')):  # make sure it is a model dir
            models_dict[model_dir] = os.path.join(models_path, model_dir)
    return models_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Audio event detection for the National Park Service',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-s', '--show',
                        action='store_true',
                        help='Show the available models and paths')

    parser.add_argument('-d', '--model_dir',
                        action='append',
                        required=False,
                        help='path to model directory',
                        default='./models/')

    args = parser.parse_args()

    model_dir = args.model_dir

    if args.show:
        print("\nAvailable Models and Locations at {}".format(model_dir))
        for model, path in get_available_models(model_dir).items():
            print("\t{}: {}".format(model, path))
