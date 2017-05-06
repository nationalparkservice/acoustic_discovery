__author__="Cameron Summers"

import logging
import imp
import os

import keras
import numpy as np

class EventModel(object):
    """
    A class to encapsulate detection models
    """
    def __init__(self, model_path):

        try:
            event_config_mod = imp.load_source('config', os.path.join(model_path, 'config.py'))
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
            scaler_var_path = os.path.join(model_path, 'scaler_var.npy')
            scaler_mean = np.load(scaler_mean_path)
            scaler_var = np.load(scaler_var_path)
        except Exception as e:
            logging.error('Could not load scaler params at: {}'.format(scaler_params_path)) 
            raise e
                
        self.event_type = event_config_mod.event_type
        self.event_names = event_config_mod.names
        self.event_codes = event_config_mod.codes
        self.fconfig = event_config_mod.feature_config
        self.model_id = event_config_mod.model_id
        self.keras_model = keras_model
        self.scaler_mean = scaler_mean
        self.scaler_var = scaler_var

    def process(self, feature_vector):
	"""
	Get the probability of this event for the feature vector.
	"""
	feature_vector_sc = (feature_vector - self.scaler_mean) / self.scaler_var
	return self.keras_model.predict_proba(feature_vector_sc, batch_size=1, verbose=0)

