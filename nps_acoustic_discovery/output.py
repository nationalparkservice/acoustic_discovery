__author__ = 'Cameron Summers'

"""
Utility functions
"""

import os

import numpy as np
import pandas as pd


def probs_to_pandas(model_probabilities, save_dir, audio_path):
    """
    Output probabilities for models to pandas df.

    Args:
        model_probabilities (dict): model to detection probabilities
        save_dir (str): where to save the dataframe
    """
    audio_filename = os.path.basename(audio_path)
    audio_name = os.path.splitext(audio_filename)[0]
    for model, probs in model_probabilities.items():
        time = [float(t) * model.fconfig['hop_size'] for i, t in enumerate(range(len(probs)))]
        df = pd.DataFrame(np.column_stack([time, probs]), columns=["Time (s)"] + model.event_codes)
        print(model.model_id)
        df.to_pickle(os.path.join(save_dir, 'mid{}_{}_probs_df.pk'.format(model.model_id, os.path.basename(audio_name))))


def probs_to_raven_detections(model_probabilities, threshold):
    """
    Output probabilities for models to pandas df.
    """
    raise NotImplementedError()
