__author__ = 'Cameron Summers'

"""
Utility functions
"""

import os
import datetime
import copy

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter


def probs_to_pandas(model_prob_map, start_datetime=None):
    """
    Output probabilities for models to pandas df.

    Args:
        model_probabilities (dict): model to detection probabilities
        start_datetime (datetime.datetime): absolute start time of audio
    """
    model_prob_df_map = dict()
    for model, probs in model_prob_map.items():
        # Time relative to the file start
        rel_time = [float(t) * model.fconfig['hop_size'] for i, t in enumerate(range(len(probs)))]
        df = pd.DataFrame(np.column_stack([rel_time, probs]), columns=["Relative Time (s)", model.event_code])

        # Create new column with absolute time if a start is provided
        if start_datetime is not None:
            abs_time = [start_datetime + datetime.timedelta(0, t) for t in rel_time]
            df['Absolute Time'] = pd.Series(abs_time)

        model_prob_df_map[model] = df

    return model_prob_df_map


def probs_to_raven_detections(model_prob_df_map):
    """
    Output probabilities for models to pandas df.

    Args:
        model_prob_df_map (dict): maps the model object to the probabilities dataframe
        threshold (float): the threshold for determining a detection

    Returns:
        dict: Map of model object to dataframe for the event code supported by the model
              that can be written as selection table files
    """
    model_raven_df_map = dict()
    for model, prob_df in model_prob_df_map.items():
        detection_window_size = model.fconfig['window_size_sec']

        filtered_signal = lowpass_filter(prob_df[model.event_code])

        # Vectorized location of detection start times
        binarized_signal = copy.copy(filtered_signal)
        binarized_signal[filtered_signal < model.detection_threshold] = 0
        binarized_signal[filtered_signal > model.detection_threshold] = 1
        rise_indices = np.where(np.diff(binarized_signal, axis=0) == 1)[0]

        # Compile detection start times into dataframe compatible with Raven
        detections = []
        detection_ctr = 1
        prev_rise_time = None
        for idx in rise_indices:
            rise_time = prob_df.iloc[idx]['Relative Time (s)']

            # Skip a rise if it's within the window
            if prev_rise_time is not None and (rise_time - prev_rise_time) < detection_window_size:
                continue

            detections.append({
                'Selection': detection_ctr,
                'Begin Time (s)': rise_time,
                'End Time (s)': rise_time + detection_window_size,
                'Species': model.event_code,
            })

            detection_ctr += 1
            prev_rise_time = rise_time

        detections_df = pd.DataFrame(detections)

        model_raven_df_map[model] = detections_df

    return model_raven_df_map


def lowpass_filter(signal):
    """
    Apply a lowpass filter to the probabilities.
    """
    b, a = butter(5, 0.1, analog=False)
    return lfilter(b, a, signal)
