__author__ = 'Cameron Summers'

"""
Utility functions
"""

import os
import datetime
import copy
import subprocess
import logging

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def probs_to_pandas(model_prob_map, start_datetime=None):
    """
    Output probabilities for models to pandas df. Optionally, can give this
    function a datetime that represents the true start of the detections. This
    is useful when you are processing multiple files in sequence and want
    to maintain their time relations.

    Args:
        model_prob_map (dict): model object to detection probabilities
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


def probs_to_raven_detections(model_prob_df_map, filter_probs=True):
    """
    Get detections at the model threshold and format to be Raven friendly.

    Args:
        model_prob_df_map (dict): maps the model object to the probabilities dataframe
        filter_probs (bool): whether to apply a low pass smoothing filter to probabilities before generating detections

    Returns:
        dict: Map of model object to dataframe that can be written as selection table files
    """
    model_raven_df_map = dict()
    for model, prob_df in model_prob_df_map.items():
        detection_window_size = model.fconfig['window_size_sec']

        signal = prob_df[model.event_code]
        if filter_probs:
            signal = lowpass_filter(prob_df[model.event_code])

        # Vectorized location of detection start times
        binarized_signal = copy.copy(signal)
        binarized_signal[signal < model.detection_threshold] = 0
        binarized_signal[signal > model.detection_threshold] = 1
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


def save_detections_to_audio(model_prob_df_map, audio_path, audio_basename, audio_ext, save_dir, ffmpeg_path):
    # Save detections at given threshold as individual corresponding audio files

    model_raven_df_map = probs_to_raven_detections(model_prob_df_map)

    for model, raven_df in model_raven_df_map.items():
        create_audio = input(
            'About to create {} audio files for {} detections. Are you sure? (y/N)'.format(len(raven_df),
                                                                                           model.event_code))

        if create_audio not in ['y', 'Y']:
            logging.info('Process aborted by user.')
            continue

        for idx, row in raven_df.iterrows():
            start_time = row['Begin Time (s)']
            end_time = start_time + model.fconfig['window_size_sec']

            out_filename = '{}_{}_s{:.1f}_e{:.1f}_m{}{}'.format(os.path.basename(audio_basename),
                                                                model.event_code,
                                                                start_time,
                                                                end_time,
                                                                model.model_id,
                                                                audio_ext
                                                                )

            outpath = os.path.join(save_dir, out_filename)

            # Save the detection's slice of audio
            ffmpeg_slice_cmd = [ffmpeg_path, '-i', audio_path,
                                '-ss', str(start_time), '-t', str(model.fconfig['window_size_sec']),
                                '-acodec', 'copy', outpath]
            subprocess.Popen(ffmpeg_slice_cmd)


def save_detections_to_raven(model_prob_df_map, audio_basename, save_dir):

    # Save detections at given threshold to Raven file

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
                os.path.join(save_dir, '{}_{}_{}_th{}_selection_table.txt'.format(os.path.basename(audio_basename),
                                                                                  model.event_code,
                                                                                  model.model_id,
                                                                                  model.detection_threshold)),
                sep='\t',
                float_format='%.1f',
                index=False
            )


def save_probs_to_csv(model_prob_df_map, audio_basename, save_dir):
    # Save raw probabilities to tsv file

    for model, df in model_prob_df_map.items():
        df.to_csv(os.path.join(save_dir, '{}_{}_{}_probs_df.tsv'.format(os.path.basename(audio_basename),
                                                                        model.event_code,
                                                                        model.model_id,
                                                                        )),
                  sep='\t',
                  float_format='%0.4f',
                  index=False
                  )


def lowpass_filter(signal):
    """
    Apply a lowpass filter to the probabilities.
    """
    b, a = butter(5, 0.1, analog=False)
    return lfilter(b, a, signal)
