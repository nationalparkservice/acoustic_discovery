__author__ = 'Cameron Summers'

"""
Command line utility for running audio event detection for the National Park Service.
"""

import logging
from collections import defaultdict
import os
import subprocess
import struct

import argparse

import numpy as np

from nps_acoustic_discovery.output import probs_to_pandas, probs_to_raven_detections
from nps_acoustic_discovery.feature import FeatureExtractor
from nps_acoustic_discovery.model import EventModel

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# Training audio used 44100Hz sample rate
MODEL_SAMPLE_RATE = 44100


class AcousticDetector(object):
    """
    A class for handling detections with various models.
    """

    def __init__(self, model_paths, thresholds, ffmpeg_path=None):
        """
        Args:
            model_paths (list): Which models to use for detection
            thresholds (list): Thresholds for the models, expected to match order
            ffmpeg_path (str): Path to ffmpeg executable
        """
        if len(thresholds) != len(model_paths):
            raise Exception('Expected same number of models and thresholds. '
                            'Instead got {} models and {} thresholds'.format(len(model_paths), len(thresholds)))

        self.ffmpeg_path = ffmpeg_path
        self.models = dict()

        # Initialize the models
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

    def get_feature_vector(self, audio_data, sample_rate):
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

    def iter_audio(self, audio_filepath, chunk_size=None):
        """
        Read an input audio file for processing. Reads chunks in a stream because soundscape recordings
        can be quite large.

        Args:
            audio_filepath (str): path to the audio file
            chunk_size (int): size in bytes of chunk to process
        """
        try:

            if chunk_size is None:
                chunk_size = int(MODEL_SAMPLE_RATE * 60 * 2 * 10)  # 10 min audio

            # FFMPEG command to modify input audio to look like training audio.
            # Audio used for training is 16 bit-depth, 44.1k sample rate, and single channel.
            decode_command = [
                self.ffmpeg_path,
                '-i',
                audio_filepath,
                '-f',
                's16le', # raw signed 16-bit little endian
                '-ac',
                '1', # force to mono if necessary
                '-ar',
                str(MODEL_SAMPLE_RATE),  # resample if necessary
                'pipe:1',
            ]
            proc = subprocess.Popen(decode_command, stdout=subprocess.PIPE)

            chunk_idx = 1
            for raw_data in iter(lambda: proc.stdout.read(chunk_size), ''):

                num_samples = len(raw_data) / 2  # 2 shorts per sample
                format_str = '%ih' % num_samples
                int_data = struct.unpack(format_str, raw_data)
                signal = np.array(int_data)

                if len(signal) > 0:
                    logging.debug('Processing chunk: {}. Audio len (s): {}'.format(chunk_idx,
                                                                                   len(signal) / float(
                                                                                       MODEL_SAMPLE_RATE)))

                yield signal, MODEL_SAMPLE_RATE
                chunk_idx += 1

        except Exception as e:
            logging.error('Could not read audio file: {}'.format(audio_filepath))
            raise e

    def process(self, audio_filepath):
        """
        Get raw probabilities of events for the audio data.

        Args:
            audio_filepath (str): path to audio

        Returns:
            dict: model obj to detection probabilities
        """
        model_probs_map = defaultdict(list)

        for sig, sample_rate in self.iter_audio(audio_filepath):

            # Finished reading file
            if len(sig) == 0:
                break

            X_win = self.get_feature_vector(sig, sample_rate)
            for model_id, model in self.models.items():
                feat = np.copy(X_win)
                prob = model.process(feat)
                model_probs_map[model].append(prob)

        for model, probs in model_probs_map.items():
            probs = np.concatenate(tuple(probs), axis=0)
            model_probs_map[model] = probs

        return model_probs_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Audio event detection for the National Park Service',
                                     formatter_class=argparse.RawTextHelpFormatter)

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
                        help='The threshold for a positive detection')

    output_help_text = """
    Type of output file:
         probs: Raw probabilities over time
         detections: Raven detections file
         audio: Audio slices for each detection
    """
    parser.add_argument('-o', '--output',
                        choices=['probs', 'detections', 'audio'],
                        default='probs',
                        help=output_help_text)

    parser.add_argument('--ffmpeg',
                        required=True,
                        help='Path to FFMPEG executable')

    args = parser.parse_args()

    thresholds = args.threshold
    model_dir_paths = args.model_dir_path
    audio_path = args.audio_path
    save_dir = args.save_dir
    output_type = args.output
    ffmpeg_path = args.ffmpeg

    detector = AcousticDetector(model_dir_paths, thresholds, ffmpeg_path=ffmpeg_path)

    model_prob_map = detector.process(audio_path)
    model_prob_df_map = probs_to_pandas(model_prob_map)

    logging.debug('Saving output...')

    audio_filename = os.path.basename(audio_path)
    audio_name = os.path.splitext(audio_filename)[0]
    audio_ext = os.path.splitext(audio_filename)[-1]

    if output_type == 'probs':
        # Save raw probabilities to tsv file

        for model, df in model_prob_df_map.items():
            df.to_csv(os.path.join(save_dir, '{}_{}_{}_probs_df.tsv'.format(os.path.basename(audio_name),
                                                                            model.event_code,
                                                                            model.model_id,
            )),
                      sep='\t',
                      float_format='%0.4f',
                      index=False
            )
    elif output_type == 'detections':
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
                    os.path.join(save_dir, '{}_{}_{}_th{}_selection_table.txt'.format(os.path.basename(audio_name),
                                                                                      model.event_code,
                                                                                      model.model_id,
                                                                                      model.detection_threshold)),
                    sep='\t',
                    float_format='%.1f',
                    index=False
                )
    elif output_type == 'audio':
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

                out_filename = '{}_{}_s{:.1f}_e{:.1f}_m{}{}'.format(os.path.basename(audio_name),
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

