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

from nps_acoustic_discovery.output import probs_to_pandas, \
    save_detections_to_audio, save_detections_to_raven, save_probs_to_csv
from nps_acoustic_discovery.feature import FeatureExtractor
from nps_acoustic_discovery.model import EventModel

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# Training audio used 44100Hz sample rate
MODEL_SAMPLE_RATE = 44100


def minutes_to_bytes(num_minutes):
    """
    Get number of bytes in num_minutes of audio.

    Args:
        num_minutes (int): number of minutes

    Returns:
        int: number of bytes
    """
    # bytes / chunk => samples / seconds * seconds / minutes * bytes / sample * minutes / chunk
    return int(MODEL_SAMPLE_RATE * 60 * 2 * num_minutes)


def get_ffmpeg_decode_cmd(ffmpeg_path, audio_filepath, start_seconds=None, duration_seconds=None, ffmpeg_quiet=False):
    """
    Get a list of arguments to pass to ffmpeg for audio decoding.

    Args:
        ffmpeg_path (str): path to ffmpeg
        audio_filepath (str): path to audio file
        ffmpeg_quiet (bool): whether to suppress output

    Returns:
        list: list of parameters for subprocess command
    """

    # FFMPEG command to modify input audio to look like training audio.
    # Audio used for training is 16 bit-depth, 44.1k sample rate, and single channel.
    decode_command = [
        ffmpeg_path,
    ]

    # See documentation for this behavior https://trac.ffmpeg.org/wiki/Seeking
    # Do fast seek on input
    if start_seconds is not None:
        decode_command += ['-ss', str(start_seconds)]

    decode_command += ['-i', audio_filepath]

    # See documentation for this behavior https://trac.ffmpeg.org/wiki/Seeking
    # Timestamps are reset with input seeking -ss above
    if duration_seconds is not None:
        decode_command += ['-to', str(duration_seconds)]

    # Normalize audio to training audio params and pipe for feature processing
    decode_command += [
        '-f',
        's16le',  # raw signed 16-bit little endian
        '-ac',
        '1',  # force to mono if necessary
        '-ar',
        str(MODEL_SAMPLE_RATE),  # resample if necessary
        'pipe:1',
    ]

    if ffmpeg_quiet:
        decode_command += ["-loglevel", "panic"]

    return decode_command


def raw_ffmpeg_data_to_ndarray(raw_data):
    """
    Convert the raw byte data from ffmpeg decoding to numpy array.

    Args:
        raw_data (bytes): raw data from ffmpeg decoding

    Returns:
        np.ndarray: numpy array of ints
    """
    num_samples = len(raw_data) / 2  # 2 shorts per sample
    format_str = '%ih' % num_samples
    int_data = struct.unpack(format_str, raw_data)
    signal = np.array(int_data)

    return signal


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

    def iter_audio(self, audio_filepath, chunk_size, ffmpeg_quiet, chunk_idx_start=1, chunk_idx_end=None):
        """
        Read an input audio file for processing. Reads chunks in a stream because soundscape recordings
        can be quite large.

        Args:
            audio_filepath (str): path to the audio file
            chunk_size (int): size in bytes of chunk to process
            ffmpeg_quiet (bool): suppress ffmpeg output
            chunk_idx_start (int): chunk idx on which to start processing, starting at 1
            chunk_idx_end (int): chunk idx on which to end processing
        """
        try:
            decode_command = get_ffmpeg_decode_cmd(self.ffmpeg_path, audio_filepath, ffmpeg_quiet=ffmpeg_quiet)

            proc = subprocess.Popen(decode_command, stdout=subprocess.PIPE)

            chunk_idx = 1
            for raw_data in iter(lambda: proc.stdout.read(chunk_size), ''):

                # Don't process before starting chunk
                if chunk_idx < chunk_idx_start:
                    logging.debug('Skipping chunk {}'.format(chunk_idx))
                    chunk_idx += 1
                    continue

                # Don't process after ending chunk
                if chunk_idx_end is not None and chunk_idx > chunk_idx_end:
                    logging.debug('Reached end chunk {}'.format(chunk_idx_end))
                    break

                signal = raw_ffmpeg_data_to_ndarray(raw_data)

                if len(signal) > 0:
                    logging.debug('Processing chunk: {}. Audio len (s): {}'.format(chunk_idx,
                                                                    len(signal) / float(MODEL_SAMPLE_RATE)))

                yield signal, MODEL_SAMPLE_RATE
                chunk_idx += 1

        except Exception as e:
            logging.error('Could not read audio file: {}'.format(audio_filepath))
            raise e

    def process(self, audio_filepath, chunk_size_minutes=10, ffmpeg_quiet=True, chunk_idx_start=1, chunk_idx_end=None):
        """
        Get raw probabilities of events for the audio data.

        Args:
            audio_filepath (str): path to audio
            chunk_size_minutes (int): number of minutes to process at a time for large files
            ffmpeg_quiet (bool): whether to suppress ffmpeg output

        Returns:
            dict: model obj to detection probabilities
        """
        if chunk_size_minutes is None or 1 > chunk_size_minutes:
            raise ValueError('Chunk size of minutes {} invalid.'.format(chunk_size_minutes))

        chunk_size_bytes = minutes_to_bytes(chunk_size_minutes)

        model_probs_map = defaultdict(list)

        for signal, sample_rate in self.iter_audio(audio_filepath, chunk_size_bytes, ffmpeg_quiet, chunk_idx_start, chunk_idx_end):

            # Finished reading file
            if len(signal) == 0:
                break

            X_win = self.get_feature_vector(signal, sample_rate)
            for model_id, model in self.models.items():
                feat = np.copy(X_win)
                prob = model.process(feat)
                model_probs_map[model].append(prob)

        for model, probs in model_probs_map.items():
            probs = np.concatenate(tuple(probs), axis=0)
            model_probs_map[model] = probs

        return model_probs_map

    def process_sampled(self, audio_filepath, num_days, offset_minutes, step_minutes, duration_minutes, num_samples, ffmpeg_quiet=False):
        """
        Get raw probabilities of events for the audio data via a sampling scheme. The purpose is to allow
         faster processing of a large file while still extracting meaningful information. In the ornithological
         case for example, this could be used to sample a few minutes from each hour through a portion of a day.
         The ffmpeg fast input seek and output seek are used for locating positions within the file.

        IMPORTANT NOTE: The sampling process is intended to speed up processing large files where not
         all of the audio needs to be read and processed. Because of this, the ffmpeg fast seek is used
         to quickly locate the samples in the file. But beware that for mp3 files, the fast seek estimates
         frame locations based on bit rates so this location is an estimate and may be not exact. Trading
         some precision for speed here.

        Also NOTE: This function assumes the user knows there is enough audio to accomodate the provided sampling scheme.

        Example Scenario: Start sampling 20 minutes into file, get fifteen 5-minute samples every 30 minutes for 2 days.
        Params:

            num_days = 2
            offset_minutes = 20
            step_minutes = 30
            duration_minutes = 5
            num_samples = 15

        Args:
            audio_filepath (str): path to audio
            num_days (int): number of days to apply the scheme, should be 1 unless file is > 24hrs
            offset_minutes (int): minutes from the start of the file to start sampling scheme
            step_minutes (int): minutes between the start of each sample
            duration_minutes (int): minutes in each sample
            num_samples (int): number of samples to take from file
            ffmpeg_quiet (bool): whether to suppress ffmpeg output

        Returns:
            list: list days with a list of samples with maps of model obj to detection probabilities
        """
        # Basic argument validation
        for var_name, var in {'num_days': num_days, 'num_samples': num_samples, 'offset_minutes': offset_minutes,
                    'step_minutes': step_minutes, 'duration_minutes': duration_minutes}.items():
            if not isinstance(var, int) or var < 1:
                raise ValueError('Invalid {}. Must be an integer great than 0.'.format(var_name))

        offset_seconds = offset_minutes * 60  # minutes * seconds / minute
        sample_duration_seconds = duration_minutes * 60

        try:
            day_results = []

            for day_idx in range(num_days):

                day_offset_seconds = day_idx * 24 * 60 * 60  # hours / day * minutes / hour * seconds / minute

                samples_probs = []

                for sample_idx in range(num_samples):

                    # Get sample offset and start
                    sample_offset_seconds = 60 * step_minutes * sample_idx
                    sample_start_seconds = offset_seconds + day_offset_seconds + sample_offset_seconds

                    decode_command = get_ffmpeg_decode_cmd(self.ffmpeg_path,
                                                           audio_filepath,
                                                           start_seconds=sample_start_seconds,
                                                           duration_seconds=sample_duration_seconds,
                                                           ffmpeg_quiet=ffmpeg_quiet)

                    proc = subprocess.Popen(decode_command, stdout=subprocess.PIPE)

                    model_probs_map = defaultdict(list)

                    raw_data = proc.stdout.read()
                    signal = raw_ffmpeg_data_to_ndarray(raw_data)

                    signal_len_minutes = len(signal) / float(MODEL_SAMPLE_RATE) / 60.0

                    # Got significantly different amount (6 sec) of audio back than expected
                    if abs(signal_len_minutes - duration_minutes) > 0.1:
                        logging.warning("Expected {} minutes of audio but got {}.".format(duration_minutes,
                                                                                          signal_len_minutes))

                    # Got no audio back
                    if len(signal) == 0:
                        raise ValueError('Signal length 0 for sample {} on day {}. Double check your sampling scheme.'
                                         .format(sample_idx + 1, day_idx + 1))

                    # Create feature vector and run through models
                    X_win = self.get_feature_vector(signal, MODEL_SAMPLE_RATE)
                    for model_id, model in self.models.items():
                        feat = np.copy(X_win)
                        prob = model.process(feat)
                        model_probs_map[model].append(prob)

                    samples_probs.append(model_probs_map)

                    logging.debug('Day {}, processing sample: {}. Audio len (min): {}'.format(day_idx + 1, sample_idx + 1,
                                                                                            signal_len_minutes))

                if len(samples_probs) < num_samples:
                    logging.warning(
                        "Asked for {} samples but only get {} before end of file {}.".format(num_samples,
                                                                                            len(samples_probs)),
                                                                                            audio_filepath)
                day_results.append(samples_probs)

            if len(day_results) < num_days:
                logging.warning(
                    "Asked for {} days but only get {} before end of file {}.".format(num_days,
                                                                                         len(day_results)),
                    audio_filepath)

        except Exception as e:
            logging.error('Could not read audio file for sampling scheme: {}'.format(audio_filepath))
            raise e

        return day_results


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

    parser.add_argument('--ffmpeg_quiet',
                        action='store_true',
                        default=False,
                        help="Suppress ffmpeg output for detection processing")

    parser.add_argument('--chunk_size_minutes',
                        type=int,
                        default=10,
                        help='Number of minutes of audio to process at a time in large files')

    args = parser.parse_args()

    thresholds = args.threshold
    model_dir_paths = args.model_dir_path
    audio_path = args.audio_path
    save_dir = args.save_dir
    output_type = args.output
    ffmpeg_path = args.ffmpeg
    ffmpeg_quiet = args.ffmpeg_quiet
    chunk_size_minutes = args.chunk_size_minutes

    detector = AcousticDetector(model_dir_paths, thresholds, ffmpeg_path=ffmpeg_path)

    model_prob_map = detector.process(audio_path, chunk_size_minutes=chunk_size_minutes, ffmpeg_quiet=ffmpeg_quiet)
    model_prob_df_map = probs_to_pandas(model_prob_map)

    logging.debug('Saving output...')

    audio_filename = os.path.basename(audio_path)
    audio_basename, audio_ext = os.path.splitext(audio_filename)

    if output_type == 'probs':
        save_probs_to_csv(model_prob_df_map, audio_basename, save_dir)

    elif output_type == 'detections':
        save_detections_to_raven(model_prob_df_map, audio_basename, save_dir)

    elif output_type == 'audio':
        save_detections_to_audio(model_prob_df_map, audio_path, audio_basename, audio_ext, save_dir, ffmpeg_path)
