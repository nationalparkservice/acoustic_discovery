__author__ = 'Cameron Summers'

import os
import unittest
import numpy as np

from nps_acoustic_discovery.output import probs_to_pandas, probs_to_raven_detections
from nps_acoustic_discovery.discover import AcousticDetector
from nps_acoustic_discovery.model import EventModel


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

FFMPEG_PATH = '/opt/local/bin/ffmpeg'


class TestSmoke(unittest.TestCase):

    def setUp(self):
        self.test_model_dir = os.path.join(THIS_DIR, '../models/SWTH')
        self.test_input = np.ones((1, 84))
        self.test_audio_filepath = os.path.join(THIS_DIR, 'SWTH_test_30s.wav')

    def test1_model(self):
        model = EventModel(self.test_model_dir)
        model.process(self.test_input)

    def test2_detector(self):
        detector = AcousticDetector([self.test_model_dir], [0.5], ffmpeg_path=FFMPEG_PATH)
        detector.process(self.test_audio_filepath)

    def test3_probs_df(self):
        detector = AcousticDetector([self.test_model_dir], [0.5], ffmpeg_path=FFMPEG_PATH)
        model_prob_map = detector.process(self.test_audio_filepath)
        model_probs_df_map = probs_to_pandas(model_prob_map)

    def test4_probs_raven(self):
        detector = AcousticDetector([self.test_model_dir], [0.5], ffmpeg_path=FFMPEG_PATH)
        model_prob_map = detector.process(self.test_audio_filepath)
        model_probs_df_map = probs_to_pandas(model_prob_map)
        model_raven_df_map = probs_to_raven_detections(model_probs_df_map)


class TestMP3(unittest.TestCase):

    def setUp(self):
        self.test_model_dir = os.path.join(THIS_DIR, '../models/SWTH')
        self.test_mp3_320k_audio_filepath = os.path.join(THIS_DIR, 'SWTH_test_30s_320k.mp3')
        self.test_mp3_60k_audio_filepath = os.path.join(THIS_DIR, 'SWTH_test_30s_60k.mp3')
        self.test_wav_audio_filepath = os.path.join(THIS_DIR, 'SWTH_test_30s.wav')

    def test1_smoke(self):
        mp3_detector = AcousticDetector([self.test_model_dir], [0.5], ffmpeg_path=FFMPEG_PATH)

        mp3_320k_model_prob_map = mp3_detector.process(self.test_mp3_320k_audio_filepath)
        mp3_320k_model_probs_df_map = probs_to_pandas(mp3_320k_model_prob_map)
        mp3_320k_model_raven_df_map = probs_to_raven_detections(mp3_320k_model_probs_df_map)
        
        mp3_60k_model_prob_map = mp3_detector.process(self.test_mp3_60k_audio_filepath)
        mp3_60k_model_probs_df_map = probs_to_pandas(mp3_60k_model_prob_map)
        mp3_60k_model_raven_df_map = probs_to_raven_detections(mp3_60k_model_probs_df_map)

        wav_detector = AcousticDetector([self.test_model_dir], [0.5], ffmpeg_path=FFMPEG_PATH)
        wav_model_prob_map = wav_detector.process(self.test_wav_audio_filepath)
        wav_model_probs_df_map = probs_to_pandas(wav_model_prob_map)
        wav_model_raven_df_map = probs_to_raven_detections(wav_model_probs_df_map)

        for model, probs_320k_df in mp3_320k_model_probs_df_map.items():
            mp3_320k_probs = probs_320k_df[model.event_code]

        for model, probs_60k_df in mp3_60k_model_probs_df_map.items():
            mp3_60k_probs = probs_60k_df[model.event_code]

        for model, probs_df in wav_model_probs_df_map.items():
            wav_probs = probs_df[model.event_code]

        import matplotlib
        matplotlib.use('TkAgg')

        import matplotlib.pyplot as plt
        plt.plot(wav_probs, label='original_wav')
        plt.plot(mp3_320k_probs, label='mp3_320kps')
        plt.plot(mp3_60k_probs, label='mp3_60kps')
        plt.xlabel('Time (centisec)')
        plt.ylabel('Detection Probabilty')
        plt.title('Encoding Interference')
        plt.plot()
        plt.legend()
        plt.show()

        print(len(mp3_320k_probs), len(wav_probs))
        assert abs(len(mp3_320k_probs) - len(wav_probs)) <= 1

        num_samples = 1000  # check over ten seconds
        prob_diff_sum = np.sum(abs(mp3_320k_probs[:num_samples] - wav_probs[:num_samples]))
        print('Probability sum of difference over 10 seconds {}'.format(prob_diff_sum))
        acceptable_prob_error_per_sample = 0.05
        assert prob_diff_sum < acceptable_prob_error_per_sample * num_samples


if __name__ == '__main__':

    unittest.main()



