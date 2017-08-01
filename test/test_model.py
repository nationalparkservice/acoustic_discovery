__author__ = 'Cameron Summers'

import os
import unittest
import numpy as np

from nps_acoustic_discovery.output import probs_to_pandas, probs_to_raven_detections
from nps_acoustic_discovery.discover import AcousticDetector
from nps_acoustic_discovery.model import EventModel


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestSmoke(unittest.TestCase):

    def setUp(self):
        self.test_model_dir = os.path.join(THIS_DIR, 'test_model')
        self.test_input = np.ones((1, 84))
        self.test_audio_filepath = os.path.join(THIS_DIR, 'test30s.wav')

    def test1_model(self):
        model = EventModel(self.test_model_dir)
        model.process(self.test_input)

    def test2_detector(self):
        detector = AcousticDetector([self.test_model_dir], [0.5], ffmpeg_path='/opt/local/bin/ffmpeg')
        detector.process(self.test_audio_filepath)

    def test3_probs_df(self):
        detector = AcousticDetector([self.test_model_dir], [0.5], ffmpeg_path='/opt/local/bin/ffmpeg')
        model_prob_map = detector.process(self.test_audio_filepath)
        model_probs_df_map = probs_to_pandas(model_prob_map)

    def test4_probs_raven(self):
        detector = AcousticDetector([self.test_model_dir], [0.5], ffmpeg_path='/opt/local/bin/ffmpeg')
        model_prob_map = detector.process(self.test_audio_filepath)
        model_probs_df_map = probs_to_pandas(model_prob_map)
        model_raven_df_map = probs_to_raven_detections(model_probs_df_map)


class TestMP3(unittest.TestCase):

    def setUp(self):
        self.test_model_dir = os.path.join(THIS_DIR, 'test_model')
        self.test_mp3_audio_filepath = os.path.join(THIS_DIR, 'test30s.mp3')
        self.test_wav_audio_filepath = os.path.join(THIS_DIR, 'test30s.wav')

    def test1_smoke(self):
        mp3_detector = AcousticDetector([self.test_model_dir], [0.5], ffmpeg_path='/opt/local/bin/ffmpeg')

        mp3_model_prob_map = mp3_detector.process(self.test_mp3_audio_filepath)
        mp3_model_probs_df_map = probs_to_pandas(mp3_model_prob_map)
        mp3_model_raven_df_map = probs_to_raven_detections(mp3_model_probs_df_map)

        wav_detector = AcousticDetector([self.test_model_dir], [0.5], ffmpeg_path='/opt/local/bin/ffmpeg')
        wav_model_prob_map = wav_detector.process(self.test_wav_audio_filepath)
        wav_model_probs_df_map = probs_to_pandas(wav_model_prob_map)
        wav_model_raven_df_map = probs_to_raven_detections(wav_model_probs_df_map)

        for model, probs_df in mp3_model_probs_df_map.items():
            mp3_num_probs = len(probs_df)
            mp3_prob_sum = probs_df[model.event_code].sum()

        for model, probs_df in wav_model_probs_df_map.items():
            wav_num_probs = len(probs_df)
            wav_prob_sum = probs_df[model.event_code].sum()

        print(mp3_num_probs, wav_num_probs)
        print (mp3_prob_sum, wav_prob_sum)
        assert mp3_num_probs == wav_num_probs
        assert abs(mp3_prob_sum - wav_prob_sum) < 1.0


if __name__ == '__main__':
    unittest.main()



