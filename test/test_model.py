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
        self.test_audio_filepath = os.path.join(THIS_DIR, 'test.wav')

    def test1_model(self):
        model = EventModel(self.test_model_dir)
        model.process(self.test_input)

    def test2_detector(self):
        detector = AcousticDetector([self.test_model_dir], [0.5])
        detector.process(self.test_audio_filepath)

    def test3_probs_df(self):
        detector = AcousticDetector([self.test_model_dir], [0.5])
        model_prob_map = detector.process(self.test_audio_filepath)
        model_probs_df_map = probs_to_pandas(model_prob_map)

    def test4_probs_raven(self):
        detector = AcousticDetector([self.test_model_dir], [0.5])
        model_prob_map = detector.process(self.test_audio_filepath)
        model_probs_df_map = probs_to_pandas(model_prob_map)
        model_raven_df_map = probs_to_raven_detections(model_probs_df_map)


if __name__ == '__main__':
    unittest.main()



