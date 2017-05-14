__author__ = 'Cameron Summers'

import os
import unittest
import numpy as np

from nps_acoustic_discovery.model import EventModel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestSmoke(unittest.TestCase):

    def smoke_test(self):
        """
        Test loading a model and putting through some data.
        """
        test_model_dir = os.path.join(THIS_DIR, 'test_model')
        model = EventModel(test_model_dir)
        model.process(np.ones((1, 84)))


if __name__ == '__main__':
    unittest.main()



