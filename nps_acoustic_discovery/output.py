__author__='Cameron Summers'

"""
Utility functions
"""
import os
import pickle as pk

import pandas as pd

def probs_to_pandas(model_probabilities, save_dir):
    """
    Output probabilities for models to pandas df.

    Args:
        model_probabilities (dict): model to detection probabilities
        save_dir (str): where to save the dataframe
    """
    for model, probs in model_probabilities.iteritems():
        df = pd.DataFrame(probs, columns=model.event_names)
        pk.dump(df, open(os.path.join(save_dir, 'model_{}_probs_df.pk'.format(model.model_id)), 'wb'))

def probs_to_raven_detections(model_probabilities, threshold):   
    """
    Output probabilities for models to pandas df.
    """
    raise NotImplementedError()
