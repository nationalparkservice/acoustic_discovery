National Park Service Acoustic Discovery
========================================

A project for automatic detection of acoustic events in audio commissioned by the
National Park Service.

Example Usage:

`python -m nps_acoustic_discovery.discover <path_to_audio> <path_to_save_dir> --m <model_dir1> -m <model_dir2> -t <threshold1> -t <threshold2>`

This will output two files of detections, one for each of the models.


For help:

`python -m nps_acoustic_discovery.discover -h`


Run tests:

`nosetests --nocapture test/test_model.py`


Dependencies
============

Keras - https://keras.io/

Pandas - http://pandas.pydata.org/

Python Speech Features - https://github.com/jameslyons/python_speech_features

h5py - http://www.h5py.org/


Installation
============

I recommend pip and virtualenv (or [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)) to manage packages.

    pip install keras
    pip install pandas
    pip install python_speech_features
    pip install h5py

Tested using Python 3.5

Recommended package versions:

keras - 2.0.4
numpy - 1.12.1
pandas - 0.20.1
h5py - 2.7.0


Troubleshooting
---------------


-`ImportError: No module named 'tensorflow'`

Installing Keras with Pip creates a configuration file in your home directory ~/.keras/keras.json with
the compute backend as Tensorflow. You may need to change this to Theano: `"backend": "theano"`








