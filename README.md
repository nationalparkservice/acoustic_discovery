National Park Service Acoustic Discovery
========================================

A project for automatic detection of acoustic events in audio commissioned by the
National Park Service.

For help:

`python nps_acoustic_discovery.py -h`


Dependencies
============

Keras - https://keras.io/

Pandas - http://pandas.pydata.org/

Python Speech Features - https://github.com/jameslyons/python_speech_features

h5py - http://www.h5py.org/


Installation
============

```
pip install keras
pip install pandas
pip install python_speech_features
pip install h5py
```


Troubleshooting
---------------


-`ImportError: No module named 'tensorflow'`

Installing Keras with Pip creates a configuration file in your home directory ~/.keras/keras.json with
the compute backend as Tensorflow. You may need to change this to Theano: `"backend": "theano"`








