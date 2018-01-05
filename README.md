# Avian Acoustic Discovery: Alaska

This library was created to automatically detect avian songs in audio. 
It was commissioned by the National Park Service to assist with biological research. 


## Table of Contents

* [Background](#background)
* [Author](#author)
* [Detection Library]
    * [How to Use](#usage)
        * [Command Line](#using-command-line)
        * [Code](#using-code)
    * [Installation](#installation)
* [Model Training](#model-training)
* [Testing](#smoke-tests)
* [Troubleshooting](#troubleshooting)
* [Dependencies](#dependencies)


### Background

Since 2001 researchers at Denali National Park have collected extensive audio recordings throughout the park
in an initiative to protect and study the natural acoustic environment. Recordings often contain sounds of birds which can be analyzed for species abundance, behavior, etc. and support conservation efforts. The identification and annotation of avian species over thousands of hours of audio would require an enormous amount of time from skilled technical staff. Recent advances in artificial intelligence technology have drastically improved the ability of machines to perceive audio signals at human levels. This
library uses machine listening models already-trained on Denali audio to help automatically identify a variety of
avian species, speeding the analysis several fold.


### Author

This library and the associated listening models were created by [Cameron Summers](mailto:scaubrey84@gmail.com) 
who is a researcher in machine learning and artificial intelligence located in the San Francisco Bay Area.


### Usage

At a high level, the library takes in an audio file stored on the local hard drive and outputs a corresponding timeline
of detection probabilities of one or more species. 

* 0.0 probability means unlikely detection of species
* 1.0 probability means likely detection of species

From these probabilities, the user can specify thresholds (or use recommended ones) for true detections and 
optionally output these detections or the detection audio slice.

The configuration for the models is carefully tuned for optimal detection performance. It is helpful to
understand some of these parameters to be able to interpret the outputs of the library:
 
* window_size_sec - Size of the detection window
* hop_size - Separation between consecutive overlapping detection windows

For the models in the initial release of this library, the window size is 4.0 seconds and the hop size is 0.1 seconds.
So for a 30 second long file, there should be 300 detections. The first detection window goes from 0.0 seconds 
in the audio to 4.0 seconds, the second window from 0.1 seconds to 4.1 seconds, and so on.


##### Models

Each species has an already-trained model in a folder and they are stored in the `models` directory
of this project. The user provides a path to one of these to use
it for detections.


##### Detection Thresholds

When running a detector, you will likely use these recommended thresholds:
 
Species | Code | Recommended Threshold
--- | --- | ---
American Robin | AMRO | 0.6
Blackpoll Warbler | BLPW | 0.2
Fox Sparrow | FOSP | 0.7
White-crowned Sparrow | WCSP | 0.99
Common Raven | CORA | 0.1
Dark-eyed Junco | DEJU | 0.2
Greater Yellowlegs | GRYE | 0.3
Hermit Thrush | HETH | 0.1
Swainson’s Thrush | SWTH | 0.6
White-tailed Ptarmigan | WTPT | 0.9


Using your own thresholds:

Knowledge of [Binary Classification](https://en.wikipedia.org/wiki/Binary_classification) and associated evaluation
techniques is useful for setting thresholds. A user might vary the detection thresholds depending on the 
application. If to goal is to answer the question "Does my species exist anywhere in this file?", this might
call for a high threshold to limit [Type I Errors](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#Type_I_error).
However, if the goal is to answer the question of "Precisely how many calls occurred in the file?", then a lower
threshold may be appropriate to limit [Type II Errors](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#Type_II_error).


#### Using Command Line

For help:

`python -m nps_acoustic_discovery.discover -h`


```
usage: Audio event detection for the National Park Service [-h]
                                                           -m MODEL_DIR_PATH
                                                           -t THRESHOLD
                                                           [-o {probs,detections,audio}]
                                                           --ffmpeg FFMPEG
                                                           audio_path save_dir

positional arguments:
  audio_path            Path to audio file on which to run the classifier
  save_dir              Directory in which to save the output.

optional arguments:
-h, --help            show this help message and exit
-m MODEL_DIR_PATH, --model_dir_path MODEL_DIR_PATH
                    Path to model(s) directories for classification
-t THRESHOLD, --threshold THRESHOLD
                    The threshold for a positive detection
-o {probs,detections,audio}, --output {probs,detections,audio}

                        Type of output file:
                             probs: Raw probabilities over time
                             detections: Raven detections file
                             audio: Audio slices for each detection

--ffmpeg FFMPEG       Path to FFMPEG executable
```


##### Command Line Examples

Running one model to generate a Raven file:

`python -m nps_acoustic_discovery.discover <path_to_audio> <path_to_save_dir> --m <model_dir> -t <threshold> -t -o detections`

Running two species models with two different thresholds generates two
Raven files describing where the model detection probabilities
exceeded the thresholds:

`python -m nps_acoustic_discovery.discover <path_to_audio> <path_to_save_dir> --m <model_dir1> -m <model_dir2> -t <threshold1> -t <threshold2> -o detections`

Running one model to generate a file with raw probabilities:

`python -m nps_acoustic_discovery.discover <path_to_audio> <path_to_save_dir> --m <model_dir> -t <threshold> -t -o probs`

Running one model to generate an audio file (possibly many) where the
model detection probabilities exceeded the threshold:

`python -m nps_acoustic_discovery.discover <path_to_audio> <path_to_save_dir> --m <model_dir> -t <threshold> -t -o audio`


#### Using Code

While inside the project directory, setup a model:

```python
>>> from nps_acoustic_discovery.discover import AcousticDetector
>>> model_dir_paths = ['./test/test_model']
>>> thresholds = [0.5]
>>> ffmpeg_path = '/usr/bin/ffmpeg'   # or where yours is
>>> detector = AcousticDetector(model_dir_paths, thresholds, ffmpeg_path=ffmpeg_path)
```

The models attribute in the detector is a dict that maps
a model id to the model object. Now the detector houses 1 model
at a threshold of 0.5 and a feature configuration. The feature
configuration is derived from the model training phase and should
not be altered since it could alter detection performance or
break detection functionality.

```python
>>> len(detector.models)
1
>>> detector.models.items()
dict_items([('13318244', <nps_acoustic_discovery.model.EventModel object at 0x10e4c2710>)])
>>> detector.models['13318244'].detection_threshold
0.5
>>> detector.models['13318244'].fconfig
{'window_size_sec': 1.5, 'nfft': 1024, 'axis_dim': 1, 'num_cepstral_coeffs': 14, 'hop_size': 0.1, 'feature_dim': 42, 'num_filters': 512, 'high_freq': 12000.0, 'low_freq': 100.0}
```

Now we can use the detector on some audio.

```python
>>> audio_path = './test/test30s.wav'
>>> model_prob_map = detector.process(audio_path)
ffmpeg version 3.3.4 Copyright (c) 2000-2017 the FFmpeg developers
  built with Apple LLVM version 6.0 (clang-600.0.56) (based on LLVM 3.5svn)
  configuration: --prefix=/usr/local/Cellar/ffmpeg/3.3.4 --enable-shared --enable-pthreads --enable-gpl --enable-version3 --enable-hardcoded-tables --enable-avresample --cc=clang --host-cflags= --host-ldflags= --enable-libmp3lame --enable-libx264 --enable-libxvid --enable-opencl --enable-videotoolbox --disable-lzma --enable-vda
  libavutil      55. 58.100 / 55. 58.100
  libavcodec     57. 89.100 / 57. 89.100
  libavformat    57. 71.100 / 57. 71.100
  libavdevice    57.  6.100 / 57.  6.100
  libavfilter     6. 82.100 /  6. 82.100
  libavresample   3.  5.  0 /  3.  5.  0
  libswscale      4.  6.100 /  4.  6.100
  libswresample   2.  7.100 /  2.  7.100
  libpostproc    54.  5.100 / 54.  5.100
Guessed Channel Layout for Input Stream #0.0 : mono
Input #0, wav, from './test/test30s.wav':
  Duration: 00:00:30.50, bitrate: 706 kb/s
    Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 44100 Hz, mono, s16, 705 kb/s
Stream mapping:
  Stream #0:0 -> #0:0 (pcm_s16le (native) -> pcm_s16le (native))
Press [q] to stop, [?] for help
Output #0, s16le, to 'pipe:1':
  Metadata:
    encoder         : Lavf57.71.100
    Stream #0:0: Audio: pcm_s16le, 44100 Hz, mono, s16, 705 kb/s
    Metadata:
      encoder         : Lavc57.89.100 pcm_s16le
size=    2627kB time=00:00:30.50 bitrate= 705.6kbits/s speed=2.39e+03x
video:0kB audio:2627kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.000000%
DEBUG:Processing chunk: 1. Audio len (s): 30.5
DEBUG:Processing features...
WARNING:frame length (1103) is greater than FFT size (1024), frame will be truncated. Increase NFFT to avoid.
DEBUG:Input vector shape: (306, 42)
```

Now we have probabilities of detection for the file.

```python
>>> for model, probabilities in model_prob_map.items():
...     print("Type: {}, Shape: {}".format(type(probabilities), probabilities.shape))
...
Type: <class 'numpy.ndarray'>, Shape: (306, 1)
```

As you can see, there are 306 raw detection probabities for each 0.1
seconds of the file. There are some convenience functions for common outputs. One is to
easily create a [Pandas](http://pandas.pydata.org/) dataframe.

```python
>>> from nps_acoustic_discovery.output import probs_to_pandas, probs_to_raven_detections
>>> model_prob_df_map = probs_to_pandas(model_prob_map)
>>> for model, prob_df in model_prob_df_map.items():
...     print(prob_df.columns)
...
Index(['Relative Time (s)', 'AMRO'], dtype='object')
```

And then to create a file that can be read by [Raven](http://www.birds.cornell.edu/brp/raven/RavenFeatures.html)
built by the Cornell Lab of Ornithology.

```python
>>> model_raven_df_map = probs_to_raven_detections(model_prob_df_map)
>>> header = ['Selection', 'Begin Time (s)', 'End Time (s)', 'Species']
>>> for model, raven_df in model_raven_df_map.items():
...     raven_df[header].to_csv('./', 'selection_table.txt', sep='\t', float_format='%.1f', index=False)
```

Presumably, we could then count the number of times the species was
detected:

```python
>>> len(raven_df)
```


## Installation

This project was developed for and tested with **Python 3.5**.

To install, clone this repository then install python dependencies
using pip: `pip install -r requirements.txt`. It is recommended to
use pip with virtualenv (or [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/))
to keep your projects tidy.

This library also requires [ffmpeg](https://ffmpeg.org/) for file
conversion - which implies it also handles many different types
of audio file encodings - and for stream processing of large files.
To install ffmpeg on Windows, see this the installation steps outline
 [here](https://github.com/nationalparkservice/ffaudIO) by a member of
 the National Park Service. For static builds on all platforms, see
 the [downloads](https://ffmpeg.org/download.html) on the ffmpeg site.


## Model Training

A significant amount of time was invested in training species models
to perform optimally. However, users can expect varied detection
performance depending on the species/background noise/etc. since
the model learns from the data and the data isn't always perfect or
complete. Some common considerations for users that affect performance:

* Species
    * The model learns from the data and some species have fewer examples to learn from
* Background Noise
    * Rain or heavy overlap in species calls
* Audio Encoding
    * The training audio was 60 and 90 kbps mp3 at 44.1kHz.
    Lower audio quality than this may reduce performance.


## Smoke Tests

To run some basic tests, use [nose](https://nose.readthedocs.io/en/latest/):

`nosetests --nocapture test/test_model.py`

This should generate no errors.


## Troubleshooting

-`ImportError: No module named 'tensorflow'`

Installing Keras with Pip creates a configuration file in your home directory ~/.keras/keras.json with
the compute backend as Tensorflow. You may need to change this to Theano: `"backend": "theano"`


## Dependencies

* [Keras](https://keras.io/)

* [Pandas](http://pandas.pydata.org/)

* [Python Speech Features](https://github.com/jameslyons/python_speech_features)

* [h5py](http://www.h5py.org/)

* [ffmpeg](https://ffmpeg.org/)



