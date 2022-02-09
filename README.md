# NINAPRO Utilities
Functions for helping work with NINAPRO databases 1 and 2.

Get:
* Raw EMG
    * Normalise
    * Get Windows
* Refined movement labels
* Refined repetition labels
* Accelerometer data (DB2 only, seperate function for memory usage reduction)

Relevant data also included:
* Number of subjects in database
* Number of channels
* Number of movements
* Number of repetitions
* Sample Frequency
* Movement labels
* Repetition labels
* Male/female subjects (subject ID and index)
* Right/Left handed subjects (subject ID and index)
* Age
* Height
* Weight

## Usage

Download this repository then install as a local package via pip

```
pip install [-e] path/to/repo
```

Use `-e` if you'd like to make local edits to the code or keep update to date with the repo.

Alternatively add this package either to python's path or put the script in your working directory.

Then ensure you have a folder somewhere with all the raw mat files (or some subset of subjects) downloadable from [here](http://ninapro.hevs.ch/).

Finally you can import the NINAPRO data from a particular subject and be away
```python
# subject_nb: 1-27
data_dict = nina_helper.import_db1(db1_path, subject_nb)
```
or
```python
# subject_nb: 1-40
data_dict = nina_helper.import_db2(db2_path, subject_nb)
```

A typical workflow to go from raw data to normalised and windowed data ready for use in your favourite machine learning library may look like ths:


```python
from nina_helper import *

db1_path = "path/to/db1"

# Decide window length (150ms window, 10ms increment)
window_len = 15
window_inc = 1

# Choose subject and get info
subject = 1
info_dict = db1_info()  # Get info

# Get EMG, repetition and movement data, don't cap maximum length of rest
data_dict = nina_helper.import_db1(db1_path, subject)

# Create a balanced test - training split based on repetition number
reps = info_dict['rep_labels']
nb_test_reps = 3
train_reps, test_reps = gen_split_balanced(reps, nb_test_reps)

# Normalise EMG data based on training set
emg_data = normalise_emg(data_dict['emg'], data_dict['rep'], train_reps[0, :])

# Window data: x_all data is 4D tensor [observation, time_step, channel, 1] for use with Keras
# y_all: movement label, length: number of windows
# r_all: repetition label, length: number of windows
moves = np.array([1, 3, 5, 20])
x_all, y_all, r_all = get_windows(reps, window_len, window_inc,
                                  emg_data, data_dict['move'],
                                  data_dict['rep'],
                                  which_moves=moves)

train_idx = get_idxs(r_all, train_reps[0, :])
train_data = x_all[train_idx, :, :, :]

one_hot_categorical = to_categorical(y_all)
```

Similarly the code is virtually identical if you wish to work with database 2 instead:

```
from nina_helper import *
import numpy as np

db2_path = "path/to/db2"

# Decide window length
window_len = 300  # Equivalent window length since sampled 20 times faster
window_inc = 20

# Choose subject and get info
subject = 7
info_dict = db2_info()  # Get info

# Get EMG, repetition and movement data, cap max length of rest data before and after each movement to 5 seconds
# Capping occurs by reducing the size of repetition segments since splitting is based on repetition number
data_dict = nina_helper.import_db2(db2_path, subject, rest_length_cap=5)

# Create a random test - training split based on repetition number (specify a set to include)
reps = info_dict['rep_labels']
nb_test_reps = 2
nb_splits = 12
train_reps, test_reps = gen_split_rand(reps, nb_test_reps, nb_splits, base=[2, 5])

# Normalise EMG data based on training set
emg_data = normalise_emg(data_dict['emg'], data_dict['rep'], train_reps[0, :])

# Window data: x_all data is 4D tensor [observation, time_step, channel, 1] for use with Keras
# y_all: movement label, length: number of windows
# r_all: repetition label, length: number of windows
x_all, y_all, r_all = get_windows(reps, window_len, window_inc,
                                  emg_data, data_dict['move'],
                                  data_dict['rep'])

test_idx = get_idxs(r_all, test_reps[0, :])
test_data = x_all[test_idx, :, :, :]

one_hot_categorical = to_categorical(y_all)
```

## Licence
MIT Licence.

If this helps you with your research please considering referencing as:

Hartwell, A. (2017) _NINAPRO Software Utilities._
