# NINAPRO Utilities
Functions for helping work with NINAPRO databases 1 and 2.

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
Simplest usage is to download this repository then install as a package via pip

```
pip install [-e] path/to/repo
```

Use `-e` if you'd like to make local edits to the code or keep update to date with the repo.

Alternatively add this package either to python's path or put the script in your working directory. Then ensure you have a folder somewhere with all the raw mat files downloadable from [here](http://ninapro.hevs.ch/).

Then you can import the NINAPRO data from a particular subject and be away
```python
# subject_nb: 1-27
data_dict = nina_helper.import_db1(db1_path, subject_nb)
```
or
```python
# subject_nb: 1-40
data_dict = nina_helper.import_db2(db2_path, subject_nb)
```



```python
from nina_helper import *

db1_path = "E:\\phd-data\\ninapro\\db1"

# Decide window length
window_len = 15
window_inc = 1

# Choose subject and get info
subject = 1
info_dict = db1_info()  # Get info

# Get EMG, repetition and movement data, cap max length of rest data before and after each movement to 5 seconds
data_dict = nina_helper.import_db1(db1_path, subject, rest_length_cap=5)

# Create a balanced test - training split based on repetition number
reps = info_dict['rep_labels']
nb_test_reps = 3
train_reps, test_reps = gen_split_balanced(reps, nb_test_reps)

# Create a random test - training split based on repetition number (specify a set to include)
reps = info_dict['rep_labels']
nb_test_reps = 3
nb_splits = 12
train_reps, test_reps = gen_split_rand(reps, nb_test_reps, nb_splits, base=[2, 5, 7])

# Normalise EMG data based on training set
emg_data = normalise_emg(data_dict['emg'], data_dict['rep'], train_reps[0, :])

# Window data: x_all data is 4D tensor [observation, time_step, channel, 1] for use with Keras
# y_all: movement label, length: number of windows
# r_all: repetition label, length: number of windows
x_all, y_all, r_all = get_windows(reps, window_len, window_inc,
                                  emg_data, data_dict['move'],
                                  data_dict['rep'],
                                  which_moves=)

train_idx = get_idxs(r_all, train_reps[0, :])
train_data = x_all[train_idx, :, :, :]

one_hot_categorical = to_categorical(y_all)
```

## Licence
MIT Licence.

If this helps you with your research please considering referencing as:

Hartwell, A. (2017) _NINAPRO Software Utilities._
