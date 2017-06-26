# NINAPRO Utilities
Functions for helping work with NINAPRO databases 1 and 2.

Relevant data also included:
* Number of subjects in database
* Number of channels
* Number of movements
* Sample Frequency

## Usage
Add this package either to python's path or put the script in your working directory. Then ensure you have a folder somewhere with all the raw mat files downloadable from [here](http://ninapro.hevs.ch/).

Then you can import the NINAPRO data from a particular subject and be away
```python
# subject_nb: 1-27
data_dict = nina_helper.import_db1(db1_path, subject_nb, rest_length_cap=5)
```
or
```python
# subject_nb: 1-40
data_dict = nina_helper.import_db2(db2_path, subject_nb, rest_length_cap=5)
```
```python
reps = [1, 2, 3, 4, 5, 6]
nb_test_reps = 2
train_reps, test_reps = nina_helper.gen_split_balanced(reps, nb_test_reps)

emg_data = nina_helper.normalise_emg(data_dict['emg'], data_dict['rep'], train_reps[0, :])

x_all, y_all, r_all = nina_helper.get_windows(window_len, window_inc, reps,
                                              emg_data, data_dict['move'], data_dict['rep'])

train_idx = nina_helper.get_idxs(r_all, train_reps[0, :])

one_hot_categorical = nina_helper.to_categorical(y_all)
```

## Licence
MIT Licence.

If this helps you with your research please considering referencing as:

Hartwell, A. (2017) _NINAPRO Software Utilities._
