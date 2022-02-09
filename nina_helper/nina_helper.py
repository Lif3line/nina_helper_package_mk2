"""Utility functions to help with working with NinaPro database."""

import os
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from itertools import combinations, chain


def db1_info():
    """Return relevant info on database 1.

    Returns:
        Dict: Useful information on database 1
    """
    # General Info
    nb_subjects = 27
    nb_channels = 10
    nb_moves = 53  # 52 + rest
    nb_reps = 10
    fs = 100

    # Labels
    rep_labels = np.array(range(1, nb_reps + 1))
    move_labels = np.array(range(1, nb_moves + 1))

    # Male - Female
    female = np.array([6, 8, 10, 14, 15, 20, 22])
    male = np.array([1, 2, 3, 4, 6, 8, 10, 11, 12, 15, 16, 17, 18, 20, 22, 23, 24, 25, 26, 27])
    female_ind = np.array([5, 7, 9, 13, 14, 19, 21])
    male_ind = np.array([0, 1, 2, 3, 5, 7, 9, 10, 11, 14, 15, 16, 17, 19, 21, 22, 23, 24, 25, 26])

    # Handedness
    left_handed = np.array([14, 16])
    right_handed = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15,
                             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
    left_handed_ind = np.array([13, 15])
    right_handed_ind = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])

    # Other Info (synced)
    ages = np.array([31, 27, 22, 27, 27, 22, 28, 27, 23, 30, 28, 25, 27,
                     29, 26, 29, 30, 29, 34, 26, 38, 35, 30, 26, 28, 40, 28])
    heights = np.array([170, 170, 180, 183, 178, 163, 170, 164, 173, 160, 170, 185, 184,
                        155, 162, 167, 175, 178, 173, 165, 178, 168, 180, 180, 180, 179, 185])
    weights = np.array([75, 62, 85, 95, 75, 48, 60, 54, 63, 60, 67, 80, 85,
                        54, 60, 67, 76, 68, 82, 54, 73, 65, 65, 65, 70, 66, 100])

    return {'nb_subjects': nb_subjects,
            'nb_channels': nb_channels,
            'nb_moves': nb_moves,
            'nb_reps': nb_reps,
            'fs': fs,
            'rep_labels': rep_labels,
            'move_labels': move_labels,
            'female': female,
            'male': male,
            'female_ind': female_ind,
            'male_ind': male_ind,
            'left_handed': left_handed,
            'right_handed': right_handed,
            'left_handed_ind': left_handed_ind,
            'right_handed_ind': right_handed_ind,
            'ages': ages,
            'heights': heights,
            'weights': weights,
            }


def db2_info():
    """Return relevant info on database 2.

    Returns:
        Dict: Useful information on database 2
    """
    # General Info
    nb_subjects = 40
    nb_channels = 12
    nb_moves = 50  # 40 + 9 force movements + rest
    nb_reps = 6
    fs = 2000

    # Labels
    rep_labels = np.array(range(1, nb_reps + 1))
    move_labels = np.array(range(1, nb_moves + 1))

    # Male - Female
    female = np.array([4, 11, 14, 18, 19, 20, 22, 28, 35, 36, 38])
    male = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 21,
                     23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 37, 39, 40])
    female_ind = np.array([3, 10, 13, 17, 18, 19, 21, 27, 34, 35, 37])
    male_ind = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 20,
                         22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 36, 38, 39])

    # Handedness
    left_handed = np.array([4, 13, 22, 25, 26])
    right_handed = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20,
                             21, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])
    left_handed_ind = np.array([3, 12, 21, 24, 25])
    right_handed_ind = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
                                 20, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])

    # Other Info (synced)
    ages = np.array([29, 29, 31, 30, 25, 35, 27, 45, 23, 34, 32, 29, 30, 30, 30, 34, 29, 30, 31,
                     26, 32, 28, 25, 28, 31, 30, 29, 29, 27, 30, 29, 28, 25, 31, 24, 27, 34, 30, 31, 31])
    heights = np.array([187, 183, 174, 154, 175, 172, 187, 173, 172, 173, 150, 184, 182, 173, 169, 173, 175, 169, 158,
                        155, 170, 162, 170, 170, 168, 186, 170, 160, 171, 173, 185, 173, 183, 192, 170, 155, 190, 163,
                        183, 173])
    weights = np.array([75, 75, 69, 50, 70, 79, 92, 73, 63, 84, 54, 90, 70, 59, 58, 76, 70, 90, 52,
                        52, 75, 54, 66, 73, 70, 90, 65, 61, 64, 68, 98, 72, 71, 78, 52, 44, 105, 62, 96, 65])

    return {'nb_subjects': nb_subjects,
            'nb_channels': nb_channels,
            'nb_moves': nb_moves,
            'nb_reps': nb_reps,
            'fs': fs,
            'rep_labels': rep_labels,
            'move_labels': move_labels,
            'female': female,
            'male': male,
            'female_ind': female_ind,
            'male_ind': male_ind,
            'left_handed': left_handed,
            'right_handed': right_handed,
            'left_handed_ind': left_handed_ind,
            'right_handed_ind': right_handed_ind,
            'ages': ages,
            'heights': heights,
            'weights': weights,
            }


def import_db1(folder_path, subject, rest_length_cap=999):
    """Function for extracting data from raw NinaiPro files for DB1.

    Args:
        folder_path (string): Path to folder containing raw mat files
        subject (int): 1-27 which subject's data to import
        rest_length_cap (int, optional): The number of seconds of rest data to keep before/after a movement

    Returns:
        Dictionary: Raw EMG data, corresponding repetition and movement labels, indices of where repetitions are
            demarked and the number of repetitions with capped off rest data
    """
    fs = 100

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_A1_E1.mat')
    data = sio.loadmat(cur_path)
    emg = np.squeeze(np.array(data['emg']))
    rep = np.squeeze(np.array(data['rerepetition']))
    move = np.squeeze(np.array(data['restimulus']))

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_A1_E2.mat')
    data = sio.loadmat(cur_path)
    emg = np.vstack((emg, np.array(data['emg'])))
    rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
    move_tmp = np.squeeze(np.array(data['restimulus']))  # Fix for numbering
    move_tmp[move_tmp != 0] += max(move)
    move = np.append(move, move_tmp)

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_A1_E3.mat')
    data = sio.loadmat(cur_path)
    emg = np.vstack((emg, np.array(data['emg'])))
    rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
    move_tmp = np.squeeze(np.array(data['restimulus']))  # Fix for numbering
    move_tmp[move_tmp != 0] += max(move)
    move = np.append(move, move_tmp)

    move = move.astype('int8')  # To minimise overhead

    # Label repetitions using new block style: rest-move-rest regions
    move_regions = np.where(np.diff(move))[0]
    rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
    nb_reps = int(round(move_regions.shape[0] / 2))
    last_end_idx = int(round(move_regions[0] / 2))
    nb_unique_reps = np.unique(rep).shape[0] - 1  # To account for 0 regions
    nb_capped = 0
    cur_rep = 1

    rep = np.zeros([rep.shape[0], ], dtype=np.int8)  # Reset rep array
    for i in range(nb_reps - 1):
        rep_regions[2 * i] = last_end_idx
        midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] +
                                  move_regions[2 * (i + 1)]) / 2)) + 1

        trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]
        if trailing_rest_samps <= rest_length_cap * fs:
            rep[last_end_idx:midpoint_idx] = cur_rep
            last_end_idx = midpoint_idx
            rep_regions[2 * i + 1] = midpoint_idx - 1

        else:
            rep_end_idx = (move_regions[2 * (i + 1) - 1] +
                           int(round(rest_length_cap * fs)))
            rep[last_end_idx:rep_end_idx] = cur_rep
            last_end_idx = ((move_regions[2 * (i + 1)] -
                             int(round(rest_length_cap * fs))))
            rep_regions[2 * i + 1] = rep_end_idx - 1
            nb_capped += 2

        cur_rep += 1
        if cur_rep > nb_unique_reps:
            cur_rep = 1

    end_idx = int(round((emg.shape[0] + move_regions[-1]) / 2))
    rep[last_end_idx:end_idx] = cur_rep
    rep_regions[-2] = last_end_idx
    rep_regions[-1] = end_idx - 1

    return {'emg': emg,
            'rep': rep,
            'move': move,
            'rep_regions': rep_regions,
            'nb_capped': nb_capped
            }


def import_db1_unrefined(folder_path, subject, rest_length_cap=999):
    """Get the original repetition and stimulus information DB1 (repetition still aligned as rest-move-rest).

    Args:
        folder_path (string): Path to folder containing raw mat files
        subject (int): 1-27 which subject's data to import
        rest_length_cap (int, optional): The number of seconds of rest data to keep before/after a movement

    Returns:
        Dictionary: Unrefined repetition and movement labels, indices of where repetitions are
            demarked and the number of repetitions with capped off rest data
    """
    fs = 100

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_A1_E1.mat')
    data = sio.loadmat(cur_path)
    rep = np.squeeze(np.array(data['repetition']))
    move = np.squeeze(np.array(data['stimulus']))

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_A1_E2.mat')
    data = sio.loadmat(cur_path)
    rep = np.append(rep, np.squeeze(np.array(data['repetition'])))
    move_tmp = np.squeeze(np.array(data['stimulus']))  # Fix for numbering
    move_tmp[move_tmp != 0] += max(move)
    move = np.append(move, move_tmp)

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_A1_E3.mat')
    data = sio.loadmat(cur_path)
    rep = np.append(rep, np.squeeze(np.array(data['repetition'])))
    move_tmp = np.squeeze(np.array(data['stimulus']))  # Fix for numbering
    move_tmp[move_tmp != 0] += max(move)
    move = np.append(move, move_tmp)

    move = move.astype('int8')  # To minimise overhead

    # Label repetitions using new block style: rest-move-rest regions
    move_regions = np.where(np.diff(move))[0]
    rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
    nb_reps = int(round(move_regions.shape[0] / 2))
    last_end_idx = int(round(move_regions[0] / 2))
    nb_unique_reps = np.unique(rep).shape[0] - 1  # To account for 0 regions
    nb_capped = 0
    cur_rep = 1

    rep = np.zeros([rep.shape[0], ], dtype=np.int8)  # Reset rep array
    for i in range(nb_reps - 1):
        rep_regions[2 * i] = last_end_idx
        midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] +
                                  move_regions[2 * (i + 1)]) / 2)) + 1

        trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]
        if trailing_rest_samps <= rest_length_cap * fs:
            rep[last_end_idx:midpoint_idx] = cur_rep
            last_end_idx = midpoint_idx
            rep_regions[2 * i + 1] = midpoint_idx - 1

        else:
            rep_end_idx = (move_regions[2 * (i + 1) - 1] +
                           int(round(rest_length_cap * fs)))
            rep[last_end_idx:rep_end_idx] = cur_rep
            last_end_idx = ((move_regions[2 * (i + 1)] -
                             int(round(rest_length_cap * fs))))
            rep_regions[2 * i + 1] = rep_end_idx - 1
            nb_capped += 2

        cur_rep += 1
        if cur_rep > nb_unique_reps:
            cur_rep = 1

    end_idx = int(round((rep.shape[0] + move_regions[-1]) / 2))
    rep[last_end_idx:end_idx] = cur_rep
    rep_regions[-2] = last_end_idx
    rep_regions[-1] = end_idx - 1

    return {'rep': rep,
            'move': move,
            'rep_regions': rep_regions,
            'nb_capped': nb_capped
            }


def import_db2(folder_path, subject, rest_length_cap=999):
    """Function for extracting data from raw NinaiPro files for DB2.

    Args:
        folder_path (string): Path to folder containing raw mat files
        subject (int): 1-40 which subject's data to import
        rest_length_cap (int, optional): The number of seconds of rest data to keep before/after a movement

    Returns:
        Dictionary: Raw EMG data, corresponding repetition and movement labels, indices of where repetitions are
            demarked and the number of repetitions with capped off rest data

    Note:
        Last 9 "movements" are actually force exercises
    """
    fs = 2000

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E1_A1.mat')
    data = sio.loadmat(cur_path)
    emg = np.squeeze(np.array(data['emg']))
    rep = np.squeeze(np.array(data['rerepetition']))
    move = np.squeeze(np.array(data['restimulus']))

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E2_A1.mat')
    data = sio.loadmat(cur_path)
    emg = np.vstack((emg, np.array(data['emg'])))
    rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
    move_tmp = np.squeeze(np.array(data['restimulus']))
    move = np.append(move, move_tmp)  # Note no fix needed for this exercise

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E3_A1.mat')
    data = sio.loadmat(cur_path)
    emg = np.vstack((emg, np.array(data['emg'])))
    data['repetition'][-1] = 0  # Fix for diffing
    rep = np.append(rep, np.squeeze(np.array(data['repetition'])))

    # Movements number in non-logical pattern [0  1  2  4  6  8  9 16 32 40]
    # Also note that for last file there is no 'rerepetition or 'restimulus'
    data['stimulus'][-1] = 0  # Fix for diffing
    data['stimulus'][np.where(data['stimulus'] == 1)] = 41
    data['stimulus'][np.where(data['stimulus'] == 2)] = 42
    data['stimulus'][np.where(data['stimulus'] == 4)] = 43
    data['stimulus'][np.where(data['stimulus'] == 6)] = 44
    data['stimulus'][np.where(data['stimulus'] == 8)] = 45
    data['stimulus'][np.where(data['stimulus'] == 9)] = 46
    data['stimulus'][np.where(data['stimulus'] == 16)] = 47
    data['stimulus'][np.where(data['stimulus'] == 32)] = 48
    data['stimulus'][np.where(data['stimulus'] == 40)] = 49
    move_tmp = np.squeeze(np.array(data['stimulus']))
    move = np.append(move, move_tmp)

    move = move.astype('int8')  # To minimise overhead

    # Label repetitions using new block style: rest-move-rest regions
    move_regions = np.where(np.diff(move))[0]
    rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
    nb_reps = int(round(move_regions.shape[0] / 2))
    last_end_idx = int(round(move_regions[0] / 2))
    nb_unique_reps = np.unique(rep).shape[0] - 1  # To account for 0 regions
    nb_capped = 0
    cur_rep = 1

    rep = np.zeros([rep.shape[0], ], dtype=np.int8)  # Reset rep array
    for i in range(nb_reps - 1):
        rep_regions[2 * i] = last_end_idx
        midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] +
                                  move_regions[2 * (i + 1)]) / 2)) + 1

        trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]
        if trailing_rest_samps <= rest_length_cap * fs:
            rep[last_end_idx:midpoint_idx] = cur_rep
            last_end_idx = midpoint_idx
            rep_regions[2 * i + 1] = midpoint_idx - 1
        else:
            rep_end_idx = (move_regions[2 * (i + 1) - 1] +
                           int(round(rest_length_cap * fs)))
            rep[last_end_idx:rep_end_idx] = cur_rep
            last_end_idx = ((move_regions[2 * (i + 1)] -
                             int(round(rest_length_cap * fs))))
            rep_regions[2 * i + 1] = rep_end_idx - 1
            nb_capped += 2

        cur_rep += 1
        if cur_rep > nb_unique_reps:
            cur_rep = 1

    end_idx = int(round((emg.shape[0] + move_regions[-1]) / 2))
    rep[last_end_idx:end_idx] = cur_rep
    rep_regions[-2] = last_end_idx
    rep_regions[-1] = end_idx - 1

    return {'emg': emg,
            'rep': rep,
            'move': move,
            'rep_regions': rep_regions,
            'nb_capped': nb_capped
            }


def import_db2_unrefined(folder_path, subject, rest_length_cap=999):
    """Get the original repetition and stimulus information DB1 (repetition still aligned as rest-move-rest).

    Args:
        folder_path (string): Path to folder containing raw mat files
        subject (int): 1-27 which subject's data to import
        rest_length_cap (int, optional): The number of seconds of rest data to keep before/after a movement

    Returns:
        Dictionary: Unrefined repetition and movement labels, indices of where repetitions are
            demarked and the number of repetitions with capped off rest data
    """
    fs = 2000

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E1_A1.mat')
    data = sio.loadmat(cur_path)
    rep = np.squeeze(np.array(data['repetition']))
    move = np.squeeze(np.array(data['stimulus']))

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E2_A1.mat')
    data = sio.loadmat(cur_path)
    rep = np.append(rep, np.squeeze(np.array(data['repetition'])))
    move_tmp = np.squeeze(np.array(data['stimulus']))
    move = np.append(move, move_tmp)  # Note no fix needed for this exercise

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E3_A1.mat')
    data = sio.loadmat(cur_path)
    data['repetition'][-1] = 0  # Fix for diffing
    rep = np.append(rep, np.squeeze(np.array(data['repetition'])))

    # Movements number in non-logical pattern [0  1  2  4  6  8  9 16 32 40]
    data['stimulus'][-1] = 0  # Fix for diffing
    data['stimulus'][np.where(data['stimulus'] == 1)] = 41
    data['stimulus'][np.where(data['stimulus'] == 2)] = 42
    data['stimulus'][np.where(data['stimulus'] == 4)] = 43
    data['stimulus'][np.where(data['stimulus'] == 6)] = 44
    data['stimulus'][np.where(data['stimulus'] == 8)] = 45
    data['stimulus'][np.where(data['stimulus'] == 9)] = 46
    data['stimulus'][np.where(data['stimulus'] == 16)] = 47
    data['stimulus'][np.where(data['stimulus'] == 32)] = 48
    data['stimulus'][np.where(data['stimulus'] == 40)] = 49
    move_tmp = np.squeeze(np.array(data['stimulus']))
    move = np.append(move, move_tmp)

    move = move.astype('int8')  # To minimise overhead

    # Label repetitions using new block style: rest-move-rest regions
    move_regions = np.where(np.diff(move))[0]
    rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
    nb_reps = int(round(move_regions.shape[0] / 2))
    last_end_idx = int(round(move_regions[0] / 2))
    nb_unique_reps = np.unique(rep).shape[0] - 1  # To account for 0 regions
    nb_capped = 0
    cur_rep = 1

    rep = np.zeros([rep.shape[0], ], dtype=np.int8)  # Reset rep array
    for i in range(nb_reps - 1):
        rep_regions[2 * i] = last_end_idx
        midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] +
                                  move_regions[2 * (i + 1)]) / 2)) + 1

        trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]
        if trailing_rest_samps <= rest_length_cap * fs:
            rep[last_end_idx:midpoint_idx] = cur_rep
            last_end_idx = midpoint_idx
            rep_regions[2 * i + 1] = midpoint_idx - 1
        else:
            rep_end_idx = (move_regions[2 * (i + 1) - 1] +
                           int(round(rest_length_cap * fs)))
            rep[last_end_idx:rep_end_idx] = cur_rep
            last_end_idx = ((move_regions[2 * (i + 1)] -
                             int(round(rest_length_cap * fs))))
            rep_regions[2 * i + 1] = rep_end_idx - 1
            nb_capped += 2

        cur_rep += 1
        if cur_rep > nb_unique_reps:
            cur_rep = 1

    end_idx = int(round((rep.shape[0] + move_regions[-1]) / 2))
    rep[last_end_idx:end_idx] = cur_rep
    rep_regions[-2] = last_end_idx
    rep_regions[-1] = end_idx - 1

    return {'rep': rep,
            'move': move,
            'rep_regions': rep_regions,
            'nb_capped': nb_capped
            }


def import_db2_acc(folder_path, subject):
    """Function for extracting acceleronmeter data from raw NinaiPro files for DB2.

    Args:
        folder_path (string): Path to folder containing raw mat files
        subject (int): 1-40 which subject's data to import

    Returns:
        array: Raw accceleronmeter from each electrode
    """
    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E1_A1.mat')
    data = sio.loadmat(cur_path)
    acc = np.squeeze(np.array(data['acc']))

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E2_A1.mat')
    data = sio.loadmat(cur_path)
    acc = np.vstack((acc, np.array(data['acc'])))

    cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E3_A1.mat')
    data = sio.loadmat(cur_path)
    acc = np.vstack((acc, np.array(data['acc'])))

    return acc


def gen_split_balanced(rep_ids, nb_test, base=None):
    """Create a balanced split for training and testing based on repetitions (all reps equally tested + trained on) .

    Args:
        rep_ids (array): Repetition identifiers to split
        nb_test (int): The number of repetitions to be used for testing in each each split
        base (array, optional): A specific test set to use (must be of length nb_test)

    Returns:
        Arrays: Training repetitions and corresponding test repetitions as 2D arrays [[set 1], [set 2] ..]
    """
    nb_reps = rep_ids.shape[0]
    nb_splits = nb_reps

    train_reps = np.zeros((nb_splits, nb_reps - nb_test,), dtype=int)
    test_reps = np.zeros((nb_splits, nb_test), dtype=int)

    # Generate all possible combinations
    all_combos = combinations(rep_ids, nb_test)
    all_combos = np.fromiter(chain.from_iterable(all_combos), int)
    all_combos = all_combos.reshape(-1, nb_test)

    if base is not None:
        test_reps[0, :] = base
        all_combos = np.delete(all_combos, np.where(np.all(all_combos == base, axis=1))[0][0], axis=0)
        cur_split = 1
    else:
        cur_split = 0

    all_combos_copy = all_combos
    reset_counter = 0
    while cur_split < (nb_splits):
        if reset_counter >= 10 or all_combos.shape[0] == 0:
            all_combos = all_combos_copy
            test_reps = np.zeros((nb_splits, nb_test), dtype=int)
            if base is not None:
                test_reps[0, :] = base
                cur_split = 1
            else:
                cur_split = 0

            reset_counter = 0

        randomIndex = np.random.randint(0, all_combos.shape[0])
        test_reps[cur_split, :] = all_combos[randomIndex, :]
        all_combos = np.delete(all_combos, randomIndex, axis=0)

        _, counts = np.unique(test_reps[:cur_split + 1], return_counts=True)

        if max(counts) > nb_test:
            test_reps[cur_split, :] = np.zeros((1, nb_test), dtype=int)
            reset_counter += 1
            continue
        else:
            cur_split += 1
            reset_counter = 0

    for i in range(nb_splits):
        train_reps[i, :] = np.setdiff1d(rep_ids, test_reps[i, :])

    return train_reps, test_reps


def gen_split_rand(rep_ids, nb_test, nb_splits, base=None):
    """Randomly generate nb_splits out of nb_reps training-test splits.

    Args:
        rep_ids (array): Repetition identifiers to split
        nb_test (int): The number of repetitions to be used for testing in each each split
        nb_splits (int): The number of splits to produce
        base (array, optional): A specific test set to use (must be of length nb_test)

    Returns:
        Arrays: Training repetitions and corresponding test repetitions as 2D arrays [[set 1], [set 2] ..]
    """
    nb_reps = rep_ids.shape[0]

    all_combos = combinations(rep_ids, nb_test)
    all_combos = np.fromiter(chain.from_iterable(all_combos), int)
    all_combos = all_combos.reshape(-1, nb_test)

    train_reps = np.zeros((nb_splits, nb_reps - nb_test,), dtype=int)
    test_reps = np.zeros((nb_splits, nb_test), dtype=int)
    cur_split = 0

    if base is not None:
        test_reps[0, :] = base
        train_reps[0, :] = np.setdiff1d(rep_ids, test_reps[0, :])
        all_combos = np.delete(all_combos, np.where(np.all(all_combos == base, axis=1))[0][0], axis=0)
        cur_split = 1

    for i in range(cur_split, nb_splits):
        rand_idx = np.random.randint(all_combos.shape[0])
        test_reps[i, :] = all_combos[rand_idx, :]
        train_reps[i, :] = np.setdiff1d(rep_ids, test_reps[i, :])
        all_combos = np.delete(all_combos, rand_idx, axis=0)

    return train_reps, test_reps


def normalise_emg(emg, reps, train_reps, movements=None, which_moves=None):
    """Preprocess train+test data to mean 0, std 1 based on training data only.

    Args:
        emg (array): Raw EMG data
        reps (array): Corresponding repetition information for each EMG observation
        train_reps (array): Which repetitions are in the training set
        movements (array, optional): Movement labels, required if using which_moves
        which_moves (array, optional): Which movements to return - if None use all

    Returns:
        array: Rescaled EMG data
    """
    train_targets = get_idxs(reps, train_reps)

    # Keep only selected movement(s)
    if which_moves is not None and movements is not None:
        move_targets = get_idxs(movements[train_targets], which_moves)
        train_targets = train_targets[move_targets]

    scaler = StandardScaler(with_mean=True,
                            with_std=True,
                            copy=False).fit(emg[train_targets, :])

    return scaler.transform(emg)


def get_windows(which_reps, window_len, window_inc, emg, movements, repetitons, which_moves=None, get_valid_windows=False, dtype=np.float32):
    """Get set of windows based on repetition and movement criteria and associated label + repetition data.

    Args:
        which_reps (array): Which repetitions to return
        window_len (int): Desired window length
        window_inc (int): Desired window increment
        emg (array): EMG data (should be normalise beforehand)
        movements (array): Movement labels
        repetitons (array): Repetition labels
        which_moves (array, optional): Which movements to return - if None use all
        dtype (TYPE, optional): What precision to use for EMG data
        get_valid_windows (BOOL, optional): option to exclude invalid windows with all zeros

    Returns:
        X_data (array): Windowed EMG data
        Y_data (array): Movement label for each window
        R_data (array): Repetition label for each window
    """
    nb_obs = emg.shape[0]
    nb_channels = emg.shape[1]

    # All possible window end locations given an increment size
    possible_targets = np.array(range(window_len - 1, nb_obs, window_inc))

    targets = get_idxs(repetitons[possible_targets], which_reps)

    # Re-adjust back to original range (for indexinging into rep/move)
    targets = (window_len - 1) + targets * window_inc

    # Keep only selected movement(s)
    if which_moves is not None:
        move_targets = get_idxs(movements[targets], which_moves)
        targets = targets[move_targets]

    valid_inds = []     # save valid window indices
    X_data = np.zeros([targets.shape[0], window_len, nb_channels, 1],
                      dtype=dtype)
    Y_data = np.zeros([targets.shape[0], ], dtype=np.int8)
    R_data = np.zeros([targets.shape[0], ], dtype=np.int8)
    for i, win_end in enumerate(targets):
        win_start = win_end - (window_len - 1)
        if movements[win_start] == movements[win_end]:
            X_data[i, :, :, 0] = emg[win_start:win_end + 1, :]  # Include end
            Y_data[i] = movements[win_end]
            R_data[i] = repetitons[win_end]
            valid_inds.append(i)
    if get_valid_windows:
        X_data = X_data[valid_inds]
        Y_data = Y_data[valid_inds]
        R_data = R_data[valid_inds]

    return X_data, Y_data, R_data


def to_categorical(y, nb_classes=None):
    """Convert a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to nb_classes).
        nb_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.

    Taken from:
    https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py
    v2.0.2 of Keras to remove unnecessary Keras dependency
    """
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, nb_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def get_idxs(in_array, to_find):
    """Utility function for finding the positions of observations of one array in another an array.

    Args:
        in_array (array): Array in which to locate elements of to_find
        to_find (array): Array of elements to locate in in_array

    Returns:
        TYPE: Indices of all elements of to_find in in_array
    """
    targets = ([np.where(in_array == x) for x in to_find])
    return np.squeeze(np.concatenate(targets, axis=1))


def db_info(db):
    """Return relevant info on databases 1 and2.

    Args:
        db (int): Which database to get info on (1 or 2 currently)

    Returns:
        Dict: Useful information on selected database
    """
    if db == 1:
        # General Info
        nb_subjects = 27
        nb_channels = 10
        nb_moves = 53  # 52 + rest
        nb_reps = 10
        fs = 100

        # Labels
        rep_labels = np.array(range(1, nb_reps + 1))
        move_labels = np.array(range(1, nb_moves + 1))

        # Male - Female
        female = np.array([6, 8, 10, 14, 15, 20, 22])
        male = np.array([1, 2, 3, 4, 6, 8, 10, 11, 12, 15, 16, 17, 18, 20, 22, 23, 24, 25, 26, 27])
        female_ind = np.array([5, 7, 9, 13, 14, 19, 21])
        male_ind = np.array([0, 1, 2, 3, 5, 7, 9, 10, 11, 14, 15, 16, 17, 19, 21, 22, 23, 24, 25, 26])

        # Handedness
        left_handed = np.array([14, 16])
        right_handed = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15,
                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
        left_handed_ind = np.array([13, 15])
        right_handed_ind = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                     14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])

        # Other Info (synced)
        ages = np.array([31, 27, 22, 27, 27, 22, 28, 27, 23, 30, 28, 25, 27,
                         29, 26, 29, 30, 29, 34, 26, 38, 35, 30, 26, 28, 40, 28])
        heights = np.array([170, 170, 180, 183, 178, 163, 170, 164, 173, 160, 170, 185, 184,
                            155, 162, 167, 175, 178, 173, 165, 178, 168, 180, 180, 180, 179, 185])
        weights = np.array([75, 62, 85, 95, 75, 48, 60, 54, 63, 60, 67, 80, 85,
                            54, 60, 67, 76, 68, 82, 54, 73, 65, 65, 65, 70, 66, 100])
    elif db == 2:
        # General Info
        nb_subjects = 40
        nb_channels = 12
        nb_moves = 50  # 40 + 9 force movements + rest
        nb_reps = 6
        fs = 2000

        # Labels
        rep_labels = np.array(range(1, nb_reps + 1))
        move_labels = np.array(range(1, nb_moves + 1))

        # Male - Female
        female = np.array([4, 11, 14, 18, 19, 20, 22, 28, 35, 36, 38])
        male = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 21,
                         23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 37, 39, 40])
        female_ind = np.array([3, 10, 13, 17, 18, 19, 21, 27, 34, 35, 37])
        male_ind = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 20,
                             22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 36, 38, 39])

        # Handedness
        left_handed = np.array([4, 13, 22, 25, 26])
        right_handed = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20,
                                 21, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])
        left_handed_ind = np.array([3, 12, 21, 24, 25])
        right_handed_ind = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
                                     20, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])

        # Other Info (synced)
        ages = np.array([29, 29, 31, 30, 25, 35, 27, 45, 23, 34, 32, 29, 30, 30, 30, 34, 29, 30, 31,
                         26, 32, 28, 25, 28, 31, 30, 29, 29, 27, 30, 29, 28, 25, 31, 24, 27, 34, 30, 31, 31])
        heights = np.array([187, 183, 174, 154, 175, 172, 187, 173, 172, 173, 150, 184, 182, 173, 169, 173, 175, 169,
                            158, 155, 170, 162, 170, 170, 168, 186, 170, 160, 171, 173, 185, 173, 183, 192, 170, 155,
                            190, 163, 183, 173])
        weights = np.array([75, 75, 69, 50, 70, 79, 92, 73, 63, 84, 54, 90, 70, 59, 58, 76, 70, 90, 52,
                            52, 75, 54, 66, 73, 70, 90, 65, 61, 64, 68, 98, 72, 71, 78, 52, 44, 105, 62, 96, 65])
    else:
        raise ValueError('db should be either 1 for database 1 or 2 for database 2')

    return {'nb_subjects': nb_subjects,
            'nb_channels': nb_channels,
            'nb_moves': nb_moves,
            'nb_reps': nb_reps,
            'fs': fs,
            'rep_labels': rep_labels,
            'move_labels': move_labels,
            'female': female,
            'male': male,
            'female_ind': female_ind,
            'male_ind': male_ind,
            'left_handed': left_handed,
            'right_handed': right_handed,
            'left_handed_ind': left_handed_ind,
            'right_handed_ind': right_handed_ind,
            'ages': ages,
            'heights': heights,
            'weights': weights,
            }


def import_subject(folder_path, subject, db, rest_length_cap=999):
    """Function for extracting data from raw NinaiPro files for DB1.

    Args:
        folder_path (string): Path to folder containing raw mat files
        subject (int): 1-27 which subject's data to import
        rest_length_cap (int, optional): The number of seconds of rest data to keep before/after a movement
        db (int): Which database to get info on (1 or 2 currently)

    Returns:
        Dictionary: Raw EMG data, corresponding repetition and movement labels, indices of where repetitions are
            demarked and the number of repetitions with capped off rest data
    """
    if db == 1:
        fs = 100

        cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_A1_E1.mat')
        data = sio.loadmat(cur_path)
        emg = np.squeeze(np.array(data['emg']))
        rep = np.squeeze(np.array(data['rerepetition']))
        move = np.squeeze(np.array(data['restimulus']))

        cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_A1_E2.mat')
        data = sio.loadmat(cur_path)
        emg = np.vstack((emg, np.array(data['emg'])))
        rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
        move_tmp = np.squeeze(np.array(data['restimulus']))  # Fix for numbering
        move_tmp[move_tmp != 0] += max(move)
        move = np.append(move, move_tmp)

        cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_A1_E3.mat')
        data = sio.loadmat(cur_path)
        emg = np.vstack((emg, np.array(data['emg'])))
        rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
        move_tmp = np.squeeze(np.array(data['restimulus']))  # Fix for numbering
        move_tmp[move_tmp != 0] += max(move)
        move = np.append(move, move_tmp)

        move = move.astype('int8')  # To minimise overhead

        # Label repetitions using new block style: rest-move-rest regions
        move_regions = np.where(np.diff(move))[0]
        rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
        nb_reps = int(round(move_regions.shape[0] / 2))
        last_end_idx = int(round(move_regions[0] / 2))
        nb_unique_reps = np.unique(rep).shape[0] - 1  # To account for 0 regions
        nb_capped = 0
        cur_rep = 1

        rep = np.zeros([rep.shape[0], ], dtype=np.int8)  # Reset rep array
        for i in range(nb_reps - 1):
            rep_regions[2 * i] = last_end_idx
            midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] +
                                      move_regions[2 * (i + 1)]) / 2)) + 1

            trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]
            if trailing_rest_samps <= rest_length_cap * fs:
                rep[last_end_idx:midpoint_idx] = cur_rep
                last_end_idx = midpoint_idx
                rep_regions[2 * i + 1] = midpoint_idx - 1

            else:
                rep_end_idx = (move_regions[2 * (i + 1) - 1] +
                               int(round(rest_length_cap * fs)))
                rep[last_end_idx:rep_end_idx] = cur_rep
                last_end_idx = ((move_regions[2 * (i + 1)] -
                                 int(round(rest_length_cap * fs))))
                rep_regions[2 * i + 1] = rep_end_idx - 1
                nb_capped += 2

            cur_rep += 1
            if cur_rep > nb_unique_reps:
                cur_rep = 1

        end_idx = int(round((emg.shape[0] + move_regions[-1]) / 2))
        rep[last_end_idx:end_idx] = cur_rep
        rep_regions[-2] = last_end_idx
        rep_regions[-1] = end_idx - 1

    elif db == 2:
        fs = 2000

        cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E1_A1.mat')
        data = sio.loadmat(cur_path)
        emg = np.squeeze(np.array(data['emg']))
        rep = np.squeeze(np.array(data['rerepetition']))
        move = np.squeeze(np.array(data['restimulus']))

        cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E2_A1.mat')
        data = sio.loadmat(cur_path)
        emg = np.vstack((emg, np.array(data['emg'])))
        rep = np.append(rep, np.squeeze(np.array(data['rerepetition'])))
        move_tmp = np.squeeze(np.array(data['restimulus']))
        move = np.append(move, move_tmp)  # Note no fix needed for this exercise

        cur_path = os.path.normpath(folder_path + '/S' + str(subject) + '_E3_A1.mat')
        data = sio.loadmat(cur_path)
        emg = np.vstack((emg, np.array(data['emg'])))
        data['repetition'][-1] = 0  # Fix for diffing
        rep = np.append(rep, np.squeeze(np.array(data['repetition'])))

        # Movements number in non-logical pattern [0  1  2  4  6  8  9 16 32 40]
        # Also note that for last file there is no 'rerepetition or 'restimulus'
        data['stimulus'][-1] = 0  # Fix for diffing
        data['stimulus'][np.where(data['stimulus'] == 1)] = 41
        data['stimulus'][np.where(data['stimulus'] == 2)] = 42
        data['stimulus'][np.where(data['stimulus'] == 4)] = 43
        data['stimulus'][np.where(data['stimulus'] == 6)] = 44
        data['stimulus'][np.where(data['stimulus'] == 8)] = 45
        data['stimulus'][np.where(data['stimulus'] == 9)] = 46
        data['stimulus'][np.where(data['stimulus'] == 16)] = 47
        data['stimulus'][np.where(data['stimulus'] == 32)] = 48
        data['stimulus'][np.where(data['stimulus'] == 40)] = 49
        move_tmp = np.squeeze(np.array(data['stimulus']))
        move = np.append(move, move_tmp)

        move = move.astype('int8')  # To minimise overhead

        # Label repetitions using new block style: rest-move-rest regions
        move_regions = np.where(np.diff(move))[0]
        rep_regions = np.zeros((move_regions.shape[0],), dtype=int)
        nb_reps = int(round(move_regions.shape[0] / 2))
        last_end_idx = int(round(move_regions[0] / 2))
        nb_unique_reps = np.unique(rep).shape[0] - 1  # To account for 0 regions
        nb_capped = 0
        cur_rep = 1

        rep = np.zeros([rep.shape[0], ], dtype=np.int8)  # Reset rep array
        for i in range(nb_reps - 1):
            rep_regions[2 * i] = last_end_idx
            midpoint_idx = int(round((move_regions[2 * (i + 1) - 1] +
                                      move_regions[2 * (i + 1)]) / 2)) + 1

            trailing_rest_samps = midpoint_idx - move_regions[2 * (i + 1) - 1]
            if trailing_rest_samps <= rest_length_cap * fs:
                rep[last_end_idx:midpoint_idx] = cur_rep
                last_end_idx = midpoint_idx
                rep_regions[2 * i + 1] = midpoint_idx - 1
            else:
                rep_end_idx = (move_regions[2 * (i + 1) - 1] +
                               int(round(rest_length_cap * fs)))
                rep[last_end_idx:rep_end_idx] = cur_rep
                last_end_idx = ((move_regions[2 * (i + 1)] -
                                 int(round(rest_length_cap * fs))))
                rep_regions[2 * i + 1] = rep_end_idx - 1
                nb_capped += 2

            cur_rep += 1
            if cur_rep > nb_unique_reps:
                cur_rep = 1

        end_idx = int(round((emg.shape[0] + move_regions[-1]) / 2))
        rep[last_end_idx:end_idx] = cur_rep
        rep_regions[-2] = last_end_idx
        rep_regions[-1] = end_idx - 1
    else:
        raise ValueError('db should be either 1 for database 1 or 2 for database 2')

    return {'emg': emg,
            'rep': rep,
            'move': move,
            'rep_regions': rep_regions,
            'nb_capped': nb_capped
            }
