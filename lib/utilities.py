"""
Utility functions
"""
import os
import time
import logging
import tensorflow as tf
import numpy as np


def set_random_seed(seed=9899):
    """ Set random seed for numpy and tensorflow """
    np.random.seed(seed)
    tf.set_random_seed(seed)


def make_config_string(config, key_len=4, max_num_key=4):
    """ Generate a name for configs files """
    str_config = ''
    num_key = 0
    for k, v in config.items():
        if num_key < max_num_key:

            str_config += k[:key_len] + '-' + str(v)[:6] + '_'
            num_key += 1
    return str_config[:-1]


def window_rolling(origin_data, window_size):
    """Rolling data over 0-dim.
    :param origin_data: ndarray of [n_records, ...]
    :param window_size: window_size
    :return: [n_records - window_size + 1, window_size, ...]
    """
    n_records = len(origin_data)
    if n_records < window_size:
        return None

    data = origin_data[:, None]
    all_data = []
    for i in range(window_size):
        all_data.append(data[i: (n_records - window_size + i + 1)])

    # shape -> [n_records - window_size + 1, window_size, ...]
    rolling_data = np.hstack(all_data)

    return rolling_data


def yield2batch_data(arr_dict, batch_size, keep_remainder=True):
    """Iterate the dictionary of array over 0-dim to get batch data.
    :param arr_dict: a dictionary containing array whose shape is [n_items, ...]
    :param batch_size:
    :param keep_remainder: Discard the remainder if False, otherwise keep it.
    :return:
    """
    if arr_dict is None or len(arr_dict) == 0:
        return

    keys = list(arr_dict.keys())

    idx = 0
    n_items = len(arr_dict[keys[0]])
    while idx < n_items:
        if idx + batch_size > n_items and keep_remainder is False:
            return
        next_idx = min(idx + batch_size, n_items)

        yield {k: arr_dict[k][idx: next_idx] for k in keys}

        # update idx
        idx = next_idx


def create_folder(*args):
    """Create path if the folder doesn't exist.
    :param args:
    :return: The folder's path depends on operating system.
    """
    path = os.path.join(*args)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def concat_arrs_of_dict(dict_list):
    """ Concatenate each ndarray with the same key in the dict_list in 0-dimension.
    :param dict_list:
    :return: dict containing concatenated values
    """
    res = dict()

    keys = dict_list[0].keys()
    for k in keys:
        arr_list = []
        for d in dict_list:
            arr_list.append(d[k])
        res[k] = np.concatenate(arr_list, axis=0)

    return res


def get_logger(filename=None):
    logger = logging.Logger(filename)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    # formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    # Add stdout stream handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Add file handler
    if filename:
        fh = logging.FileHandler(filename, mode='a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class Timer(object):
    """
    Count the elapse between start and end time.
    """

    def __init__(self, unit='s'):
        SECOND_UNIT = 1
        MINUTE_UNIT = 60
        HOUR_UNIT = 1440

        unit = unit.lower()
        if unit == 's':
            self._unit = SECOND_UNIT
        elif unit == 'm':
            self._unit = MINUTE_UNIT
        elif unit == 'h':
            self._unit = HOUR_UNIT
        else:
            raise RuntimeError('Unknown unit:', unit)
        # default start time is set to the time the object initialized
        self._start_time = time.time()

    def start(self):
        self._start_time = time.time()

    def end(self):
        end_time = time.time()
        return (end_time - self._start_time) / self._unit
