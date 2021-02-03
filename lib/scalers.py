"""
Data Scalers
"""

import numpy as np
from abc import abstractmethod


class AbstractScaler(object):
    """
    Scaler for scaling data into proper range.
    """

    @abstractmethod
    def fit(self, records):
        pass

    @abstractmethod
    def is_fit(self):
        pass

    @abstractmethod
    def scaling(self, records):
        pass

    @abstractmethod
    def inverse_scaling(self, scaled_records):
        pass

    def fit_scaling(self, records):
        self.fit(records)
        return self.scaling(records)


class DictScaler(AbstractScaler):
    """
    Scaler for dictionary that scales the items specified by key scaler in initialization. If the item doesn't exist,
    do nothing.
    """
    def __init__(self, **kwargs):
        self.scaler_dict = {}
        for k, scaler_class in kwargs.items():
            self.scaler_dict[k] = scaler_class()

    def is_fit(self):
        res = True
        for _, scaler in self.scaler_dict.items():
            res = res and scaler.is_fit()
        return res

    def fit(self, dict_data):
        for k, records in dict_data.items():
            if k in self.scaler_dict.keys():
                self.scaler_dict[k].fit(records)

    def scaling(self, dict_data):
        scaled_dict_data = dict()
        for k, records in dict_data.items():
            if k in self.scaler_dict.keys():
                scaled_dict_data[k] = self.scaler_dict[k].scaling(records)
            else:
                scaled_dict_data[k] = records
        return scaled_dict_data

    def inverse_scaling(self, scaled_dict_data):
        dict_data = dict()

        for k, scaled_records in scaled_dict_data.items():
            if k in self.scaler_dict.keys():
                dict_data[k] = self.scaler_dict[k].inverse_scaling(scaled_records)
            else:
                dict_data[k] = scaled_records
        return dict_data


class VoidScaler(AbstractScaler):
    def __init__(self):
        pass

    def is_fit(self):
        return True

    def fit(self, records):
        pass

    def scaling(self, records):
        return records

    def inverse_scaling(self, scaled_records):
        return scaled_records


class SingletonZeroMaxScaler(AbstractScaler):
    def __init__(self, epsilon=1e-8):
        self._max_val = None
        self._epsilon = epsilon

    def is_fit(self):
        return self._max_val is not None

    def fit(self, records):
        if self.is_fit():
            raise RuntimeError('Try to fit a fitted scaler!')
        self._max_val = np.max(records)

    def scaling(self, records):
        if not self.is_fit():
            raise RuntimeError('Try to scaling records using a uninitialized scaler!')
        return records / (self._max_val + self._epsilon)

    def inverse_scaling(self, scaled_records):
        if not self.is_fit():
            raise RuntimeError('Try to inverse_scaling records using a uninitialized scaler!')
        return scaled_records * (self._max_val + self._epsilon)


class MinMaxScaler(AbstractScaler):
    def __init__(self, epsilon=1e-8):
        self._min_val = None
        self._max_val = None
        self._epsilon = epsilon

    def is_fit(self):
        return self._max_val is not None

    def fit(self, records):
        if self.is_fit():
            raise RuntimeError('Try to fit a fitted scaler!')
        self._max_val = np.max(records, axis=0)
        self._min_val = np.min(records, axis=0)

    def scaling(self, records):
        if not self.is_fit():
            raise RuntimeError('Try to scaling records using a uninitialized scaler!')
        return (records - self._min_val) / (self._max_val - self._min_val + self._epsilon)

    def inverse_scaling(self, scaled_records):
        if not self.is_fit():
            raise RuntimeError('Try to inverse_scaling records using a uninitialized scaler!')
        return (scaled_records * (self._max_val - self._min_val + self._epsilon)) + self._min_val


class ZeroMaxScaler(AbstractScaler):
    def __init__(self, epsilon=1e-8):
        self._min_val = 0.0
        self._max_val = None
        self._epsilon = epsilon

    def is_fit(self):
        return self._max_val is not None

    def fit(self, records):
        if self.is_fit():
            raise RuntimeError('Try to fit a fitted scaler!')
        self._max_val = np.max(records, axis=1, keepdims=True)

    def scaling(self, records):
        if not self.is_fit():
            raise RuntimeError('Try to scaling records using a uninitialized scaler!')
        return (records - self._min_val) / (self._max_val - self._min_val + self._epsilon)

    def inverse_scaling(self, scaled_records):
        if not self.is_fit():
            raise RuntimeError('Try to inverse_scaling records using a uninitialized scaler!')
        return (scaled_records * (self._max_val - self._min_val + self._epsilon)) + self._min_val


class SingletonStandScaler(AbstractScaler):
    def __init__(self, epsilon=1e-8):
        self._mean = None
        self._std = None
        self._epsilon = epsilon

    def is_fit(self):
        return self._mean is not None

    def fit(self, records):
        if self.is_fit():
            raise RuntimeError('Try to fit a fitted scaler!')
        self._mean = np.mean(records)
        self._std = np.std(records)

    def scaling(self, records):
        if not self.is_fit():
            raise RuntimeError('Try to scaling records using a uninitialized scaler!')
        return (records - self._mean) / (self._std + self._epsilon)

    def inverse_scaling(self, scaled_records):
        if not self.is_fit():
            raise RuntimeError('Try to inverse_scaling records using a uninitialized scaler!')
        return scaled_records * (self._std + self._epsilon) + self._mean
