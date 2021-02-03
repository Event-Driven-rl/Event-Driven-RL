"""
Define the data provider that process raw data that will feed into the models
"""
from abc import abstractmethod
import numpy as np

from lib import yield2batch_data, get_metrics_callback_from_names
from lib import DictScaler, VoidScaler


class AbstractDataProvider(object):

    @abstractmethod
    def iterate_batch_data(self):
        """ Get batch models input of one epoch.
        Remark: batch -> partition -> epoch
        :return: yield a list containing batch inputs until the end of the epoch.
        """
        pass

    @abstractmethod
    def get_metrics(self, preds, labels):
        """ Calculate the metrics of preds and labels.
        :param preds:
        :param labels:
        :return: a dictionary of the metrics.
        """
        pass

    @abstractmethod
    def epoch_inverse_scaling(self, scaled_records):
        """ Inverse the scaled_records to real scale.
        :param scaled_records:
        :return: real scale records.
        """
        pass


class DataProvider(AbstractDataProvider):
    """
    Data provider for processing models inputs.
    """

    def __init__(self, data_source, data_config, scaler=None):
        self.data_source = data_source
        self.batch_size = data_config['batch_size']
        self.metrics_function = get_metrics_callback_from_names(data_config['metrics'])
        self.scaler = scaler if scaler else DictScaler(dtimes=VoidScaler)
        self.is_first_iterate = True

        self.type_padding = data_config['process_dim'] + 1

    def get_scaler(self):
        return self.scaler

    def epoch_inverse_scaling(self, scaled_records):
        return self.scaler.inverse_scaling(scaled_records)

    def get_metrics(self, preds, labels):
        labels['marks'] = np.squeeze(labels['marks'], -1)
        seq_mask = labels['types'] < self.type_padding
        marks_mask = labels['marks'] == 0
        seq_mask = np.logical_and(seq_mask, marks_mask)
        return self.metrics_function(preds, labels, seq_mask=seq_mask)

    def iterate_batch_data(self):
        # record_data of a partition whose shape is [n_records, ...]
        for data in self.data_source.load_partition_data():
            if self.is_first_iterate:
                data_stats = self._dataset_statistics(data)
                print(f'Load dataset {self.data_source.data_name}: {data_stats}')

            inputs = self._process_model_input(data)
            if self.scaler.is_fit():
                scaled_inputs = self.scaler.scaling(inputs)
            else:
                scaled_inputs = self.scaler.fit_scaling(inputs)

            # yield records to batch data separately
            for batch_data in yield2batch_data(scaled_inputs, self.batch_size, keep_remainder=True):
                yield batch_data

        if self.is_first_iterate:
            self.is_first_iterate = False

    def iterate_batch_data_initial(self):
        # record_data of a partition whose shape is [n_records, ...]
        for data in self.data_source.load_partition_data():
            inputs = self._process_model_input(data)
            if self.scaler.is_fit():
                scaled_inputs = self.scaler.scaling(inputs)
            else:
                scaled_inputs = self.scaler.fit_scaling(inputs)

            # yield records to batch data separately
            for batch_data in yield2batch_data(scaled_inputs, 1, keep_remainder=True):
                yield batch_data

    def _process_model_input(self, records):
        type_padding = self.type_padding
        dt_padding = 0.0
        action_padding = 1

        type_seqs = records['types']
        time_seqs = records['timestamps']
        # dt_i = t_i - t_{i-1}
        # [0, t_1 - t_0, t_2 - t_1, ...]
        dt_seqs = [[t_seq[i] - t_seq[max(i - 1, 0)] for i in range(len(t_seq))]
                   for t_seq in time_seqs]

        n_records = len(type_seqs)
        max_len = max([len(seq) for seq in type_seqs])

        # padding
        type_seqs_padded = np.ones([n_records, max_len]) * type_padding
        dt_seqs_padded = np.ones([n_records, max_len]) * dt_padding

        for i in range(n_records):
            len_seq = len(type_seqs[i])
            type_seqs_padded[i, :len_seq] = type_seqs[i]
            dt_seqs_padded[i, : len_seq] = dt_seqs[i]

        ret = dict()
        ret['types'] = type_seqs_padded
        ret['dtimes'] = dt_seqs_padded

        # add to models input if original input has marks
        if 'marks' in records.keys():
            marks_seqs = records['marks']
            marks_padding = 1.0
            marks_padded = np.ones([n_records, max_len, 1]) * marks_padding
            for i in range(n_records):
                len_seq = len(marks_seqs[i])
                marks_padded[i, : len_seq, 0] = marks_seqs[i]
            ret['marks'] = marks_padded

        return ret

    def _dataset_statistics(self, data):
        statistics = {}
        # get target seqs
        type_seqs = data['types']
        dt_seqs = [[t_seq[i] - t_seq[i - 1] for i in range(1, len(t_seq))]
                   for t_seq in data['timestamps']]

        mark_seqs = data.get('marks')
        if mark_seqs is not None:
            type_seqs_mark = []
            for i in range(len(type_seqs)):
                type_seqs_mark.append([])
                for j in range(len(type_seqs[i])):
                    if mark_seqs[i][j] == 0:
                        type_seqs_mark[i].append(type_seqs[i][j])
        type_seqs = type_seqs_mark
        event_num = self.type_padding

        if isinstance(type_seqs[0], list):
            types = sum(type_seqs, [])
        else:
            types = np.concatenate(type_seqs)
        dts = sum(dt_seqs, [])

        # get statistics
        statistics['n_records'] = len(type_seqs)
        statistics['max_len_of_record'] = max([len(seq) for seq in type_seqs])
        statistics['min_len_of_record'] = min([len(seq) for seq in type_seqs])
        statistics['max_dt'] = np.max(dts)
        statistics['mean_dt'] = np.mean(dts)

        type_count = [0] * event_num
        for t in types:
            type_count[int(t)] += 1

        type_ratio = np.divide(type_count, np.sum(type_count))
        statistics['max_type_ratio'] = np.max(type_ratio)

        # plot marks
        mark_seqs = data.get('marks')
        if mark_seqs is not None:

            if isinstance(mark_seqs, list):
                marks = sum(mark_seqs, [])
            else:
                marks = np.concatenate(mark_seqs)
            statistics['mean_marks'] = np.mean(marks)
            marks_rmse = np.sqrt(np.mean(np.subtract(marks, np.mean(marks)) ** 2))
            print(marks_rmse)

        return statistics