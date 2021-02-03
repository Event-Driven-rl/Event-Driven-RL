import numpy as np
from lib import yield2batch_data


class BufferDataProvider:
    def __init__(self, data_source, data_config, env_config, reward_func):
        self.data_source = data_source
        self.buffer_size = data_config['buffer_size']
        self.type_padding = data_config['process_dim'] + 1
        self.batch_size = data_config['batch_size']
        self.data_filename = data_config['data_filename']
        self.process_dim = data_config['process_dim']

        self.state_flag = data_config['state_flag']
        self.action_flag = data_config['action_flag']
        self.next_state_flag = data_config['next_state_flag']
        self.act_dim = len(env_config['baseline']) - env_config['candidate_dim'] + 1
        self._init_buffer_setting(reward_func)
        self.use_emergent = env_config['use_emergent']


    def _init_buffer_setting(self, reward_func):
        self.events_buf = [[] for _ in range(int(self.buffer_size))]
        self.times_buf = [[] for _ in range(int(self.buffer_size))]
        self.marks_buf = [[] for _ in range(int(self.buffer_size))]
        self.flag_buf = [[] for _ in range(int(self.buffer_size))]
        self.filter_buf = [[] for _ in range(int(self.buffer_size))]


        self.rews_buf = np.zeros(self.buffer_size, dtype=np.float32)
        self.done_buf = np.zeros(self.buffer_size, dtype=np.float32)
        self.lambdas_buf = np.zeros((self.buffer_size, self.process_dim), dtype=np.float32)
        self.data_source = self.data_source.load_records(self.data_filename.format('train'))
        self.size = 0
        self.buffer_index = 0

    def store(self, history, reward, done):
        index = False
        if history[-1][2] == 0:
            for i in range(len(history)-2, 0, -1):
                if history[i][2] == 1:
                    index = True
                    break
            if index:
                self.events_buf[self.buffer_index] = np.asarray([item[1] for item in history])
                self.times_buf[self.buffer_index] = np.asarray([item[0] for item in history])
                self.marks_buf[self.buffer_index] = np.asarray([item[2] for item in history])

                self.flag_buf[self.buffer_index] = np.zeros_like(self.marks_buf[self.buffer_index])
                self.flag_buf[self.buffer_index][-1] = self.next_state_flag
                self.flag_buf[self.buffer_index][i] = self.action_flag
                self.flag_buf[self.buffer_index][i-1] = self.state_flag

                self.rews_buf[self.buffer_index] = reward
                self.done_buf[self.buffer_index] = done

                self.buffer_index = (self.buffer_index + 1) % self.buffer_size
                self.size = min(self.size + 1, self.buffer_size)

    def sample_and_iterate(self, repeat_num):
        idxs = np.random.randint(0, self.size, size=self.batch_size*repeat_num)
        records = dict(types=[self.events_buf[i] for i in list(idxs)],
                       timestamps=[self.times_buf[i] for i in list(idxs)],
                       flag=[self.flag_buf[i] for i in list(idxs)],
                       marks=[self.marks_buf[i] for i in list(idxs)],
                       rews=self.rews_buf[idxs],
                       done=self.done_buf[idxs])
        records = self.process_input(records)
        for batch_data in yield2batch_data(records, self.batch_size, keep_remainder=True):
            yield batch_data

    def process_input(self, records):
        dt_padding = 0.0
        marks_padding = 1.0

        type_seqs = records['types']
        time_seqs = records['timestamps']
        flag_seq = records['flag']
        marks_seqs = records['marks']

        dt_seqs = [[t_seq[i] - t_seq[max(i - 1, 0)] for i in range(len(t_seq))]
                   for t_seq in time_seqs]

        n_records = len(type_seqs)
        max_len = max([len(seq) for seq in type_seqs])

        # padding
        type_seqs_padded = np.ones([n_records, max_len]) * self.type_padding
        dt_seqs_padded = np.ones([n_records, max_len]) * dt_padding
        flag_seq_padded = np.zeros([n_records, max_len])
        marks_padded = np.ones([n_records, max_len, 1]) * marks_padding

        for i in range(n_records):
            len_seq = len(type_seqs[i])
            type_seqs_padded[i, :len_seq] = type_seqs[i]
            dt_seqs_padded[i, : len_seq] = dt_seqs[i]
            flag_seq_padded[i, : len_seq] = flag_seq[i]
            marks_padded[i, : len_seq, 0] = marks_seqs[i]

        records['types'] = type_seqs_padded
        records['dtimes'] = dt_seqs_padded
        records['flag'] = flag_seq_padded
        records['marks'] = marks_padded

        return records

    def prepare(self, history):
        records = {'dtimes': [item[0] for item in history], 'types': np.asarray([item[1] for item in history]),
                   'marks': np.asarray([item[2] for item in history])}
        records['dtimes'] = np.asarray([records['dtimes'][i] - records['dtimes'][max(i - 1, 0)] for i in range(len(records['dtimes']))])
        records['dtimes'] = np.reshape(records['dtimes'], (1, -1))

        records['types'] = np.reshape(records['types'], (1, -1))
        records['marks'] = np.reshape(records['marks'], (1, -1, 1))
        records['flag'] = np.zeros_like(records['types'])
        records['flag'][0, -1] = self.state_flag
        return records
