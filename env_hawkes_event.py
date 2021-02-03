import numpy as np

class hawkes_env(object):
    """
    First implementation of the hawkes + rl test env in openai gym style
    """
    def __init__(self, args):
        self.baseline = args['baseline']
        self.adjacency = args['adjacency']
        self.omega = args['omega']
        self.candidate_dim = args['candidate_dim']
        self.event_candidate = list(np.arange(self.candidate_dim))
        self.event_intervene = list(np.arange(start=self.candidate_dim, stop=len(self.baseline)))

        # candidate initial density
        self.baseline_candidate = self.baseline[:self.candidate_dim]

        # intervene initial density
        self.baseline_intervene = self.baseline[self.candidate_dim:]

        self.num_max_event = args.get('num_max_event', np.iinfo(np.int32).max)
        self.dt_intervene = args.get('dt_intervene', 1.0)
        self.dt_remain = self.dt_intervene
        self.horizon = args['horizon']
        self.target_intensity = args['target_intensity']

        self.history = list()
        self.initial_time, self.cur_time, self.prev_time = 0, 0, 0
        self.num_total_event = 0
        self.emergent_event = args['emergent_event']
        self.use_emergent = args['use_emergent']

        self.filter = np.zeros(len(self.baseline_intervene)+1)
        self.filter[-1] = 1
        self.filter_full = np.ones(len(self.baseline_intervene)+1)

        self.done = 0

    def kernel(self, cur_time, prev_time):
        """
        calculates the exponentially decay kernel for difference of x-y with bandwidth b
        """
        return np.exp(-self.omega * (cur_time - prev_time))

    def compute_intensity(self, event_history, cur_time):
        """
        Overwrite intensity function computation
        """
        num_event_history = len(event_history)
        intensity = np.copy(self.baseline)
        for i in range(num_event_history):
            ts, src, _ = event_history[i]
            if src < len(self.baseline):
                intensity = intensity + self.adjacency[src, :] * self.kernel(cur_time, ts)
            else:
                continue
        intensity = np.squeeze(intensity)
        return intensity

    def reset(self):
        """
        reset the simulation and return the history with first event
        :return:
        history: [(time, event_index)]
        """
        self.history = list()
        self.initial_time, self.cur_time, self.prev_time = 0, 0, 0
        self.num_total_event = 0
        self.done = 0
        self.dt_remain = self.dt_intervene

        while self.num_total_event == 0:
            max_cum_intensity = np.sum(self.compute_intensity(self.history, self.prev_time))
            dt_sample = self.draw_exp_rv(max_cum_intensity)
            self.cur_time += dt_sample
            assert (self.cur_time < self.horizon), 'cannot get the initial event when reach the horizon'
            inst_intensity = self.compute_intensity(self.history, self.cur_time)
            cum_inst_intensity = np.sum(inst_intensity)
            d = np.random.uniform()
            if d < (cum_inst_intensity / max_cum_intensity):
                u = self.attribute(d, cum_inst_intensity, inst_intensity)
                self.history.append((self.cur_time, u, 0))
                self.num_total_event += 1
            self.prev_time = self.cur_time

        self.initial_time = self.cur_time
        return self.history

    def step(self, action):
        """
        one unit time step from the previous time step.
        :param action: the event_index of intervene event, e.g. 0,1,2.
        :return:
        history: [(time, event_index),...]
        done: 0/1
        """
        if action < len(self.event_intervene):
            self.history.append((self.cur_time,
                                 self.event_intervene[int(action)], 1))
            self.num_total_event += 1

        self.initial_time = self.cur_time

        max_cum_intensity = np.sum(self.compute_intensity(self.history, self.prev_time))
        dt_sample = self.draw_exp_rv(max_cum_intensity)

        while dt_sample < self.dt_remain:
            self.cur_time += dt_sample
            inst_intensity = self.compute_intensity(self.history, self.cur_time)
            cum_inst_intensity = np.sum(inst_intensity)
            d = np.random.uniform()
            if d < (cum_inst_intensity / max_cum_intensity):
                u = self.attribute(d, cum_inst_intensity, inst_intensity)
                self.history.append((self.cur_time, u, 0))
                self.num_total_event += 1
            self.prev_time = self.cur_time
            self.dt_remain -= dt_sample

            max_cum_intensity = np.sum(self.compute_intensity(self.history, self.prev_time))
            dt_sample = self.draw_exp_rv(max_cum_intensity)

        self.dt_remain = self.dt_intervene
        self.cur_time = self.initial_time + self.dt_intervene
        self.prev_time = self.cur_time

        if self.cur_time > self.horizon or self.num_total_event >= self.num_max_event:
            self.done = 1

        return self.history, self.done

    def reward_func(self, history):
        intensity = self.compute_intensity(history, history[-1][0])
        result_list = [abs(i-j) for i, j in zip(intensity[0:len(self.target_intensity)], self.target_intensity)]
        return -sum(result_list)

    @staticmethod
    def draw_exp_rv(param):
        """
        Return exp random variable
        """
        # using the built-in numpy function
        return np.random.exponential(scale=param)

    @staticmethod
    def attribute(uniform_rv, i_star, mlambda):
        # this is the recommended
        S = np.cumsum(mlambda) / i_star
        return (uniform_rv > S).sum()