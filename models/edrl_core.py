import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from lib.tf_utils import Attention
from enum import Enum

EPS = 1e-8
TEMPERATURE = 0.1


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


class EDRL:
    class TrainValidTest(Enum):
        TRAIN = 0
        VALID = 1
        TEST = 2
        @staticmethod
        def to_string(train_valid_test):
            if train_valid_test == EDRL.TrainValidTest.TRAIN:
                return "Train"
            elif train_valid_test == EDRL.TrainValidTest.VALID:
                return "Valid"
            elif train_valid_test == EDRL.TrainValidTest.TEST:
                return "Test"
            else:
                return "UNKNOW_TYPE"

    def __init__(self, model_config):
        # get hyper-parameters from configs
        self.process_dim = model_config['process_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.candicate_dim = model_config['candidate_dim']
        self.output_dim = model_config['process_dim'] - model_config['candidate_dim'] + 1
        self.polyak = model_config['polyak']

        self.state_flag = model_config['state_flag']
        self.action_flag = model_config['action_flag']
        self.next_state_flag = model_config['next_state_flag']
        self.use_filter = model_config['use_filter']

        self.q_net = model_config['q_net']
        self.pi_net = model_config['pi_net']
        self.v_net = model_config['v_net']

        with tf.variable_scope('model_input'):
            self.learning_rate = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)
            self.types_seq = tf.placeholder(tf.int32, shape=[None, None])
            self.dtimes_seq = tf.placeholder(tf.float32, shape=[None, None])
            self.flag_seq = tf.placeholder(tf.float32, shape=[None, None])
            self.marks_seq = tf.placeholder(tf.float32, shape=[None, None, 1])

            self.reward = tf.placeholder(tf.float32, shape=[None])
            self.done = tf.placeholder(tf.float32, shape=[None])
            self.alpha = tf.get_variable('alpha', dtype=tf.float32, initializer=model_config['alpha'])
            self.gamma = tf.get_variable('gamma', dtype=tf.float32, initializer=model_config['gamma'])

            self.types_seq_one_hot = tf.one_hot(self.types_seq, self.process_dim)
            # EOS padding type is all zeros in the last dim of the tensor
            self.seq_mask = tf.reduce_sum(self.types_seq_one_hot, axis=-1) > 0
            marks_mask = tf.squeeze(self.marks_seq, axis=-1)
            marks_mask = marks_mask < 1
            self.seq_mask = tf.math.logical_and(self.seq_mask, marks_mask)

        with tf.variable_scope('EDRL'):
            # 1. Embedding of input
            emb = self.emb_layer(self.types_seq, self.marks_seq)
            # 2. Intensity layer
            # shape -> [batch_size, max_len, process_dim]
            lambdas, h_states = self.intensity_layer(emb, self.dtimes_seq)

        s = tf.gather_nd(h_states, tf.where(tf.equal(self.flag_seq, self.state_flag)))
        next_s = tf.gather_nd(h_states, tf.where(tf.equal(self.flag_seq, self.next_state_flag)))

        time_cum = tf.cumsum(self.dtimes_seq, axis=1)
        time = tf.gather_nd(time_cum, tf.where(tf.equal(self.flag_seq, self.state_flag)))
        time_next = tf.gather_nd(time_cum, tf.where(tf.equal(self.flag_seq, self.next_state_flag)))
        time = tf.reshape(time, [-1])
        time_next = tf.reshape(time_next, [-1])
        time = time_next - time

        a_event = tf.gather_nd(self.types_seq, tf.where(tf.equal(self.flag_seq, self.action_flag)))
        a = tf.one_hot(a_event - self.candicate_dim, self.output_dim)

        with tf.variable_scope('pi'):
            mu, pi, diff_pi, action_probs, log_action_probs = self.mlp_gaussian_policy(s, self.output_dim)

        with tf.variable_scope('pi', reuse=True):
            mu_filter, pi_filter, _, _, _ = self.mlp_gaussian_policy(s, self.output_dim)

        with tf.variable_scope('main'):
            q1_pi, q2_pi, q1_a, q2_a, v = self.mlp_actor_critic(s, a, diff_pi)

        self.deterministic_action = mu
        self.stochastic_action = pi

        r_m, r_m_pi, v_prime, t_m, t_m_pi = self.reward_model(s, a, diff_pi, time)

        with tf.variable_scope('target'):
            _, _, _, _, v_targ = self.mlp_actor_critic(next_s, a, diff_pi)

        # Targets for Q/V regression
        min_q_pi = tf.minimum(q1_pi, q2_pi)
        q_backup = tf.stop_gradient(((1 - tf.exp(-self.gamma*time))/self.gamma)*self.reward + tf.exp(-self.gamma*time) * (1 - self.done) * v_targ)
        v_backup = tf.stop_gradient(tf.reduce_sum(action_probs * (-self.alpha * log_action_probs), axis=-1) + min_q_pi)

        # critic losses
        q1_loss = 0.5 * tf.reduce_mean((q_backup - q1_a) ** 2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - q2_a) ** 2)
        v_loss = 0.5 * tf.reduce_mean((v_backup - v) ** 2)
        value_loss = q1_loss + q2_loss + v_loss

        # models loss
        r_loss = 0.5 * tf.reduce_mean((r_m - self.reward) ** 2)
        t_loss = 0.5 * tf.reduce_mean((t_m - next_s) ** 2)
        model_loss = r_loss + t_loss

        # policy loss
        pi_backup = tf.reduce_sum(action_probs * self.alpha * log_action_probs, axis=-1) - ((1 - tf.exp(-self.gamma*time))/self.gamma) * r_m_pi - tf.exp(-self.gamma*time) * (1 - self.done) * v_prime
        pi_loss = tf.reduce_mean(pi_backup)

        self.pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.model_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.train_pi_op = self.pi_optimizer.minimize(pi_loss, var_list=get_vars('pi'))
        with tf.control_dependencies([self.train_pi_op]):
            self.train_value_op = self.value_optimizer.minimize(value_loss, var_list=get_vars('main/q') + get_vars('main/v') + get_vars('EDRL'))
        with tf.control_dependencies([self.train_value_op]):
            self.train_model_op = self.model_optimizer.minimize(model_loss, var_list=get_vars('rm') + get_vars('tm'))

        with tf.control_dependencies([self.train_value_op]):
            self.target_update = tf.group([tf.assign(v_targ, self.polyak * v_targ + (1 - self.polyak) * v_main)
                                      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        self.step_ops = [pi_loss, q1_loss, q2_loss, q1_a, q2_a, self.train_pi_op, self.train_value_op, self.target_update,
                    self.train_model_op]

        self.target_init = tf.group(
            [tf.assign(v_targ, v_main) for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # 3. Inference layer and loss function
        # [batch_size, max_len, process_dim], [batch_size, max_len]
        pred_type_logits, pred_time = self.hybrid_inference(lambdas)
        self.loss = self.hybrid_loss(pred_type_logits, pred_time)

        # 4. train step
        self.opt = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.opt.minimize(self.loss)

        # assign prediction
        # shape -> [batch_size, max_len]
        self.pred_types = tf.argmax(pred_type_logits, axis=-1)
        # shape -> [batch_size, max_len]
        self.pred_time = pred_time

        self.train_valid_test = EDRL.TrainValidTest.TRAIN
        self.loss_summ_list = [tf.summary.scalar('loss', self.loss),
                                   tf.summary.scalar('valid_loss', self.loss),
                                   tf.summary.scalar('test_loss', self.loss)]
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()

    def mlp_actor_critic(self, h_input, a, diff_pi):

        x_a = tf.concat([h_input, a], -1)
        x_pi = tf.concat([h_input, diff_pi], -1)

        with tf.variable_scope('q1'):
            q1_a = self.mlp(x_a, self.q_net, tf.nn.relu, None, True)

        with tf.variable_scope('q1', reuse=True):
            q1_pi = self.mlp(x_pi, self.q_net, tf.nn.relu, None, True)

        with tf.variable_scope('q2'):
            q2_a = self.mlp(x_a, self.q_net, tf.nn.relu, None, True)

        with tf.variable_scope('q2', reuse=True):
            q2_pi = self.mlp(x_pi, self.q_net, tf.nn.relu, None, True)

        with tf.variable_scope('v'):
            v = self.mlp(h_input, self.v_net, tf.nn.relu, None, True)

        return q1_pi, q2_pi, q1_a, q2_a, v

    def reward_model(self, h_input, a, diff_pi, time):
        time = tf.expand_dims(time, axis=1)

        x_a = tf.concat([h_input, a, time], -1)
        x_pi = tf.concat([h_input, diff_pi, time], -1)

        transition_dim = h_input.get_shape().as_list()[-1]

        with tf.variable_scope('tm'):
            t_m = self.mlp(x_a, self.q_net[:-1] + [transition_dim], tf.nn.relu, None, False)

        with tf.variable_scope('tm', reuse=True):
            t_m_pi = self.mlp(x_pi, self.q_net[:-1] + [transition_dim], tf.nn.relu, None, False)

        with tf.variable_scope('rm'):
            r_m = self.mlp(x_a, self.q_net, tf.nn.relu, None, True)

        with tf.variable_scope('rm', reuse=True):
            r_m_pi = self.mlp(x_pi, self.q_net, tf.nn.relu, None, True)

        with tf.variable_scope('main/v', reuse=True):
            v_prime = self.mlp(t_m_pi, self.v_net, tf.nn.relu, None, True)

        return r_m, r_m_pi, v_prime, t_m, t_m_pi

    def emb_layer(self, types_seq, marks_seq):
        type_seq_emb = self.type_embedding_layer(types_seq)
        marks_emb = self.marks_embedding_layer(marks_seq)
        pos_encoding = self.pos_encoding_layer(types_seq)
        emb = type_seq_emb + pos_encoding
        emb = tf.concat([emb, marks_emb], -1)
        emb = self.mlp(emb, [128, self.hidden_dim], tf.nn.relu, None)
        return emb

    def type_embedding_layer(self, types_seq):
        """ Equation (7) """
        # add 1 dim because of EOS padding
        emb_layer = layers.Embedding(
            self.process_dim + 2, self.hidden_dim, name='type_embedding')
        # shape -> [batch_size, max_len, hidden_dim]
        emb = emb_layer(types_seq)
        return emb

    def marks_embedding_layer(self, marks_seq):
        emb_layer = layers.Embedding(
            2, self.hidden_dim, name='marks_embedding')
        marks_seq = tf.squeeze(marks_seq, -1)
        emb = emb_layer(marks_seq)
        return emb

    def pos_encoding_layer(self, types_seq):
        """
        Equation (8)
        Use the same positional encoding formula as in Transformer
        https://github.com/Kyubyong/transformer/blob/master/modules.py
        """
        batch_size, max_len = tf.shape(types_seq)[0], tf.shape(types_seq)[1]  # dynamic

        # shape -> [batch_size, max_len]
        position_ind = tf.tile(tf.expand_dims(tf.range(max_len), 0), [batch_size, 1])

        # First part of the PE function: sin and cos argument
        PREDEFINED_MAX_LENGTH = 1000
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / self.hidden_dim) for i in range(self.hidden_dim)]
            for pos in range(PREDEFINED_MAX_LENGTH)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # shape -> [PREDEFINED_MAX_LENGTH, hidden_dim]
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)

        # lookup
        # shape -> [batch_size, max_len, hidden_dim]
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # shape -> [batch_size, max_len, hidden_dim]
        return tf.to_float(outputs)

    @staticmethod
    def gelu(x):
        cdf = 0.5 * (1.0 + tf.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf

    def mlp(self, x,  hidden_sizes=(32,), activation=tf.tanh, output_activation=None, squeeze=False):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(x, units=h, activation=activation)
        if squeeze:
            x = tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)
            x = tf.squeeze(x, -1)
            return x
        else:
            return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

    def attention_layers(self, x_input, num_heads):
        attention_layer = Attention(self.hidden_dim, 'exp_dot')
        q_ = tf.layers.dense(x_input, self.hidden_dim * num_heads, activation=tf.nn.tanh)
        k_ = tf.layers.dense(x_input, self.hidden_dim * num_heads, activation=tf.nn.tanh)
        v_ = tf.layers.dense(x_input, self.hidden_dim * num_heads, activation=None)
        context_vector, _ = attention_layer.compute_attention_weight(q_,
                                                                     k_,
                                                                     v_,
                                                                     num_heads=num_heads,
                                                                     drop_proba=0.0,
                                                                     pos_mask='right')
        return context_vector

    def intensity_layer(self, x_input, dtimes_seq, reuse=tf.AUTO_REUSE):
        num_heads = 2
        with tf.variable_scope('intensity_layer', reuse=reuse):
            x_input = self.attention_layers(x_input, num_heads)
            x_input = self.attention_layers(x_input, num_heads)
            etas = self.mlp(x_input, [128, self.process_dim], tf.nn.relu, None)
            decays = self.mlp(x_input, [128, self.process_dim], tf.nn.relu, None)
            mus = self.mlp(x_input, [128, self.process_dim], tf.nn.relu, None)
            lambdas = mus + (etas - mus) * tf.exp(-decays * dtimes_seq[:, :, None])
        return lambdas, x_input

    def mlp_gaussian_policy(self, h_input, output_dim):
        h_input = self.mlp(h_input, self.pi_net + [output_dim], tf.nn.relu, None)
        softmax = tf.nn.softmax(h_input, axis=-1)
        log_action_probs = tf.nn.log_softmax(h_input, axis=-1)
        mu = tf.math.argmax(softmax, 1)
        gumbel_dist = tf.contrib.distributions.RelaxedOneHotCategorical(TEMPERATURE, logits=h_input)
        diff_pi = gumbel_dist.sample()
        policy_dist = tf.distributions.Categorical(probs=softmax)
        pi = policy_dist.sample()
        return mu, pi, diff_pi, softmax, log_action_probs

    def hybrid_inference(self, lambdas, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('hybrid_inference', reuse=reuse):
            pred_type_logits = self.mlp(lambdas, [128, self.process_dim], tf.nn.relu, None)
            pred_time = self.mlp(lambdas, [128, 1], tf.nn.relu, None)
            pred_time = tf.squeeze(pred_time, axis=-1)
        return pred_type_logits, pred_time

    def hybrid_loss(self, pred_type_logits, pred_times):
        label_types = self.types_seq_one_hot
        label_times = self.dtimes_seq
        seq_mask = self.seq_mask
        with tf.variable_scope('shuffle_hybrid_loss'):
            # shape -> [batch_size, max_len - 1]
            seq_mask = seq_mask[:, 1:]
            # (batch_size, max_len - 1, process_dim)
            type_label = label_types[:, 1:]
            pred_type_logits = pred_type_logits[:, :-1]

            # (batch_size, max_len - 1)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=type_label, logits=pred_type_logits)
            type_loss = tf.reduce_mean(tf.boolean_mask(cross_entropy, seq_mask))

            dtimes_pred = pred_times[:, :-1]
            dtimes_label = label_times[:, 1:]

            time_diff = tf.boolean_mask((dtimes_pred - dtimes_label), seq_mask)
            time_loss = tf.reduce_mean(tf.abs(time_diff))

        return type_loss + time_loss

    def train(self, sess, batch_data, **kwargs):
        type_seqs = batch_data['types']
        dtime_seqs = batch_data['dtimes']
        marks_seqs = batch_data['marks']

        lr = kwargs.get('lr')
        fd = {
            self.types_seq: type_seqs,
            self.dtimes_seq: dtime_seqs,
            self.learning_rate: lr,
            self.marks_seq: marks_seqs
        }

        _, loss, pred_types, pred_time = sess.run([self.train_op, self.loss, self.pred_types, self.pred_time],
                                                  feed_dict=fd)

        # shape -> [batch_size, max_len - 1]
        preds = {
            'types': pred_types[:, :-1],
            'dtimes': pred_time[:, :-1]
        }
        labels = {
            'types': type_seqs[:, 1:],
            'dtimes': dtime_seqs[:, 1:],
            'marks': marks_seqs[:, 1:]
        }
        return loss, preds, labels

    def predict(self, sess, batch_data, **kwargs):
        type_seqs = batch_data['types']
        dtime_seqs = batch_data['dtimes']
        marks_seqs = batch_data['marks']
        fd = {
            self.types_seq: type_seqs,
            self.dtimes_seq: dtime_seqs,
            self.marks_seq: marks_seqs
        }
        loss, pred_types, pred_time = sess.run([self.loss, self.pred_types, self.pred_time],
                                               feed_dict=fd)

        # shape -> [batch_size, max_len - 1]
        preds = {
            'types': pred_types[:, :-1],
            'dtimes': pred_time[:, :-1]
        }
        labels = {
            'types': type_seqs[:, 1:],
            'dtimes': dtime_seqs[:, 1:],
            'marks': marks_seqs[:, 1:]
        }
        return loss, preds, labels

    def run_target_init(self, sess):
        sess.run(self.target_init)

    def rl_train(self, sess, batch_data, **kwargs):
        type_seqs = batch_data['types']
        dtime_seqs = batch_data['dtimes']
        marks_seqs = batch_data['marks']
        flag_seqs = batch_data['flag']
        lr = kwargs.get('lr')

        reward = batch_data['rews']
        done = batch_data['done']

        # reward = np.ones(np.shape(type_seqs)[0])
        # done = np.zeros(np.shape(type_seqs)[0])

        fd = {
            self.types_seq: type_seqs,
            self.dtimes_seq: dtime_seqs,
            self.marks_seq: marks_seqs,
            self.reward: reward,
            self.done: done,
            self.learning_rate: lr,
            self.flag_seq: flag_seqs
        }

        loss, _, _, _, _ = sess.run([self.loss, self.train_pi_op, self.train_model_op, self.train_value_op, self.target_update], feed_dict=fd)
        # sess.run(self.print_op, feed_dict=fd)

        return loss

    def get_action(self, sess, batch_data, stochastic=True):
        type_seqs = batch_data['types']
        dtime_seqs = batch_data['dtimes']
        marks_seqs = batch_data['marks']
        flag_seqs = batch_data['flag']
        action_filter = np.ones((np.shape(batch_data['types'])[0], self.output_dim))
        fd = {
            self.types_seq: type_seqs,
            self.dtimes_seq: dtime_seqs,
            self.marks_seq: marks_seqs,
            self.flag_seq: flag_seqs,
        }

        stochastic_action, deterministic_action = sess.run([self.stochastic_action, self.deterministic_action],
                                                            feed_dict=fd)
        if stochastic:
            return stochastic_action
        else: return deterministic_action

    def predict_next(self, sess, batch_data, **kwargs):
        type_seqs = batch_data['types']
        dtime_seqs = batch_data['dtimes']
        marks_seqs = batch_data['marks']
        fd = {
            self.types_seq: type_seqs,
            self.dtimes_seq: dtime_seqs,
            self.marks_seq: marks_seqs
        }
        pred_types, pred_time = sess.run([self.pred_types, self.pred_time], feed_dict=fd)
        pred_types = pred_types[0, -1]
        pred_time = pred_time[0, -1]
        return pred_types, pred_time
