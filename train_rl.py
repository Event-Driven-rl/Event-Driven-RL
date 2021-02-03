import yaml
from absl import app
from absl import flags
import tensorflow as tf
import os
import sys
import numpy as np
from env_hawkes_event import hawkes_env

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.split(CUR_PATH)[0]
sys.path.append(ROOT_PATH)
from training.model_runner import ModelRunner
from preprocess.data_loader import DataLoader
from preprocess.data_provider import DataProvider
from preprocess.data_provider_buffer import BufferDataProvider
from lib import get_logger

FLAGS = flags.FLAGS

# Data input params
flags.DEFINE_string('config_filename', 'configs/config.yaml', 'Config file for the models')


def main(argv):
    config_filename = FLAGS.config_filename
    with open(config_filename) as config_file:
        config = yaml.load(config_file)
        data_config = config['data']
        env_config = config['env']
        rl_config = config['rl_train']

    data_loader = DataLoader(data_config)
    train_ds, valid_ds, test_ds = data_loader.get_three_datasource()
    train_dp = DataProvider(train_ds, data_config)
    valid_dp = DataProvider(valid_ds, data_config)
    test_dp = DataProvider(test_ds, data_config)

    env_config['baseline'] = np.array(env_config['baseline'])
    env_config['adjacency'] = np.array(env_config['adjacency'])

    env = hawkes_env(env_config)
    env_test = hawkes_env(env_config)
    buffer_dp = BufferDataProvider(train_ds, data_config, env_config, env.reward_func)

    def test_agent(test_env, buffer_dp, sess, model_runner, env_config, random=False):
        history_test = test_env.reset()
        done_test = 0
        ep_test_ret = 0
        while not done_test:
            if random:
                a_test = np.random.choice(np.arange(0, len(env_config['baseline']) - env_config['candidate_dim'] + 1))
            else: a_test = model_runner.get_action(sess, history_test, buffer_dp, stochastic=False)
            history_test, done_test = test_env.step(a_test)
            reward_test = test_env.reward_func(history_test)
            ep_test_ret += reward_test
        return ep_test_ret

    with tf.Session() as sess:
        model_runner = ModelRunner(config)
        model_runner.train_model(sess, train_dp, valid_dp, test_dp)
        preds, labels, metrics = model_runner.evaluate_model(sess, test_dp)
        print(metrics)
        model_runner.run_target_init(sess)
        history, done = env.reset(), 0
        total_steps = rl_config['steps_per_epoch'] * rl_config['epochs']
        ep_ret, ep_len, result_list = 0, 0, []
        rl_logger = get_logger()
        for t in range(total_steps):
            action = model_runner.get_action(sess, history, buffer_dp, stochastic=True)
            history, done = env.step(action)
            reward = env.reward_func(history)
            if not env.use_emergent:
                buffer_dp.store(history, reward, done)
            ep_ret += reward
            ep_len += 1
            if done == 1:
                history, done = env.reset(), 0
            if t > 0 and t % rl_config['steps_per_epoch'] == 0 and buffer_dp.size > rl_config['steps_per_epoch']:
                model_runner.run_rl_train(sess, buffer_dp, min(rl_config['steps_per_epoch'],buffer_dp.size//data_config['batch_size']), rl_config['model_train_steps_per_epoch'])
                ep_test_ret = test_agent(env_test, buffer_dp, sess, model_runner, env_config)
                result_list.append(ep_test_ret)
                rl_logger.info('Epoch ' + str(t // rl_config['steps_per_epoch']) + ' is finished')
                rl_logger.info('The test result is ' + str(ep_test_ret))


if __name__ == '__main__':
    app.run(main)
