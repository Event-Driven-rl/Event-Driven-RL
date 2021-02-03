"""
Model train, validation and prediction pipeline
"""

import os
import datetime

import numpy as np
import tensorflow as tf
import yaml

from lib import get_num_trainable_params, make_config_string, \
    create_folder, concat_arrs_of_dict, Timer, get_logger
from training.lr_scheduler import LRScheduler
from models.edrl_core import EDRL

class ModelRunner(object):
    """
    Train, evaluate, save and restore models.
    """

    def __init__(self, config):
        self.config = config
        self.train_config = config['train']
        self.model_config = config['model']
        self.data_config = config['data']
        self.tensorboard = config.get('tensorboard')

        # the folders for models and tensor board
        self.training_folder = None
        self.model_folder = None
        self.tfb_folder = None

        # build models
        self.model = EDRL(self.model_config)
        self.model_saver = tf.train.Saver(max_to_keep=0)

        # other setting
        self.timer = Timer('m')

        # the tensorboard record writer
        self.write_summary = self.tensorboard['write_summary']
        self.tensorboard_writer = None

    def train_model(self, sess,
                    train_data_provider, valid_data_provider, test_data_provider=None):

        epoch_num, max_epoch, lr_scheduler, continue_training = self._load_train_status()

        # get logger for this training
        logger = get_logger(os.path.join(self.training_folder, 'training.log'))

        # training from scratch or continue training
        if continue_training:
            # trained models existed, then restore it.
            model_path = self.restore_model(sess)
            epoch_num += 1
            logger.info(f'Restore models from {model_path}')
        else:
            # initialize variables
            sess.run([tf.global_variables_initializer()])

        logger.info(f'Training starts on dataset {self.data_config["data_name"]}')
        logger.info(f'----------Trainable parameter count: {get_num_trainable_params()} of models {self.model_folder}')

        best_valid_loss = float('inf')
        lr = lr_scheduler.get_lr()
        while lr > 0 and epoch_num <= max_epoch:

            # Train
            loss, _, _, elapse = self.run_one_epoch(sess,
                                                    train_data_provider,
                                                    lr,
                                                    is_train=EDRL.TrainValidTest.TRAIN,
                                                    epoch_num=epoch_num)
            logger.info(f'Epoch {epoch_num}: train loss - {loss}, learning rate - {lr}. Cost time: {elapse:.3f}m')

            # Update after train
            # update lr
            lr = lr_scheduler.update_lr(loss=loss, epoch_num=epoch_num)
            # update train_config
            self.update_train_config(lr, epoch_num)

            # Valid
            valid_loss, _, _, _ = self.run_one_epoch(sess,
                                                     valid_data_provider,
                                                     lr,
                                                     is_train=EDRL.TrainValidTest.VALID,
                                                     epoch_num=epoch_num)

            # save the best models on valid dataset.
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                # save best models
                self.save_model_with_config(sess)

                # Test
                test_loss, preds, labels, elapse = self.run_one_epoch(sess,
                                                                      test_data_provider,
                                                                      lr,
                                                                      is_train=EDRL.TrainValidTest.TEST,
                                                                      epoch_num=epoch_num)
                metrics = test_data_provider.get_metrics(preds, labels)
                str_metrics = str(metrics)
                logger.info(f'---Test Loss: {loss}, metrics: {str_metrics}. Cost time: {elapse:.3f}m')

            epoch_num += 1
        logger.info('Training Finished!')

    def evaluate_model(self, sess, data_provider):
        """ Evaluate the models """
        self.restore_model(sess)
        loss, preds, labels, elapse = self.run_one_epoch(sess,
                                                         data_provider,
                                                         lr=0,
                                                         is_train=EDRL.TrainValidTest.VALID)
        metrics = data_provider.get_metrics(preds, labels)
        return preds, labels, metrics

    def model_predict(self, sess, data_provider):
        """ Get prediction using models """
        self.restore_model(sess)
        loss, preds, labels, elapse = self.run_one_epoch(sess,
                                                         data_provider,
                                                         lr=0,
                                                         is_train=EDRL.TrainValidTest.TEST)
        return preds

    def restore_model(self, sess):
        """ Restore models from checkpoints """
        train_config = self.train_config
        model_path = train_config['model_path']
        self.model_saver.restore(sess, model_path)
        return model_path

    def _load_train_status(self):
        """ Load training status. Create base folders if the configs presents a new training.
        :return:
        """
        train_config = self.train_config
        # assign parameters
        epoch_num = train_config.get('epoch')
        max_epoch = train_config.get('max_epoch')

        # get lr scheduler
        lr_scheduler = LRScheduler.generate_scheduler_by_name(train_config.get('lr_scheduler'), **train_config)
        model_path = train_config.get('model_path')

        if model_path:
            # continue last training
            continue_training = True
            # read corresponding training path
            self.model_folder = os.path.dirname(model_path)
            self.training_folder = os.path.dirname(self.model_folder)
            self.tfb_folder = create_folder(self.training_folder, 'tfbs')

        else:
            # training from scratch
            continue_training = False
            # create models and tensorflow board folder
            time = datetime.datetime.now()
            timestamp = datetime.datetime.strftime(time, '%m%d%H%M%S')
            model_foldername = make_config_string(self.config['model']) + '_' + timestamp

            self.training_folder = create_folder(self.config['base_dir'], model_foldername)
            self.model_folder = create_folder(self.training_folder, 'models')
            self.tfb_folder = create_folder(self.training_folder, 'tfbs')

        return epoch_num, max_epoch, lr_scheduler, continue_training

    def save_model_with_config(self, sess):
        train_config = self.train_config
        # update models path in train configs
        train_config['model_path'] = os.path.join(self.model_folder, 'models-' + str(train_config['epoch']))

        # save models
        self.model_saver.save(sess, train_config['model_path'])
        # save configs to yaml file
        config_path = os.path.join(self.model_folder, 'configs-' + str(train_config['epoch']) + '.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)

        if self.write_summary and not self.tensorboard_writer:
            self.tensorboard_writer = tf.summary.FileWriter(train_config['model_path'])
            self.tensorboard_writer.add_graph(sess.graph)

    def update_train_config(self, lr, epoch):
        train_config = self.train_config
        train_config['lr'] = lr
        train_config['epoch'] = epoch

    def run_one_epoch(self, sess, data_provider, lr, is_train, epoch_num=0):
        """
        :param sess:
        :param data_provider:
        :param lr:
        :param is_train: TrainValidTest
        :return: [epoch_loss, epoch_pred, epoch_label, epoch_cost_time]
        """
        self.timer.start()
        model = self.model
        model.train_valid_test = is_train
        if is_train == EDRL.TrainValidTest.TRAIN:
            run_func = model.train
        elif is_train == EDRL.TrainValidTest.VALID:
            run_func = model.predict
        else:
            run_func = model.predict

        loss_list = []
        pred_list = []
        real_list = []
        for i, batch_data in enumerate(data_provider.iterate_batch_data()):
            loss, pred, real = run_func(sess, batch_data, lr=lr)

            loss_list.append(loss)
            pred_list.append(pred)
            real_list.append(real)

        # shape -> [n_items, horizon, D]
        epoch_preds = concat_arrs_of_dict(pred_list)
        epoch_reals = concat_arrs_of_dict(real_list)

        epoch_avg_loss = np.mean(loss_list)
        if self.write_summary and self.tensorboard_writer:
            if is_train == EDRL.TrainValidTest.TRAIN:
                print("Writing meta data %d"% epoch_num)
                self.tensorboard_writer.add_run_metadata(self.model.run_metadata, 'step %d' % epoch_num)
            summ = tf.Summary()
            summ.value.add(tag="Ave Loss %s"% EDRL.TrainValidTest.to_string(is_train), simple_value=epoch_avg_loss)
            self.tensorboard_writer.add_summary(summ, epoch_num)
            self.tensorboard_writer.flush()
            print("  add_summary", loss, epoch_num, is_train)
        # inverse scaling data
        epoch_preds = data_provider.epoch_inverse_scaling(epoch_preds)
        epoch_reals = data_provider.epoch_inverse_scaling(epoch_reals)

        return epoch_avg_loss, epoch_preds, epoch_reals, self.timer.end()

    def run_target_init(self, sess):
        self.model.run_target_init(sess)

    def run_global_init(self, sess):
        sess.run(tf.global_variables_initializer())

    def run_part_init(self, sess, name):
        params = [v for v in tf.global_variables() if name in v.name]
        sess.run(tf.variables_initializer(var_list=params))

    def get_action(self, sess, current_history, buffer_provider, stochastic):
        batch_data = buffer_provider.prepare(current_history)
        action = self.model.get_action(sess, batch_data, stochastic=stochastic)
        return action

    def run_rl_train(self, sess, buffer_dp, repeat_num, model_repeat_num):
        for batch_data in buffer_dp.sample_and_iterate(repeat_num):
            _ = self.model.rl_train(sess, batch_data, lr=self.train_config['lr'])
        for batch_data in buffer_dp.sample_and_iterate(model_repeat_num):
            _ = self.model.train(sess, batch_data, lr=self.train_config['lr'])