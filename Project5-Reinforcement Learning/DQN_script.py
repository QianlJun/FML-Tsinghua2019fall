#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: QianJun
# File Name: DQN_script.py
# Created Time:Sat 11 Jan 2020 10:24:45 PM CST

from __future__ import division
from __future__ import print_function
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color
import skimage.transform
import tensorflow as tf
from tqdm import trange
import vizdoom as vzd
from argparse import ArgumentParser
import pandas as pd

# Q-learning settings
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 100
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64
# How often to perform a training step.
# update_freq = 4

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
# frame_repeat = 4
resolution = (60, 108)
episodes_to_watch = 1

# for soft update of target parameters
tau = 0.001
# tau = 1

save_model = True
load_model = True
skip_learning = True

# Configuration file path
# DEFAULT_MODEL_SAVEFILE = "save/model_dqn_1"  # model1: tau=1
# DEFAULT_MODEL_SAVEFILE = "save/model_dqn_2"  # model2: tau=0.001
# DEFAULT_HISTORY_SAVEFILE = "logs/model_dqn_2.csv"
# DEFAULT_HISTORY_SAVEFILE = "logs/model_dqn_map1_6.csv"
# DEFAULT_MODEL_SAVEFILE = "save/model_dqn_map1_6.ckpt"  # model2: tau=0.001
# DEFAULT_HISTORY_SAVEFILE = "logs/model_dqn_map1_6_2.csv"  #
# DEFAULT_MODEL_SAVEFILE = "save/model_dqn_map1_6_2.ckpt"  # model: uodate_freq=4, frame_repeat=4
# DEFAULT_HISTORY_SAVEFILE = "logs/model_dqn_mapdfc_6_1.csv"  #
# DEFAULT_MODEL_SAVEFILE = "save/model_dqn_mapdfc_6_1.ckpt"  # model: uodate_freq=1, frame_repeat=12,50epoch+time
# DEFAULT_HISTORY_SAVEFILE = "logs/model_dqn_mapdfc_6_2.csv"  #
# DEFAULT_MODEL_SAVEFILE = "save/model_dqn_mapdfc_6_2.ckpt"  # model: uodate_freq=1, frame_repeat=12, 200epoch+time
# DEFAULT_HISTORY_SAVEFILE = "logs/model_dqn_map2_6_2.csv"  #
DEFAULT_MODEL_SAVEFILE = "restore/model_dqn_map2_6_2.ckpt-100"  # model: uodate_freq=1, frame_repeat=12,100epoch


# DEFAULT_CONFIG = "D:/software/Anaconda3/envs/python-3.7.4/Lib/site-packages/vizdoom/scenarios/simpler_basic.cfg"
# DEFAULT_CONFIG = "D:/software/Anaconda3/envs/python-3.7.4/Lib/site-packages/vizdoom/scenarios/basic_6action.cfg"
DEFAULT_CONFIG = "D:/software/Anaconda3/envs/python-3.7.4/Lib/site-packages/vizdoom/scenarios/test_map2.cfg"

# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


class Qnetwork:
    def __init__(self, available_actions_count):
        # Create the input variables
        self.s1_ = tf.placeholder(tf.float32, [None] + list(resolution) + [1], name="State")
        self.a_ = tf.placeholder(tf.int32, [None], name="Action")
        self.target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

        # Add 2 convolutional layers with ReLu activation
        self.conv1 = tf.contrib.layers.convolution2d(self.s1_,
                                                     num_outputs=8,
                                                     kernel_size=[6, 6],
                                                     stride=[3, 3],
                                                     activation_fn=tf.nn.relu,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                     biases_initializer=tf.constant_initializer(0.1))
        self.conv2 = tf.contrib.layers.convolution2d(self.conv1,
                                                     num_outputs=8,
                                                     kernel_size=[3, 3],
                                                     stride=[2, 2],
                                                     activation_fn=tf.nn.relu,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                     biases_initializer=tf.constant_initializer(0.1))
        self.conv2_flat = tf.contrib.layers.flatten(self.conv2)
        self.fc1 = tf.contrib.layers.fully_connected(self.conv2_flat,
                                                     num_outputs=128,
                                                     activation_fn=tf.nn.relu,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                     biases_initializer=tf.constant_initializer(0.1))
        self.q = tf.contrib.layers.fully_connected(self.fc1,
                                                   num_outputs=available_actions_count,
                                                   activation_fn=None,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                   biases_initializer=tf.constant_initializer(0.1))
        self.best_a = tf.argmax(self.q, 1)

        self.loss = tf.losses.mean_squared_error(self.q, self.target_q_)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        # Update the parameters according to the computed gradient using RMSProp.
        self.train_step = self.optimizer.minimize(self.loss)


def exploration_rate(epoch):
    """# Define exploration rate change over time"""
    start_eps = 1.0
    end_eps = 0.1
    const_eps_epochs = 0.1 * epochs  # 10% of learning time
    eps_decay_epochs = 0.6 * epochs  # 60% of learning time

    if epoch < const_eps_epochs:
        return start_eps
    elif epoch < eps_decay_epochs:
        # Linear decay
        return start_eps - (epoch - const_eps_epochs) / \
               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
    else:
        return end_eps


# These functions allow us to update the parameters of our target network
# Soft update model parameters.
# θ_target = tau*θ_eval + (1 - tau)*θ_target
def updateTargetGraph(params, tau):
    total_params = len(params)
    op_holder = []
    for index, variable in enumerate(params[0:total_params // 2]):
        op_holder.append(params[index + total_params // 2].assign(
            (variable.value() * tau) + ((1 - tau) * params[index + total_params // 2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X360)
    game.init()
    print("Doom initialized.")
    return game


if __name__ == '__main__':
    parser = ArgumentParser("ViZDoom example showing how to train a simple agent using simplified DQN.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")

    args = parser.parse_args()

    # Create Doom instance
    game = initialize_vizdoom(args.config)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Create evaluation_Q-network and target_Q-network
    eval_QN = Qnetwork(len(actions))
    target_QN = Qnetwork(len(actions))

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    session = tf.Session()

    saver = tf.train.Saver()
    # init = tf.global_variables_initializer()
    # params = tf.trainable_variables()
    # targetOps = updateTargetGraph(params, tau)

    # session.run(init)
    if load_model:
        print("Loading model from: ", DEFAULT_MODEL_SAVEFILE)
        saver.restore(session, DEFAULT_MODEL_SAVEFILE)
    else:
        init = tf.global_variables_initializer()
        session.run(init)
    print("Starting the training!")
    params = tf.trainable_variables()
    targetOps = updateTargetGraph(params, tau)
    time_start = time()

    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []
            total_steps = 0

            print("Training...")
            game.new_episode()
            for learning_step in trange(learning_steps_per_epoch, leave=False):
                s1 = preprocess(game.get_state().screen_buffer)
                eps = exploration_rate(epoch)
                if random() <= eps:
                    a = randint(0, len(actions) - 1)
                else:
                    # Choose the best action according to the network.
                    a = session.run(eval_QN.best_a,
                                    feed_dict={eval_QN.s1_: s1.reshape([1, resolution[0], resolution[1], 1])})[0]
                reward = game.make_action(actions[a], frame_repeat)
                isterminal = game.is_episode_finished()
                s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None
                # Remember the transition that was just experienced.
                memory.add_transition(s1, a, s2, isterminal, reward)
                total_steps += 1

                # learn_from_memory()
                """ Learns from a single transition (making use of replay memory).
                s2 is ignored if s2_isterminal """
                # Get a random minibatch from the replay memory and learns from it.
                if memory.size > batch_size:
                    # if memory.size > batch_size and total_steps % update_freq == 0:
                    s1, a, s2, isterminal, r = memory.get_sample(batch_size)

                    q2 = np.max(session.run(target_QN.q, feed_dict={target_QN.s1_: s2}), axis=1)
                    target_q = session.run(target_QN.q, feed_dict={target_QN.s1_: s1})
                    # target differs from q only for the selected action. The following means:
                    # target_Q(s,a) = r + gamma * max Q(s2,_) if not isterminal else r
                    target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2

                    # Update the network with our target values.
                    session.run(eval_QN.train_step, feed_dict={eval_QN.s1_: s1,
                                                               eval_QN.target_q_: target_q})
                    # Update the target network toward the primary network.
                    updateTarget(targetOps, session)

                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()),
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
            # print([epoch+1, train_scores.mean(), round(train_scores.std(), 2), train_scores.min(),
            # train_scores.max()]) save training history with open(DEFAULT_HISTORY_SAVEFILE, 'wb') as f: writer =
            # csv.writer(f) writer.writerows([epoch+1, train_scores.mean(), round(train_scores.std(), 2),
            # train_scores.min(), train_scores.max()])

            # save log file
            log_file = DEFAULT_HISTORY_SAVEFILE
            history = pd.DataFrame({'epoch': [epoch + 1],
                                    'mean': [train_scores.mean()],
                                    'std': [round(train_scores.std(), 2)],
                                    'min': [train_scores.min()],
                                    'max': [train_scores.max()],
                                    'time': ((time() - time_start) / 60.0)})

            history.to_csv(log_file, index=False, mode='a+', header=False)

            # print("\nTesting...")
            # test_episode = []
            # test_scores = []
            # for test_episode in trange(test_episodes_per_epoch, leave=False):
            #     game.new_episode()
            #     while not game.is_episode_finished():
            #         state = preprocess(game.get_state().screen_buffer)
            #         best_action_index = session.run(eval_QN.best_a,
            #                                         feed_dict={eval_QN.s1_: state.reshape(
            #                                             [1, resolution[0], resolution[1], 1])})[0]
            #         game.make_action(actions[best_action_index], frame_repeat)
            #     r = game.get_total_reward()
            #     test_scores.append(r)
            #
            # test_scores = np.array(test_scores)
            # print("Results: mean: %.1f±%.1f," % (
            #     test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
            #       "max: %.1f" % test_scores.max())

            if (epoch + 1) % 10 == 0:
                print("Saving the network weigths to:", DEFAULT_MODEL_SAVEFILE)
                saver.save(session, DEFAULT_MODEL_SAVEFILE, global_step=epoch + 1)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_sound_enabled(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = session.run(eval_QN.best_a,
                                            feed_dict={
                                                eval_QN.s1_: state.reshape([1, resolution[0], resolution[1], 1])})[0]
            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
