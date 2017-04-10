import tensorflow as tf
import gym

from envs import gym_2048
from agent.base import DQN


flags = tf.app.flags

flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('is_train', True, 'wheather to train the model')


def main():
    agent.train()
    pass
