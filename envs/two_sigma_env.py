import pandas as pd
import numpy as np
import gym
from sklearn.metrics import r2_score


def r_score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    r = np.sign(r2) * np.sqrt(np.abs(r2))
    if r <= -1:
        return -1
    else:
        return r


class Container(object):
    def __init__(self, target, features, train):
        self.target = target
        self.features = features
        self.train = train


class TwoSigmaExperimentsTraining(gym.Env):
    """2016 Two sigma Kaggle competition.

    For each steps we predict "y" targets in pandas format.
        1. We have to deal with missing values.
    """

    with pd.HDFStore("./data/train.h5", "r") as train:
        df = train.get("train")
        timestamp = np.sort(df['timestamp'].unique())  # accent
        max_memory_size = len(df['timestamp'].unique())

    def __init__(self, init_replay_memory_size=None):
        assert init_replay_memory_size < self.max_memory_size
        assert init_replay_memory_size > 2

        if init_replay_memory_size is None:
            self.init_replay_memory_size = len(self.timestamp) // 2
        else:
            self.init_replay_memory_size = init_replay_memory_size

        # setup the parameters
        self.timesteps = self.init_replay_memory_size
        self.current_t = self.timestamp[self.init_replay_memory_size]
        self.done = False

    def reward(self, action, y):
        """
        Linear Correlation
        """
        return r_score(y, action)

    def _reset(self):
        # reset
        # reset generator

        train = self.df[self.df['timestamp'] <
                        self.timestamp[self.init_replay_memory_size]]

        features = self.df[self.df['timestamp'] ==
                           self.timestamp[self.init_replay_memory_size]
                           ].drop('y', 1)

        target = self.df[self.df['timestamp'] ==
                         self.timestamp[self.init_replay_memory_size]
                         ][['timestamp', 'id', 'y']]

        target['y'] = 0.0

        # make this as numpy arry this will help value assigmnet faster
        self.y_full_true = self.df[['timestamp', 'id', 'y']][
            self.df['timestamp'] >=
            self.timestamp[self.init_replay_memory_size]].values[:, 2]

        self.y_full_pred = np.zeros_like(self.y_full_true)

        self.contrainer = Container(target=target,
                                    features=features, train=train)

        self.timesteps = 0
        self.current_t = self.timestamp[self.init_replay_memory_size]
        self.done = False
        self.current_pred_size = 0

        return self.contrainer

    def make(self):
        return self

    def _step(self, action):
        current_idx = self.current_pred_size
        batch_size = action.shape[0]
        # setup for pubulic score
        # pandas iloc assign is very slow
        self.y_full_pred[current_idx:current_idx+batch_size] = \
            action['y'].values
        # including y and features
        target = self.y_full_true[current_idx:current_idx+batch_size]
        reward = self.reward(action['y'].values, target)

        if self.current_t == self.timestamp[-1]:
            observations = None
            reward = None
            self.done = True
            info = r_score(self.y_full_true, self.y_full_pred)
        else:
            # time iter
            self.timesteps += 1
            self.current_t = self.timestamp[
                self.timesteps+self.init_replay_memory_size]
            self.current_pred_size += batch_size

            next_batch = self.df_generator[self.timesteps][1]
            features = next_batch.drop('y', 1)
            target = next_batch[['timestamp', "id", "y"]].copy()
            target['y'] = 0.0
            observations = Container(target=target,
                                     features=features, train=self.train)
            info = []

        return observations, reward, self.done, info


class TwoSigmaExperiments(gym.Env):
    """2016 Two sigma Kaggle competition.

    For each steps we predict "y" targets in pandas format.
        1. We have to deal with missing values.
    """

    with pd.HDFStore("./data/train.h5", "r") as train:
        df = train.get("train")
        timestamp = np.sort(df['timestamp'].unique())  # accent
        max_memory_size = len(df['timestamp'].unique())

    def __init__(self, init_replay_memory_size=None):
        assert init_replay_memory_size < self.max_memory_size
        assert init_replay_memory_size > 2

        if init_replay_memory_size is None:
            self.init_replay_memory_size = len(self.timestamp) // 2
        else:
            self.init_replay_memory_size = init_replay_memory_size

        # setup the parameters
        self.timesteps = self.init_replay_memory_size
        self.current_t = self.timestamp[self.init_replay_memory_size]
        self.done = False

    def reward(self, action, y):
        """
        Linear Correlation
        """
        return r_score(y, action)

    def _reset(self):
        # reset
        # reset generator

        train = self.df[self.df['timestamp'] <
                        self.timestamp[self.init_replay_memory_size]]

        features = self.df[self.df['timestamp'] ==
                           self.timestamp[self.init_replay_memory_size]
                           ].drop('y', 1)
        target = self.df[self.df['timestamp'] ==
                         self.timestamp[self.init_replay_memory_size]
                         ][['timestamp', 'id', 'y']]

        target['y'] = 0.0

        # make this as numpy arry this will help value assigmnet faster
        self.y_full_true = self.df[['timestamp', 'id', 'y']][
            self.df['timestamp'] >=
            self.timestamp[self.init_replay_memory_size]].values[:, 2]

        self.y_full_pred = np.zeros_like(self.y_full_true)

        self.df_generator = list(
            self.df.groupby('timestamp').__iter__())[
                self.init_replay_memory_size:]
        self.contrainer = Container(target=target,
                                    features=features, train=train)
        self.timesteps = 0
        self.current_t = self.timestamp[self.init_replay_memory_size]
        self.done = False
        self.current_pred_size = 0

        return self.contrainer

    def make(self):
        return self

    def _step(self, action):
        current_idx = self.current_pred_size
        batch_size = action.shape[0]
        # setup for pubulic score
        # pandas iloc assign is very slow
        self.y_full_pred[current_idx:current_idx+batch_size] = \
            action['y'].values
        # including y and features
        target = self.y_full_true[current_idx:current_idx+batch_size]
        reward = self.reward(action['y'].values, target)

        if self.current_t == self.timestamp[-1]:
            observations = None
            reward = None
            self.done = True
            info = {r_score(self.y_full_true, self.y_full_pred)}
        else:
            # time iter
            self.timesteps += 1
            self.current_t = self.timestamp[
                self.timesteps+self.init_replay_memory_size]
            self.current_pred_size += batch_size

            next_batch = self.df_generator[self.timesteps][1]
            features = next_batch.drop('y', 1)
            target = next_batch[['timestamp', "id", "y"]].copy()
            target['y'] = 0.0
            observations = Container(target=target,
                                     features=features, train=self.train)
            info = {}

        return observations, reward, self.done, info
