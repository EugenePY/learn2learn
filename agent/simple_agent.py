from agent import AGENT, train_iter
from functools import partial, wraps
import tensorflow as tf


class Staff_v1(AGENT):
    """
    Staticiant Agent: Hyperparameter Choosing, online learning.
    Currculiumn learning.
    """
    __trained__ = False  # check the agent if trained
    __model_fn__ = None  # compiled agent computation graph

    def __init__(self, env, config=None):
        self.env = env
        self.config = config
        if self.config is None:
            self.config = {'lr': .001}

    def initial_params(self):

        pass

    def build_network(self):
        pass

    def act(self, state):
        pass

    def one_step_update(self, obs):
        """
        For training usuage
        """
        pass

    def online_update(self, obs):
        """
        Explore and Learning trade-off
        """
        pass

    def train(self):
        """
        train
        """
        pass

if __name__ == '__main__':

    def func(*arg, **kwarg):
        print('this is first fun')

    def func_sec(f):
        @wraps(f)
        def wrapper(*arg, **kwarg):
            print('simple test'+ kwarg['name'])
            return f(*arg, **kwarg)
        return wrapper

    func_sec(func)(name='name')

    print Staff_v1(env=None).train.__dict__





