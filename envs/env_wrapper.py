import gym
from gym.spaces import Discrete
from gym import Space

from types import FunctionType
from gym.envs.registration import register, registry
from tensorflow.contrib.distributions.python.ops import Categorical

"""
Wrap gym space with a new method called input_tf_placeholder
"""
def factory(name):
    """
    making lab class make it compatible with gym
    """
    pass


class MetaLabSpace(type):
    def __new__(cls, name, bases, classDict):
        newClassDict = classDict  # initial the new class dict

        # add new method
        for new_method_name, new_method in cls.__dict__.items():
            if new_method_name.endswith('_wrapper'):
                newClassDict[new_method_name[:-8]] = getattr(
                    cls, new_method_name)(name)

        return type.__new__(cls, name, bases, newClassDict)

    @classmethod
    def make_tf_placeholder_wrapper(cls, cls_name):
        def make_tf_placeholder(self, name):
            return tf.placeholder(dtype=self._dtype, shape=[None, self.n],
                                  name='{}_{}_placeholder'.format(
                                      str(cls_name), name))
        return make_tf_placeholder


class LabSpace(Space):
    __metaclass__ = MetaLabSpace

    def __init__(self, seed):
        self.seed = 1126 if seed is None else seed
        self.rng = np.random.RandomState(self.seed)

    def numpy_sampling(self, batch_size):
        raise NotImplementedError

    def tf_smapling(self, batch_size, name):
        raise NotImplementedError


class LabGaussian(LabSpace):
    __metaclass__ = MetaLabSpace

    def __init__(self, n_dim, **kwarg):
        super(LabGaussian, self).__init__(**kwarg)

        self.n = n_dim
        self._dtype = np.float32

    def numpy_sampling(self, batch_size):
        return self.rng.normal(size=(batch_size, self.n))

    def tf_sampling(self, mean, std, batch_size, name):
        return tf.random_normal(shape=(batch_size, self.n), dtype=self._dtype,
                         name=name, seed=self.seed)


class LabDiscrete(LabSpace):
    _dtype = np.float32
    __metaclass__ = MetaLabSpace

    def __init__(self, n_actions):
        super(LabDiscrete, self).__init__(n=n_actions)
        self.n = n_actions

    def tf_sampling(self, batch_size, name):
        return Categorical(shape=(batch_size, self.n), dtype=self._dtype,
                         name=name, seed=self.seed)


class LabMetaEnv(gym.Env):
    """
    make the gym.Env required observation_space and action_space reward_range
    """

    def __new__(cls):
        pass

    def __init__(self, reward_range, action_space, observation_space, **kwarg):
        super(LabMetaEnv, self).__init__(**kwarg)
        assert all([isinstance(space, MetaLabSpace) for space in \
                    [observation_space, action_space]])
        assert self._assert_space([observation_space, action_space])  # this is not good practice

         # assign the value
        self.__state_space = observation_space
        self.__action_space = action_space
        if kwarg.get('step_spec') is not None:
            # TODO: not complete need to split the spec string
            # and assert the vaid spec
            self._step_spec = state_spec
        else:
            self._step_spec = 'SASRT'

    def _make_names(self):
        pass

    def _assert_space(self, spaces_list):
        return all([space is not None for space in spaces_list])

    @property
    def state_space(self):
        return self.__state_space

    @property
    def action_space(self):
        return self.__action_space

    def make_output_tf_placeholders(self):
        """
        state
        """
        return self.observation_space.make_tf_placeholder('states'), \
            self.action_space.make_tf_placeholder('actions'),\
            tf.placeholder(dtype=np.float32, shape=(None, 1),
                           name=self.__name__+'_rewards'), \
            self.observation_space('states_next'), \
            tf.placeholder(dtype=np.int32, shape=(None, 1),
                           name=self.__name__+'_terminal')


def make(id):
    pass

if __name__ == '__main__':
    # make('AirRaid-ram-v0')
    import pkg_resources
    import gym
    print LabDiscrete(5).make_tf_placeholder('state')
