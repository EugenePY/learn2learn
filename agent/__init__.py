import tensorflow as tf
import numpy as np

TENSORDTYPE = np.float32


class BaseModel(object):
    pass


class AGENT(object):
    """
    An agent object defines a MDP.
    """
    __actions__ = {}
    __states__ = None

    def training(self):
        """
        Base-Agent
        """
        pass

    @property
    def __actions__(self):
        return self.__actions__

    @property
    def __states__(self):
        return self.__states__

    @__states__.setter
    def __state__(self, state):
        """
        Assign state space
        """
        self.__states__ = state

    @__actions__.setter
    def __actions__(self, action):
        self.__actions__ = action

    def update(self):
        pass


class MCMCAgnet(AGENT):
    pass


class PolicyGradientAgent(AGENT):
    def __init__(self, batch_size, agent, **kwarg):
        super(AGENT, self).__init__(**kwarg)

        self.agent = agent
        self.obs_dim = self.agent.obs_dim
        self.act_dim = self.agent.act_dim
        # tf graphs
        self.batch_size = batch_size

    def sample_action_space(self, x):
        if not hasattr(self.agent, 'sampling_action'):
            print('object %s do not have attribute: sampling_action'.format(
                str(self.agent)))
        return self.agent.sampling_action(x, batch_size=self.batch_size)

    def policy_gradients(self, X, reward):
        """
        Policy gradients
        """
        x = tf.placeholder(TENSORDTYPE, shape=(self.batch_size, self.obs_dim))
        y = self.sample_action_space(x)
        # sampled actions do not take gradients, sampled action act as a fake
        # targget
        tf.stop_gradient(y, name='sample_actions')

        y_hat = self.agent.predict_prop(x)

        loss = tf.math_ops.reduce_mean(tf.square(y-y_hat))

        gradients = tf.gradients(loss, self.agent.params)
        baseline = self.basline_net(reward, x)
        reinforce_grad = (reward - baseline) * gradients

        return reinforce_grad

    def basline_net(self, reward, x):
        """
        simple baseline
        """
        return tf.math_ops.reduce_mean(reward)


class TDAgent(AGENT):
    pass


class QLearningAgent(AGENT):
    def __init__(self, agent):
        pass


class SARSAAgent(AGENT):
    pass
