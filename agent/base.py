import tensorflow as tf
import numpy as np
from agent.replay_memory import MemoryReplay
from tensorflow.contrib.layers.python.layers import conv2d, linear
from tqdm import tqdm

TENSORFLOAT = np.float32
TENSORINT = np.int32


def check_input(input):
    """
    Check the input is symbolic variable
    """
    pass


class Monitors(object):
    """
    Access the data from Agents
    """
    pass


class AgentGraph(object):
    """
    Define the compution graph of an agent.
    """
    nn_topo = {}

    def build_model(self):
        pass

    def _initial_params(self):
        pass

    def _shape_infer(self):
        pass

    def chech_valid_shape(self, topo_dict):
        pass

    def _topo_dict(self):
        """
        Build the simple graph topo
        """
        self.topo_dict = {}
        for i, mappings in enumerate(self.nn_topo[1:]):
            w_name = 'W_%i'.format(i)
            bias_name = 'bias_%i'.format(i)
            self.topo_dict[w_name] = [self.nn_topo[i-1], mappings]
            self.topo_dict[bias_name] = mappings

        return self

    def save(self):
        pass


class DQN(AgentGraph):
    """
    DQN:
        Buiding graph do not use name_scope api
    """
    def __init__(self, env, obs_spec, config, sensor=None):
        self.env = env
        self.raw_input_spec = env
        self.env.action_space.n_actions = env.action_space.n  # hmmm
        self.obs_spec = obs_spec
        self.sensor = sensor
        self.memory = MemoryReplay(memory_size=1000, env=env, config=config)
        self.config = config

        self.default_config = {'filter_shapes': [(8, 8), (4, 4), (3, 3)],
                               'channels': [32, 64, 64],
                               'strides': [(4, 4), (2, 2), (1, 1)],
                               'padding': 'VALID',
                               'linear': [512, self.env.action_space.n]}

    def _build_sensor(self, input, **kwarg):
        if self.sensor is not None:
            sensor = self.sensor(input, **kwarg)
        else:
            sensor = self._build_atari_cnn(input, self.default_config, **kwarg)
        return sensor

    def _initial_params(self):
        self.params = {}
        self.bootrap_params = {}
        return self

    def make_input_tf_placeholder(self, env, state_spec):
        """
        just for testing
        """
        act = tf.placeholder(TENSORINT, [None, 1], 'actions')
        state = tf.placeholder(TENSORFLOAT, (None,) + state_spec, 'states')
        state_next = tf.placeholder(TENSORFLOAT, (None,) + state_spec,
                                    'state_next')
        reward = tf.placeholder(TENSORFLOAT, (None, 1), 'reward')
        terminal = tf.placeholder(TENSORINT, (None, 1), 'done')

        return state, act, state_next, reward, terminal

    def _build_atari_cnn(self, input, config, variable_scope):
        """
        Sensor network
        """
        states = input
        filter_shapes = config['filter_shapes']
        channels = config['channels']
        strides = config['strides']
        padding = config['padding']

        # fully connect
        output_dims = config['linear']

        with tf.variable_scope(variable_scope):
            # TODO: make spec can have input type
            # if spec.input_type == 'image':
            if True:
                # build the conv
                for i, (channel, shapes, strides) in enumerate(
                        zip(channels, filter_shapes, strides)):
                    if i == 0:
                        hiddens_i = states

                    hiddens_i = conv2d(inputs=hiddens_i,
                                       num_outputs=channel,
                                       kernel_size=shapes,
                                       stride=strides,
                                       padding=padding,
                                       weights_initializer=tf.contrib.layers.
                                       xavier_initializer(),
                                       activation_fn=tf.nn.relu)

                conv2d_out = hiddens_i
                flatten = tf.contrib.layers.flatten(conv2d_out)

                for i, output_dim in enumerate(output_dims):
                    if i == 0:
                        input_ = flatten
                    input_ = linear(input_, output_dim,
                                    weights_initializer=tf.contrib.layers.
                                    xavier_initializer(),
                                    activation_fn=tf.nn.relu)

                sensor = input_  # Q(s, a) fn
        return sensor

    def initialize(self):
        # prepare the inputs
        with tf.variable_scope('inputs'):
            try:
                self.states, self.actions, self.states_next, self.rewards, \
                    self.terminals = self.env.make_input_tf_placeholder()
            except:
                self.states, self.actions, self.states_next, self.rewards, \
                    self.terminals = self.make_input_tf_placeholder(
                        self.env, self.obs_spec)

        with tf.name_scope('q_network'):
            self.q_network = self._build_sensor(
                self.states, variable_scope='dqn')  # create the dqn params

        with tf.name_scope('q_boostrap'):
            self.q_boostrap = self._build_sensor(
                self.states_next, variable_scope='boostrap_params')
            # create the lag dqn params

        with tf.name_scope('update_boostrap_params'):
            with tf.variable_scope('dqn', reuse=True):
                self.q_params = {q_param.name: q_param for q_param in
                                 tf.get_collection(
                                     tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope='dqn')}

            with tf.variable_scope('boostrap_params', reuse=True):
                self.boostrap_params = {boostrap_param.name: boostrap_param for
                                        boostrap_param in
                                        tf.get_collection(
                                            tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope='boostrap_params')}

            self.update_boostrap_ops = self._update_boostrap_params(
                self.q_params, self.boostrap_params)

        with tf.name_scope('boostrap_targ'):
            self.loss, self.action_one_hot = self._build_training_op(
                self.states, self.rewards, self.actions,
                self.states_next, self.terminals)

        with tf.name_scope('summary'):
            # build the short summary for each minibatch
            self.q_mean, self.q_std = self._build_summary(self.q_network)

        with tf.name_scope('tensorboard_monitor') as scope:
            # create the tensorboard summary
            # plot for each epoch
            self.q_hist = tf.placeholder(shape=[None, self.env.action_space.n],
                                         dtype=np.float32)
            self.loss_hist = tf.placeholder(shape=[None, 1], dtype=np.float32)
            self.rewards_hist = tf.placeholder(shape=[None, 1],
                                               dtype=np.float32)

            self._tensorboard_monitor(self.q_hist, self.loss_hist,
                                      self.rewards_hist, scope)

        with tf.name_scope('optimizer'):
            self.optimizer = self._build_optimizer(self.loss)

    def _update_boostrap_params(self, q_params, boostrap_params):
        update_boostrap_ops = {}
        for name, param in q_params.items():
            update_boostrap_ops[name+'_update_op'] = \
                tf.assign(param,
                          boostrap_params.get(name.replace('dqn',
                                                           'boostrap_params')))
        return update_boostrap_ops

    def _build_optimizer(self, loss, learning_rate_op=0.00025):
        optim = tf.train.RMSPropOptimizer(
            learning_rate_op, momentum=0.95, epsilon=0.01).minimize(
                loss, global_step=tf.contrib.framework.get_global_step())
        return optim

    def _greed_eposilon_policy(self, epsilon, q):
        action_greedy = tf.argmax(q, axis=1)
        action_random = tf.multinomial(tf.log([self.env.action_space.n *
                                              [1./self.env.action_space.n]]),
                                       1)[0][0]  # make it one dim
        pred = tf.greater_equal(tf.random_uniform(shape=(1,)), 1-epsilon)[0]
        actions = tf.cond(pred, lambda: action_greedy,
                          lambda: action_random)
        return actions

    def _build_learning_rate_op(self, lr, lr_step, lr_decay_step, lr_decay,
                                lr_minimum):
        """
        Optional Operation
        """
        decay_op = tf.train.exponential_decay(lr, lr_step, lr_decay_step,
                                              lr_decay, staircase=True)

        learning_rate_op = tf.maximum(lr_minimum, decay_op)
        return learning_rate_op

    def _build_training_op(self, states, rewards, actions, states_next,
                           terminal, beta=None):

        if beta is None:
            beta = tf.get_variable('beta', [1, ],
                                   initializer=tf.constant_initializer(1.))

        q_network = self.q_network

        action_one_hot = tf.one_hot(tf.squeeze(actions),
                                    self.env.action_space.n,
                                    1.0, 0.0, axis=-1,
                                    name='action_one_hot')
        q_ij = tf.reduce_sum(q_network * action_one_hot,
                             reduction_indices=1, name='q_acted')

        target = rewards + beta * tf.reduce_max(self.q_boostrap, axis=1) * \
            tf.cast(terminal, tf.float32)

        target = tf.stop_gradient(target)  # treat target as a contant

        loss = tf.nn.l2_loss(q_ij - target)
        return loss, action_one_hot

    def _build_summary(self, q_network):
        q = tf.reduce_mean(q_network, axis=0)
        q_mean, q_std = tf.nn.moments(q, axes=[0])
        return q_mean, q_std

    def _tensorboard_monitor(self, q_mean_hist, loss_mean_hist,
                             reward_mean_hist, name_scope):
        """
        Monitor the gradients dist
        """
        # the variance of average q
        # summary_tags = ['avarage_q', 'std_q', 'loss_mean', 'loss_std',
        #                'weights', 'bias', 'reward_max', 'reward_min',
        #                'reward_std']
        q = tf.reduce_mean(q_mean_hist, axis=0)
        # q_mean = tf.reduce_mean(q, name='average_q_mean')

        tf.summary.scalar('q_mean', q)

    def learn(self, sess):
        """
        Evaluate the Graph and do one step training
        """
        # Doing training
        q, loss = self.train_mini_batch(sess)
        return q, loss

    def train_mini_batch(self, sess):
        """
        Training the agent
        """
        mini_batch = self.memory.sample_mini_batch(self.config.batch_size)

        feed_dict = {self.states: mini_batch['states'],
                     self.rewards: mini_batch['rewards'],
                     self.actions: mini_batch['actions'],
                     self.states_next: mini_batch['states_next'],
                     self.terminals: mini_batch['terminals']}
        evals = [self.q_network, self.loss, self.optimizer]

        q, loss, opt = sess.run(evals, feed_dict=feed_dict)
        return q, loss

    def act(self, state):
        """
        input is np.array
        """
        act = self._greed_eposilon_policy(self.config.epsilon_init,
                                          self.q_network)
        return act.eval({self.states: [state]})

    def train(self):
        # initial the iter
        self.n_update = 0
        self.n_frame_past = 0
        n_games = 0
        env = self.env
        self.summary_asic = {'q_hist': [], 'loss_hist': []}

        state = env.reset()

        DONE = False
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in tqdm(range(self.config.max_epoch), ncols=100):
                print '\n'
                print "EPOCH: {}".format(epoch)
                for _ in tqdm(range(self.config.steps_within_epoch)):
                    # action with greed-eposilon policy (behavior policy)
                    action = self.act(state=state)
                    # interact with enviorment
                    state_next, reward, terminal, info = env.step(action)

                    # update states & adding memory *** this is bad.... code
                    self.memory.add_memory(
                        **{'states': np.expand_dims(state, 0),
                            'actions': np.expand_dims(action, 0),
                            'states_next': np.expand_dims(state_next, 0),
                            'rewards': np.expand_dims([reward], 0),
                            'terminals': np.expand_dims([terminal], 0)})

                    if self.n_frame_past % 100 == 0:
                        self.memory.pop_memory()  # clear some memories

                    # learning form experience
                    q, loss = self.learn(sess)
                    print loss, np.mean(self.summary_asic['loss_hist'])
                    print reward
                    sess.run(self.update_boostrap_ops)

                    state = state_next

                    if terminal:
                        state = self.env.reset()
                        n_games += 1
                    # make summary data
                    self.summary_asic['q_hist'].append(q)
                    self.summary_asic['loss_hist'].append(loss)
                    self.n_frame_past += 1
                    self.n_update += 1

                print "epoch: {} value_mean: {}, loss: {}".format(
                    epoch, np.mean(self.summary_asic['q_hist']),
                    np.mean(self.summary_asic['loss_hist'])
                )
                self._done(DONE, self.n_frame_past, n_games)

    def _done(self, done, iters, n_games):
        if (iters / self.config.steps_within_epoch) >= \
                self.config.max_epoch:
            done = True
        return done


class POMDPNet(AgentGraph):
    def __init__(self):
        pass

    def _initial_params(self):
        pass


class SimpleAgent(object):
    pass

if __name__ == '__main__':
    import gym
    env = gym.make("AirRaid-v0")
    obs = env.reset()
    action = env.action_space.sample()

    observation, reward, done, info = env.step(action)

    agent = DQN(nn_topo={}, env=env, sess=tf.Session(),
                obs_spec=observation.shape)
    agent.initialize()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print sess.run(agent.q_network, feed_dict={
                agent.states: obs.reshape((1, )+observation.shape)})
