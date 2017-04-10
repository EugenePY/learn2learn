import numpy as np


class MemoryReplay(object):
    __count = 0
    __current = 0
    __initilized__ = False

    def __init__(self, memory_size, env, config, seed=None):

        self.config = config
        self.batch_size = getattr(config, 'batch_size', 32)
        self.env = env

        self.state_space = env.observation_space
        self.action_space = env.action_space
        self.memory_size = memory_size
        self.buffers_maps = self._make_buffers_maps()

        self.buffers = self.inital_memory()
        if seed is None:
            seed = 123

        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.random_index = np.arange(self.memory_size)

    def _make_buffers_maps(self):
        """
        making bufer shape info and dtype info.
        """
        self.buffer_names = ['actions', 'states', 'states_next', 'rewards',
                             'terminals']

        self.buffer_types = ['1-dim', 'state_space', 'state_space',
                             '1-dim', '1-dim']
        # TODO: make this more general buffer type will cause the image crash
        self.buffer_dtype = [np.int32, np.float32, np.float32,
                             np.float32, np.bool]

        buffers = {}
        for i, name in enumerate(self.buffer_names):
            if self.buffer_types[i] != '1-dim':
                try:
                    buffers[name] = ((eval('self.{}.n'.format(
                        self.buffer_types[i])),),
                                    self.buffer_dtype[i])
                except AttributeError:
                    buffers[name] = (eval('self.{}.shape'.format(
                        self.buffer_types[i])), self.buffer_dtype[i])
            else:
                buffers[name] = ((1,), self.buffer_dtype[i])

        return buffers

    def inital_memory(self):
        env = self.env
        states = env.reset()
        n_memory = 0
        DONE = False
        buffers = self._make_buffer(self.memory_size)

        while not DONE:
            actions = self.action_space.sample()

            states_next, rewards, terminals, info = env.step(actions)

            if terminals:
                states = env.reset()
            else:
                states = states_next

            for name in self.buffer_names:
                buffers[name][n_memory] = eval(name)

            n_memory += 1

            if n_memory >= self.memory_size:
                DONE = True

        return buffers

    def _make_buffer(self, memory_size):
        buffers = {}
        for i, name in enumerate(self.buffer_names):
            buffers[name] = np.empty(shape=(memory_size,) +
                                     self.buffers_maps[name][0],
                                     dtype=self.buffers_maps[name][1])
        return buffers

    def add_memory(self, **kwarg):
        """
        this method requred keword arguments only.
        """
        for name, obs in kwarg.items():
            self.buffers[name] = np.concatenate((self.buffers[name], obs), axis=0)
            # get recent memory
        return self

    def pop_memory(self, **kwarg):
        """
        remove the old memory which is older than memory size
        """
        for name, obs in kwarg.items():
            self.buffers[name] = obs[-self.memory_size:]
        return self

    def sample_mini_batch(self, batch_size):
        self.rng.shuffle(self.random_index)
        idx = self.random_index[:batch_size]
        mini_batch = {}

        for name in self.buffer_names:
            mini_batch[name] = self.buffers[name][idx]

        return mini_batch

if __name__ == '__main__':

    def test_memroy():
        # Memory is stack
        import gym
        from configs.config_dqn_test import model_config
        from PIL import Image

        env = gym.make('Breakout-v0')
        env.frameskip = (4, 5)
        memory = MemoryReplay(memory_size=1000,
                              env=env,
                              config=model_config)

        mini_batch = memory.sample_mini_batch(30)
        # print memory.buffers['rewards'].mean(1)
        print mini_batch['states'][0, :, :, :].shape
        img = Image.fromarray(mini_batch['states'][0, :, :, :], 'RGB')
        img.show()

    test_memroy()
