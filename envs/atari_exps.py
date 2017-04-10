from gym.envs.atari.atari_env import AtariEnv
from gym.spaces import Box
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize


class BreakoutDeepMind(AtariEnv):
    def __init__(self, action_repeat=4, **kwarg):
        super(BreakoutDeepMind, self).__init__(
            game='breakout', obs_type='image', **kwarg)
        self.observation_space = Box(low=0, high=255,
                                     shape=(84, 84, action_repeat))
        self.action_repeat = action_repeat

    def phi(self, state):
        """
        Preprocess the image
        """
        return resize(rgb2gray(state), (110, 84))[18:102, :] # cut the top

    def _reset(self):
        obs = self.phi(super(BreakoutDeepMind, self)._reset()).reshape(
            self.observation_space.shape[:-1]+(1,))
        return np.tile(obs, (1, 1, self.action_repeat))

    def _step(self, action):
        reward_cum = 0
        stack_state = np.empty(shape=self.observation_space.shape,
                               dtype=np.float32)
        terminal_all = []
        for i in range(self.action_repeat):
            state, reward, terminal, info = \
                super(BreakoutDeepMind, self)._step(action)

            reward_cum += reward
            stack_state[:, :, i] = self.phi(state)
            terminal_all.append(terminal)

        terminal_all = any(terminal_all)
        return stack_state, reward_cum, terminal_all, info

if __name__ == '__main__':
    from PIL import Image
    env = BreakoutDeepMind()
    print env.step(0)[0].mean(2)*255
    img = Image.fromarray(env.step(0)[0][:, :, 0]*255)
    img.show()
