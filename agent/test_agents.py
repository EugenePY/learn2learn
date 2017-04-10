import gym
import envs
import tensorflow as tf
from agent.base import DQN
from configs.config_dqn_test import model_config
from PIL import Image


def test_dqn():
    env = gym.make("BreakoutDQN-v0")
    obs = env.reset()
    action = env.action_space.sample()
    config = model_config
    observation, reward, done, info = env.step(action)

    agent = DQN(env=env, obs_spec=observation.shape, config=model_config)
    agent.initialize()
    mini_batch = agent.memory.sample_mini_batch(32)
    # print mini_batch['states'][0]
    # img = Image.fromarray(mini_batch['states'][10].mean(2)*255)
    # img.show()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print sess.run(agent.q_network, feed_dict={
                agent.states: mini_batch['states']})
        print sess.run(agent.action_one_hot,
                     feed_dict = {agent.states: mini_batch['states'],
                     agent.rewards: mini_batch['rewards'],
                     agent.actions: mini_batch['actions'],
                     agent.states_next: mini_batch['states_next'],
                     agent.terminals: mini_batch['terminals']})
        print sess.run(agent.loss,
                       feed_dict = {agent.states: mini_batch['states'],
                     agent.rewards: mini_batch['rewards'],
                     agent.actions: mini_batch['actions'],
                     agent.states_next: mini_batch['states_next'],
                     agent.terminals: mini_batch['terminals']})


    agent.train()

if __name__ == '__main__':
    test_dqn()
