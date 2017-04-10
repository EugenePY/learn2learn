class env_config:
    env_name = 'Breakout-v0'
    env_type = 'gray'
    img_width = 84
    img_height = 84
    max_reward = 1
    min_reward = -1


class model_config(env_config):
    #
    max_epoch = 1000
    steps_within_epoch = 500
    train_freq = 2
    # optimizer
    batch_size = 32
    learning_rate = 0.00025
    momentum = 0.95
    q_lag_update_step = 1  # n step lag
    learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5

    # greedy policy(behavior policy)
    epsilon_init = 1
    epsilon_end = 0.1

    # memory replay
    memory_size = 1000
    repeat_actions = 4
    beta = 0.87
    cnn_format = 'NCHW'

    # summary
    gradient_summary_freq = 1000

if __name__ == '__main__':
    pass
