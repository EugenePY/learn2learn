from gym.envs.registration import register

register(
    id='2048-v0',
    entry_point='envs.gym_2048.envs:Game2048Env',
    timestep_limit=1000,
)

register(
    id='BreakoutDQN-v0',
    entry_point='envs.atari_exps:BreakoutDeepMind'
)

register(
    id='TwoSigma-v0',
    entry_point='envs.two_sigma_env:TwoSigmaExperiments'
)
