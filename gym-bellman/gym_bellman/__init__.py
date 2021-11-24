from gym.envs.registration import register

register(
    id='kbc-a-v0',
    entry_point='gym_bellman.envs:BellmanDpA',
)

register(
    id='kbc-b-v0',
    entry_point='gym_bellman.envs:BellmanDpB',
)

register(
    id='kbc-c-v0',
    entry_point='gym_bellman.envs:BellmanDpC',
)
