from gym.envs.registration import register

register(
    id='drill-v0',
    entry_point='gym_drill.envs:DrillEnv',
)