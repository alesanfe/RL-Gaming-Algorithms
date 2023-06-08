from gym.envs.registration import register
from .golf_env import GolfEnv

register(
    id='Golf-v0',
    entry_point='games.golf_env:GolfEnv',
)