import gymnasium
import highway_env
from matplotlib import pyplot as plt

env = gymnasium.make('highway-v0', render_mode = 'rgb_array',
                     config = {"vehicles_count": 50,
                               "vehicles_density": 0.5, 
                               "manual_control": False})
env.reset()
for _ in range(100):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.show()