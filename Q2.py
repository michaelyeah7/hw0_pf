import numpy as np
from env import Env
from utils import plot_trajectory_Q2

#pos [x, y, theta]
#command [v_l, v_w, duration]
init_pos = [0, 0, 0]
commands = [[0.5, 0, 1],
            [0, -1/(2*np.pi), 1],
            [0.5,0, 1],
            [0, 1/(2*np.pi), 1],
            [0.5, 0, 1]]
env = Env(init_pos)
for command in commands:
    env.step(command[:-1], command[2])
#plot trajectory named trajectory_Q2.png
plot_trajectory_Q2(env.trajectory)
