import numpy as np
from env import Env
from utils import load_data, plot_trajectory_Q3, plot_trajectory_Q3_ground_truth

#pos [x, y, theta]
#command [time, v_l, v_w]
init_pos = [0, 0, 0]

commands = load_data('./ds1_Odometry.dat', 3, 0, [0,4,5])
# print("commands[0][0] : % f" % (commands[0][0]))

env = Env(init_pos)
#step (v, duration)
for i in range(len(commands)-1):
    duration = commands[i+1][0] - commands[i][0]
    vel = [commands[i][1]] 
    vel.append(commands[i][2]) 
    env.step(vel, duration)

#plot trajectory named trajectory_Q3.png
plot_trajectory_Q3(env.trajectory)

ground_truth = load_data('./ds1_Groundtruth.dat', 3, 0, [0,4,5])
x = []
y = []
for i in range(len(ground_truth)):
    x.append(ground_truth[i][1])
    y.append(ground_truth[i][2])
#plot trajectory named trajectory_Q3_ground_truth.png   
plot_trajectory_Q3_ground_truth(x, y)

