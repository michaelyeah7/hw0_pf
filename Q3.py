import numpy as np
from env import Env
from utils import load_data, plot_trajectory_Q3

def Q3():
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

    ground_truth = load_data('ds1_Groundtruth.dat', 3, 0, [0,3,5,7]) 
    plot_trajectory_Q3(env.trajectory, ground_truth)

