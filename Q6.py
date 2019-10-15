import numpy as np
from env import Env

def Q6():
    poses = [[2, 3, 0], [0, 3, 0], [1, -2, 0]]
    landmarker_ground_truth = [[1.88032539, -5.57229508], [3.07964257, 0.24942861], [-1.04151642, 2.80020985]]
    errors = []
    env = Env()
    for i in range(len(poses)):
        env.pos = poses[i]
        landmark_measurement = env.measure(landmarker_ground_truth[i])
        print('The %ith pose prediction'% i)
        print("prediction:",landmark_measurement)
        landmark_measurement_global = env.rel_to_global(landmark_measurement)
        error = [(landmark_measurement_global[0] - landmarker_ground_truth[i][0]), (landmark_measurement_global[1] - landmarker_ground_truth[i][1])]
        errors.append(error)

    print("errors",errors)

