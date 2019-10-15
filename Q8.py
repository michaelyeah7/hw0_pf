import copy
import numpy as np
from particleFilter import ParticleFilter
from particle_env import ParticleEnv
from utils import load_data, generate_barcode_dict, whether_landmark, plot_predict_trajectory_Q8

def Q8():
    #load data, need odometry.dat for command, measurement.dat for comparision, landmark_gt.dat for measurement model
    # odom_data = load_data('ds1_Odometry.dat', 3, 0, [0, 4, 5])
    odom_data = [[0.5, 0, 1],
                [0, -1/(2*np.pi), 1],
                [0.5,0, 1],
                [0, 1/(2*np.pi), 1],
                [0.5, 0, 1]]
    landmark_gt_data = load_data('ds1_Landmark_Groundtruth.dat', 3, 0, [0, 2, 4, 6, 8])
    barcode_data = load_data('ds1_Barcodes.dat', 3, 0, [0, 3])
    barcode_dict = generate_barcode_dict(barcode_data)

    gt_data = load_data('ds1_Groundtruth.dat', 3, 0, [0, 3, 5, 7])

    #init particles
    particle_num = 50
    filter = ParticleFilter(particle_num)
    # init_pose = [gt_data[0][1], gt_data[0][2], gt_data[0][3]]
    init_pose = [0,0,0]
    filter.generate_particles(init_pose)

    #init environment with robot and particles
    particle_env = ParticleEnv(init_pose, filter.particles)


    maxsteps = len(odom_data)
    k = 0
    robot_trajectory = [init_pose]
    robot_pos_predict_trajectory = []
    for i in range(maxsteps):
        vel = [odom_data[i][0], odom_data[i][1]]
        duration = odom_data[i][2]
        particle_env.forward(vel, duration)

        #record the robot's trajectory and filter' prediction of robot's pos
        robot_trajectory.append(particle_env.robot_pos)

        #get robot's measurement in a duration, if null go next duration 
        # k, robot_measurements = particle_env.find_available_measurement(k, odom_data[i-1][0], odom_data[i][0])
        # print("robot_measurements length",len(robot_measurements))
        robot_measurements=[]
        if (len(robot_measurements)==0):
            robot_pos_predict_trajectory.append(particle_env.robot_pos_predict)
            i+=1
        else:       
            #robot_measurements is the list of [barcode, range, bearing]
            #given one duration's robot_measurements, update all particles' weights
            filter.update(robot_measurements)

            #record the filter's prediction
            robot_pos_predict_trajectory.append(particle_env.robot_pos_predict)

            # if degeneracy is too high, resample
            filter.resample()
            i+=1
    print("prediction",robot_pos_predict_trajectory)
    plot_predict_trajectory_Q8(robot_trajectory, robot_pos_predict_trajectory)
    # print("robot_pos_predict_trajectory:",robot_pos_predict_trajectory)





