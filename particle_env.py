import numpy as np
import copy
from utils import load_data, generate_barcode_dict, generate_landmark_gt_dict, whether_landmark

class ParticleEnv:

    def __init__(self, robot_init_pos, particles):
        #current position [x, y, theta], global coordinates x, y and orientation in radius.
        self.robot_pos = robot_init_pos
        #list of class Particle
        self.partcles = particles
        #filter' prediction of robot's pos
        self.robot_pos_predict = robot_init_pos
        measurement_data = load_data('ds1_Measurement.dat', 3, 0, [0, 4, 6, 7])
        self.measurement_data = measurement_data

    def find_available_measurement(self, k, t_prev, t_now):
        measurement_data = self.measurement_data
        robot_measurements = []
        while(measurement_data[k][0] < t_now and k<6160):
            if (measurement_data[k][0] >= t_prev and whether_landmark(measurement_data[k][1])):
                #landmark, range, bearing
                measurement = [measurement_data[k][1], measurement_data[k][2], measurement_data[k][3]]
                robot_measurements.append(measurement)
            k+=1
        return k, robot_measurements

    def predict_robot_pos(self):
        mean_x = 0
        mean_y = 0
        for i in range(len(self.partcles)):
            mean_x += self.partcles[i].x * self.partcles[i].weight
            mean_y += self.partcles[i].y * self.partcles[i].weight
        self.robot_pos_predict = [mean_x, mean_y, 0]
                


    def forward(self, vel, duration):
        """Move robot and particles in the env one time step

        Args:
            vel: [v_l, v_w], linear velocity v_l and angular velocity v_w.
            duration: execute time.
        """
        #move one time step of robot
        self.step_robot(self.robot_pos, vel, duration)
        #move one time step of particles
        for i in range(len(self.partcles)):
            self.step_particle(i, self.partcles[i], vel, duration)
        
        #update the mean of particles
        self.predict_robot_pos()

    def step_robot(self, pose, vel, duration):
        """Predict the next pos given a velocity in a duration.

        Args:
            pose: [x, y, theta], the pose of robot.
            vel: [v_l, v_w], linear velocity v_l and angular velocity v_w.
            duration: execute time.

        """
        pos = pose
        new_pos = copy.deepcopy(pos)
        #calculate the new x and y
        new_pos[0] = pos[0] + vel[0] * np.cos(pos[2]) * duration
        new_pos[1] = pos[1] + vel[0] * np.sin(pos[2]) * duration
        #calculate the new theta
        new_pos[2] = pos[2] + vel[1] * duration
        self.robot_pos = new_pos

    def step_particle(self,i, particle, vel, duration):
        """Predict the next pos given a velocity in a duration.

        Args:
            i: the i_th particle
            particle: class Particle
            vel: [v_l, v_w], linear velocity v_l and angular velocity v_w.
            duration: execute time.

        """
        pos = [particle.x, particle.y, particle.theta]
        new_pos = copy.deepcopy(pos)
        #calculate the new x and y
        new_pos[0] = pos[0] + vel[0] * np.cos(pos[2]) * duration
        new_pos[1] = pos[1] + vel[0] * np.sin(pos[2]) * duration
        #calculate the new theta
        new_pos[2] = pos[2] + vel[1] * duration

        self.partcles[i].x = new_pos[0]
        self.partcles[i].y = new_pos[1]
        self.partcles[i].theta = new_pos[2]

