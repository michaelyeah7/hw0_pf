import numpy as np
import copy
from utils import load_data, generate_barcode_dict, generate_landmark_gt_dict
from math import pow

class Particle:
    def __init__(self, x, y, theta, weight=1):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
    

class ParticleFilter:
    def __init__(self, particle_num=100):
        self.particle_num = particle_num
        self.particles = []
        #landmarks_groundtruth
        barcode_data = load_data('ds1_Barcodes.dat', 3, 0, [0, 3])
        self.barcode_dict = generate_barcode_dict(barcode_data)
        landmark_gt_data = load_data('ds1_Landmark_Groundtruth.dat', 3, 0, [0, 2, 4, 6, 8])
        self.landmark_gt_dict = generate_landmark_gt_dict(landmark_gt_data)

    def generate_particles(self, init_pose):
        pos_var = 0
        theta_var = 0
        for i in range(self.particle_num):
            x = np.random.normal(init_pose[0], pos_var)
            y = np.random.normal(init_pose[1], pos_var)
            theta = np.random.normal(init_pose[2], theta_var)
            weight = 1 / self.particle_num
            particle =  Particle(x, y, theta, weight)
            self.particles.append(particle)

    def update(self, robot_measurements):
        """update the i_th particle's weight

        Args:
            robot_measurements: list of [barcode, range, bearing]

        """

        for i in range(self.particle_num):  
            # print("particles[i].x",self.particles[i].x)   
            # print("particles[i].y",self.particles[i].y)     
            # print("particles[i].weight",self.particles[i].weight)    
            l = 1
            for robot_measurement in robot_measurements:
                l = l * self.measure(i,robot_measurement)
            self.particles[i].weight = l
            self.particles[i].weight += 1e-31
            # print("pos_particles[i].x",self.particles[i].x)   
            # print("pos_particles[i].y",self.particles[i].y)     
            # print("pos_particles[i].weight",self.particles[i].weight) 
        
        #sum the weight
        weight_sum = 0
        for i in range(self.particle_num):
            weight_sum += self.particles[i].weight
        #normalize the weights
        for i in range(self.particle_num):
            # print("self.particles[i].weight",self.particles[i].weight)
            self.particles[i].weight =  np.divide(self.particles[i].weight, weight_sum)
            # print("particle %f nomarlized weight %f" % (i,self.particles[i].weight))

            

    def measure(self, i, robot_measurement):
        """ Return the landmark's relative location regarding to robot local coordinates

        Args:
            i: particle id
            robot_measurement: [barcode, range, bearing]
        
        Returns:
            l: likelihood of particle measurement and robot measurement
        """
        landmark_barcode = robot_measurement[0]

        landmark_id = self.barcode_dict[landmark_barcode]
        # landmark[landmark_gt_x, landmark_gt_y]
        landmark = self.landmark_gt_dict[landmark_id]
        landmark_x = landmark[0]
        landmark_y = landmark[1]

        pos = [self.particles[i].x, self.particles[i].y, self.particles[i].theta]
        particle_landmark_range = np.sqrt((landmark_x - pos[0])**2 + (landmark_y - pos[1])**2)
        particle_landmark_bearing = np.arctan2((landmark_y - pos[1]), (landmark_x - pos[0])) - pos[2]

        diff_range = robot_measurement[1] - particle_landmark_range
        # diff_bearing = np.abs(particle_landmark_bearing - robot_measurement[2])
        diff_bearing = np.arccos(np.cos(particle_landmark_bearing) * np.cos(robot_measurement[2]) \
             + np.sin(particle_landmark_bearing) * np.sin(robot_measurement[2]))
        # print("diff_range",diff_range)
        # print("diff_bearing",diff_bearing)

        l_range = self.norm_pdf(diff_range,2)
        l_bearing = self.norm_pdf(diff_bearing,np.pi/4)
        l = l_range * l_bearing

        # l = np.exp(-diff_range) + np.exp(-diff_bearing)
        # print("l_range",l_range)
        # print("l_bearing",l_bearing)
        # print("l",l)
        return l

    def norm_pdf(self, v, sigma):
        """The probability density function of a univariate zero-mean normal distribution.
        Args:
            v: the value for which the probability shall be evaluated
            sigma: the standard devation sigma > 0.
        """
        return 1 / np.sqrt(2*np.pi*pow(sigma, 2)) * np.exp(-pow(v, 2) / (2*pow(sigma, 2)))

    def resample(self):
        """if degeneracy is too high, resample

        """
        tmp_particles = copy.deepcopy(self.particles)
        weights_square = 0
        for i in range(self.particle_num):
            weights_square += tmp_particles[i].weight **2

        #effective N as an indicator to whether resample
        n_eff = 1/weights_square
        n_eff_threshold = 0.5 * self.particle_num
        # if n_eff < n_eff_threshold:
        for i in range(self.particle_num):
            psample = self.particle_sample(tmp_particles)
            self.particles[i].x = psample.x
            self.particles[i].y = psample.y
            self.particles[i].theta = psample.theta
            self.particles[i].weight = psample.weight
    
    # def particle_sample(self, tmp_particles):
    #     #find some big particle to replace the old one, using multinomial resampling
    #     threshold = np.random.random_sample()
    #     cumulative_sum = 0
    #     for i in range(self.particle_num):
    #         cumulative_sum += tmp_particles[i].weight
    #         if (cumulative_sum > threshold):
    #             break
    #         # if (threshold < tmp_particles[i].weight):
    #         #     break
    #         # r -= w
    #         # if r < 0:
    #         #     break
    #     return tmp_particles[i]

    def particle_sample(self, tmp_particles):
        weight_sum = 0
        for i in range(self.particle_num):
            weight_sum += tmp_particles[i].weight
        r = np.random.random_sample() * weight_sum
        for i in range(self.particle_num):
            r -= tmp_particles[i].weight
            if r < 0:
                break
        return tmp_particles[i]


