import numpy as np
import copy
class Env:
    """
    A class generate a path with init_state and a series of commands."""
    def __init__(self, init_pos = [0, 0, 0]):
        #current position [x, y, theta], global coordinates x, y and orientation in radius.
        self.pos = init_pos
        #list of positions
        self.trajectory = []
        self.trajectory.append(self.pos)

    def step(self, vel, duration):
        """Predict the next pos given a velocity in a duration.

        Args:
            vel: [v_l, v_w], linear velocity v_l and angular velocity v_w.
            duration: execute time.

        Returns:
            new_pos: [x, y, theta], the new pos.
        """
        pos = self.pos
        new_pos = copy.deepcopy(pos)
        #calculate the new x and y
        new_pos[0] = pos[0] + vel[0] * np.cos(pos[2]) * duration
        new_pos[1] = pos[1] + vel[0] * np.sin(pos[2]) * duration
        #calculate the new theta
        new_pos[2] = pos[2] + vel[1] * duration
        self.pos = new_pos
        self.trajectory.append(self.pos)


    def measure(self, landmark):
        """ Return the landmark's relative location regarding to robot local coordinates

        Args:
            landmark: [landmark_x, landmark_y], global position of landmark
        
        Returns:
            landmark_measurement: [landmark_range, landmark_bearing], range in m, bearing in rad
        """
        landmark_range = np.sqrt((landmark[0] - self.pos[0])**2 + (landmark[1] - self.pos[1])**2)
        landmark_bearing = np.arctan2((landmark[1] - self.pos[1]), (landmark[0] - self.pos[0])) - self.pos[2]

        #add fake noise on range and bearing
        landmark_range += np.random.normal(0,0.01)
        landmark_bearing += np.random.normal(0,0.01)

        landmark_measurement = [landmark_range, landmark_bearing]
        return landmark_measurement

    def rel_to_global(self, landmark_measurement):
        """Return the landmark's global location given the relative location

        Args:
            landmark_measurement: [landmark_range, landmark_bearing], range in m, bearing in rad

        Returns:
            landmark_measurement_global: [landmark_measurement_x, landmark_measurement_y], global position of landmark_measurement
        """
        [landmark_range, landmark_bearing] = landmark_measurement
        landmark_measurement_x = self.pos[0] + landmark_range * np.cos(self.pos[2] + landmark_bearing)
        landmark_measurement_y = self.pos[1] + landmark_range * np.sin(self.pos[2] + landmark_bearing)
        landmark_measurement_global = [landmark_measurement_x, landmark_measurement_y]
        return landmark_measurement_global

