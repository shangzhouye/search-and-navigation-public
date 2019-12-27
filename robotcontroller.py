# Created by Shangzhou Ye, MSR, Northwestern
# The script is written in Python 2.7

from __future__ import print_function, division
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math



class Controller(object):
    ''' Implement go-to-goal control

    Implemented in cell_x, cell_y
    The robot should calculate the error: linear and angular velocity.
    Using a PID controller to reach the goal.
    Tune the parameters to make the robot not going in circles.
    Limit the maximum linear and angular acceleration.
    '''
    def __init__(self,start_position=[0,0,-math.pi/2]):
        ''' Initialize the controller

        Args:
            start_position (list e.g. [0,0,0]) : three entries are initial x, y (in m) and heading respectively (in rad)
        '''
        # the robot model
        self.self_x = start_position[0]
        self.self_y = start_position[1]
        self.self_theta = start_position[2]


        #### propotional controller gain ####
        self.K_linear = 0.5
        self.K_angular = 4
        
        # tolerance in m
        self.tolerance = 0.01

        # record the path
        self.trajectory = [start_position]

        # record the current v and w for not allowing a and w_dot larger than the limitation
        self.v = 0
        self.w = 0

    

    # robot model
    def model(self, v, w, delta_t):
        ''' Robot model: move the robot for one time step 
        
        Args:
            v (m/s) : the linear velocity of this step
            w (m/s) : the angular velocity of this step
            delta_t (s) : the time step for each iteration
        '''
        self.self_x = self.self_x + v * delta_t * math.cos(self.self_theta) + np.random.normal(0, 0.002, 1)[0]
        self.self_y = self.self_y + v * delta_t * math.sin(self.self_theta) + np.random.normal(0, 0.002, 1)[0]
        self.self_theta = self.self_theta + w * delta_t + np.random.normal(0, 0.0001, 1)[0]
        self.self_theta = (self.self_theta / (math.pi * 2) - int(self.self_theta / (math.pi * 2))) * (math.pi * 2)
        if abs(self.self_theta) > math.pi:
            self.self_theta = -(self.self_theta/abs(self.self_theta)) * abs((math.pi * 2) - abs(self.self_theta))


    
    def euclidean_distance(self, goal_position):
        ''' calculate the distance between the goal and the current position

        Args:
            goal_position ([2,5]) : the goal position in [x,y]

        Returns:
            distance (float) : the distance between the current robot position and the goal
        '''
        return math.sqrt((goal_position[0] - self.self_x) ** 2 +
                         (goal_position[1] - self.self_y) ** 2)


    def linear_v(self, goal_position):
        ''' Caculate v command by linear propotional gain * distance '''
        return self.K_linear * self.euclidean_distance(goal_position)

    
    def steering_angle(self, goal_position):
        ''' Remember to use atan2 '''
        return math.atan2(goal_position[1] - self.self_y, goal_position[0] - self.self_x)
    

    def angular_w(self, goal_position):
        ''' Caculate w command
        
        Calculated by angular propotional gain * angle difference 
        Limit the angle within -pi to pi range
        '''
        angle_difference = self.steering_angle(goal_position) - self.self_theta
        angle_difference = (angle_difference / (math.pi * 2) - int(angle_difference / (math.pi * 2))) * (math.pi * 2)
        if abs(angle_difference) > math.pi:
            angle_difference = -(angle_difference/abs(angle_difference)) * abs((math.pi * 2) - abs(angle_difference))

        return self.K_angular * angle_difference



    def move_it(self, goal_position, delta_t = 0.1):
        ''' Move the robot continuously until it reaches the waypoint

        Args:
            goal_position (list e.g. [5,5]) : the goal waypoint
            delta_t (s) : time step for each iteration (default to be 0.001)
        
        Returns:
            self.trajectory (list) : a list of position after each iteration
        '''

        # move the robot when it is out of the tolerance
        while self.euclidean_distance(goal_position) > self.tolerance:

            # set target v and w
            target_v = self.linear_v(goal_position)
            target_w = self.angular_w(goal_position)

            # if the acceleration is larger than limitation, use the largest limitation
            if abs(target_v - self.v)/delta_t > 0.288:
                self.v += (target_v - self.v)/abs(target_v - self.v) * 0.288 * delta_t
            else:
                self.v = target_v

            if abs(target_w - self.w)/delta_t > 5.579:
                self.w += (target_w - self.w)/abs(target_w - self.w) * 5.579 * delta_t
            else:
                self.w = target_w

            # move the robot for one step
            self.model(self.v, self.w, delta_t)

            # append the position after one step movement to the trajectory list
            self.trajectory.append([self.self_x,self.self_y,self.self_theta])

        return self.trajectory


    def move_within_cell(self, goal_position, delta_t = 0.1):
        ''' Move the robot. If it goes out to the neighbor cell, return the cell coordinate

        Args:
            goal_position (list e.g. [5,5]) : the goal waypoint
            delta_t (s) : time step for each iteration (default to be 0.001)
        
        Returns:
            if_reached (bool) : if the robot reached the goal without goes into neighbor cells
            reached_cell (list) : the cell coordinate the robot ends up at
        '''

        # the original cell
        start_cell = [int(list(np.around([self.self_x, self.self_y]))[0]),int(list(np.around([self.self_x, self.self_y]))[1])]
        goal_cell = [int(list(np.around(goal_position))[0]), int(list(np.around(goal_position))[1])]



        # move the robot when it is out of the tolerance
        while self.euclidean_distance(goal_position) > self.tolerance:

            # set target v and w
            target_v = self.linear_v(goal_position)
            target_w = self.angular_w(goal_position)

            # if the acceleration is larger than limitation, use the largest limitation
            if abs(target_v - self.v)/delta_t > 0.288:
                self.v += (target_v - self.v)/abs(target_v - self.v) * 0.288 * delta_t
            else:
                self.v = target_v

            if abs(target_w - self.w)/delta_t > 5.579:
                self.w += (target_w - self.w)/abs(target_w - self.w) * 5.579 * delta_t
            else:
                self.w = target_w

            # move the robot for one step
            self.model(self.v, self.w, delta_t)

            # append the position after one step movement to the trajectory list
            self.trajectory.append([self.self_x,self.self_y,self.self_theta])
            
            # when goes into another cell
            end_cell = [int(list(np.around([self.self_x, self.self_y]))[0]), int(list(np.around([self.self_x, self.self_y]))[1])]
            if end_cell != start_cell and end_cell != goal_cell:
                return False, end_cell

        return True, end_cell


    

if __name__ == '__main__':
    ''' Test only '''
    my_controller = Controller()
    print(my_controller.move_within_cell([0,1]))
    print(my_controller.self_x, my_controller.self_y)
    trajectory = np.array(my_controller.trajectory)
    fig_grid, ax_grid = plt.subplots()

    ax_grid.scatter(trajectory[:,0], trajectory[:,1], color = 'green', \
            s=3, label='path')

    for arrow in range(trajectory.shape[0]):
            # notice this is in plot coordinate, y should be reversed
            ax_grid.arrow(trajectory[arrow,0], trajectory[arrow,1], 0.001*np.cos(trajectory[arrow,2]), \
                0.001*np.sin(trajectory[arrow,2]), color='green')
    ax_grid.set_ylabel('Y axis')
    ax_grid.set_xlabel('X axis')
    ax_grid.set_title('Proportional Controller')
    # ax_grid.set_xlim(-0.03,0.04)
    # ax_grid.set_ylim(-0.04,0.12)
    plt.show()