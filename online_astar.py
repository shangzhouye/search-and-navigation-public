# Created by Shangzhou Ye, MSR, Northwestern
# The script is written in Python 2.7



from __future__ import print_function, division
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
from astar import *
from robotcontroller import *

# logging for debug
import logging
# logging.basicConfig(filename='test.txt', level=logging.INFO)





class OnlineAStar(object):
    ''' Online Astar algorithms implemented. 
    
    The frontier list can only have eight neghbours of
    the current node.
    '''
    def __init__(self, start_pos, goal_pos):
        ''' Initialize the online search '''
        self.obstacle_memory = np.zeros((nrows, ncols))

        goal_cell_x, goal_cell_y = cart2cell(goal_pos[0], goal_pos[1])
        self.goal_pos = [goal_cell_x, goal_cell_y]
        start_cell_x, start_cell_y = cart2cell(start_pos[0], start_pos[1])
        self.start_cell = [start_cell_x, start_cell_y]

        self.path = [self.start_cell]
        self.current = self.start_cell
        
        logging.info('memory %s', self.obstacle_memory)
        logging.info('goal_pos %s', self.goal_pos)
        logging.info('start_cell %s', self.start_cell)
        logging.info('path %s', self.path)
        logging.info('current %s', self.current)

    

    def online_search_step(self, start_pos, goal_pos):
        ''' One step of online astar search

        cost is based on current memory; when move a step, plan again.
        The process is:
            - initialize an A star search
            - overwrite the initialization of the astar search algorithm
                with current position as the starting point
            - update the memory on known obstacles
            - complete the astar search
            - move the to next point on the planned path (move for one step)
            - repeat the process

        Args:
            start_pos (list e.g. [3,5]) : the start position
            goal_pos (list e.g. [1,5]) : the goal position
        '''
        astar_search = AStar(start_pos, goal_pos)

        # re-initialize it
        # overwrite the initialization of the offline search
        astar_search.start_cell = self.current
        start_h = astar_search.h_cost(astar_search.start_cell)
        astar_search.open = [[astar_search.start_cell, start_h, 0, start_h, [None, None]]]

        logging.info('astar goal_pos %s', astar_search.goal_pos)
        logging.info('astar start_cell %s', astar_search.start_cell)
        logging.info('astar open %s', astar_search.open)
        logging.info('astar closed %s', astar_search.closed)

        # update memory
        step_neighbours = astar_search.find_neighbour(self.current)
        for each in step_neighbours:
                each_c, each_r = cell2rc(each[0], each[1])
                self.obstacle_memory[each_r, each_c] = obstacles[each_r, each_c]

        # search it
        astar_search.search_it(self.obstacle_memory)
        step_path = astar_search.find_path()
        logging.info('step_path %s', step_path)
        self.current = step_path[-2]
        self.path.append(self.current)



    def search_process(self, start_pos, goal_pos):
        ''' The searching process: keep searching until move to the goal

        Args:
            start_pos (list e.g. [3,5]) : the start position
            goal_pos (list e.g. [1,5]) : the goal position

        Returns:
            path (list) : the complete path to the goal 
                (it begins with starting position of ends with goal position)
            grid_plot (Grid) : return the Grid object for future use
        '''
        grid_plot = Grid()

        while True:
            if self.current == self.goal_pos:
                break
            logging.info('current position is %s', self.current)
            self.online_search_step(start_pos, goal_pos)

        grid_plot.print_grid(self.path)

        return self.path, grid_plot




    def search_and_execute(self, start_pos, goal_pos):
        ''' Search and execute simutaniously

        If the robot moves to the neighbor cell during execution
        It should re-plan and goes to the next target cell

        Args:
            start_pos (list e.g. [3,5]) : the start position
            goal_pos (list e.g. [1,5]) : the goal position

        Returns:
            path (list) : the complete path to the goal 
                (it begins with starting position of ends with goal position)
            grid_plot (Grid) : return the Grid object for future use
            my_controller (Controller): return the controller object for future use
        '''
        grid_plot = Grid()
        start_x, start_y = cart2cell(start_pos[0], start_pos[1])
        my_controller = Controller([start_x, start_y, -math.pi/2])

        while True:
            if self.current == self.goal_pos:
                break
            logging.info('current position is %s', self.current)

            self.online_search_step(start_pos, goal_pos)
            if_reached, reached_cell = my_controller.move_within_cell(self.current)
            if if_reached == False:
                # because it did not reach the goal, the last node should be removed from the path
                self.path.pop(-1)
                self.current = reached_cell
                self.path.append(self.current)

        grid_plot.print_grid(self.path)

        return self.path, grid_plot, my_controller



if __name__ == '__main__':
    ''' test only '''

    def run_online_astar(start_pos,goal_pos):
        ''' Run the online search '''
        my_online_astar = OnlineAStar(start_pos,goal_pos)
        return my_online_astar.search_process(start_pos, goal_pos)

    def waypoints_tracking(waypoints):
        ''' Move the robot following a series of waypoints 
        
        Returns:
            trajectory (np.array) : a trajectory of robot positions with time step of 0.001 (by default)
        '''
        my_controller = Controller([waypoints[0][0], waypoints[0][1], -math.pi/2])
        for waypoint in waypoints:
            my_controller.move_it(waypoint)
        trajectory = np.array(my_controller.trajectory)

        return trajectory

    def run_plan_and_execute(start_pos,goal_pos):
        my_online_astar = OnlineAStar(start_pos,goal_pos)
        path, grid, my_controller = my_online_astar.search_and_execute(start_pos,goal_pos)
        traj = np.array(my_controller.trajectory)
        grid.print_traj(traj)

    # path1, grid1 = run_online_astar([2.45,-3.55], [0.95,-1.55])
    # traj1 = waypoints_tracking(path1)
    # grid1.print_traj(traj1)
    run_plan_and_execute([2.45,-3.55], [0.95,-1.55])
    plt.show()
        

