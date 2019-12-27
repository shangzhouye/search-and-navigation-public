# Created by Shangzhou Ye, MSR, Northwestern
# The script is written in Python 2.7


from __future__ import print_function, division
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
from astar import *
from online_astar import *
from robotcontroller import *


def run_astar(start_pos, goal_pos):
    ''' Run the offline search '''
    my_grid = Grid()
    my_astar = AStar(start_pos,goal_pos)
    my_astar.search_it(obstacles)
    path = my_astar.find_path()
    my_grid.print_grid(path)

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


if __name__ == '__main__':
    ''' A3 '''
    #### Uncomment this to see the results for A3 and A5 ####
    # run_astar([0.5,-1.5], [0.5,1.5])
    # run_astar([4.5,3.5], [4.5,-1.5])
    # run_astar([-0.5,5.5], [1.5,-3.5])

    # run_online_astar([0.5,-1.5], [0.5,1.5])
    # run_online_astar([4.5,3.5], [4.5,-1.5])
    # run_online_astar([-0.5,5.5], [1.5,-3.5])

    #### Comment this to see the results for A3 and A5 ####
    path1, grid1 = run_online_astar([2.45,-3.55], [0.95,-1.55])
    traj1 = waypoints_tracking(path1)
    grid1.print_traj(traj1)
    path2, grid2 = run_online_astar([4.95,-0.05], [2.45,0.25])
    traj2 = waypoints_tracking(path2)
    grid2.print_traj(traj2)
    path3, grid3 = run_online_astar([-0.55,1.45], [1.95,3.95])
    traj3 = waypoints_tracking(path3)
    grid3.print_traj(traj3)

    def run_plan_and_execute(start_pos,goal_pos):
        my_online_astar = OnlineAStar(start_pos,goal_pos)
        path, grid, my_controller = my_online_astar.search_and_execute(start_pos,goal_pos)
        traj = np.array(my_controller.trajectory)
        grid.print_traj(traj)
    
    run_plan_and_execute([2.45,-3.55], [0.95,-1.55])
    run_plan_and_execute([4.95,-0.05], [2.45,0.25])
    run_plan_and_execute([-0.55,1.45], [1.95,3.95])

    run_plan_and_execute([0.5,-1.5], [0.5,1.5])
    run_plan_and_execute([4.5,3.5], [4.5,-1.5])
    run_plan_and_execute([-0.5,5.5], [1.5,-3.5])

    plt.show()