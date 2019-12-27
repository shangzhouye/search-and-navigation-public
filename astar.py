# Created by Shangzhou Ye, MSR, Northwestern
# The script is written in Python 2.7



from __future__ import print_function, division
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
import copy





''' Parameters '''
cart_xlim = [-2, 5]
cart_ylim = [-6, 6]

#### Modify this to change the grid size ####
grid_size = 0.1

# range of x determines no. of columns
ncols = int(math.ceil((cart_xlim[1] - cart_xlim[0]) / grid_size))
nrows = int(math.ceil((cart_ylim[1] - cart_ylim[0]) / grid_size))
obstacles = np.zeros((nrows, ncols))




''' Helper functions

Positions have four representations:
1. cart_x, cart_y: position in the cartesian coordinates;
                    can have negative value;
2. cell_x, cell_y: cell index in the grid with origin at bottom left;
3. cell_c, cell_r: column and row number in the grid;
                    origin at the top left;
4. plot coordinate: when doing scattar plot on the matshow plot,
                    the origin is at the center of the top-left grid
                    it is offset by 0.5 grid from the top-left corner of the plot;
'''
def cart2cell(cart_x, cart_y):
    ''' Transfer cartesian coordinates to cell x, y '''
    return int((cart_x - cart_xlim[0]) / grid_size), int((cart_y - cart_ylim[0]) / grid_size)


def cell2rc(cell_x, cell_y):
    ''' Transfer cell x, y to rows and columns '''
    cell_c = cell_x
    cell_r = nrows - 1 - cell_y
    return cell_c, cell_r


def cart2rc(cart_x, cart_y):
    ''' Transfer cartesian coordinates to rows and columns 
    
    Returns:
        cell_c (int) : the column number in the grid
        cell_r (int) : the row number in the grid
    '''
    cell_x, cell_y = cart2cell(cart_x, cart_y)
    return cell2rc(cell_x, cell_y)


def cart2plot(cart_x,cart_y):
    ''' Transfers the cartesian coordinates to the coordinates that scatter plot needed

    The origin of the scatter plot is at top-left
    x, y in cartesian coordinate -> (2+x)/grid_size, (6-y)/grid_size in plot coordinates

    It is important to notice that the ticks are offset by 0.5 grid
    so do not align the position in plot coordinate with the ticks
    '''

    return (2+cart_x)/grid_size - 0.5, (6-cart_y)/grid_size - 0.5

def cell2plot(cell_x, cell_y):
    cart_x = (cell_x + 0.5) * grid_size + cart_xlim[0]
    cart_y = (cell_y + 0.5) * grid_size + cart_ylim[0]
    return cart2plot(cart_x,cart_y)




class Grid(object):
    ''' Grid used for A* searching algorithm
    
    The origin of the grid is at the left bottom.
    Function cart2rc serves as a tranformation that transfers the
    cartesian coordinates to rows and columns (which
    has its origin at the left top) of the grid.
    '''


    def __init__(self):
        ''' Initialize the position of obstacles '''

        # range of x determines no. of columns
        ncols = (cart_xlim[1] - cart_xlim[0]) / grid_size
        nrows = (cart_ylim[1] - cart_ylim[0]) / grid_size

        self.landmark = np.loadtxt('ds1_Landmark_Groundtruth.dat')
        self.position_obstacles()


    
    def position_obstacles(self):
        ''' Position the obstacles

        This function positions the obstacles into the grid
        based on the information from the landmark.dat. It should 
        be run before the searching.

        x_pos ~ cols of the grid ~ int(x - x0)
        y_pos ~ rows of the grid ~ (nrows - 1) - int(y - y0)
        '''

        #### Uncomment this to see the results without inflation ####
        # for i in range(self.landmark.shape[0]):
        #     col_pos, row_pos = cart2rc(self.landmark[i,1], self.landmark[i,2])
        #     obstacles[row_pos, col_pos] = 1

        #### Comment this to see the results without inflation ####
        for i in range(self.landmark.shape[0]):
            col_pos_l, row_pos_l = cart2rc(self.landmark[i,1] - 0.3, self.landmark[i,2])
            col_pos_r, row_pos_r = cart2rc(self.landmark[i,1] + 0.3, self.landmark[i,2])
            col_pos_t, row_pos_t = cart2rc(self.landmark[i,1], self.landmark[i,2] + 0.3)
            col_pos_b, row_pos_b = cart2rc(self.landmark[i,1], self.landmark[i,2] - 0.3)
            for j in range(col_pos_l, col_pos_r+1):
                for k in range(row_pos_t, row_pos_b+1):
                    obstacles[k, j] = 1
    

    def print_grid(self, path):
        ''' Print the grid using matplotlib 
        
        Args:
            path : a list of path in cell_x and cell_y
        '''

        # make a copy, otherwise it will overwrite the previous path
        path = copy.deepcopy(path)
        for i in range(len(path)):
            path[i][0], path[i][1] = cell2rc(path[i][0], path[i][1])
        # in each point of path, it becomes [colnum, rownum]

        show_path = np.copy(obstacles)
        for i in range(len(path)):
            show_path[path[i][1]][path[i][0]] = 2

        fig_grid, self.ax_grid = plt.subplots()
        self.ax_grid.matshow(show_path)

        '''How ticks are defined

        - the origin of the matrix is at left-top
        - set_xtick is setting at which position have a tick
        - 8 positions should be set in x direction (horizontal)
        - set_xticklabels is asking what label to set at each tick
        - the sequence should be same as the increasing direction of the axis
        '''
        ticks_x = np.arange(0,ncols+0.0001,ncols/7)
        ticks_y = np.arange(0,nrows+0.0001,nrows/12)
        self.ax_grid.set_xticks(ticks_x)
        self.ax_grid.set_xticklabels(range(-2,6))
        self.ax_grid.set_yticks(ticks_y)
        self.ax_grid.set_yticklabels(range(6,-7,-1))
        self.ax_grid.tick_params(top=False, bottom=True, \
                            labeltop=False, labelbottom=True)
        
        self.ax_grid.set_title('A* Grid')

    
    def print_traj(self, traj):
        ''' print the trajectory on the existing grid '''
        traj_plot = np.copy(traj)
        for i in range(traj.shape[0]):
            traj_plot[i,0:2] = cell2plot(traj[i,0], traj[i,1])
        
        self.ax_grid.scatter(traj_plot[:,0], traj_plot[:,1], color = 'red', \
            s=0.5, label='path')
        for arrow in range(traj_plot.shape[0]):
            # notice this is in plot coordinate, y should be reversed
            self.ax_grid.arrow(traj_plot[arrow,0], traj_plot[arrow,1], 2*np.cos(traj_plot[arrow,2]), \
                -2*np.sin(traj_plot[arrow,2]), color='red', head_width=0.01)
        self.ax_grid.set_ylabel('Y axis')
        self.ax_grid.set_xlabel('X axis')
        # self.ax_grid.legend()






class AStar(object):
    ''' Astar algorithms implemented. '''
    def __init__(self, start_pos, goal_pos):
        ''' Initialization

        cart2cell serves as an interface function.
        start_pos, goal_pos are in cartesian coordinates
        others in the algorithm are by cell_x and cell_y.
        
        Args:
            start_pos (list e.g. [3,5]) : the start position
            goal_pos (list e.g. [1,5]) : the goal position
        '''
        goal_cell_x, goal_cell_y = cart2cell(goal_pos[0], goal_pos[1])
        self.goal_pos = [goal_cell_x, goal_cell_y]
        # each element in open list has the form: [pos, F, G, H, parent]
        # pos and parent is a list e.g. [5, 5]
        start_cell_x, start_cell_y = cart2cell(start_pos[0], start_pos[1])
        self.start_cell = [start_cell_x, start_cell_y]
        start_h = self.h_cost(self.start_cell)
        self.open = [[self.start_cell, start_h, 0, start_h, [None, None]]]
        self.closed = []


    def lowest_f(self):
        ''' Find the position with lowest F value 
        
        Return:
            lowest_pos (list) : position with lowest F
            lowest_idx (int) : index (row) in the open list
        '''
        lowest_idx = 0
        lowest_cost = self.open[0][1]
        for i in range(len(self.open)):
            if self.open[i][1] < lowest_cost:
                lowest_idx = i
                lowest_cost = self.open[i][1]
        return self.open[lowest_idx][0], lowest_idx


    def h_cost(self, cell_pos):
        ''' Calculate the herostic 
        
        The herostic used is the straight-line distance between the current points
        and the goal position.
        '''
        return math.sqrt((cell_pos[1] - self.goal_pos[1]) ** 2 + \
                            (cell_pos[0] - self.goal_pos[0]) ** 2)


    def find_neighbour(self, center):
        '''  Find neighbours.

        If it is in closed list or out of the range of the grid,
        ignore it.

        Args:
            center (list) : the center position
        '''
        neighbours = [[center[0]-1, center[1]-1], \
                      [center[0], center[1]-1], \
                      [center[0]+1, center[1]-1], \
                      [center[0]-1, center[1]], \
                      [center[0]+1, center[1]], \
                      [center[0]-1, center[1]+1], \
                      [center[0], center[1]+1], \
                      [center[0]+1, center[1]+1]]

        list_to_cut = []
        for member in neighbours:
            # debug finding: if using list.remove removed 
            # the current member, next member will skip one index
            if member[0] < 0 or member[0] > (ncols - 1) \
                    or member[1] < 0 or member[1] > (nrows - 1):
                list_to_cut.append(member)
                continue

            for j in range(len(self.closed)):
                if member == self.closed[j][0]:
                    list_to_cut.append(member)

        for to_cut in list_to_cut:
            neighbours.remove(to_cut)
 
        return neighbours

    

    def search_it(self, obstacles):
        ''' Do the search 
        
        Args:
            obstacles (array) : an array of the position of obstacles.
            position with obstacles have value of 1.
            This variable can be the ground truth obstacles (off-line search) or
            the internal memory of obstacles (online search).
        '''
        while True:
            # find the node in open with lowest F cost
            self.current_pos, self.current_idx = self.lowest_f()

            # delete current from open and add it to closed
            self.closed.append(self.open.pop(self.current_idx))

            if self.current_pos == self.goal_pos:
                break

            self.neighbours = self.find_neighbour(self.current_pos)
            for each in self.neighbours:
                each_c, each_r = cell2rc(each[0], each[1])
                if obstacles[each_r, each_c] == 1:
                    g_step = self.closed[-1][2] + 1000
                # for testing different cost functions
                # elif abs(each[0]-self.current_pos[0]) + abs(each[1]-self.current_pos[1]) == 2:
                #     g_step = self.closed[-1][2] + 2
                else:
                    g_step = self.closed[-1][2] + 1
                h_step = self.h_cost(each)
                f_step = g_step + h_step
                info_step = [each, f_step, g_step, h_step, self.closed[-1][0]]
                for i in range(len(self.open)):
                    if self.open[i][0] == each:
                        if self.open[i][1] > f_step:
                            self.open[i] = info_step
                        break
                else:
                    self.open.append(info_step)



    def find_path(self):
        ''' Find the path after the search 
        
        Returns:
            path (list e.g. [[2,5], [3,6]]) : a list of points. It starts with the 
            goal position and ends with the starting position. (in cell_x, y)
        '''
        current = self.goal_pos
        path = []
        while True:
            if current == self.start_cell:
                path.append(current)
                break
            for i in self.open:
                if i[0] == current:
                    current = i[4]
                    path.append(i[0])
                    break
            else:
                for j in self.closed:
                    if j[0] == current:
                        current = j[4]
                        path.append(j[0])
                        break
        
        return path
        




if __name__ == '__main__':
    ''' Only for testing '''
    start_pos = [-0.5,5.5]
    goal_pos = [1.5,-3.5]
    my_grid = Grid()
    my_astar = AStar(start_pos,goal_pos)
    my_astar.search_it(obstacles)
    path = my_astar.find_path()
    my_grid.print_grid(path)

    plt.show()
    

