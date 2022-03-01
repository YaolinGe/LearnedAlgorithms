"""
This script tests the rrt* algorithm for collision avoidance
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-22
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from scipy.spatial.distance import cdist
from scipy.stats import mvn
import time
from scipy.interpolate import griddata


#== GP kernel
SIGMA = .08
LATERAL_RANGE = .7
NUGGET = .01
np.random.seed(0)
THRESHOLD = .3
#==


MAXNUM = 1000
XLIM = [0, 1]
YLIM = [0, 1]
GOAL_SAMPLE_RATE = .01
STEP = .15
RADIUS_NEIGHBOUR = .2
DISTANCE_TOLERANCE = .05
# OBSTACLES = [[[.1, .1], [.2, .1], [.2, .2], [.1, .2]],
#              [[.4, .4], [.6, .5], [.5, .6], [.3, .4]],
#              [[.8, .8], [.95, .8], [.95, .95], [.8, .95]]]
# OBSTACLES = [[[.1, .0], [.2, .0], [.2, .5], [.1, .5]],
#              [[.0, .6], [.6, .6], [.6, 1.], [.0, 1.]],
#              [[.8, .0], [1., .0], [1., .9], [.8, .9]],
#              [[.3, .1], [.4, .1], [.4, .6], [.3, .6]],
#              [[.5, .0], [.6, .0], [.6, .4], [.5, .4]]]
# OBSTACLES = [[[1.2, 1.2], [1.4, 1.2], [1.4, 1.4], [1.2, 1.4]]]
OBSTACLES = [[]]


FIGPATH = "/Users/yaoling/OneDrive - NTNU/Self-improvements/LearnedAlgorithms/pathplanning/fig/rrt_star/"

def plotf(grid, values, title):
    x = grid[:, 0]
    y = grid[:, 1]
    nx = 100
    ny = 100
    # plt.figure()
    # plt.scatter(x, y, c=values, cmap="Paired")
    # plt.colorbar()
    # plt.title(title)
    # plt.show()

    xmin, ymin = map(np.amin, [x, y])
    xmax, ymax = map(np.amax, [x, y])

    xv = np.linspace(xmin, xmax, nx)
    yv = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(xv, yv)

    grid_values = griddata(grid, values, (grid_x, grid_y))
    plt.figure()
    plt.scatter(grid_x, grid_y, c=grid_values, cmap="Paired")
    plt.colorbar()
    plt.title(title)
    plt.show()


class GPKernel:

    def __init__(self):
        self.getGrid()
        self.getMean()
        self.getCov()
        self.getGroundTruth()
        self.getEIBVField()
        pass

    def getGrid(self):
        x = np.linspace(XLIM[0], XLIM[1], 25)
        y = np.linspace(YLIM[0], YLIM[1], 25)
        self.x_matrix, self.y_matrix = np.meshgrid(x, y)
        self.grid_x_vector = self.x_matrix.reshape(-1, 1)
        self.grid_y_vector = self.y_matrix.reshape(-1, 1)
        self.grid = np.hstack((self.grid_x_vector, self.grid_y_vector))
        pass

    def setCoef(self):
        self.sigma = SIGMA
        self.eta = 4.5 / LATERAL_RANGE
        self.tau = NUGGET
        self.R = np.diagflat(self.tau ** 2)
        self.threshold = THRESHOLD

    def getMean(self):
        self.mu_prior = (1 - np.exp(- ((self.grid[:, 0] - 1.) ** 2 + (self.grid[:, 1] - .5) ** 2))).reshape(-1, 1)
        self.mu_prior_matrix = np.zeros_like(self.x_matrix)
        for i in range(self.x_matrix.shape[0]):
            for j in range(self.x_matrix.shape[1]):
                self.mu_prior_matrix[i, j] = 1 - np.exp(- ((self.x_matrix[i, j] - 1.) ** 2 +
                                                           (self.y_matrix[i, j] - .5) ** 2))
        # plotf(self.grid, self.mu_prior, "Prior")
        plt.imshow(self.mu_prior_matrix, cmap='Paired')
        plt.colorbar()
        plt.show()

        pass

    def getCov(self):
        self.setCoef()
        DistanceMatrix = cdist(self.grid, self.grid)
        self.Sigma_prior = self.sigma ** 2 * (1 + self.eta * DistanceMatrix) * np.exp(-self.eta * DistanceMatrix)
        pass

    def getGroundTruth(self):
        self.mu_truth = self.mu_prior.reshape(-1, 1) + np.linalg.cholesky(self.Sigma_prior) @ np.random.randn(len(self.mu_prior)).reshape(-1, 1)
        plotf(self.grid, self.mu_truth, "Truth")
        pass

    def getF(self, location):
        x_loc = np.array(location.x).reshape(-1, 1)
        y_loc = np.array(location.y).reshape(-1, 1)

        DM_x = x_loc @ np.ones([1, len(self.grid_x_vector)]) - np.ones([len(x_loc), 1]) @ self.grid_x_vector.T
        DM_y = y_loc @ np.ones([1, len(self.grid_y_vector)]) - np.ones([len(y_loc), 1]) @ self.grid_y_vector.T
        DM = DM_x ** 2 + DM_y ** 2
        ind_F = np.argmin(DM, axis = 1) # interpolated vectorised indices
        # F = np.zeros([1, len(self.grid_x)])
        # F[0, ind_F] = True
        return ind_F

    @staticmethod
    def GPupd(mu, Sigma, F, R, measurement):
        C = F @ Sigma @ F.T + R
        mu = mu + Sigma @ F.T @ np.linalg.solve(C, (measurement - F @ mu))
        Sigma = Sigma - Sigma @ F.T @ np.linalg.solve(C, F @ Sigma)
        return mu, Sigma

    @staticmethod
    def getEIBV(mu, Sigma, F, R, threshold):
        Sigma_updated = Sigma - Sigma @ F.T @ np.linalg.solve(F @ Sigma @ F.T + R, F @ Sigma)
        Variance = np.diag(Sigma_updated).reshape(-1, 1)
        EIBV = 0
        for i in range(mu.shape[0]):
            EIBV += (mvn.mvnun(-np.inf, threshold, mu[i], Variance[i])[0] -
                     mvn.mvnun(-np.inf, threshold, mu[i], Variance[i])[0] ** 2)
        return EIBV

    def getEIBVField(self):
        self.eibv = []
        t1 = time.time()
        for i in range(self.grid.shape[0]):
            print(i)
            F = np.zeros([1, self.grid.shape[0]])
            F[0, i] = True
            self.eibv.append(GPKernel.getEIBV(self.mu_prior, self.Sigma_prior, F, self.R, THRESHOLD))
        self.eibv = np.array(self.eibv)
        self.eibv -= np.amin(self.eibv)
        t2 = time.time()
        print("Time consumed: ", t2 - t1)
        plotf(self.grid, self.eibv, "EIBV")
        pass

gp = GPKernel()

#%%

class Location:

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        pass


class TreeNode:

    def __init__(self, location=None, parent=None, cost=None, mu=None, Sigma=None, F=None, EIBV=None):
        self.location = location
        self.parent = parent
        self.cost = cost
        self.mu = mu
        self.Sigma = Sigma
        self.F = F
        self.EIBV = EIBV
        pass


class RRTConfig:

    def __init__(self, starting_location=None, ending_location=None, goal_sample_rate=None, step=None, GPKernel=None):
        self.starting_location = starting_location
        self.ending_location = ending_location
        self.goal_sample_rate = goal_sample_rate
        self.step = step
        self.GPKernel = GPKernel
        pass


class RRTStar:

    nodes = []
    obstacles = np.array(OBSTACLES)
    polygon_obstacles = []

    def __init__(self, config=None):
        self.config = config
        self.path = []
        self.starting_node = TreeNode(self.config.starting_location, None, 0,
                                      self.config.GPKernel.mu_prior, self.config.GPKernel.Sigma_prior)
        self.ending_node = TreeNode(self.config.ending_location, None, 0)
        self.starting_node.EIBV = 0
        self.setF(self.starting_node)
        self.setF(self.ending_node)
        pass

    def setF(self, node):
        node.F = self.config.GPKernel.getF(node.location)
        # print(node.F)

    def expand_trees(self):
        self.nodes.append(self.starting_node)
        for i in range(MAXNUM):
            print(i)
            if np.random.rand() <= self.config.goal_sample_rate:
                new_location = self.config.ending_location
            else:
                new_location = self.get_new_location()

            nearest_node = self.get_nearest_node(self.nodes, new_location)
            next_node = self.get_next_node(nearest_node, new_location)

            next_node, nearest_node = self.rewire_tree(next_node, nearest_node)

            # if self.iscollided(next_node):
            #     continue

            if self.isarrived(next_node):
                self.ending_node.parent = next_node
                # self.nodes.append(self.ending_node)
                # break
            else:
                self.nodes.append(next_node)
            pass

            # plt.clf()
            # for j in range(self.obstacles.shape[0]):
            #     obstacle = np.append(self.obstacles[j], self.obstacles[j][0, :].reshape(1, -1), axis=0)
            #     plt.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')
            #
            # for node in self.nodes:
            #     if node.parent is not None:
            #         plt.plot([node.location.x, node.parent.location.x],
            #                  [node.location.y, node.parent.location.y], "-g")
            # # path = np.array(self.path)
            # # plt.plot(path[:, 0], path[:, 1], "-r")
            # plt.plot(self.config.starting_location.x, self.config.starting_location.y, 'k*', ms=10)
            # plt.plot(self.config.ending_location.x, self.config.ending_location.y, 'g*', ms=10)
            # plt.grid()
            # plt.savefig(FIGPATH + "P_{:05d}.png".format(i))
            # plt.show()
        pass

    @staticmethod
    def get_new_location():
        x = np.random.uniform(XLIM[0], XLIM[1])
        y = np.random.uniform(YLIM[0], YLIM[1])
        location = Location(x, y)
        return location

    def get_nearest_node(self, nodes, location):
        dist = []
        node_new = TreeNode(location)
        for node in nodes:
            dist.append(self.get_distance_between_nodes(node, node_new))
        return nodes[dist.index(min(dist))]

    @staticmethod
    def get_cost_from_field(location):
        # return 0
        return 2**2 * np.exp(-((location.x - .5) ** 2 + (location.y - .5) ** 2) / .09)

    @staticmethod
    def get_cost_along_path(location1, location2):
        N = 10
        x = np.linspace(location1.x, location2.x, N)
        y = np.linspace(location1.y, location2.y, N)
        cost = []
        for i in range(N):
            cost.append(RRTStar.get_cost_from_field(Location(x[i], y[i])))
        cost_total = np.trapz(cost) / N
        # cost_total = np.sum(cost) / RRTStar.get_distance_between_nodes(TreeNode(location1), TreeNode(location2))
        # print("cost total: ", cost_total)
        return cost_total

    @staticmethod
    def get_distance_between_nodes(node1, node2):
        dist_x = node1.location.x - node2.location.x
        dist_y = node1.location.y - node2.location.y
        dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
        return dist

    def get_next_node(self, node, location):
        node_temp = TreeNode(location)
        if RRTStar.get_distance_between_nodes(node, node_temp) <= self.config.step:
            return TreeNode(location, node)
        else:
            angle = np.math.atan2(location.y - node.location.y, location.x - node.location.x)
            x = node.location.x + self.config.step * np.cos(angle)
            y = node.location.y + self.config.step * np.sin(angle)
            location_next = Location(x, y)
        return TreeNode(location_next, node)

    def rewire_tree(self, node_current, node_nearest):
        ind_neighbour_nodes = self.get_neighbour_nodes(node_current)

        for i in range(len(ind_neighbour_nodes)):
            if self.get_cost_between_nodes(self.nodes[ind_neighbour_nodes[i]], node_current) < \
                    self.get_cost_between_nodes(node_nearest, node_current):
                node_nearest = self.nodes[ind_neighbour_nodes[i]]
        node_current.cost = self.get_cost_between_nodes(node_nearest, node_current)
        node_current.parent = node_nearest

        for i in range(len(ind_neighbour_nodes)):
            cost_current_neighbour = self.get_cost_between_nodes(node_current, self.nodes[ind_neighbour_nodes[i]])
            if cost_current_neighbour < self.nodes[ind_neighbour_nodes[i]].cost:
                self.nodes[ind_neighbour_nodes[i]].cost = cost_current_neighbour
                self.nodes[ind_neighbour_nodes[i]].parent = node_current

        return node_current, node_nearest
        # for i in range(len(self.nodes)):
            # distance_between_nodes = RRTStar.get_distance_between_nodes(self.nodes[i], node_current)

            # if distance_between_nodes <= RADIUS_NEIGHBOUR:
            #     if self.nodes[i].cost + distance_between_nodes < \
            #             node_nearest.cost + RRTStar.get_distance_between_nodes(node_nearest, node_current):
            #         node_nearest = self.nodes[i]
        # node_current.cost = node_nearest.cost + RRTStar.get_distance_between_nodes(node_nearest, node_current)
        # node_current.parent = node_nearest
        # return node_current, node_nearest
        pass

    def get_cost_between_nodes(self, node1, node2):
        cost = node1.cost + RRTStar.get_distance_between_nodes(node1, node2) + \
               self.get_reward_between_nodes(node1, node2)
               # RRTStar.get_cost_along_path(node1.location, node2.location)

        print("Cost: ", cost)
        return cost

    def get_reward_between_nodes(self, node1, node2):
        self.setF(node2)
        node1.EIBV = self.get_eibv_for_node(node1)
        node2.EIBV = self.get_eibv_for_node(node2)
        reward = (node1.EIBV + node2.EIBV) / 2 * RRTStar.get_distance_between_nodes(node1, node2)
        return reward

    def get_eibv_for_node(self, node):
        return self.config.GPKernel.eibv[node.F]

    def get_neighbour_nodes(self, node_current):
        distance_between_nodes = []
        for i in range(len(self.nodes)):
            distance_between_nodes.append(RRTStar.get_distance_between_nodes(self.nodes[i], node_current))
        ind_neighbours = np.where(np.array(distance_between_nodes) <= RADIUS_NEIGHBOUR)[0]
        return ind_neighbours

    def isarrived(self, node):
        dist = self.get_distance_between_nodes(self.ending_node, node)
        if dist < DISTANCE_TOLERANCE:
            return True
        else:
            return False

    def set_obstacles(self):
        for i in range(self.obstacles.shape[0]):
            self.polygon_obstacles.append(Polygon(list(map(tuple, self.obstacles[i]))))

    def iscollided(self, node):
        point = Point(node.location.x, node.location.y)
        line = LineString([(node.parent.location.x, node.parent.location.y),
                           (node.location.x, node.location.y)])
        collision = False
        for i in range(len(self.polygon_obstacles)):
            if self.polygon_obstacles[i].contains(point) or self.polygon_obstacles[i].intersects(line):
                # print("Collision detected")
                collision = True
        return collision

    def get_shortest_path(self):
        self.path.append([self.ending_node.location.x, self.ending_node.location.y])
        pointer_node = self.ending_node
        while pointer_node.parent is not None:

            node = pointer_node.parent
            self.path.append([node.location.x, node.location.y])
            pointer_node = node

        self.path.append([self.starting_node.location.x, self.starting_node.location.y])
        pass

    def plot_tree(self):

        plt.clf()
        for i in range(self.obstacles.shape[0]):
            if i > 0:
                obstacle = np.append(self.obstacles[i], self.obstacles[i][0, :].reshape(1, -1), axis=0)
                plt.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')
            else:
                pass

        for node in self.nodes:
            if node.parent is not None:
                plt.plot([node.location.x, node.parent.location.x],
                         [node.location.y, node.parent.location.y], "-g")
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], "-r")
        plt.plot(self.config.starting_location.x, self.config.starting_location.y, 'k*', ms=10)
        plt.plot(self.config.ending_location.x, self.config.ending_location.y, 'g*', ms=10)

        plt.grid()
        plt.title("rrt_star")
        plt.show()


if __name__ == "__main__":
    starting_loc = Location(.0, .0)
    ending_loc = Location(.0, .99)
    # gp = GPKernel()
    rrtconfig = RRTConfig(starting_location=starting_loc, ending_location=ending_loc, goal_sample_rate=GOAL_SAMPLE_RATE,
                          step=STEP, GPKernel=gp)
    rrt = RRTStar(rrtconfig)
    rrt.set_obstacles()
    rrt.expand_trees()
    rrt.get_shortest_path()
    rrt.plot_tree()
    pass





